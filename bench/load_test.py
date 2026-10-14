"""Closed-loop load test for an OpenAI-compatible completions endpoint.

For each concurrency level C, C clients send streaming requests back to back
until the level's requests are done. Per request it records the time to the
first token, the gaps between tokens and the total latency; while the level
runs it samples the server's /metrics and, given its PID, its memory.

    python load_test.py --url http://localhost:8000 --concurrency 1 4 16 --out results.json
"""

import argparse
import asyncio
import json
import random
import re
import time
from pathlib import Path

import httpx
import numpy as np


def workload(args, count, seed):
    """Token-id prompts: an optional shared prefix, then a random tail."""
    rng = random.Random(seed)
    shared = [rng.randrange(args.vocab) for _ in range(args.shared_prefix)]
    lo, hi = args.prompt_len
    return [shared + [rng.randrange(args.vocab) for _ in range(rng.randint(lo, hi))] for _ in range(count)]


async def one_request(client, url, prompt, max_tokens):
    start = time.perf_counter()
    token_times = []
    body = {"model": "bench", "prompt": prompt, "max_tokens": max_tokens, "stream": True}
    async with client.stream("POST", f"{url}/v1/completions", json=body) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            if "error" in chunk:
                raise RuntimeError(chunk["error"]["message"])
            if chunk["choices"][0]["text"]:
                token_times.append(time.perf_counter())
    end = time.perf_counter()
    return {
        "prompt_tokens": len(prompt),
        "output_tokens": len(token_times),
        "ttft": token_times[0] - start,
        "itl": list(np.diff(token_times)),
        "latency": end - start,
    }


def parse_metrics(text):
    """Every unlabelled or labelled sample, keyed by its full name."""
    values = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        name, _, value = line.rpartition(" ")
        try:
            values[name] = float(value)
        except ValueError:
            pass
    return values


def rss_mib(pid):
    try:
        status = Path(f"/proc/{pid}/status").read_text()
    except OSError:
        return None
    match = re.search(r"VmRSS:\s+(\d+) kB", status)
    return int(match.group(1)) / 1024 if match else None


async def sample(client, url, pid, samples, stop):
    while not stop.is_set():
        metrics = parse_metrics((await client.get(f"{url}/metrics")).text)
        samples.append(
            {
                "kv_utilization": metrics.get("paged_kv_kv_cache_utilization", 0.0),
                "running": metrics.get("paged_kv_running_requests", 0.0),
                "waiting": metrics.get("paged_kv_waiting_requests", 0.0),
                "rss_mib": rss_mib(pid) if pid else None,
            }
        )
        try:
            await asyncio.wait_for(stop.wait(), 0.25)
        except asyncio.TimeoutError:
            pass


def percentiles(values):
    if not values:
        return None
    a = np.asarray(values)
    return {
        "mean": float(a.mean()),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
    }


async def run_level(args, concurrency, prompts):
    limits = httpx.Limits(max_connections=concurrency + 4)
    async with httpx.AsyncClient(timeout=None, limits=limits) as client:
        # Warm up so the first measured request doesn't pay one-time costs.
        await one_request(client, args.url, prompts[0][:8], 2)
        before = parse_metrics((await client.get(f"{args.url}/metrics")).text)

        queue = asyncio.Queue()
        for p in prompts:
            queue.put_nowait(p)
        results, errors = [], []
        samples, stop = [], asyncio.Event()
        sampler = asyncio.create_task(sample(client, args.url, args.server_pid, samples, stop))

        async def client_loop():
            while not queue.empty():
                prompt = queue.get_nowait()
                try:
                    results.append(await one_request(client, args.url, prompt, args.output_len))
                except Exception as e:  # recorded, not fatal: the report counts them
                    errors.append(str(e))

        started = time.perf_counter()
        await asyncio.gather(*(client_loop() for _ in range(concurrency)))
        duration = time.perf_counter() - started
        stop.set()
        await sampler
        after = parse_metrics((await client.get(f"{args.url}/metrics")).text)

    def delta(name):
        return after.get(name, 0.0) - before.get(name, 0.0)

    steps = delta("paged_kv_batch_size_count")
    cached = delta("paged_kv_prefill_tokens_cached_total")
    computed = delta("paged_kv_prefill_tokens_computed_total")
    rss = [s["rss_mib"] for s in samples if s["rss_mib"] is not None]
    output_tokens = sum(r["output_tokens"] for r in results)
    return {
        "concurrency": concurrency,
        "requests": len(results),
        "errors": len(errors),
        "error_examples": errors[:3],
        "duration_s": duration,
        "requests_per_s": len(results) / duration,
        "output_tokens_per_s": output_tokens / duration,
        "ttft_s": percentiles([r["ttft"] for r in results]),
        "inter_token_s": percentiles([gap for r in results for gap in r["itl"]]),
        "latency_s": percentiles([r["latency"] for r in results]),
        "server": {
            "mean_batch_size": delta("paged_kv_batch_size_sum") / steps if steps else 0.0,
            "steps": steps,
            "prefill_tokens_cached": cached,
            "prefill_tokens_computed": computed,
            "prefill_cache_hit_rate": cached / (cached + computed) if cached + computed else 0.0,
            "preemptions": delta("paged_kv_preemptions_total"),
            "peak_kv_utilization": max((s["kv_utilization"] for s in samples), default=0.0),
            "peak_waiting": max((s["waiting"] for s in samples), default=0.0),
            "peak_rss_mib": max(rss) if rss else None,
        },
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4, 8, 16])
    parser.add_argument(
        "--requests",
        type=int,
        default=0,
        help="requests per level; default 4 per client, at least 8",
    )
    parser.add_argument("--prompt-len", type=int, nargs=2, default=[32, 128], metavar=("MIN", "MAX"))
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--shared-prefix", type=int, default=0, help="tokens every prompt starts with")
    parser.add_argument("--vocab", type=int, default=151936)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--server-pid", type=int, help="to sample the server's memory")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    levels = []
    for concurrency in args.concurrency:
        count = args.requests or max(8, 4 * concurrency)
        prompts = workload(args, count, args.seed + concurrency)
        level = await run_level(args, concurrency, prompts)
        levels.append(level)
        ttft = level["ttft_s"] or {}
        print(
            f"c={concurrency:>3}  {level['requests']:>3} ok {level['errors']} errors  "
            f"{level['output_tokens_per_s']:7.1f} tok/s  "
            f"TTFT p50 {ttft.get('p50', 0):6.2f}s p99 {ttft.get('p99', 0):6.2f}s  "
            f"batch {level['server']['mean_batch_size']:4.1f}",
            flush=True,
        )

    config = {k: v for k, v in vars(args).items() if k not in ("out", "server_pid")}
    Path(args.out).write_text(json.dumps({"config": config, "levels": levels}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
