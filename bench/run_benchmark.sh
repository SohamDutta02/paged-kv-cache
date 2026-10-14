#!/usr/bin/env bash
# Reproduces docs/benchmarks.md: builds paged-kv-serve, serves a model, load
# tests it, and writes results, charts and tables to $OUT.
#
#   bench/run_benchmark.sh                     # Qwen2.5-0.5B's architecture, random weights
#   MODEL=path/to/checkpoint bench/run_benchmark.sh
#   QUICK=1 bench/run_benchmark.sh             # tiny model, seconds; what CI runs
set -euo pipefail
cd "$(dirname "$0")/.."

OUT=${OUT:-bench/results}
PORT=${PORT:-8099}
PYTHON=${PYTHON:-python3}
mkdir -p "$OUT"

if [ -n "${QUICK:-}" ]; then
  CONFIG=bench/tiny/config.json
  SWEEP=(--concurrency 1 4 8 --requests 16 --prompt-len 8 32 --output-len 16 --vocab 512)
  PREFIX=(--concurrency 4 --requests 16 --shared-prefix 64 --prompt-len 4 12 --output-len 8 --vocab 512)
else
  CONFIG=bench/qwen2.5-0.5b/config.json
  SWEEP=(--concurrency 1 2 4 8 16 --prompt-len 32 128 --output-len 32)
  PREFIX=(--concurrency 8 --requests 24 --shared-prefix 256 --prompt-len 16 48 --output-len 16)
fi

SERVE_ARGS=()
if [ -z "${MODEL:-}" ]; then
  MODEL=$(mktemp -d)
  trap 'rm -rf "$MODEL"' EXIT
  cp "$CONFIG" "$MODEL/config.json"
  "$PYTHON" bench/make_tokenizer.py "$MODEL"
  SERVE_ARGS+=(--dummy-weights)
fi

cargo build --release -p paged-kv-serve
SERVER=./target/release/paged-kv-serve
PID=

start_server() {
  "$SERVER" --model "$MODEL" "${SERVE_ARGS[@]}" --port "$PORT" \
    --num-blocks 1024 --block-size 16 --max-batch-size 16 "$@" >>"$OUT/server.log" 2>&1 &
  PID=$!
  for _ in $(seq 1 120); do
    curl -sf "http://localhost:$PORT/health" >/dev/null && return
    kill -0 "$PID" 2>/dev/null || { cat "$OUT/server.log"; exit 1; }
    sleep 1
  done
  echo "the server did not start" >&2
  exit 1
}

stop_server() {
  kill "$PID"
  wait "$PID" 2>/dev/null || true
}

run() {
  "$PYTHON" bench/load_test.py --url "http://localhost:$PORT" --server-pid "$PID" "$@"
}

echo "== concurrency sweep"
start_server
run "${SWEEP[@]}" --label "Concurrency sweep" --out "$OUT/concurrency.json"
stop_server

echo "== shared system prompt, prefix caching on"
start_server
run "${PREFIX[@]}" --label "Prefix caching on" --out "$OUT/prefix-on.json"
stop_server

echo "== shared system prompt, prefix caching off"
start_server --no-prefix-caching
run "${PREFIX[@]}" --label "Prefix caching off" --out "$OUT/prefix-off.json"
stop_server

"$PYTHON" bench/report.py "$OUT"
