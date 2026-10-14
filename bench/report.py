"""Turns load-test results into charts and Markdown tables.

    python report.py <results-dir>

Reads concurrency.json, prefix-on.json and prefix-off.json, and writes PNG
charts in light and dark versions plus summary.md. Exits non-zero if a request
failed or batching or prefix caching stopped working, so CI can gate on it.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "muted": "#52514e",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#f0efec",
        "muted": "#c3c2b7",
        "grid": "#383835",
        "series": ["#3987e5", "#d95926"],
    },
}


def load(directory, name):
    path = directory / name
    return json.loads(path.read_text()) if path.exists() else None


def style(ax, theme):
    ax.set_facecolor(theme["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(theme["grid"])
    ax.tick_params(colors=theme["muted"], length=0, labelsize=9)
    ax.grid(axis="y", color=theme["grid"], linewidth=1)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(theme["muted"])
    ax.yaxis.label.set_color(theme["muted"])


def figure(theme, ncols=1, width=6.4, height=3.4):
    fig, axes = plt.subplots(1, ncols, figsize=(width, height), dpi=150, squeeze=False)
    fig.patch.set_facecolor(theme["surface"])
    for ax in axes[0]:
        style(ax, theme)
    return fig, axes[0]


def line(ax, theme, x, y, color, label, label_end=True, fmt="{:.2f}"):
    ax.plot(
        x,
        y,
        color=color,
        linewidth=2,
        marker="o",
        markersize=6,
        markeredgecolor=theme["surface"],
        markeredgewidth=2,
        label=label,
    )
    if label_end:
        ax.annotate(
            f"{label} {fmt.format(y[-1])}",
            (x[-1], y[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=9,
            color=theme["text"],
        )


def concurrency_axis(ax, levels):
    ax.set_xscale("log", base=2)
    ax.set_xticks(levels, [str(c) for c in levels])
    ax.minorticks_off()
    ax.set_xlabel("Concurrent clients")
    ax.margins(x=0.08)


def throughput_chart(sweep, theme):
    levels = [level["concurrency"] for level in sweep["levels"]]
    tps = [level["output_tokens_per_s"] for level in sweep["levels"]]
    fig, (ax,) = figure(theme)
    line(ax, theme, levels, tps, theme["series"][0], "tokens/s", label_end=False)
    for i in (0, len(levels) - 1):
        ax.annotate(
            f"{tps[i]:.1f}",
            (levels[i], tps[i]),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            color=theme["text"],
        )
    concurrency_axis(ax, levels)
    ax.set_ylim(0, max(tps) * 1.2)
    ax.set_title("Output tokens per second", loc="left", fontsize=11, color=theme["text"])
    fig.tight_layout()
    return fig


def latency_chart(sweep, theme):
    levels = [level["concurrency"] for level in sweep["levels"]]
    fig, axes = figure(theme, ncols=2, width=9.6)
    for ax, key, title in (
        (axes[0], "ttft_s", "Time to first token (s)"),
        (axes[1], "inter_token_s", "Time between tokens (s)"),
    ):
        series = {pct: [level[key][pct] for level in sweep["levels"]] for pct in ("p50", "p99")}
        # Lines that end close together share one spot: the legend names them.
        ends = [v[-1] for v in series.values()]
        apart = abs(ends[0] - ends[1]) > 0.08 * max(max(v) for v in series.values())
        for color, (pct, values) in zip(theme["series"], series.items()):
            line(ax, theme, levels, values, color, pct, label_end=apart)
        concurrency_axis(ax, levels)
        ax.set_ylim(bottom=0)
        ax.set_title(title, loc="left", fontsize=11, color=theme["text"])
        ax.margins(x=0.25)
        legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
        for text in legend.get_texts():
            text.set_color(theme["text"])
    fig.tight_layout()
    return fig


def prefix_chart(on, off, theme):
    (l_on,), (l_off,) = on["levels"], off["levels"]
    groups = ["p50", "p99"]
    # Two bands in about 300px of plot: 0.16 keeps each bar near 24px.
    fig, (ax,) = figure(theme, width=6.4, height=2.9)
    height, gap = 0.16, 0.02
    for offset, (level, label, color) in enumerate(
        ((l_off, "Prefix caching off", theme["series"][1]), (l_on, "Prefix caching on", theme["series"][0]))
    ):
        values = [level["ttft_s"][g] for g in groups]
        ys = [i + (offset - 0.5) * (height + gap) for i in range(len(groups))]
        ax.barh(ys, values, height=height, color=color, label=label)
        for y, v in zip(ys, values):
            ax.annotate(
                f"{v:.2f} s",
                (v, y),
                xytext=(6, 0),
                textcoords="offset points",
                va="center",
                fontsize=9,
                color=theme["text"],
            )
    ax.set_yticks(range(len(groups)), [f"TTFT {g}" for g in groups])
    ax.invert_yaxis()
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", color=theme["grid"], linewidth=1)
    ax.set_xlim(0, max(l_off["ttft_s"]["p99"], l_on["ttft_s"]["p99"]) * 1.25)
    ax.set_xlabel("Seconds")
    ax.set_title(
        "Time to first token with a shared system prompt",
        loc="left",
        fontsize=11,
        color=theme["text"],
        pad=26,
    )
    legend = ax.legend(
        frameon=False,
        fontsize=9,
        loc="lower left",
        bbox_to_anchor=(0, 1.0),
        ncol=2,
        borderaxespad=0.2,
        handlelength=1.2,
    )
    for text in legend.get_texts():
        text.set_color(theme["text"])
    fig.tight_layout()
    return fig


def tables(sweep, on, off):
    lines = []
    if sweep:
        lines += [
            "| Clients | Requests | Tokens/s | Requests/s | TTFT p50 | TTFT p99 "
            "| Between tokens p50 | p99 | Latency p99 | Mean batch | Errors |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for level in sweep["levels"]:
            ttft, gap, server = level["ttft_s"], level["inter_token_s"], level["server"]
            lines.append(
                f"| {level['concurrency']} | {level['requests']} | {level['output_tokens_per_s']:.1f} "
                f"| {level['requests_per_s']:.2f} | {ttft['p50']:.2f} s | {ttft['p99']:.2f} s "
                f"| {gap['p50']:.2f} s | {gap['p99']:.2f} s | {level['latency_s']['p99']:.1f} s "
                f"| {server['mean_batch_size']:.1f} | {level['errors']} |"
            )
        lines.append("")
    if on and off:
        lines += [
            "| Prefix caching | TTFT p50 | TTFT p99 | Latency p50 "
            "| Prompt tokens computed | From cache | Tokens/s | Errors |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for label, run in (("Off", off), ("On", on)):
            (level,) = run["levels"]
            s = level["server"]
            lines.append(
                f"| {label} | {level['ttft_s']['p50']:.2f} s | {level['ttft_s']['p99']:.2f} s "
                f"| {level['latency_s']['p50']:.1f} s "
                f"| {s['prefill_tokens_computed']:.0f} | {s['prefill_tokens_cached']:.0f} "
                f"| {level['output_tokens_per_s']:.1f} | {level['errors']} |"
            )
        lines.append("")
    if sweep:
        peak = max((level["server"]["peak_rss_mib"] or 0) for level in sweep["levels"])
        kv = max(level["server"]["peak_kv_utilization"] for level in sweep["levels"])
        lines.append(f"Peak server memory (RSS): {peak:.0f} MiB. Peak KV cache use: {kv:.1%} of the pool.")
    return "\n".join(lines) + "\n"


def main():
    directory = Path(sys.argv[1])
    sweep = load(directory, "concurrency.json")
    on, off = load(directory, "prefix-on.json"), load(directory, "prefix-off.json")

    for mode, theme in THEMES.items():
        charts = []
        if sweep:
            charts += [
                ("throughput", throughput_chart(sweep, theme)),
                ("latency", latency_chart(sweep, theme)),
            ]
        if on and off:
            charts.append(("prefix-ttft", prefix_chart(on, off, theme)))
        for name, fig in charts:
            fig.savefig(directory / f"{name}-{mode}.png", facecolor=theme["surface"])
            plt.close(fig)

    (directory / "summary.md").write_text(tables(sweep, on, off))
    print((directory / "summary.md").read_text())

    problems = regressions(sweep, on, off)
    for problem in problems:
        print(f"FAIL: {problem}", file=sys.stderr)
    if problems:
        sys.exit(1)


def regressions(sweep, on, off):
    """Behaviour that must hold on any hardware; timings are too noisy to gate on."""
    problems = []
    failed = sum(level["errors"] for run in (sweep, on, off) if run for level in run["levels"])
    if failed:
        problems.append(f"{failed} requests failed")
    if sweep:
        top = max(sweep["levels"], key=lambda level: level["concurrency"])
        if top["concurrency"] >= 4 and top["server"]["mean_batch_size"] < 2:
            problems.append(
                f"{top['concurrency']} clients averaged a batch of {top['server']['mean_batch_size']:.1f}: "
                "concurrent requests are not being batched"
            )
    if on and off:
        computed_on = on["levels"][0]["server"]["prefill_tokens_computed"]
        computed_off = off["levels"][0]["server"]["prefill_tokens_computed"]
        if computed_on > 0.5 * computed_off:
            problems.append(
                f"prefix caching computed {computed_on:.0f} prompt tokens against {computed_off:.0f} "
                "without it: expected at least a halving"
            )
    return problems


if __name__ == "__main__":
    main()
