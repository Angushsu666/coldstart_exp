#!/usr/bin/env python3
"""Generate all figures from results/*.json and results/*.jsonl.

Usage: python scripts/make_figures.py
Outputs: figures/fig1_warmlevel.png, fig2_breakdown.png, fig3_nccl.png, fig4_cost.png
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

COLORS = {
    "scale-to-zero": "#d62728",
    "container-warm": "#ff7f0e",
    "gpu-warm": "#2ca02c",
}

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 130,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def load_warmlevel(name: str) -> dict:
    path = RESULTS / f"final_{name.replace('-', '_')}.json"
    with open(path) as f:
        data = json.load(f)
    return data["warm_levels"][name]


def fig1_warmlevel_comparison():
    """Hero figure: cold mean + p95 across three strategies."""
    levels = ["scale-to-zero", "container-warm", "gpu-warm"]
    means, p95s = [], []
    for lv in levels:
        d = load_warmlevel(lv)["cold_start"]["wall_time_ms"]
        means.append(d["mean"])
        p95s.append(d["p95"])

    x = np.arange(len(levels))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    bars_mean = ax.bar(x - width/2, means, width, label="Cold mean",
                      color=[COLORS[l] for l in levels], alpha=0.7)
    bars_p95 = ax.bar(x + width/2, p95s, width, label="Cold p95",
                     color=[COLORS[l] for l in levels], hatch="//", edgecolor="white")

    for bar, v in zip(bars_mean, means):
        ax.text(bar.get_x() + bar.get_width()/2, v + 60, f"{v:.0f}",
                ha="center", fontsize=9)
    for bar, v in zip(bars_p95, p95s):
        ax.text(bar.get_x() + bar.get_width()/2, v + 60, f"{v:.0f}",
                ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel("Cold start latency (ms)")
    ax.set_title("GPU-warm cuts p95 cold start 8.1× vs scale-to-zero\n(GKE, Tesla T4, DistilGPT-2, n=20)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    out = FIGURES / "fig1_warmlevel.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def fig2_component_breakdown():
    """Stacked bars showing component decomposition."""
    levels = ["scale-to-zero", "container-warm", "gpu-warm"]
    components = [
        ("t_sched_container_ms", "Sched + container", "#7f7f7f"),
        ("t_model_ms",           "Model load (T_model)", "#1f77b4"),
        ("t_cuda_ms",            "CUDA init (T_cuda)", "#9467bd"),
        ("t_to_gpu_ms",          "Model→GPU", "#8c564b"),
        ("t_warmup_ms",          "GPU warmup", "#e377c2"),
        ("t_infer_ms",           "Inference (T_infer)", "#17becf"),
    ]

    bottoms = np.zeros(len(levels))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    for key, label, color in components:
        vals = []
        for lv in levels:
            d = load_warmlevel(lv)["cold_start"]
            if key == "t_sched_container_ms":
                v = d["component_breakdown_ms"]["t_sched_container_ms"]["mean"]
            else:
                v = d["component_breakdown_ms"][key]["mean"]
            vals.append(v)
        vals = np.array(vals)
        ax.bar(levels, vals, bottom=bottoms, label=label, color=color, edgecolor="white", linewidth=0.5)
        for i, v in enumerate(vals):
            if v >= 80:
                ax.text(i, bottoms[i] + v/2, f"{v:.0f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
        bottoms += vals

    totals = bottoms
    for i, t in enumerate(totals):
        ax.text(i, t + 60, f"{t:.0f} ms", ha="center", fontsize=10, fontweight="bold")

    ax.set_ylabel("Cold start latency (ms)")
    ax.set_title("All strategies perform the same work — they differ only in *when*\n(GKE, Tesla T4, DistilGPT-2, n=20)")
    ax.legend(loc="upper right", framealpha=0.95)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_ylim(0, max(totals) * 1.15)

    out = FIGURES / "fig2_breakdown.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def fig3_nccl():
    """NCCL cold vs warm comparison."""
    rows = [json.loads(l) for l in open(RESULTS / "nccl.jsonl")]
    vals = [r["nccl_init_ms"] for r in rows]
    cold = vals[0]
    warm = vals[1:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.0), gridspec_kw={"width_ratios": [1, 1.4]})

    runs = list(range(1, len(vals) + 1))
    colors = ["#d62728"] + ["#2ca02c"] * len(warm)
    ax1.bar(runs, vals, color=colors)
    ax1.axhline(np.mean(warm), color="#2ca02c", ls="--", alpha=0.6, label=f"Warm mean ({np.mean(warm):.0f} ms)")
    ax1.set_xlabel("Trial #")
    ax1.set_ylabel("NCCL init time (ms)")
    ax1.set_title("Per-trial NCCL init")
    ax1.set_xticks([1, 5, 10, 15, 20])
    ax1.legend(loc="upper right")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_axisbelow(True)
    ax1.annotate("Cold driver init", xy=(1, cold), xytext=(4.5, cold - 200),
                 fontsize=9, color="#d62728",
                 arrowprops=dict(arrowstyle="->", color="#d62728", alpha=0.7))

    bp = ax2.boxplot([[cold], warm], tick_labels=["Cold\n(run 1)", f"Warm\n(runs 2–{len(vals)})"],
                     patch_artist=True, widths=0.5, showfliers=True)
    for patch, color in zip(bp["boxes"], ["#d62728", "#2ca02c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax2.set_ylabel("NCCL init time (ms)")
    ax2.set_title("Cold vs warm distribution")
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    ratio = cold / np.mean(warm)
    ax2.text(0.5, 0.95, f"Cold is {ratio:.1f}× slower than warm",
             transform=ax2.transAxes, ha="left", va="top",
             fontsize=10, fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff3cd", edgecolor="#856404"))

    fig.suptitle("First NCCL communicator init is 4.5× slower than steady-state\n(CURC Alpine, 2×A100 80GB PCIe, world_size=2, n=20)",
                 fontsize=12, y=1.02)

    out = FIGURES / "fig3_nccl.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


def fig4_cost_pareto():
    """Cost vs latency tradeoff as a function of request rate."""
    # GCP T4 cost roughly $0.35/hr — paper section 4.8
    gpu_hourly_usd = 0.35
    seconds_per_hour = 3600

    qps = np.logspace(-2, 2, 200)  # 0.01 to 100 req/s

    # cost per request, USD
    # scale-to-zero: only pay for time pod is alive serving the request
    # approximation: each cold request keeps pod alive ~ cold_mean + steady_ttl seconds
    # then scales back. We approximate as cold_mean ms of GPU per request.
    cold_mean = {"scale-to-zero": 3.873, "container-warm": 1.300, "gpu-warm": 0.475}
    steady = {"scale-to-zero": 0.389, "container-warm": 0.384, "gpu-warm": 0.383}

    # for warm strategies: pay $gpu_hourly_usd / hour regardless
    # so cost/request = $gpu_hourly_usd / (qps * 3600)
    # for scale-to-zero: pay only during request (cold or steady)
    #   ratio of cold requests is hard to model. Assume 1 cold per minute of idle.
    #   Simplification: cost/req = qps * steady_sec * cost/sec + 1cold_per_min penalty
    cost_per_sec = gpu_hourly_usd / seconds_per_hour

    # scale-to-zero: amortize one cold start per minute of idle + steady cost per req
    #   cold/min = 1 if qps < 1/60 else qps (i.e., every request is a cold start if rare)
    # Simplified: assume one cold start triggered per ~60s of activity gap
    cold_per_hour = np.minimum(qps * 3600, 60)  # cap at 60 cold/hour
    sz_cost_per_hour = cold_per_hour * cold_mean["scale-to-zero"] * cost_per_sec \
                       + qps * 3600 * steady["scale-to-zero"] * cost_per_sec
    sz_cost_per_req = sz_cost_per_hour / (qps * 3600)

    # warm strategies: pod is alive 24/7, cost = $0.35/hr
    cw_cost_per_req = gpu_hourly_usd / (qps * 3600)
    gw_cost_per_req = gpu_hourly_usd / (qps * 3600)

    # multiply by 1M to get $/1M req
    sz = sz_cost_per_req * 1e6
    cw = cw_cost_per_req * 1e6
    gw = gw_cost_per_req * 1e6

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.loglog(qps, sz, label="scale-to-zero", color=COLORS["scale-to-zero"], lw=2)
    ax1.loglog(qps, cw, label="container-warm", color=COLORS["container-warm"], lw=2, ls="--")
    ax1.loglog(qps, gw, label="gpu-warm", color=COLORS["gpu-warm"], lw=2, ls=":")
    ax1.set_xlabel("Request rate (req/s)")
    ax1.set_ylabel("Cost per 1M requests (USD)")
    ax1.set_title("Cost amortization vs request rate")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend()

    # break-even: where sz_cost crosses warm_cost
    crossover_idx = np.argmin(np.abs(sz - gw))
    crossover_qps = qps[crossover_idx]
    ax1.axvline(crossover_qps, color="gray", ls="--", alpha=0.5)
    ax1.annotate(f"break-even ≈ {crossover_qps:.2f} req/s",
                 xy=(crossover_qps, sz[crossover_idx]),
                 xytext=(crossover_qps * 2, sz[crossover_idx] * 5),
                 fontsize=9, color="gray",
                 arrowprops=dict(arrowstyle="->", color="gray", alpha=0.7))

    # right panel: latency p95 (constant, just a reference table)
    p95 = {"scale-to-zero": 4044, "container-warm": 1386, "gpu-warm": 497}
    labels = list(p95.keys())
    ax2.barh(labels, [p95[k] for k in labels],
             color=[COLORS[k] for k in labels])
    for i, k in enumerate(labels):
        ax2.text(p95[k] + 60, i, f"{p95[k]} ms", va="center", fontsize=10)
    ax2.set_xlabel("Cold start p95 latency (ms)")
    ax2.set_title("Latency cost (constant)")
    ax2.set_xlim(0, max(p95.values()) * 1.15)
    ax2.grid(axis="x", alpha=0.3)
    ax2.set_axisbelow(True)

    fig.suptitle("Cost-latency tradeoff: GPU-warm pays for itself above ~{:.2f} req/s".format(crossover_qps),
                 fontsize=12, y=1.02)

    out = FIGURES / "fig4_cost.png"
    plt.savefig(out)
    plt.close()
    print(f"wrote {out}")


if __name__ == "__main__":
    fig1_warmlevel_comparison()
    fig2_component_breakdown()
    fig3_nccl()
    fig4_cost_pareto()
    print("\nAll figures saved to figures/")
