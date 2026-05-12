#!/usr/bin/env python3
"""
finalize_results.py
Reads a raw JSONL file and produces a final summary JSON.

Usage:
  python finalize_results.py                                          # defaults
  python finalize_results.py --input results/server_gcp_container_warm_raw.jsonl --output results/final_container_warm.json
"""
import argparse, json, math, os
from collections import defaultdict

DEFAULT_INPUT  = "results/server_gcp_scale_to_0_raw.jsonl"
DEFAULT_OUTPUT = "results/final_scale_to_zero.json"

COLD_COMPONENTS = [
    "t_sched_container_ms",
    "t_model_ms",
    "t_cuda_ms",
    "t_to_gpu_ms",
    "t_warmup_ms",
    "t_infer_ms",
]

STARTUP_COMPONENTS = [
    "startup_model_load_ms",
    "startup_cuda_init_ms",
    "startup_to_gpu_ms",
    "startup_warmup_ms",
]

def percentile(xs_sorted, p):
    """xs_sorted must already be sorted."""
    if not xs_sorted:
        return None
    k = (len(xs_sorted) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    if f == c:
        return xs_sorted[int(k)]
    return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)

def stats(xs):
    if not xs:
        return {}
    xs_s = sorted(xs)
    n = len(xs)
    mean = sum(xs) / n
    variance = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0  # sample stdev
    return {
        "n":      n,
        "mean":   round(mean, 3),
        "median": round(percentile(xs_s, 0.50), 3),
        "p95":    round(percentile(xs_s, 0.95), 3),
        "p99":    round(percentile(xs_s, 0.99), 3),
        "min":    round(xs_s[0], 3),
        "max":    round(xs_s[-1], 3),
        "stdev":  round(math.sqrt(variance), 3),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=DEFAULT_INPUT)
    ap.add_argument("--output", default=DEFAULT_OUTPUT)
    args = ap.parse_args()
    INPUT  = args.input
    OUTPUT = args.output

    data = []
    with open(INPUT, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # group by warm_level
    by_wl = defaultdict(lambda: {"cold": [], "steady": []})
    for d in data:
        wl  = d.get("warm_level", "unknown")
        rtype = d.get("request_type", "unknown")
        if rtype in ("cold", "steady"):
            by_wl[wl][rtype].append(d)

    result = {
        "source": INPUT,  # noqa: F821  (assigned above via argparse)
        "gpu": list({d.get("gpu_name") for d in data}),
        "warm_levels": {},
    }

    for wl, groups in by_wl.items():
        cold_rows   = groups["cold"]
        steady_rows = groups["steady"]

        # --- cold start summary ---
        cold_wall  = [r["wall_time_ms"] for r in cold_rows]
        cold_total = [r["total_request_ms"] for r in cold_rows]

        breakdown = {}
        for comp in COLD_COMPONENTS:
            vals = [r.get(comp, 0.0) for r in cold_rows]
            breakdown[comp] = stats(vals)

        # startup components (container-warm / gpu-warm)
        startup_breakdown = {}
        for comp in STARTUP_COMPONENTS:
            vals = [r.get(comp, 0.0) for r in cold_rows]
            if any(v > 0 for v in vals):
                startup_breakdown[comp] = stats(vals)

        # percentage share of each component vs mean wall_time
        mean_wall = sum(cold_wall) / len(cold_wall) if cold_wall else 1
        breakdown_pct = {
            comp: round(breakdown[comp]["mean"] / mean_wall * 100, 1)
            for comp in COLD_COMPONENTS
        }

        # --- steady state summary ---
        steady_wall = [r["wall_time_ms"] for r in steady_rows]
        steady_infer = [r.get("t_infer_ms", 0.0) for r in steady_rows]

        cold_mean   = sum(cold_wall) / len(cold_wall) if cold_wall else None
        steady_mean = sum(steady_wall) / len(steady_wall) if steady_wall else None
        slowdown = round(cold_mean / steady_mean, 2) if (cold_mean and steady_mean) else None

        result["warm_levels"][wl] = {
            "cold_start": {
                "wall_time_ms":            stats(cold_wall),
                "total_request_ms":        stats(cold_total),
                "component_breakdown_ms":  breakdown,
                "component_pct_of_wall":   breakdown_pct,
                **({"startup_breakdown_ms": startup_breakdown} if startup_breakdown else {}),
            },
            "steady_state": {
                "wall_time_ms": stats(steady_wall),
                "t_infer_ms":   stats(steady_infer),
            },
            "cold_vs_steady_slowdown_x": slowdown,
        }

    os.makedirs(os.path.dirname(OUTPUT) or ".", exist_ok=True)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"Written to {OUTPUT}")

    # also print a quick human-readable summary
    for wl, entry in result["warm_levels"].items():
        cs = entry["cold_start"]["wall_time_ms"]
        ss = entry["steady_state"]["wall_time_ms"]
        bd = entry["cold_start"]["component_pct_of_wall"]
        print(f"\n=== warm_level: {wl} ===")
        print(f"  Cold wall_time  — mean: {cs['mean']} ms | p95: {cs['p95']} ms | n={cs['n']}")
        print(f"  Steady wall_time— mean: {ss['mean']} ms | p95: {ss['p95']} ms | n={ss['n']}")
        print(f"  Cold/Steady slowdown: {entry['cold_vs_steady_slowdown_x']}x")
        print("  Breakdown (% of wall):")
        for comp, pct in bd.items():
            mean_ms = entry["cold_start"]["component_breakdown_ms"][comp]["mean"]
            print(f"    {comp:<25} {pct:>5}%  ({mean_ms} ms)")

if __name__ == "__main__":
    main()
