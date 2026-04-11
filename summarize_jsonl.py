#!/usr/bin/env python3
import argparse, json, math
from collections import defaultdict

def percentile(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs)-1) * p
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs[int(k)]
    return xs[f] * (c-k) + xs[c] * (k-f)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True)
    ap.add_argument("--metric", default="total_cold_local_ms")
    args = ap.parse_args()

    buckets = defaultdict(list)
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            key = (r.get("warm_level"), r.get("gpu_name"))
            val = r.get(args.metric)
            if val is not None:
                buckets[key].append(float(val))

    for (wl, gpu), xs in buckets.items():
        print(f"\n== {args.metric} | warm_level={wl} | gpu={gpu} | n={len(xs)} ==")
        print(f"p50: {percentile(xs, 0.50):.3f} ms")
        print(f"p95: {percentile(xs, 0.95):.3f} ms")
        print(f"p99: {percentile(xs, 0.99):.3f} ms")

if __name__ == "__main__":
    main()
