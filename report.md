# Project Update: Cold Start Measurement on GCP

## Overview

We ran three experiments on GKE with a DistilGPT-2 inference server on a Tesla T4 GPU. Each experiment uses a different warm level strategy: scale-to-zero (baseline), container-warm, and gpu-warm. The goal is to see how much each strategy reduces cold start latency and what the trade-offs are.

Each experiment ran 20 cold start trials and 5 steady-state requests per trial. A "cold" request means the pod was scaled to zero replicas first, then scaled back up, and the first request was sent. Steady-state requests were sent immediately after without restarting the pod.

---

## Experiment 1: Scale-to-Zero (Baseline)

| Metric | Value |
|--------|-------|
| Cold mean | 3873 ms |
| Cold p95 | 4044 ms |
| Steady mean | 389 ms |
| Cold/Steady slowdown | 9.95x |

When a request hits a scaled-to-zero pod, the system has to do everything from scratch: schedule the pod, start the container, load the model weights, initialize CUDA, move the model to GPU, and then run inference. This is why the first request takes about 3.9 seconds while steady-state is only ~390ms — almost a 10x difference.

**Breakdown of where the time goes:**

| Stage | Mean | % of wall time |
|-------|------|----------------|
| Model load | 2603 ms | 67.2% |
| First inference | 607 ms | 15.7% |
| Scheduling + container | 269 ms | 7.0% |
| CUDA init | 249 ms | 6.4% |
| Model to GPU | 143 ms | 3.7% |

Model loading dominates at 67% of total cold start time. This makes sense because DistilGPT-2 weights need to be read from disk and transferred into memory every time a new pod starts.

One thing we noticed: `t_warmup_ms` is 0 for scale-to-zero. This is because the server code only runs a warmup forward pass for container-warm, not for scale-to-zero. So the first inference in scale-to-zero is a "cold" inference — GPU kernels haven't been compiled yet, which is why `t_infer_ms` (~607ms) is much higher compared to the other strategies.

---

## Experiment 2: Container-Warm

| Metric | Value |
|--------|-------|
| Cold mean | 1300 ms |
| Cold p95 | 1386 ms |
| Steady mean | 384 ms |
| Cold/Steady slowdown | 3.38x |
| vs. scale-to-zero | **3.0x faster** |

Container-warm loads the model to CPU memory at container startup, before any request arrives. So when the first request comes in, `t_model_ms = 0` — the model is already there. This eliminates the biggest bottleneck from Experiment 1 and brings cold start down from ~3.9s to ~1.3s.

**Breakdown:**

| Stage | Mean | % of wall time |
|-------|------|----------------|
| GPU warmup | 420 ms | 32.3% |
| Scheduling + container | 271 ms | 20.9% |
| CUDA init | 257 ms | 19.8% |
| First inference | 209 ms | 16.1% |
| Model to GPU | 141 ms | 10.9% |
| Model load | 0 ms | 0.0% ✅ |

Now that model loading is gone, warmup (420ms) becomes the biggest component. This is the dummy forward pass that "warms up" the GPU kernels before real inference. Because of this warmup, `t_infer_ms` drops from ~607ms (scale-to-zero) to ~209ms — the warmup basically pre-pays the kernel compilation cost so actual inference is faster.

**One interesting thing:** waiting for the server to become ready took noticeably longer for container-warm compared to scale-to-zero. This is because container-warm loads the model (~2.5s) during container startup before the HTTP server is ready to accept requests. This startup cost (~2531ms on average) doesn't show up in `wall_time_ms` because the measurement only starts after the server is confirmed ready — but the cost is still real, it's just paid upfront rather than during the request.

---

## Experiment 3: GPU-Warm

| Metric | Value |
|--------|-------|
| Cold mean | 475 ms |
| Cold p95 | 497 ms |
| Steady mean | 383 ms |
| Cold/Steady slowdown | 1.24x |
| vs. scale-to-zero | **8.1x faster** |

GPU-warm moves everything to startup: model load, CUDA init, model-to-GPU transfer, and a warmup forward pass. By the time the first request arrives, the GPU is completely ready. Cold start drops to ~475ms.

**Breakdown:**

| Stage | Mean | % of wall time |
|-------|------|----------------|
| Scheduling + container | 270 ms | 56.7% |
| First inference | 204 ms | 42.9% |
| Model load | 0 ms | 0.0% ✅ |
| CUDA init | 0 ms | 0.0% ✅ |
| Model to GPU | 0 ms | 0.0% ✅ |
| GPU warmup | 0 ms | 0.0% ✅ |

Four components are now 0ms. The only things left are Kubernetes pod scheduling (~270ms) and the actual inference (~204ms). At this point there's basically nothing left to optimize from the application side — the scheduling delay is a Kubernetes infrastructure cost that can't be avoided without keeping the pod alive permanently.

The cold/steady slowdown is only 1.24x, meaning a cold request is barely slower than a warm one.

**Startup cost (paid before any request, not in wall_time):**

| Stage | Mean |
|-------|------|
| Model load | 2458 ms |
| CUDA init | 247 ms |
| Model to GPU | 141 ms |
| GPU warmup | 401 ms |
| **Total** | **~3247 ms** |

Similar to container-warm, the server ready wait time was the longest of the three experiments because gpu-warm has to complete all of the above before it can serve requests.

---

## Three-Way Comparison

| Strategy | Cold mean | Cold p95 | Steady mean | Cold/Steady |
|----------|-----------|----------|-------------|-------------|
| scale-to-zero | 3873 ms | 4044 ms | 389 ms | 9.95x |
| container-warm | 1300 ms | 1386 ms | 384 ms | 3.38x |
| gpu-warm | 475 ms | 497 ms | 383 ms | 1.24x |

Each strategy progressively moves more work out of the request path and into startup, which reduces cold start latency at the cost of longer startup time and higher idle resource usage.

---

## Cost Trade-Off

The strategies differ not just in latency but in how much GPU time they consume.

**Scale-to-zero** only uses GPU resources when actively serving requests. When there are no requests, the pod is gone and you pay nothing. This is the cheapest option when traffic is low or unpredictable.

**Container-warm and GPU-warm** require the pod to stay alive all the time. On GCP, a T4 GPU costs roughly $0.35/hr. If the pod runs 24/7, that's about $8.40/day regardless of how many requests come in.

The break-even point depends on how often users hit a cold start. If requests are frequent enough that the always-on cost is cheaper than the penalty of repeated cold starts, warm strategies make sense. If traffic is sparse, scale-to-zero is more cost-efficient even with the latency penalty.

| Strategy | Idle GPU cost | Cold latency | Good for |
|----------|--------------|--------------|----------|
| scale-to-zero | $0 when idle | ~3.9s | Low / unpredictable traffic |
| container-warm | Always-on | ~1.3s | Medium traffic, cost-sensitive |
| gpu-warm | Always-on | ~475ms | Latency-critical, consistent traffic |

In practice, gpu-warm gives the best user experience but is the most expensive to maintain. Scale-to-zero is free when idle but users feel the cold start every time. Container-warm sits in between — it cuts latency by 3x compared to scale-to-zero while costing the same as gpu-warm to run.

---

## Summary

The results confirm what we expected from the decomposition model. Model loading (~2.6s) is the dominant cost in a full cold start, and moving it to startup eliminates most of the latency. GPU initialization and warmup contribute another ~700ms combined, and gpu-warm eliminates those too. What's left (~475ms) is mostly unavoidable Kubernetes scheduling overhead.

The key insight is that all three strategies have roughly the same total work to do — the difference is just *when* that work happens: during the request (scale-to-zero), at container startup (container-warm / gpu-warm), or spread across both.
