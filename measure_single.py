#!/usr/bin/env python3
import argparse, json, os, socket, time, uuid
from datetime import datetime, timezone

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "distilgpt2"

def now_ms():
    return time.perf_counter() * 1000.0

def utc_ts():
    return datetime.now(timezone.utc).isoformat()

def to_ms(x):
    return round(float(x), 3)

def run_once(warm_level, warmup_iters, steady_iters):
    torch.backends.cudnn.benchmark = True
    t0 = now_ms()

    t_model_load_start = t_model_load_end = None
    t_first_cuda_touch = t_cuda_init_end = None
    t_to_gpu_end = None
    t_warmup_start = t_warmup_end = None
    t_first_infer_start = t_first_infer_end = None
    device_name = "cpu"

    def load_model_cpu():
        nonlocal t_model_load_start, t_model_load_end
        t_model_load_start = now_ms() - t0
        tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        m = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
        t_model_load_end = now_ms() - t0
        return tok, m

    def ensure_cuda_init():
        nonlocal t_first_cuda_touch, t_cuda_init_end, device_name
        if not torch.cuda.is_available():
            return
        if t_first_cuda_touch is None:
            t_first_cuda_touch = now_ms() - t0
            torch.cuda.init()
            torch.cuda.synchronize()
            t_cuda_init_end = now_ms() - t0
            device_name = torch.cuda.get_device_name(0)

    def to_gpu(m):
        nonlocal t_to_gpu_end
        if not torch.cuda.is_available():
            return m
        m = m.to("cuda")
        torch.cuda.synchronize()
        t_to_gpu_end = now_ms() - t0
        return m

    def warmup(tok, m):
        nonlocal t_warmup_start, t_warmup_end
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tok("warm up", return_tensors="pt").to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        t_warmup_start = now_ms() - t0
        with torch.no_grad():
            for _ in range(max(1, warmup_iters)):
                m.generate(**inputs, max_new_tokens=1)
        if device == "cuda":
            torch.cuda.synchronize()
        t_warmup_end = now_ms() - t0

    if warm_level == "gpu-warm":
        tok, m = load_model_cpu()
        ensure_cuda_init()
        m = to_gpu(m)
        warmup(tok, m)

    if warm_level == "container-warm":
        tok, m = load_model_cpu()
        ensure_cuda_init()
        if torch.cuda.is_available():
            m = to_gpu(m)
            warmup(tok, m)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tok("Hello, how are you?", return_tensors="pt").to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t_first_infer_start = now_ms() - t0
    with torch.no_grad():
        m.generate(**inputs, max_new_tokens=20)
    if device == "cuda":
        torch.cuda.synchronize()
    t_first_infer_end = now_ms() - t0

    if steady_iters > 0:
        if device == "cuda":
            torch.cuda.synchronize()
        t_s = now_ms()
        with torch.no_grad():
            for _ in range(steady_iters):
                m.generate(**inputs, max_new_tokens=20)
        if device == "cuda":
            torch.cuda.synchronize()
        t_e = now_ms()
        steady_avg = (t_e - t_s) / steady_iters
    else:
        steady_avg = 0.0

    model_load_ms  = (t_model_load_end - t_model_load_start) if t_model_load_start else 0.0
    cuda_init_ms   = (t_cuda_init_end - t_first_cuda_touch)  if t_first_cuda_touch  else 0.0
    to_gpu_ms      = (t_to_gpu_end - t_cuda_init_end)        if t_to_gpu_end        else 0.0
    warmup_ms      = (t_warmup_end - t_warmup_start)         if t_warmup_start      else 0.0
    first_infer_ms = (t_first_infer_end - t_first_infer_start)
    total_cold_local_ms = model_load_ms + cuda_init_ms + to_gpu_ms + warmup_ms + first_infer_ms

    return {
        "timestamp_utc": utc_ts(),
        "run_id": str(uuid.uuid4()),
        "host": socket.gethostname(),
        "mode": "gpu" if torch.cuda.is_available() else "cpu",
        "model": MODEL_NAME,
        "warm_level": warm_level,
        "gpu_name": device_name,
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "model_load_ms":       to_ms(model_load_ms),
        "cuda_init_ms":        to_ms(cuda_init_ms),
        "to_gpu_ms":           to_ms(to_gpu_ms),
        "warmup_ms":           to_ms(warmup_ms),
        "first_infer_ms":      to_ms(first_infer_ms),
        "steady_infer_avg_ms": to_ms(steady_avg),
        "total_cold_local_ms": to_ms(total_cold_local_ms),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm-level", choices=["container-warm","gpu-warm"], required=True)
    ap.add_argument("--warmup-iters", type=int, default=1)
    ap.add_argument("--steady-iters", type=int, default=10)
    ap.add_argument("--out", default="results/single.jsonl")
    args = ap.parse_args()
    rec = run_once(args.warm_level, args.warmup_iters, args.steady_iters)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))

if __name__ == "__main__":
    main()
