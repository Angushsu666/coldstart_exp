#!/usr/bin/env python3
"""
FastAPI inference server for DistilGPT-2 cold start measurement.

Warm levels (set via ENV var WARM_LEVEL at container startup):
  - scale-to-zero   : nothing pre-loaded; model loads on first request  (baseline)
  - container-warm  : model loaded at startup, CUDA init on first request
  - gpu-warm        : model loaded + moved to GPU + warmup at startup
"""

import os
import time
import socket
import uuid
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI
from pydantic import BaseModel

MODEL_NAME = "distilgpt2"
WARM_LEVEL = os.environ.get("WARM_LEVEL", "scale-to-zero")

# Global state
_state = {
    "tokenizer": None,
    "model": None,
    "device": "cpu",
    "gpu_name": "cpu",
    "t0_process": time.perf_counter() * 1000.0,
    "startup_model_load_ms": 0.0,
    "startup_cuda_init_ms": 0.0,
    "startup_to_gpu_ms": 0.0,
    "startup_warmup_ms": 0.0,
}

def now_ms() -> float:
    return time.perf_counter() * 1000.0

def utc_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_model_cpu():
    t = now_ms()
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME).eval()
    return tok, m, now_ms() - t


def _init_cuda():
    if not torch.cuda.is_available():
        return 0.0
    t = now_ms()
    torch.cuda.init()
    torch.cuda.synchronize()
    _state["gpu_name"] = torch.cuda.get_device_name(0)
    return now_ms() - t


def _to_gpu(m):
    if not torch.cuda.is_available():
        return m, 0.0
    t = now_ms()
    m = m.to("cuda")
    torch.cuda.synchronize()
    return m, now_ms() - t


def _warmup(tok, m, device):
    t = now_ms()
    inputs = tok("warmup", return_tensors="pt").to(device)
    with torch.no_grad():
        m.generate(**inputs, max_new_tokens=1)
    if device == "cuda":
        torch.cuda.synchronize()
    return now_ms() - t


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup logic depending on WARM_LEVEL:
      scale-to-zero  -> do nothing
      container-warm -> load model to CPU only
      gpu-warm       -> load model + move to GPU + warmup
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _state["device"] = device

    if WARM_LEVEL == "container-warm":
        tok, m, load_ms = _load_model_cpu()
        _state["tokenizer"] = tok
        _state["model"] = m
        _state["startup_model_load_ms"] = load_ms

    elif WARM_LEVEL == "gpu-warm":
        tok, m, load_ms = _load_model_cpu()
        cuda_ms = _init_cuda()
        m, to_gpu_ms = _to_gpu(m)
        warmup_ms = _warmup(tok, m, device)
        _state["tokenizer"] = tok
        _state["model"] = m
        _state["startup_model_load_ms"] = load_ms
        _state["startup_cuda_init_ms"] = cuda_ms
        _state["startup_to_gpu_ms"] = to_gpu_ms
        _state["startup_warmup_ms"] = warmup_ms

    # scale-to-zero: nothing loaded at startup
    yield


app = FastAPI(lifespan=lifespan)


class InferRequest(BaseModel):
    prompt: str = "Hello, how are you?"
    max_new_tokens: int = 20


class InferResponse(BaseModel):
    request_id: str
    timestamp_utc: str
    host: str
    warm_level: str
    gpu_name: str
    prompt: str
    generated_text: str
    # per-request timings (ms)
    t_model_ms: float    # model load on first request (0 if pre-loaded)
    t_cuda_ms: float     # CUDA init on first request (0 if pre-init)
    t_to_gpu_ms: float   # model->GPU transfer (0 if pre-done)
    t_warmup_ms: float   # warmup forward pass (0 if pre-done)
    t_infer_ms: float    # actual inference (TTFT proxy)
    total_request_ms: float
    # startup timings (ms) — non-zero only when pre-loaded
    startup_model_load_ms: float
    startup_cuda_init_ms: float
    startup_to_gpu_ms: float
    startup_warmup_ms: float


@app.post("/infer", response_model=InferResponse)
def infer(req: InferRequest):
    t_req_start = now_ms()
    device = _state["device"]

    t_model_ms = t_cuda_ms = t_to_gpu_ms = t_warmup_ms = 0.0

    # --- Load model if not already loaded (scale-to-zero / container-warm lazy) ---
    if _state["model"] is None:
        tok, m, t_model_ms = _load_model_cpu()
        _state["tokenizer"] = tok
        _state["model"] = m

    tok = _state["tokenizer"]
    m = _state["model"]

    # --- CUDA init if not done ---
    if device == "cuda" and not torch.cuda.is_initialized():
        t_cuda_ms = _init_cuda()

    # --- Move to GPU if still on CPU ---
    if device == "cuda" and next(m.parameters()).device.type == "cpu":
        m, t_to_gpu_ms = _to_gpu(m)
        _state["model"] = m

    # --- Warmup if not done (container-warm first request) ---
    if WARM_LEVEL == "container-warm" and t_to_gpu_ms > 0:
        t_warmup_ms = _warmup(tok, m, device)

    # --- Actual inference ---
    inputs = tok(req.prompt, return_tensors="pt").to(device)
    if device == "cuda":
        torch.cuda.synchronize()
    t_infer_start = now_ms()
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=req.max_new_tokens)
    if device == "cuda":
        torch.cuda.synchronize()
    t_infer_ms = now_ms() - t_infer_start

    generated = tok.decode(out[0], skip_special_tokens=True)
    total_ms = now_ms() - t_req_start

    return InferResponse(
        request_id=str(uuid.uuid4()),
        timestamp_utc=utc_ts(),
        host=socket.gethostname(),
        warm_level=WARM_LEVEL,
        gpu_name=_state["gpu_name"],
        prompt=req.prompt,
        generated_text=generated,
        t_model_ms=round(t_model_ms, 3),
        t_cuda_ms=round(t_cuda_ms, 3),
        t_to_gpu_ms=round(t_to_gpu_ms, 3),
        t_warmup_ms=round(t_warmup_ms, 3),
        t_infer_ms=round(t_infer_ms, 3),
        total_request_ms=round(total_ms, 3),
        startup_model_load_ms=round(_state["startup_model_load_ms"], 3),
        startup_cuda_init_ms=round(_state["startup_cuda_init_ms"], 3),
        startup_to_gpu_ms=round(_state["startup_to_gpu_ms"], 3),
        startup_warmup_ms=round(_state["startup_warmup_ms"], 3),
    )


@app.get("/health")
def health():
    return {"status": "ok", "warm_level": WARM_LEVEL, "gpu": _state["gpu_name"]}
