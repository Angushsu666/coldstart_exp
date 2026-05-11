#!/usr/bin/env python3
import os, json, time, socket, uuid
from datetime import datetime, timezone
import torch
import torch.distributed as dist

def now_ms():
    return time.perf_counter() * 1000.0

def utc_ts():
    return datetime.now(timezone.utc).isoformat()

N_TRIALS = 20
OUT_FILE = "results/dist_warm.jsonl"

rank       = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

# Startup: pay NCCL init exactly once for the lifetime of this process pair.
t_startup_start = now_ms()
dist.init_process_group(backend="nccl")
torch.cuda.set_device(rank)
dist.barrier(device_ids=[rank])
torch.cuda.synchronize()
t_startup_ms = round(now_ms() - t_startup_start, 3)

# run_id ties the startup record to its 20 trial records in analysis.
run_id = str(uuid.uuid4()) if rank == 0 else None

# Both ranks loop in lockstep; all_reduce itself is the collective barrier —
# no external signaling needed.
trial_records = []
for i in range(N_TRIALS):
    tensor = torch.zeros(1, device=f"cuda:{rank}")
    torch.cuda.synchronize()          # flush CPU queue before timing
    t0 = now_ms()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()          # wait for GPU completion before stopping timer
    t_req_ms = round(now_ms() - t0, 3)
    if rank == 0:
        trial_records.append({
            "record_type":   "trial",
            "run_id":        run_id,
            "trial_idx":     i,
            "t_request_ms":  t_req_ms,
            "rank":          rank,
            "world_size":    world_size,
            "host":          socket.gethostname(),
            "timestamp_utc": utc_ts(),
        })

if rank == 0:
    os.makedirs("results", exist_ok=True)
    startup_rec = {
        "record_type":  "startup",
        "run_id":       run_id,
        "t_startup_ms": t_startup_ms,
        "rank":         rank,
        "world_size":   world_size,
        "host":         socket.gethostname(),
        "gpu_name":     torch.cuda.get_device_name(rank),
        "timestamp_utc": utc_ts(),
    }
    with open(OUT_FILE, "a") as f:
        f.write(json.dumps(startup_rec) + "\n")
        for rec in trial_records:
            f.write(json.dumps(rec) + "\n")
    print(json.dumps(startup_rec, indent=2))
    print(f"Wrote {len(trial_records)} trial records to {OUT_FILE}")

dist.destroy_process_group()
