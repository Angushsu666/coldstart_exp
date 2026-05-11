#!/usr/bin/env python3
import os, json, time, socket, uuid
from datetime import datetime, timezone
import torch
import torch.distributed as dist

def now_ms():
    return time.perf_counter() * 1000.0

def utc_ts():
    return datetime.now(timezone.utc).isoformat()

t0 = now_ms()
rank = int(os.environ.get("RANK", 0))
world_size = int(os.environ.get("WORLD_SIZE", 1))

t_nccl_start = now_ms() - t0
dist.init_process_group(backend="nccl")
torch.cuda.set_device(rank)
dist.barrier(device_ids=[rank])
torch.cuda.synchronize()
t_nccl_end = now_ms() - t0

nccl_init_ms = round(t_nccl_end - t_nccl_start, 3)

rec = {
    "timestamp_utc": utc_ts(),
    "run_id": str(uuid.uuid4()),
    "host": socket.gethostname(),
    "rank": rank,
    "world_size": world_size,
    "gpu_name": torch.cuda.get_device_name(rank),
    "nccl_init_ms": nccl_init_ms,
}

if rank == 0:
    os.makedirs("results", exist_ok=True)
    with open("results/nccl.jsonl", "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(json.dumps(rec, indent=2))

dist.destroy_process_group()
