# Project context for Claude

This repo benchmarks cold-start mitigation strategies for GPU LLM inference,
on two independent setups: GKE (Tesla T4, DistilGPT-2) and CURC Alpine HPC
(2×A100 80GB PCIe, NCCL distributed init). See `README.md` for findings.

## Hardware ground truth (do not get this wrong)

- **CURC Alpine `aa100` partition** = NVIDIA A100 80GB PCIe, **full GPU**, not MIG.
  Host pattern: `c3gpu-a9-*`.
- **CURC Alpine MIG partition** (older `c3gpu-c2-*` hosts) = A100 40GB split into
  MIG `3g.20gb` instances. Only `results/single.jsonl` and `results/test2.jsonl`
  came from there; the paper does **not** use this data.
- **GCP GKE** = NVIDIA Tesla T4. Used for the three warm-level deployment experiments.

If you see `results/single.jsonl` or `results/test2.jsonl`, that's archival
ResNet50/MIG data and not part of the paper's measurements.

## Critical concept: cold pattern vs distributed-warm

- `measure_nccl.py` measures the **cold pattern**: each invocation spawns
  fresh processes that pay full NCCL setup. Results in `results/nccl.jsonl`.
- The paper Section 4.9 motivates but does **not** implement
  **distributed-warm**: one long-running process pair that pays NCCL setup
  once, then handles many "requests" without re-init.
- New experiments under `measure_dist_warm.py` (when added) should be the
  *implementation* of distributed-warm, not another cold-pattern variant.

## CURC Alpine workflow (NEVER violate)

- **NEVER run heavy Python on the login node.** No `python measure_*.py`,
  no `python -c "import torch; torch.zeros(...).cuda()"` on login. Login
  nodes have memory/CPU limits and sysadmin will kill the process.
- **ALWAYS submit GPU work via `sbatch`**. See `submit_coldstart.sh` as the
  template. Output lands in `slurm-<JOBID>.out`.
- **Conda env**: `module load anaconda && conda activate coldstart`. Already
  has torch 2.5.1 + cu121.
- **HuggingFace cache**: set `HF_HOME=/scratch/alpine/$USER/hf_cache` in
  sbatch scripts. `/home` has tight quota; large model weights will fill it.
- **Quick partition check**: `sinfo -p aa100` to see availability before
  submitting. Request short `--time=` (e.g. `00:30:00`) to schedule faster.

## NCCL/torchrun specifics

- `dist.barrier()` without `device_ids` emits a warning and uses guesswork
  for the device. Always pass `dist.barrier(device_ids=[rank])`.
- For 2×A100 on a single node: `torchrun --nproc-per-node=2 measure_*.py`.
  Do not set `MASTER_ADDR`/`MASTER_PORT` manually — `torchrun` handles it.
- Each NCCL trial currently spawns a new process pair. The first run on a
  fresh node pays ~2300ms (driver cold); subsequent runs pay ~520ms.

## Repo layout

```
scripts/         # Tools: figure generation, summary stats
results/         # Raw JSONL + final_*.json summaries (committed)
figures/         # Generated PNGs (committed for README visibility)
measure_*.py     # Experiment scripts (run via sbatch)
server.py        # FastAPI server for GKE (not used on CURC)
Dockerfile       # GKE deployment only
```

## Reproduce commands

```bash
make figures            # regenerate figures/ from results/ (uses .venv)
sbatch submit_coldstart.sh   # CURC: run NCCL cold-pattern experiment
```

## Don't propose without asking

- Adding ML frameworks (vLLM, Triton, TGI) — these are bigger experiments,
  discuss scope first.
- Switching base model (DistilGPT-2 → Llama-3-8B etc.) — requires HF auth
  setup and `/scratch` cache; user must confirm.
- Rewriting `server.py`'s warm-level state machine — currently works on GKE,
  don't break it for a CURC-only need.

## Output style

- Stage-level timing fields use the `_ms` suffix (e.g. `model_load_ms`).
  Match this when adding new fields.
- Append-only JSONL for raw trials; aggregated stats go to `final_*.json`.
- Cite specific file lines when discussing code (the IDE makes them clickable).
