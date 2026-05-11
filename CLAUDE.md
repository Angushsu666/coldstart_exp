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
- **Module tree differs by node type**:
    - **Login nodes** (`login-ci*`): `anaconda` Lmod module is NOT available.
      Do NOT try `module load anaconda` here. Login is for editing + git +
      sbatch submission only — never run `python` interactively here.
    - **Compute nodes** (sinteractive / sbatch job): `anaconda` Lmod module
      IS available. The pattern `module purge && module load anaconda &&
      conda activate coldstart` works on compute nodes. This is what
      `submit_coldstart.sh` uses and it works correctly.
- **Conda envs** live in `/projects/$USER/software/anaconda/envs/`. The
  `coldstart` env has torch 2.5.1 + cu121. For Llama-3 work (Day 4+),
  create a separate `vllm-coldstart` env (inside an sbatch job, not on
  login) to avoid version conflicts.
- **Interactive GPU testing** (not normally needed): use sinteractive on
  the `atesting_a100` partition (qos=testing, short time limit) — faster
  to schedule than `aa100` because of the testing qos.
- **HuggingFace cache**: set `HF_HOME=/scratch/alpine/$USER/hf_cache` in
  sbatch scripts. `/home` has tight quota; large model weights will fill it.
- **Quick partition check**: `sinfo -p aa100` to see availability before
  submitting. Request short `--time=` (e.g. `00:30:00`) to schedule faster.

## CURC sbatch queue policy

- Max 16 GPUs/user on `aa100`, 22 total across all GPU types. Our jobs use
  ≤2 GPUs so this is not a constraint, but never submit-spam.
- Wait times can exceed 24h during peak. Plan accordingly; do not last-
  minute submit.
- `squeue` order is temporal, NOT real queue position. Use
  `squeue -u $USER --start` for expected start time.
- **Recent usage lowers fair-share priority.** NEVER submit-then-cancel
  hoping to re-roll the queue — it makes the next submit even slower.
- Backfill rewards small + short jobs. Prefer:
    - `--time=00:15:00` for dist-warm (5min real work)
    - `--time=00:45:00` per warm-level run for Llama-3 vLLM (one cold start
      + N trials)
- ALWAYS validate first: `sbatch --test-only <script>` checks syntax and
  partition without consuming a queue slot.
- Useful queries:
    - `squeue -u $USER --start`   expected start time
    - `sinfo -p aa100`            partition state
    - `sshare -u $USER`           fair-share score (lower = lower priority)

## NCCL/torchrun specifics

- `dist.barrier()` without `device_ids` emits a warning and uses guesswork
  for the device. Always pass `dist.barrier(device_ids=[rank])`.
- For 2×A100 on a single node: `torchrun --nproc-per-node=2 measure_*.py`.
  Do not set `MASTER_ADDR`/`MASTER_PORT` manually — `torchrun` handles it.
- Each NCCL trial currently spawns a new process pair. The first run on a
  fresh node pays ~2300ms (driver cold); subsequent runs pay ~520ms.

## Tensor parallelism context (TP=2 Llama-3-8B on vLLM)

We extend the cold-start study to a real production-scale model: Llama-3-8B
served with **tensor parallelism degree 2 (TP=2)** across the 2×A100 80GB on
`aa100`. Each layer's matrices are split column-/row-wise; each forward pass
issues an `all_reduce` per Transformer block. This is where the
`distributed-warm` strategy from `measure_dist_warm.py` becomes practically
load-bearing: without it, every cold start pays NCCL communicator setup
before producing the first token.

- **Framework**: **vLLM** with `--tensor-parallel-size 2`. Reasons: industry
  standard for LLM serving, one-flag TP, gives "free" PagedAttention +
  continuous batching as comparison artifacts. Do NOT use HF accelerate
  `device_map="auto"` — that's pipeline-parallel sharding, not TP.
- **Model**: `meta-llama/Llama-3-8B` (gated, requires HF token + license
  accept on huggingface.co before first download).
- **Cache**: set `HF_HOME=/scratch/alpine/$USER/hf_cache`. Llama-3-8B FP16
  ≈ 16GB on disk; never download to `/home`.
- **TP degree must divide attention head count**. Llama-3-8B has 32 heads;
  TP ∈ {1, 2, 4, 8, 16, 32} valid. We use 2 (one per GPU).
- **vLLM cold start is non-trivial**: in addition to T_model and T_cuda, vLLM
  profiles available KV cache memory and pre-allocates PagedAttention pages
  at startup. Expect overall startup ~30–60s for 8B at TP=2; instrument it
  similarly to `server.py`'s warm-level pattern.
- **Warm levels for Llama-3 experiment** (port the GKE pattern to vLLM):
    - scale-to-zero: cold launch vLLM per trial
    - container-warm: vLLM process up, model on CPU, GPU not loaded
    - gpu-warm: vLLM fully ready (weights on GPU + KV cache allocated)
- Output to `results/vllm_llama3_<warm_level>.jsonl`; aggregate to
  `results/final_vllm_llama3.json`.

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

- Adding **other** ML frameworks (Triton, TGI, TensorRT-LLM, DeepSpeed) —
  vLLM is already approved for the Llama-3-8B TP=2 experiment; other
  frameworks need scope discussion.
- Increasing TP degree beyond 2 — we have 2 GPUs allocated; TP=4+ would
  require partition change and user confirmation.
- Switching model beyond what's planned (currently: DistilGPT-2 on GKE,
  Llama-3-8B on CURC). Other models (Mistral, Qwen, Llama-3-70B) need
  user confirmation.
- Rewriting `server.py`'s warm-level state machine — currently works on GKE,
  don't break it for a CURC-only need.

## Output style

- Stage-level timing fields use the `_ms` suffix (e.g. `model_load_ms`).
  Match this when adding new fields.
- Append-only JSONL for raw trials; aggregated stats go to `final_*.json`.
- Cite specific file lines when discussing code (the IDE makes them clickable).
