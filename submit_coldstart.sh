#!/bin/bash
#SBATCH --job-name=coldstart_distilgpt2
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=results/slurm-%j.out
#SBATCH --account=ucb-general
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chhs8331@colorado.edu

module purge
module load anaconda
conda activate coldstart

cd ~/coldstart_exp
mkdir -p results

echo "=== Job started on $(hostname) at $(date) ==="
nvidia-smi -L

# Ensure model is cached before running experiments
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
cache = os.path.expanduser('~/.cache/huggingface/hub')
print('Cache path:', cache, '| exists:', os.path.exists(cache))
AutoTokenizer.from_pretrained('distilgpt2')
AutoModelForCausalLM.from_pretrained('distilgpt2')
print('Model ready.')
"

# --- container-warm: N=20 ---
echo "=== container-warm ==="
rm -f results/single.jsonl
for i in $(seq 1 20); do
  python measure_single.py --warm-level container-warm \
    --warmup-iters 1 --steady-iters 3 --out results/single.jsonl
done

# --- gpu-warm: N=20 ---
echo "=== gpu-warm ==="
for i in $(seq 1 20); do
  python measure_single.py --warm-level gpu-warm \
    --warmup-iters 1 --steady-iters 3 --out results/single.jsonl
done

# --- NCCL T_dist: N=20 ---
echo "=== NCCL ==="
rm -f results/nccl.jsonl
for i in $(seq 1 20); do
  torchrun --nproc_per_node=2 measure_nccl.py
done

# --- Print summary ---
echo "=== RESULTS: total_cold_local_ms ==="
python summarize_jsonl.py --infile results/single.jsonl --metric total_cold_local_ms

echo "=== RESULTS: first_infer_ms ==="
python summarize_jsonl.py --infile results/single.jsonl --metric first_infer_ms

echo "=== RESULTS: nccl_init_ms ==="
python -c "
import json, sys
data = [json.loads(l) for l in open('results/nccl.jsonl') if l.strip()]
vals = sorted(r['nccl_init_ms'] for r in data)
n = len(vals)
print(f'n={n}')
print(f'p50={vals[n//2]:.3f} ms')
print(f'p95={vals[int(n*0.95)]:.3f} ms')
print(f'p99={vals[int(n*0.99)]:.3f} ms')
"

echo "=== Done at $(date) ==="
