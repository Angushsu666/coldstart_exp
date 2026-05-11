#!/bin/bash
#SBATCH --job-name=dist_warm
#SBATCH --partition=aa100
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gres=gpu:2
#SBATCH --time=00:15:00
#SBATCH --output=slurm-%j.out
#SBATCH --account=ucb-general
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chhs8331@colorado.edu

module purge
module load anaconda
conda activate coldstart

cd "$SLURM_SUBMIT_DIR" || exit 1
mkdir -p results

echo "=== Job started on $(hostname) at $(date) ==="
nvidia-smi -L

rm -f results/dist_warm.jsonl

torchrun --nproc_per_node=2 measure_dist_warm.py

echo "=== RESULTS: dist_warm ==="
python -c "
import json, statistics
data = [json.loads(l) for l in open('results/dist_warm.jsonl') if l.strip()]
startup = next(r for r in data if r['record_type'] == 'startup')
trials  = [r for r in data if r['record_type'] == 'trial']
first   = trials[0]['t_request_ms']
warm    = sorted(r['t_request_ms'] for r in trials[1:])
n       = len(warm)
print(f'  t_startup_ms       = {startup[\"t_startup_ms\"]:.3f} ms  (NCCL init, paid once)')
print(f'  trial[0] (first)   = {first:.3f} ms  (first all_reduce after init)')
print(f'  trials[1-{len(trials)-1}] n={n}  mean={statistics.mean(warm):.3f}  p50={warm[n//2]:.3f}  p95={warm[int(n*0.95)]:.3f} ms')
"

echo "=== Done at $(date) ==="
