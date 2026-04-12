#!/bin/bash
# Run all three warm-level experiments sequentially on GKE.
# Each deployment gets N trials, sharing one T4 GPU (scale others to 0).
#
# Usage:
#   bash run_all_gcp.sh <IP_SCALE_TO_ZERO> <IP_CONTAINER_WARM> <IP_GPU_WARM> [N]
#
# Example:
#   bash run_all_gcp.sh 34.91.210.213 35.204.119.174 34.178.68.41 20

IP_STZ=$1
IP_CW=$2
IP_GW=$3
N=${4:-20}
OUT=results/server_gcp.jsonl

if [ -z "$IP_STZ" ] || [ -z "$IP_CW" ] || [ -z "$IP_GW" ]; then
  echo "Usage: bash run_all_gcp.sh <IP_scale-to-zero> <IP_container-warm> <IP_gpu-warm> [N]"
  exit 1
fi

mkdir -p results

# ── 1. scale-to-zero ──────────────────────────────────────────────────────────
echo "========================================"
echo "Experiment 1/3: scale-to-zero"
echo "========================================"
kubectl scale deployment coldstart-container-warm --replicas=0
kubectl scale deployment coldstart-gpu-warm --replicas=0
bash run_trials.sh http://$IP_STZ:8000 coldstart-scale-to-zero $N $OUT

# ── 2. container-warm ─────────────────────────────────────────────────────────
echo "========================================"
echo "Experiment 2/3: container-warm"
echo "========================================"
kubectl scale deployment coldstart-scale-to-zero --replicas=0
bash run_trials.sh http://$IP_CW:8000 coldstart-container-warm $N $OUT

# ── 3. gpu-warm ───────────────────────────────────────────────────────────────
echo "========================================"
echo "Experiment 3/3: gpu-warm"
echo "========================================"
kubectl scale deployment coldstart-container-warm --replicas=0
bash run_trials.sh http://$IP_GW:8000 coldstart-gpu-warm $N $OUT

# ── Summary ───────────────────────────────────────────────────────────────────
echo "========================================"
echo "All experiments done. Summary:"
echo "========================================"
python summarize_jsonl.py --infile $OUT --metric wall_time_ms
echo "---"
python summarize_jsonl.py --infile $OUT --metric t_infer_ms
