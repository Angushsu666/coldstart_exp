#!/bin/bash
# Run N cold-start trials for a single deployment.
# Each trial: scale to 0 → scale to 1 → wait for server → send request.
#
# Usage:
#   bash run_trials.sh <URL> <DEPLOYMENT> [N] [OUT]
#
# Examples:
#   bash run_trials.sh http://34.91.210.213:8000 coldstart-scale-to-zero 20 results/server_gcp.jsonl
#   bash run_trials.sh http://35.204.119.174:8000 coldstart-container-warm 20 results/server_gcp.jsonl
#   bash run_trials.sh http://34.178.68.41:8000   coldstart-gpu-warm       20 results/server_gcp.jsonl

URL=$1
DEPLOYMENT=$2
N=${3:-20}
OUT=${4:-results/server_gcp.jsonl}

if [ -z "$URL" ] || [ -z "$DEPLOYMENT" ]; then
  echo "Usage: bash run_trials.sh <URL> <DEPLOYMENT> [N] [OUT]"
  exit 1
fi

mkdir -p results
echo "Starting $N trials for $DEPLOYMENT → $OUT"

for i in $(seq 1 $N); do
  echo "=== Trial $i/$N ==="

  # Kill pod
  kubectl scale deployment $DEPLOYMENT --replicas=0
  kubectl wait --for=delete pod -l app=$DEPLOYMENT --timeout=60s 2>/dev/null || true
  sleep 2

  # Start fresh pod
  kubectl scale deployment $DEPLOYMENT --replicas=1
  kubectl wait --for=condition=ready pod -l app=$DEPLOYMENT --timeout=180s

  # Wait for server to actually respond (uvicorn + model startup)
  echo "Waiting for server..."
  for j in $(seq 1 60); do
    if curl -sf $URL/health > /dev/null 2>&1; then
      echo "Server ready"
      break
    fi
    sleep 2
  done

  # Measure: 1 cold request + 5 steady-state requests
  python measure_server.py --url $URL --trials 1 --steady-n 5 --out $OUT
done

echo "All $N trials done. Results in $OUT"
