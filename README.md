# Cold Start Measurement — DistilGPT-2 on GPU

## Project Structure

```
coldstart_exp/
├── server.py            # FastAPI inference server (for GCP/Kubernetes)
├── measure_single.py    # Standalone cold start measurement script (for CURC)
├── measure_nccl.py      # NCCL distributed init measurement (for CURC, 2 GPU)
├── summarize_jsonl.py   # Stats summary (p50/p95/p99) for CURC results
├── submit_coldstart.sh  # Slurm batch job script (CURC only, do NOT use on GCP)
├── Dockerfile           # Container image for GCP deployment
├── requirements.txt     # Python dependencies
└── results/             # Output directory for measurement data
```

---

## For Angus (CURC / Alpine)

### One-time setup
```bash
module purge
module load anaconda
conda activate coldstart

# Cache the model locally (only needed once)
python -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
AutoTokenizer.from_pretrained('distilgpt2')
AutoModelForCausalLM.from_pretrained('distilgpt2')
print('Cached.')
"
```

### Run experiments (sbatch)
```bash
cd ~/coldstart_exp
sbatch submit_coldstart.sh
```

Results saved to `results/slurm-<JOBID>.out`.  
Email notifications sent to chhs8331@colorado.edu on BEGIN/END/FAIL.

### What you measure
| Field | Corresponds to |
|-------|---------------|
| model_load_ms | T_model |
| cuda_init_ms | T_cuda |
| nccl_init_ms | T_dist |
| first_infer_ms | T_infer |
| total_cold_local_ms | T_model + T_cuda + T_dist + T_infer |

---

## For Jonathan (GCP / Kubernetes)

### Prerequisites
- GCP project with GKE cluster
- NVIDIA T4 GPU node pool
- Docker + gcloud CLI installed locally

### Step 1: Build and push Docker image
```bash
PROJECT_ID=your-gcp-project-id
IMAGE=gcr.io/$PROJECT_ID/coldstart-distilgpt2:latest

docker build -t $IMAGE .
docker push $IMAGE
```

### Step 2: Deploy to GKE — three warm level variants

Deploy each variant as a separate Kubernetes Deployment.
The only difference between them is the `WARM_LEVEL` environment variable.

**scale-to-zero (baseline):**
```bash
kubectl create deployment coldstart-scale-to-zero \
  --image=$IMAGE \
  --env="WARM_LEVEL=scale-to-zero"
kubectl expose deployment coldstart-scale-to-zero --port=8000
```

**container-warm:**
```bash
kubectl create deployment coldstart-container-warm \
  --image=$IMAGE \
  --env="WARM_LEVEL=container-warm"
kubectl expose deployment coldstart-container-warm --port=8000
```

**gpu-warm:**
```bash
kubectl create deployment coldstart-gpu-warm \
  --image=$IMAGE \
  --env="WARM_LEVEL=gpu-warm"
kubectl expose deployment coldstart-gpu-warm --port=8000
```

> For GPU nodes, add `--limits=nvidia.com/gpu=1` to each deployment
> or configure it in a proper YAML manifest.

### Step 3: Test the endpoint
```bash
# Health check
curl http://<SERVICE_IP>:8000/health

# Inference request
curl -X POST http://<SERVICE_IP>:8000/infer \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello, how are you?", "max_new_tokens": 20}'
```

### Step 4: Measure T_sched + T_container

Scale deployment to zero replicas, then send a request and record the
end-to-end wall time. The server response includes per-request timings
(t_model_ms, t_cuda_ms, t_infer_ms) so you can subtract them to get:

```
T_sched + T_container = wall_time - (t_model_ms + t_cuda_ms + t_infer_ms)
```

### API Response Fields

| Field | Description |
|-------|-------------|
| t_model_ms | T_model: model load time on this request |
| t_cuda_ms | T_cuda: CUDA init time on this request |
| t_to_gpu_ms | model transfer to GPU time |
| t_warmup_ms | warmup forward pass time |
| t_infer_ms | T_infer: actual inference time (TTFT proxy) |
| total_request_ms | total time server spent on this request |
| startup_model_load_ms | T_model at container startup (gpu-warm only) |
| startup_cuda_init_ms | T_cuda at container startup (gpu-warm only) |
| warm_level | which strategy this container is running |

---

## WARM_LEVEL Reference

| WARM_LEVEL | What happens at startup | What happens on first request |
|------------|------------------------|-------------------------------|
| scale-to-zero | nothing | load model + CUDA init + to GPU + warmup + infer |
| container-warm | load model to CPU | CUDA init + to GPU + warmup + infer |
| gpu-warm | load model + CUDA init + to GPU + warmup | infer only |

---

## Experiment Results

### Scale-to-Zero Baseline (GCP, Tesla T4, n=21 cold / 105 steady)

| Metric | Value |
|--------|-------|
| Cold mean | 3873.6 ms |
| Cold p95 | 4043.7 ms |
| Steady mean | 389.5 ms |
| Cold/Steady slowdown | 9.95x |

**Cold start component breakdown:**

| Stage | Field | Mean (ms) | % of wall time |
|-------|-------|-----------|----------------|
| Model load | `t_model_ms` | 2603 ms | 67.2% |
| First inference (TTFT) | `t_infer_ms` | 607 ms | 15.7% |
| Scheduling + container | `t_sched_container_ms` | 269 ms | 7.0% |
| CUDA init | `t_cuda_ms` | 249 ms | 6.4% |
| Model to GPU | `t_to_gpu_ms` | 143 ms | 3.7% |
| GPU warmup | `t_warmup_ms` | 0 ms | 0.0% |

Full stats (mean/median/p95/p99/stdev) available in `results/final_scale_to_zero.json`.

---

### Container-Warm (GCP, Tesla T4, n=20 cold / 100 steady)

| Metric | Value |
|--------|-------|
| Cold mean | 1299.8 ms |
| Cold p95 | 1385.7 ms |
| Steady mean | 384.0 ms |
| Cold/Steady slowdown | 3.38x |
| vs. scale-to-zero | **3.0x faster** |

**Cold start component breakdown (request path only):**

| Stage | Field | Mean (ms) | % of wall time |
|-------|-------|-----------|----------------|
| GPU warmup | `t_warmup_ms` | 420 ms | 32.3% |
| First inference (TTFT) | `t_infer_ms` | 209 ms | 16.1% |
| Scheduling + container | `t_sched_container_ms` | 271 ms | 20.9% |
| CUDA init | `t_cuda_ms` | 257 ms | 19.8% |
| Model to GPU | `t_to_gpu_ms` | 141 ms | 10.9% |
| Model load | `t_model_ms` | 0 ms | 0.0% ✅ |

**Startup cost (paid before request, not in wall_time):**

| Stage | Field | Mean (ms) |
|-------|-------|-----------|
| Model load to CPU | `startup_model_load_ms` | 2531 ms |

> Note: `t_model_ms` is eliminated from the request path (moved to startup).
> `t_warmup_ms` (420 ms) is now the largest single component in the cold request.

Full stats available in `results/final_container_warm.json`.

---

### GPU-Warm (GCP, Tesla T4, n=20 cold / 100 steady)

| Metric | Value |
|--------|-------|
| Cold mean | 475.5 ms |
| Cold p95 | 496.9 ms |
| Steady mean | 383.0 ms |
| Cold/Steady slowdown | 1.24x |
| vs. scale-to-zero | **8.1x faster** |

**Cold start component breakdown (request path only):**

| Stage | Field | Mean (ms) | % of wall time |
|-------|-------|-----------|----------------|
| Scheduling + container | `t_sched_container_ms` | 270 ms | 56.7% |
| First inference (TTFT) | `t_infer_ms` | 204 ms | 42.9% |
| Model load | `t_model_ms` | 0 ms | 0.0% ✅ |
| CUDA init | `t_cuda_ms` | 0 ms | 0.0% ✅ |
| Model to GPU | `t_to_gpu_ms` | 0 ms | 0.0% ✅ |
| GPU warmup | `t_warmup_ms` | 0 ms | 0.0% ✅ |

**Startup cost (paid before request, not in wall_time):**

| Stage | Field | Mean (ms) |
|-------|-------|-----------|
| Model load to CPU | `startup_model_load_ms` | 2458 ms |
| CUDA init | `startup_cuda_init_ms` | 247 ms |
| Model to GPU | `startup_to_gpu_ms` | 141 ms |
| GPU warmup | `startup_warmup_ms` | 401 ms |
| **Total startup** | | **~3247 ms** |

> All initialization moved to startup. Cold request = K8s scheduling + inference only.

Full stats available in `results/final_gpu_warm.json`.

---

### Three-Way Comparison Summary

| warm_level | Cold mean | Cold p95 | Steady mean | Cold/Steady |
|------------|-----------|----------|-------------|-------------|
| scale-to-zero | 3873 ms | 4044 ms | 389 ms | 9.95x |
| container-warm | 1300 ms | 1386 ms | 384 ms | 3.38x |
| gpu-warm | 475 ms | 497 ms | 383 ms | 1.24x |

---

## Notes

- `submit_coldstart.sh` is CURC/Slurm specific. Do NOT use it on GCP.
- Model weights are baked into the Docker image (in `RUN python -c ...`),
  so there is no HuggingFace download at runtime.
- CUDA version in the image (12.1) matches CURC torch build for consistency.
