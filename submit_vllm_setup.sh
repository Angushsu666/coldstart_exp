#!/bin/bash
#SBATCH --job-name=vllm_setup
#SBATCH --partition=amilan
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=slurm-%j.out
#SBATCH --account=ucb-general
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=chhs8331@colorado.edu

set -euo pipefail

echo "=== Job started on $(hostname) at $(date) ==="

# Fail fast if HF_TOKEN wasn't passed through the environment.
# Submit with: sbatch --export=ALL,HF_TOKEN submit_vllm_setup.sh
if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set in sbatch environment"
    exit 1
fi

# Check headroom before creating the env (~12-15GB needed: torch ~3GB + CUDA libs + vLLM).
echo "=== Disk usage: /projects/$USER/software/anaconda/ ==="
du -sh /projects/$USER/software/anaconda/

ENV_PREFIX="/projects/$USER/software/anaconda/envs/vllm-coldstart"
export HF_HOME="/scratch/alpine/$USER/hf_cache"

module purge
module load anaconda

# --- 1. Create conda env (Python 3.11, best-tested vLLM target) ---
echo "=== Creating conda env: $ENV_PREFIX ==="
conda create --prefix "$ENV_PREFIX" python=3.11 -y

conda activate "$ENV_PREFIX"

# --- 2. Install vLLM (latest stable; hint CUDA 12.1 for torch dep) ---
echo "=== Installing vLLM ==="
pip install vllm --extra-index-url https://download.pytorch.org/whl/cu121

# --- 3. Pre-download Llama-3-8B weights to scratch (not /home) ---
echo "=== Downloading meta-llama/Meta-Llama-3-8B to $HF_HOME ==="
mkdir -p "$HF_HOME"
huggingface-cli download meta-llama/Meta-Llama-3-8B

# --- 4. Sanity checks (no GPU load; cuda.is_available() will be False on amilan) ---
echo "=== Sanity checks ==="
python -c "import vllm; print('vllm', vllm.__version__)"
python -c "import torch; print('torch', torch.__version__, '| CUDA avail (expect False on amilan):', torch.cuda.is_available())"
du -sh "$HF_HOME/models--meta-llama--Meta-Llama-3-8B/"

echo "=== Done at $(date) ==="
