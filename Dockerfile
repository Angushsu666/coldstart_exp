# Base image with CUDA 12.1 + Python 3.11 (matches CURC torch 2.5.1+cu121)
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python3 && \
    ln -sf /usr/bin/python3 /usr/bin/python

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download and cache distilgpt2 into the image
# This eliminates the HuggingFace download on every cold start
RUN python -c "\
from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('distilgpt2'); \
AutoModelForCausalLM.from_pretrained('distilgpt2'); \
print('Model cached inside image.')"

# Copy application code
COPY server.py .
COPY measure_single.py .
COPY measure_nccl.py .
COPY summarize_jsonl.py .

# WARM_LEVEL controls cold start strategy:
#   scale-to-zero  -> nothing pre-loaded (baseline)
#   container-warm -> model loaded to CPU at startup
#   gpu-warm       -> model loaded to GPU + warmup at startup
ENV WARM_LEVEL=scale-to-zero

EXPOSE 8000

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
