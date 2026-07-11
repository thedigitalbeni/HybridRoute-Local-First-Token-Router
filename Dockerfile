FROM python:3.11-slim

WORKDIR /app

# build-essential/cmake needed to build llama-cpp-python's C++ backend
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py router.py local_infer.py remote_infer.py ./

# model.gguf must exist before building -- run download_model.sh first.
# This is the single biggest contributor to image size; keep it well under
# the 10GB compressed limit (a 1.5-3B Q4_K_M quant is ~1-2GB).
COPY model.gguf ./model.gguf

ENTRYPOINT ["python", "main.py"]
