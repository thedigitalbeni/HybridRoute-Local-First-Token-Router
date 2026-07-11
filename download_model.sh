#!/usr/bin/env bash
set -e

# Run this on YOUR machine or the AMD notebook (notebooks.amd.com/hackathon) --
# it needs real internet access to huggingface.co, which this sandbox can't reach.
#
# Downloads Qwen2.5-3B-Instruct, Q4_K_M quant (~1.9-2.0GB).
#
# WHY 3B INSTEAD OF 1.5B: local answers cost ZERO tokens on the leaderboard
# regardless of model size -- token count only gets charged on calls through
# FIREWORKS_BASE_URL. So there's no token-efficiency tradeoff to running the
# larger model locally, only a RAM/latency one. 3B in 4-bit still fits well
# under the 4GB RAM / 2 vCPU grading budget (per the participant guide: "2B-3B
# 4-bit quantized models are safe"), and it's simply more likely to get the
# local categories (factual/sentiment/summary/ner) right -- which matters a
# lot given the accuracy gate is unforgiving. This is the closest thing to a
# free upgrade in the whole pipeline.
#
# Alternatives if you want to experiment (edit the URL below):
#   - Gemma-2-2b-it-GGUF (Q4_K_M)   -- smaller, different style.
#   - Phi-3.5-mini-instruct-GGUF (Q4_K_M) -- ~2.2GB, similar size class.
#
# Whichever you pick, re-run practice tasks locally (see run_practice.sh) and
# eyeball the answers before trusting it on the real submission.

curl -L -o model.gguf \
  "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"

echo "Downloaded model.gguf ($(du -h model.gguf | cut -f1))"
