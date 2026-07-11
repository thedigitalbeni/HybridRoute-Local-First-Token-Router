#!/usr/bin/env bash
set -e

# Fast local iteration loop -- runs main.py directly against practice_tasks.json,
# no Docker involved. Use this to sanity-check answers before burning a real
# submission slot (rate-limited to 10/hour per team).
#
# Requires:
#   - model.gguf present (run download_model.sh first)
#   - your team's real Fireworks credentials + ALLOWED_MODELS list

: "${FIREWORKS_API_KEY:?set FIREWORKS_API_KEY first}"
: "${FIREWORKS_BASE_URL:=https://api.fireworks.ai/inference/v1}"
: "${ALLOWED_MODELS:?set ALLOWED_MODELS to the comma-separated list from launch day}"

export FIREWORKS_API_KEY
export FIREWORKS_BASE_URL
export ALLOWED_MODELS
export LOCAL_MODEL_PATH="$(pwd)/model.gguf"
export TASKS_INPUT_PATH="$(pwd)/practice_tasks.json"
export TASKS_OUTPUT_PATH="$(pwd)/practice_output.json"

python3 main.py

echo ""
echo "=== practice_output.json ==="
cat practice_output.json
