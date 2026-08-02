#!/usr/bin/env bash
# Launches the teacher as an OpenAI-compatible vLLM server, tensor-parallel
# across a dedicated pair of GPUs, so it can be queried independently of
# (and concurrently with) student training on the remaining GPUs.
#
# Usage:
#   bash serve_teacher_vllm.sh [model] [gpus] [port]
#   bash serve_teacher_vllm.sh                                   # defaults: config.yaml's teacher on GPUs 0,1, port 8000
#   bash serve_teacher_vllm.sh Qwen/Qwen2.5-32B-Instruct 0,1 8000
#
# Leave this running in its own terminal/tmux pane. Run the other scripts
# (which query http://localhost:$PORT/v1) from a different shell.
set -euo pipefail

MODEL=${1:-Qwen/Qwen2.5-32B-Instruct}
GPUS=${2:-0,1}
PORT=${3:-8000}
TP_SIZE=$(echo "$GPUS" | awk -F',' '{print NF}')

echo "Serving $MODEL on GPUs [$GPUS] (tensor-parallel-size=$TP_SIZE) at port $PORT ..."
CUDA_VISIBLE_DEVICES="$GPUS" vllm serve "$MODEL" \
  --tensor-parallel-size "$TP_SIZE" \
  --port "$PORT" \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
