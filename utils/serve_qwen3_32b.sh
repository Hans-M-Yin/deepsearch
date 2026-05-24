#!/usr/bin/env bash
set -euo pipefail

# Notice: Qwen3-32B by default.
# Set USE_VL=true for Qwen3-VL-32B; set USE_VL=false for text-only Qwen3-32B.
# Usage:
#   bash utils/serve_qwen3_32b.sh 2 true
PORT="${PORT:-6658}"
TENSOR_PARALLEL_SIZE="${1:-${TENSOR_PARALLEL_SIZE:-8}}"
USE_VL="${2:-${USE_VL:-false}}"

if [[ "${USE_VL}" == "true" ]]; then
  MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-vl-32b}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-vl-32b}"
else
  MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-32b}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
fi
echo $SERVED_MODEL_NAME
echo $PORT
vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
  --max-model-len 12000 \
  --port "${PORT}" \
  --gpu-memory-utilization 0.8 \
  --trust-remote-code
