#!/usr/bin/env bash
set -euo pipefail

# Edit these two defaults, or override them when running:
#   CHECKPOINT=30b-a3b TP_SIZE=8 bash opensearch_vl/serve_vllm.sh
CHECKPOINT="${CHECKPOINT:-OpenSearch-VL-8B}"
TP_SIZE="${TP_SIZE:-2}"

PORT=6657
HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"

if [[ "${TP_SIZE}" == "1" ]]; then
  MAX_MODEL_LEN=32768
  MAX_NUM_SEQS=8
  MAX_NUM_BATCHED_TOKENS=32768
  GPU_MEMORY_UTILIZATION=0.90
elif [[ "${TP_SIZE}" == "2" ]]; then
  MAX_MODEL_LEN=49152
  MAX_NUM_SEQS=16
  MAX_NUM_BATCHED_TOKENS=65536
  GPU_MEMORY_UTILIZATION=0.92
elif [[ "${TP_SIZE}" == "4" ]]; then
  MAX_MODEL_LEN=65536
  MAX_NUM_SEQS=32
  MAX_NUM_BATCHED_TOKENS=98304
  GPU_MEMORY_UTILIZATION=0.93
elif [[ "${TP_SIZE}" == "8" ]]; then
  MAX_MODEL_LEN=131072
  MAX_NUM_SEQS=64
  MAX_NUM_BATCHED_TOKENS=131072
  GPU_MEMORY_UTILIZATION=0.94
else
  echo "Unsupported TP_SIZE=${TP_SIZE}; expected 1, 2, 4, or 8." >&2
  exit 1
fi

if [[ "${CHECKPOINT}" == "OpenSearch-VL-8B" ]]; then
  MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/OpenSearch-VL-8B}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-OpenSearch-VL-8B}"
elif [[ "${CHECKPOINT}" == "VDR-8B" ]]; then
  MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/Vision-DeepResearch-8B}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Vision-DeepResearch-8B}"
elif [[ "${CHECKPOINT}" == "30b-a3b" ]]; then
  MODEL_PATH="${MODEL_PATH:-/path/to/OpenSearch-VL-30B-A3B}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-OpenSearch-VL-30B-A3B}"
else
  MODEL_PATH="${MODEL_PATH:-${CHECKPOINT}}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$(basename "${CHECKPOINT}")}"
fi

echo "Serving ${SERVED_MODEL_NAME} from ${MODEL_PATH}"
echo "AGENT_BASE_URL=http://localhost:${PORT}/v1"
echo "AGENT_MODEL_NAME=${SERVED_MODEL_NAME}"

vllm serve "${MODEL_PATH}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --dtype "${DTYPE}" \
  --max-model-len "${MAX_MODEL_LEN}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
  --trust-remote-code
