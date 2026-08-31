#!/usr/bin/env bash
# Restart Qwen3 vLLM whenever it exits unexpectedly.  Stop the loop with
# `kill <this-script-pid>` (SIGTERM) or Ctrl-C; SIGKILL cannot be trapped.
set -uo pipefail

# Usage:
#   bash utils/serve_qwen3_32b_loop.sh 2 true
# Optional: RESTART_DELAY_S=0 bash utils/serve_qwen3_32b_loop.sh 2 true
PORT=6658
TENSOR_PARALLEL_SIZE="${1:-${TENSOR_PARALLEL_SIZE:-8}}"
USE_VL="${2:-${USE_VL:-false}}"
RESTART_DELAY_S="${RESTART_DELAY_S:-1}"

if [[ "${USE_VL}" == "true" ]]; then
  MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-vl-32b}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-32B}"
else
  MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-32b}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
fi

child_pid=""
stopping=0

stop_loop() {
  stopping=1
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    echo "[$(date '+%F %T')] stopping vLLM pid=${child_pid}" >&2
    kill -TERM "${child_pid}" 2>/dev/null || true
  fi
}

trap stop_loop INT TERM

echo "served_model=${SERVED_MODEL_NAME} port=${PORT} restart_delay_s=${RESTART_DELAY_S}" >&2
while (( ! stopping )); do
  echo "[$(date '+%F %T')] starting vLLM" >&2
  vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}" \
    --max-model-len 48000 \
    --port "${PORT}" \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --max-num-seqs 512 \
    --allowed-local-media-path /mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL &
  child_pid=$!

  wait "${child_pid}"
  exit_status=$?
  child_pid=""

  if (( stopping )); then
    break
  fi
  echo "[$(date '+%F %T')] vLLM exited with status=${exit_status}; restarting in ${RESTART_DELAY_S}s" >&2
  sleep "${RESTART_DELAY_S}" &
  wait $! || true
done

echo "[$(date '+%F %T')] vLLM restart loop stopped" >&2
