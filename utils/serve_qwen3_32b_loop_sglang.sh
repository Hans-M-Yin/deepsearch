#!/usr/bin/env bash
# Restart Qwen3 SGLang whenever it exits unexpectedly. Stop the loop with
# `kill <this-script-pid>` (SIGTERM) or Ctrl-C; SIGKILL cannot be trapped.
set -uo pipefail

# Usage:
#   bash utils/serve_qwen3_32b_loop_sglang.sh 2 true
# Optional:
#   RESTART_DELAY_S=0 SGLANG_PYTHON=/path/to/python \
#     bash utils/serve_qwen3_32b_loop_sglang.sh 2 true
#
# The first argument is the tensor-parallel size. The second argument selects
# Qwen3-VL when it is exactly "true"; otherwise the text-only Qwen3 model is
# served.
HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-6658}"
TENSOR_PARALLEL_SIZE="${1:-${TENSOR_PARALLEL_SIZE:-8}}"
USE_VL="${2:-${USE_VL:-false}}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-48000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.9}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-512}"
RESTART_DELAY_S="${RESTART_DELAY_S:-1}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python}"

if [[ "${USE_VL}" == "true" ]]; then
  MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-vl-32b}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-32B}"
  SGLANG_MM_ARGS=(--enable-multimodal)
else
  MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-32b}"
  SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-32b}"
  SGLANG_MM_ARGS=()
fi

child_pid=""
stopping=0

stop_loop() {
  stopping=1
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    echo "[$(date '+%F %T')] stopping SGLang pid=${child_pid}" >&2
    kill -TERM "${child_pid}" 2>/dev/null || true
  fi
}

trap stop_loop INT TERM

echo "served_model=${SERVED_MODEL_NAME} port=${PORT} tp=${TENSOR_PARALLEL_SIZE} restart_delay_s=${RESTART_DELAY_S}" >&2
while (( ! stopping )); do
  echo "[$(date '+%F %T')] starting SGLang" >&2
  "${SGLANG_PYTHON}" -m sglang.launch_server \
    --model-path "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --tp "${TENSOR_PARALLEL_SIZE}" \
    --context-length "${MAX_MODEL_LEN}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --mem-fraction-static "${MEM_FRACTION_STATIC}" \
    --max-running-requests "${MAX_RUNNING_REQUESTS}" \
    --trust-remote-code \
    "${SGLANG_MM_ARGS[@]}" &
  child_pid=$!

  wait "${child_pid}"
  exit_status=$?
  child_pid=""

  if (( stopping )); then
    break
  fi
  echo "[$(date '+%F %T')] SGLang exited with status=${exit_status}; restarting in ${RESTART_DELAY_S}s" >&2
  sleep "${RESTART_DELAY_S}" &
  wait $! || true
done

echo "[$(date '+%F %T')] SGLang restart loop stopped" >&2
