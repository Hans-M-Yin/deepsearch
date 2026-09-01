#!/usr/bin/env bash
# Start Qwen3-VL-32B with SGLang Model Gateway and two TP4 replicas.
#
# The integrated SGLang Router owns the worker processes and exposes one
# OpenAI-compatible endpoint.  This uses all eight visible GPUs as:
#
#   data parallel size 2 x tensor parallel size 4 = 8 GPUs
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#     bash utils/serve_qwen3_vl_32b_dp2_loop_sglang.sh
#
# The loop restarts the router and its managed workers if the launcher exits.
# For independently managed workers with fixed backend ports (for example
# 6661 and 6662), use separate `sglang.launch_server` processes and
# `sglang_router.launch_router` instead of this co-launch mode.
set -uo pipefail

HOST="${SGLANG_HOST:-0.0.0.0}"
PORT="${SGLANG_PORT:-6658}"
MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/qwen3-vl-32b}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-32B}"

# These two values must multiply to the number of visible GPUs.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-2}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-48000}"
MEM_FRACTION_STATIC="${MEM_FRACTION_STATIC:-0.88}"
MAX_RUNNING_REQUESTS="${MAX_RUNNING_REQUESTS:-64}"
CHUNKED_PREFILL_SIZE="${CHUNKED_PREFILL_SIZE:-12288}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-fa3}"
MM_ATTENTION_BACKEND="${MM_ATTENTION_BACKEND:-fa3}"

ROUTER_POLICY="${ROUTER_POLICY:-cache_aware}"
ROUTER_HEALTH_CHECK_INTERVAL_S="${ROUTER_HEALTH_CHECK_INTERVAL_S:-30}"
ROUTER_PROMETHEUS_PORT="${ROUTER_PROMETHEUS_PORT:-6661}"
RESTART_DELAY_S="${RESTART_DELAY_S:-1}"
SGLANG_PYTHON="${SGLANG_PYTHON:-python}"

stopping=0
child_pid=""

stop_loop() {
  stopping=1
  if [[ -n "${child_pid}" ]] && kill -0 "${child_pid}" 2>/dev/null; then
    echo "[$(date '+%F %T')] stopping SGLang Router pid=${child_pid}" >&2
    kill -TERM "${child_pid}" 2>/dev/null || true
  fi
}

trap stop_loop INT TERM

# SGLang has used both --tp and --tp-size, and both --policy and
# --router-policy across releases.  Detect the spelling supported by the
# installed router so this script remains usable across those releases.
if ! router_help="$("${SGLANG_PYTHON}" -m sglang_router.launch_server --help 2>&1)"; then
  echo "Unable to inspect sglang_router.launch_server. Is sglang-router installed in this environment?" >&2
  echo "${router_help}" >&2
  exit 1
fi

has_router_option() {
  grep -Fq -- "$1" <<<"${router_help}"
}
echo "${PORT} ${HOST}"
router_args=(
  -m sglang_router.launch_server
  --model-path "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --dp-size "${DATA_PARALLEL_SIZE}"
  --context-length "${MAX_MODEL_LEN}"
  --host "${HOST}"
  --port "${PORT}"
  --mem-fraction-static "${MEM_FRACTION_STATIC}"
  --max-running-requests "${MAX_RUNNING_REQUESTS}"
  --chunked-prefill-size "${CHUNKED_PREFILL_SIZE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --mm-attention-backend "${MM_ATTENTION_BACKEND}"
  --enable-multimodal
  --trust-remote-code
)

if has_router_option "--tp-size"; then
  router_args+=(--tp-size "${TENSOR_PARALLEL_SIZE}")
elif has_router_option "--tp "; then
  router_args+=(--tp "${TENSOR_PARALLEL_SIZE}")
else
  echo "The installed SGLang Router does not expose --tp-size or --tp." >&2
  exit 1
fi

if [[ -n "${ROUTER_POLICY}" ]]; then
  if has_router_option "--router-policy"; then
    router_args+=(--router-policy "${ROUTER_POLICY}")
  elif has_router_option "--policy"; then
    router_args+=(--policy "${ROUTER_POLICY}")
  else
    echo "Warning: this SGLang Router has no policy flag; using its default policy." >&2
  fi
fi

if has_router_option "--router-health-check-interval-secs"; then
  router_args+=(--router-health-check-interval-secs "${ROUTER_HEALTH_CHECK_INTERVAL_S}")
elif has_router_option "--health-check-interval-secs"; then
  router_args+=(--health-check-interval-secs "${ROUTER_HEALTH_CHECK_INTERVAL_S}")
else
  echo "Warning: this SGLang Router has no recognized health-check interval flag." >&2
fi

if [[ -n "${ROUTER_PROMETHEUS_PORT}" ]]; then
  if has_router_option "--router-prometheus-port"; then
    router_args+=(--router-prometheus-port "${ROUTER_PROMETHEUS_PORT}")
  elif has_router_option "--prometheus-port"; then
    router_args+=(--prometheus-port "${ROUTER_PROMETHEUS_PORT}")
  else
    echo "Warning: this SGLang Router has no recognized Prometheus port flag." >&2
  fi
fi

echo "served_model=${SERVED_MODEL_NAME} port=${PORT} tp=${TENSOR_PARALLEL_SIZE} dp=${DATA_PARALLEL_SIZE} policy=${ROUTER_POLICY} restart_delay_s=${RESTART_DELAY_S}" >&2
echo "model_path=${MODEL_PATH} context_length=${MAX_MODEL_LEN} mem_fraction_static=${MEM_FRACTION_STATIC} max_running_requests=${MAX_RUNNING_REQUESTS}" >&2
echo "attention_backend=${ATTENTION_BACKEND} mm_attention_backend=${MM_ATTENTION_BACKEND} chunked_prefill_size=${CHUNKED_PREFILL_SIZE}" >&2

while (( ! stopping )); do
  echo "[$(date '+%F %T')] starting SGLang Model Gateway with ${DATA_PARALLEL_SIZE} replicas" >&2
  "${SGLANG_PYTHON}" "${router_args[@]}" &
  child_pid=$!

  wait "${child_pid}"
  exit_status=$?
  child_pid=""

  if (( stopping )); then
    break
  fi
  echo "[$(date '+%F %T')] SGLang Router exited with status=${exit_status}; restarting in ${RESTART_DELAY_S}s" >&2
  sleep "${RESTART_DELAY_S}" &
  wait $! || true
done

echo "[$(date '+%F %T')] SGLang Router restart loop stopped" >&2
