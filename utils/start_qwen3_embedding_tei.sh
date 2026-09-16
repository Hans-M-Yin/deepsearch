#!/usr/bin/env bash
# Start a native (non-Docker) Qwen3 embedding service for search-similarity reward shaping.
#
# Prerequisite: build TEI once with
#   cargo install --path router --features candle-cuda
# from the text-embeddings-inference source root.
set -euo pipefail

MODEL_SIZE="${EMBEDDING_MODEL_SIZE:-0.6B}"
case "${MODEL_SIZE,,}" in
  0.6b|0p6b)
    MODEL_LABEL="0.6B"
    DEFAULT_MODEL_SUBDIR="qwen3-embedding-0p6b"
    DEFAULT_PORT="8086"
    DEFAULT_SERVICE_NAME="qwen3-embedding-0p6b"
    ;;
  4b)
    MODEL_LABEL="4B"
    DEFAULT_MODEL_SUBDIR="qwen3-embedding-4b"
    DEFAULT_PORT="8087"
    DEFAULT_SERVICE_NAME="qwen3-embedding-4b"
    ;;
  *)
    echo "EMBEDDING_MODEL_SIZE must be 0.6B or 4B, got: ${MODEL_SIZE}" >&2
    exit 1
    ;;
esac

MODEL_ROOT="${EMBEDDING_MODEL_ROOT:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models}"
MODEL_DIR="${EMBEDDING_MODEL_DIR:-${MODEL_ROOT}/${DEFAULT_MODEL_SUBDIR}}"
PORT="${EMBEDDING_PORT:-${DEFAULT_PORT}}"
BIND_HOST="${EMBEDDING_BIND_HOST:-0.0.0.0}"
SERVICE_NAME="${EMBEDDING_SERVICE_NAME:-${DEFAULT_SERVICE_NAME}}"
ROUTER_BIN="${EMBEDDING_ROUTER_BIN:-${HOME}/.cargo/bin/text-embeddings-router}"
GPU_DEVICE="${EMBEDDING_GPU_DEVICE:?Set EMBEDDING_GPU_DEVICE to one dedicated physical GPU, e.g. 0}"
STARTUP_TIMEOUT_S="${EMBEDDING_STARTUP_TIMEOUT_S:-600}"
HEALTH_URL="${EMBEDDING_HEALTH_URL:-http://127.0.0.1:${PORT}/health}"
PID_FILE="${EMBEDDING_PID_FILE:-/tmp/${SERVICE_NAME}.pid}"
LOG_FILE="${EMBEDDING_LOG_FILE:-/tmp/${SERVICE_NAME}.log}"
MAX_CONCURRENT_REQUESTS="${EMBEDDING_MAX_CONCURRENT_REQUESTS:-128}"
MAX_BATCH_REQUESTS="${EMBEDDING_MAX_BATCH_REQUESTS:-64}"
MAX_CLIENT_BATCH_SIZE="${EMBEDDING_MAX_CLIENT_BATCH_SIZE:-64}"
MAX_BATCH_TOKENS="${EMBEDDING_MAX_BATCH_TOKENS:-8192}"

has_weights=false
if [[ -f "${MODEL_DIR}/model.safetensors" ]]; then
  has_weights=true
elif [[ -f "${MODEL_DIR}/model.safetensors.index.json" ]] \
  && compgen -G "${MODEL_DIR}/model-*.safetensors" >/dev/null; then
  has_weights=true
fi
if [[ ! -f "${MODEL_DIR}/config.json" || "${has_weights}" != "true" ]]; then
  echo "Checkpoint is absent or incomplete: ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -x "${ROUTER_BIN}" ]]; then
  echo "Native TEI router is not executable: ${ROUTER_BIN}" >&2
  echo "Build it first from text-embeddings-inference: cargo install --path router --features candle-cuda" >&2
  exit 1
fi

if [[ -f "${PID_FILE}" ]]; then
  existing_pid="$(<"${PID_FILE}")"
  if kill -0 "${existing_pid}" 2>/dev/null; then
    existing_cmd="$(ps -p "${existing_pid}" -o args= 2>/dev/null || true)"
    if [[ "${existing_cmd}" == *"text-embeddings-router"* ]]; then
      echo "[embedding] stopping existing native service pid=${existing_pid}"
      kill "${existing_pid}"
      for _ in {1..30}; do
        if ! kill -0 "${existing_pid}" 2>/dev/null; then
          break
        fi
        sleep 1
      done
    else
      echo "Refusing to kill pid=${existing_pid}: it is not text-embeddings-router." >&2
      echo "Remove stale PID file only after checking that process." >&2
      exit 1
    fi
  fi
  rm -f "${PID_FILE}"
fi

router_args=(
  --model-id "${MODEL_DIR}"
  --hostname "${BIND_HOST}"
  --port "${PORT}"
  --dtype float16
  --max-concurrent-requests "${MAX_CONCURRENT_REQUESTS}"
  --max-batch-requests "${MAX_BATCH_REQUESTS}"
  --max-client-batch-size "${MAX_CLIENT_BATCH_SIZE}"
  --max-batch-tokens "${MAX_BATCH_TOKENS}"
  --auto-truncate
)

echo "[embedding] backend=native size=${MODEL_LABEL} model=${MODEL_DIR}"
echo "[embedding] gpu=${GPU_DEVICE} bind=${BIND_HOST}:${PORT} router=${ROUTER_BIN}"
echo "[embedding] log=${LOG_FILE} pid_file=${PID_FILE}"
CUDA_VISIBLE_DEVICES="${GPU_DEVICE}" nohup "${ROUTER_BIN}" "${router_args[@]}" >"${LOG_FILE}" 2>&1 &
service_pid=$!
printf '%s\n' "${service_pid}" >"${PID_FILE}"

deadline=$((SECONDS + STARTUP_TIMEOUT_S))
until curl --noproxy '*' -fsS "${HEALTH_URL}" >/dev/null 2>&1; do
  if ! kill -0 "${service_pid}" 2>/dev/null; then
    echo "[embedding] native service exited during startup; recent log:" >&2
    tail -n 120 "${LOG_FILE}" >&2 || true
    rm -f "${PID_FILE}"
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "[embedding] startup timed out after ${STARTUP_TIMEOUT_S}s; recent log:" >&2
    tail -n 120 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  sleep 2
done

echo "[embedding] healthy: ${HEALTH_URL}"
echo "[embedding] pid=${service_pid} endpoint=http://<this-host>:${PORT}/embed"
