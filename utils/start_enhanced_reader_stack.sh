#!/usr/bin/env bash
set -euo pipefail
# Check log
#tail -f /workspace/reader/reader_log
 #tail -f /mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL/vllm_reader_lm_log
 #tail -f /mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL/enhanced_reader_log
# Paths can be overridden by exporting env vars before running this script.
READER_DIR="${READER_DIR:-/workspace/reader}"
PROJECT_DIR="${PROJECT_DIR:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL}"
READERLM_MODEL_PATH="${READERLM_MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/reader_lm}"

# Jina Reader starts an h2c server on PORT and an HTTP/1.1 alternative server
# on PORT+1. We set PORT=8001 so the wrapper can call the HTTP/1.1 endpoint at 8002.
READER_PORT="${READER_PORT:-8001}"
RAW_READER_PORT="${RAW_READER_PORT:-8002}"
READERLM_PORT="${READERLM_PORT:-8003}"
ENHANCED_READER_PORT="${ENHANCED_READER_PORT:-8004}"

READER_LOG="${READER_LOG:-${READER_DIR}/reader_log}"
VLLM_READER_LM_LOG="${VLLM_READER_LM_LOG:-${PROJECT_DIR}/vllm_reader_lm_log}"
ENHANCED_READER_LOG="${ENHANCED_READER_LOG:-${PROJECT_DIR}/enhanced_reader_log}"

READERLM_API_BASE="${READERLM_API_BASE:-http://127.0.0.1:${READERLM_PORT}/v1}"
RAW_READER_URL="${RAW_READER_URL:-http://127.0.0.1:${RAW_READER_PORT}}"
READERLM_SERVED_MODEL_NAME="${READERLM_SERVED_MODEL_NAME:-jinaai/ReaderLM-v2}"
READERLM_MODEL_NAME="${READERLM_MODEL_NAME:-${READERLM_SERVED_MODEL_NAME}}"
READERLM_MAX_HTML_CHARS="${READERLM_MAX_HTML_CHARS:-120000}"
READERLM_MAX_TOKENS="${READERLM_MAX_TOKENS:-8192}"
ENHANCED_READER_TIMEOUT="${ENHANCED_READER_TIMEOUT:-180}"

# vLLM throughput knobs for ReaderLM. ReaderLM-v2 is small, so on an H100 80G
# it is usually better to admit more concurrent sequences than to shard one
# replica with tensor parallelism. Override these env vars before running this
# script if the server OOMs or if throughput is still low.
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-1}"
VLLM_DISABLE_LOG_REQUESTS="${VLLM_DISABLE_LOG_REQUESTS:-1}"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

port_in_use() {
  local port="$1"
  ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
}

start_reader() {
  require_dir "${READER_DIR}" "Reader directory"
  if port_in_use "${RAW_READER_PORT}"; then
    echo "Reader appears to already be listening on ${RAW_READER_PORT}; skipping."
    return
  fi

  echo "Starting Jina Reader from ${READER_DIR} ..."
  (
    cd "${READER_DIR}"
    nohup env PORT="${READER_PORT}" npm run start > "${READER_LOG}" 2>&1 &
    echo $! > "${READER_DIR}/reader.pid"
  )
  echo "Reader log: ${READER_LOG}"
}

start_readerlm() {
  require_dir "${READERLM_MODEL_PATH}" "ReaderLM model path"
  if port_in_use "${READERLM_PORT}"; then
    echo "ReaderLM appears to already be listening on ${READERLM_PORT}; skipping."
    return
  fi

  echo "Starting ReaderLM with vLLM from ${READERLM_MODEL_PATH} ..."
  (
    cd "${PROJECT_DIR}"
    nohup vllm serve "${READERLM_MODEL_PATH}" \
      --host 0.0.0.0 \
      --port "${READERLM_PORT}" \
      --served-model-name "${READERLM_SERVED_MODEL_NAME}" \
      --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION}" \
      --max-num-seqs "${VLLM_MAX_NUM_SEQS}" \
      --max-num-batched-tokens "${VLLM_MAX_NUM_BATCHED_TOKENS}" \
      $([[ "${VLLM_ENABLE_PREFIX_CACHING}" == "1" ]] && echo "--enable-prefix-caching") \
      $([[ "${VLLM_DISABLE_LOG_REQUESTS}" == "1" ]] && echo "--disable-log-requests") \
      > "${VLLM_READER_LM_LOG}" 2>&1 &
    echo $! > "${PROJECT_DIR}/vllm_reader_lm.pid"
  )
  echo "ReaderLM vLLM log: ${VLLM_READER_LM_LOG}"
}

start_enhanced_reader() {
  require_dir "${PROJECT_DIR}" "OpenSearch-VL project directory"
  if port_in_use "${ENHANCED_READER_PORT}"; then
    echo "Enhanced Reader appears to already be listening on ${ENHANCED_READER_PORT}; skipping."
    return
  fi

  echo "Starting Enhanced Reader from ${PROJECT_DIR} ..."
  (
    cd "${PROJECT_DIR}"
    nohup env \
      RAW_READER_URL="${RAW_READER_URL}" \
      READERLM_API_BASE="${READERLM_API_BASE}" \
      READERLM_MODEL_NAME="${READERLM_MODEL_NAME}" \
      READERLM_MAX_HTML_CHARS="${READERLM_MAX_HTML_CHARS}" \
      READERLM_MAX_TOKENS="${READERLM_MAX_TOKENS}" \
      ENHANCED_READER_TIMEOUT="${ENHANCED_READER_TIMEOUT}" \
      uvicorn utils.enhanced_reader:app \
        --host 0.0.0.0 \
        --port "${ENHANCED_READER_PORT}" \
      > "${ENHANCED_READER_LOG}" 2>&1 &
    echo $! > "${PROJECT_DIR}/enhanced_reader.pid"
  )
  echo "Enhanced Reader log: ${ENHANCED_READER_LOG}"
}

main() {
  require_dir "${PROJECT_DIR}" "OpenSearch-VL project directory"

  start_reader
  start_readerlm
  start_enhanced_reader

  cat <<EOF

Startup commands have been issued.

Endpoints:
  Raw Reader HTML endpoint: ${RAW_READER_URL}
  ReaderLM API endpoint:    ${READERLM_API_BASE}
  Enhanced Reader endpoint: http://127.0.0.1:${ENHANCED_READER_PORT}

Use this for OpenSearch-VL:
  export JINA_READER_URL="http://127.0.0.1:${ENHANCED_READER_PORT}"

Logs:
  Reader:          ${READER_LOG}
  ReaderLM vLLM:   ${VLLM_READER_LM_LOG}
  Enhanced Reader: ${ENHANCED_READER_LOG}

Quick test after services finish loading:
  curl -H 'Accept: application/json' 'http://127.0.0.1:${ENHANCED_READER_PORT}/https://example.com'
EOF
}

main "$@"
