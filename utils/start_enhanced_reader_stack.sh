#!/usr/bin/env bash
set -euo pipefail
# Check log
#tail -f /workspace/reader/reader_log
 #tail -f /mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL/vllm_reader_lm_log
 #tail -f /mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL/enhanced_reader_log
# Paths can be overridden by exporting env vars before running this script.
READER_DIR="${READER_DIR:-/workspace/reader}"
PROJECT_DIR="${PROJECT_DIR:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL}"

# Jina Reader starts an h2c server on PORT and an HTTP/1.1 alternative server
# on PORT+1. Port 8002 is now the public dispatcher; backend Readers use a
# separate range so the dispatcher can keep the existing endpoint stable.
RAW_READER_PORT="${RAW_READER_PORT:-8002}"
READER_BACKEND_BASE_PORT="${READER_BACKEND_BASE_PORT:-8101}"
READER_BACKEND_COUNT="${READER_BACKEND_COUNT:-8}"
READER_BACKEND_PORT_STRIDE="${READER_BACKEND_PORT_STRIDE:-10}"
RAW_MARKDOWN_READER_PORT="${RAW_MARKDOWN_READER_PORT:-8003}"
ENHANCED_READER_PORT="${ENHANCED_READER_PORT:-8004}"
ENHANCED_READER_WORKERS="${ENHANCED_READER_WORKERS:-4}"

READER_LOG="${READER_LOG:-${READER_DIR}/reader_log}"
READER_WORKER_LOG_DIR="${READER_WORKER_LOG_DIR:-${READER_DIR}/reader_workers}"
READER_DISPATCHER_LOG="${READER_DISPATCHER_LOG:-${PROJECT_DIR}/reader_dispatcher_log}"
ENHANCED_READER_LOG="${ENHANCED_READER_LOG:-${PROJECT_DIR}/enhanced_reader_log}"
RAW_MARKDOWN_READER_LOG="${RAW_MARKDOWN_READER_LOG:-${PROJECT_DIR}/raw_markdown_reader_log}"

# These variables are retained only for compatibility with an externally
# deployed ReaderLM. This script never starts a ReaderLM/vLLM process.
READERLM_API_BASE="${READERLM_API_BASE:-http://127.0.0.1:8005/v1}"
READERLM_API_BASES="${READERLM_API_BASES:-}"
RAW_READER_URL="${RAW_READER_URL:-http://127.0.0.1:${RAW_READER_PORT}}"
RAW_MARKDOWN_READER_URL="${RAW_MARKDOWN_READER_URL:-http://127.0.0.1:${RAW_MARKDOWN_READER_PORT}}"
READERLM_SERVED_MODEL_NAME="${READERLM_SERVED_MODEL_NAME:-jinaai/ReaderLM-v2}"
READERLM_MODEL_NAME="${READERLM_MODEL_NAME:-${READERLM_SERVED_MODEL_NAME}}"
READERLM_MAX_HTML_CHARS="${READERLM_MAX_HTML_CHARS:-120000}"
READERLM_MAX_TOKENS="${READERLM_MAX_TOKENS:-8192}"
ENHANCED_READER_TIMEOUT="${ENHANCED_READER_TIMEOUT:-180}"
ENHANCED_READER_FETCH_STRATEGY="${ENHANCED_READER_FETCH_STRATEGY:-markdown_first}"
READER_NODE_MAX_OLD_SPACE_MB="${READER_NODE_MAX_OLD_SPACE_MB:-32768}"
READER_AUTO_RESTART="${READER_AUTO_RESTART:-1}"
READER_RESTART_DELAY_S="${READER_RESTART_DELAY_S:-5}"
READER_DISPATCHER_MAX_CONNECTIONS="${READER_DISPATCHER_MAX_CONNECTIONS:-4096}"
READER_DISPATCHER_MAX_KEEPALIVE_CONNECTIONS="${READER_DISPATCHER_MAX_KEEPALIVE_CONNECTIONS:-1024}"
READER_DISPATCHER_READ_TIMEOUT="${READER_DISPATCHER_READ_TIMEOUT:-300}"
READER_DISPATCHER_CONNECT_TIMEOUT="${READER_DISPATCHER_CONNECT_TIMEOUT:-10}"
READER_BIND_HOST="${READER_BIND_HOST:-0.0.0.0}"
IPV6="${IPV6:-0}"

case "${IPV6,,}" in
  1|true|yes|on)
    READER_BIND_HOST="::"
    ;;
esac

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

validate_positive_int() {
  local value="$1"
  local label="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -le 0 ]]; then
    echo "${label} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

start_reader() {
  require_dir "${READER_DIR}" "Reader directory"
  if port_in_use "${RAW_READER_PORT}"; then
    echo "Reader dispatcher port ${RAW_READER_PORT} is already in use; stop the old Reader stack before starting the multi-worker stack." >&2
    return 1
  fi

  validate_positive_int "${READER_BACKEND_COUNT}" "READER_BACKEND_COUNT"
  validate_positive_int "${READER_BACKEND_PORT_STRIDE}" "READER_BACKEND_PORT_STRIDE"
  mkdir -p "${READER_WORKER_LOG_DIR}"

  local backend_urls=()
  for ((i=0; i<READER_BACKEND_COUNT; i++)); do
    local backend_base_port=$((READER_BACKEND_BASE_PORT + i * READER_BACKEND_PORT_STRIDE))
    local backend_http_port=$((backend_base_port + 1))
    if port_in_use "${backend_base_port}" || port_in_use "${backend_http_port}"; then
      echo "Reader backend ports ${backend_base_port}/${backend_http_port} are already in use; refusing to start." >&2
      return 1
    fi
    backend_urls+=("http://127.0.0.1:${backend_http_port}")
  done

  echo "Starting ${READER_BACKEND_COUNT} Jina Reader backends from ${READER_DIR} ..."
  for ((i=0; i<READER_BACKEND_COUNT; i++)); do
    local backend_base_port=$((READER_BACKEND_BASE_PORT + i * READER_BACKEND_PORT_STRIDE))
    local backend_http_port=$((backend_base_port + 1))
    local worker_log="${READER_WORKER_LOG_DIR}/reader_worker_${i}_${backend_http_port}.log"
    local worker_pid="${READER_DIR}/reader_worker_${i}.pid"
    nohup bash "${PROJECT_DIR}/utils/reader_supervisor.sh" \
      "${READER_DIR}" \
      "${backend_base_port}" \
      "${READER_NODE_MAX_OLD_SPACE_MB}" \
      "${READER_AUTO_RESTART}" \
      "${READER_RESTART_DELAY_S}" \
      > "${worker_log}" 2>&1 &
    echo $! > "${worker_pid}"
    echo "Reader backend ${i}: http://127.0.0.1:${backend_http_port} log=${worker_log}"
  done

  local backend_urls_csv
  backend_urls_csv="$(IFS=,; echo "${backend_urls[*]}")"
  echo "Starting Reader dispatcher on ${RAW_READER_PORT} ..."
  (
    cd "${PROJECT_DIR}"
    nohup env \
      READER_BACKEND_URLS="${backend_urls_csv}" \
      READER_DISPATCHER_MAX_CONNECTIONS="${READER_DISPATCHER_MAX_CONNECTIONS}" \
      READER_DISPATCHER_MAX_KEEPALIVE_CONNECTIONS="${READER_DISPATCHER_MAX_KEEPALIVE_CONNECTIONS}" \
      READER_DISPATCHER_READ_TIMEOUT="${READER_DISPATCHER_READ_TIMEOUT}" \
      READER_DISPATCHER_CONNECT_TIMEOUT="${READER_DISPATCHER_CONNECT_TIMEOUT}" \
      uvicorn utils.reader_dispatcher:app \
        --host "${READER_BIND_HOST}" \
        --port "${RAW_READER_PORT}" \
        --workers 1 \
      > "${READER_DISPATCHER_LOG}" 2>&1 &
    echo $! > "${PROJECT_DIR}/reader_dispatcher.pid"
  )
  echo "Reader dispatcher log: ${READER_DISPATCHER_LOG}"
  echo "Reader backend URLs: ${backend_urls_csv}"
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
      ENHANCED_READER_MODE="full" \
      RAW_READER_URL="${RAW_READER_URL}" \
      RAW_MARKDOWN_READER_URL="${RAW_MARKDOWN_READER_URL}" \
      READERLM_API_BASE="${READERLM_API_BASE}" \
      READERLM_API_BASES="${READERLM_API_BASES}" \
      READERLM_MODEL_NAME="${READERLM_MODEL_NAME}" \
      READERLM_MAX_HTML_CHARS="${READERLM_MAX_HTML_CHARS}" \
      READERLM_MAX_TOKENS="${READERLM_MAX_TOKENS}" \
      ENHANCED_READER_TIMEOUT="${ENHANCED_READER_TIMEOUT}" \
      ENHANCED_READER_FETCH_STRATEGY="${ENHANCED_READER_FETCH_STRATEGY}" \
      uvicorn utils.enhanced_reader:app \
        --host "${READER_BIND_HOST}" \
        --port "${ENHANCED_READER_PORT}" \
        --workers "${ENHANCED_READER_WORKERS}" \
      > "${ENHANCED_READER_LOG}" 2>&1 &
    echo $! > "${PROJECT_DIR}/enhanced_reader.pid"
  )
  echo "Enhanced Reader log: ${ENHANCED_READER_LOG}"
  echo "Enhanced Reader workers: ${ENHANCED_READER_WORKERS}"
  echo "Enhanced Reader fetch strategy: ${ENHANCED_READER_FETCH_STRATEGY}"
}

start_raw_markdown_reader() {
  require_dir "${PROJECT_DIR}" "OpenSearch-VL project directory"
  if port_in_use "${RAW_MARKDOWN_READER_PORT}"; then
    echo "Raw Markdown Reader appears to already be listening on ${RAW_MARKDOWN_READER_PORT}; skipping."
    return
  fi

  echo "Starting Raw Markdown Reader from ${PROJECT_DIR} ..."
  (
    cd "${PROJECT_DIR}"
    nohup env \
      ENHANCED_READER_MODE="raw_only" \
      RAW_READER_URL="${RAW_READER_URL}" \
      RAW_MARKDOWN_READER_URL="${RAW_MARKDOWN_READER_URL}" \
      READERLM_API_BASE="${READERLM_API_BASE}" \
      READERLM_API_BASES="${READERLM_API_BASES}" \
      READERLM_MODEL_NAME="${READERLM_MODEL_NAME}" \
      READERLM_MAX_HTML_CHARS="${READERLM_MAX_HTML_CHARS}" \
      READERLM_MAX_TOKENS="${READERLM_MAX_TOKENS}" \
      ENHANCED_READER_TIMEOUT="${ENHANCED_READER_TIMEOUT}" \
      uvicorn utils.enhanced_reader:app \
        --host "${READER_BIND_HOST}" \
        --port "${RAW_MARKDOWN_READER_PORT}" \
        --workers 1 \
      > "${RAW_MARKDOWN_READER_LOG}" 2>&1 &
    echo $! > "${PROJECT_DIR}/raw_markdown_reader.pid"
  )
  echo "Raw Markdown Reader log: ${RAW_MARKDOWN_READER_LOG}"
}

main() {
  require_dir "${PROJECT_DIR}" "OpenSearch-VL project directory"

  start_reader
  echo "ReaderLM deployment disabled: this script will not start vLLM/ReaderLM."
  if [[ "${ENHANCED_READER_FETCH_STRATEGY}" != "markdown_clean" ]]; then
    echo "Warning: fetch strategy '${ENHANCED_READER_FETCH_STRATEGY}' uses ReaderLM; configure an external ReaderLM API or use ENHANCED_READER_FETCH_STRATEGY=markdown_clean." >&2
  fi
  start_raw_markdown_reader
  start_enhanced_reader

  cat <<EOF

Startup commands have been issued.

Endpoints:
  Raw Reader HTML endpoint: ${RAW_READER_URL}
  Raw Reader cache endpoint: ${RAW_MARKDOWN_READER_URL}
  Uvicorn bind host:        ${READER_BIND_HOST}
  Reader backend count:     ${READER_BACKEND_COUNT}
  Reader backend base port: ${READER_BACKEND_BASE_PORT}
  Raw Reader Node heap:     ${READER_NODE_MAX_OLD_SPACE_MB} MB each
  Reader auto-restart:      ${READER_AUTO_RESTART}
  ReaderLM deployment:      disabled
  External ReaderLM API:    ${READERLM_API_BASES:-${READERLM_API_BASE} (only if configured/needed)}
  Enhanced Reader endpoint: http://127.0.0.1:${ENHANCED_READER_PORT}
  Enhanced Reader workers:  ${ENHANCED_READER_WORKERS}

Use this for OpenSearch-VL:
  export JINA_READER_URL="http://127.0.0.1:${ENHANCED_READER_PORT}"

Logs:
  Reader dispatcher: ${READER_DISPATCHER_LOG}
  Reader workers:    ${READER_WORKER_LOG_DIR}
  Raw Markdown:    ${RAW_MARKDOWN_READER_LOG}
  Enhanced Reader: ${ENHANCED_READER_LOG}

Quick test after services finish loading:
  curl -H 'Accept: application/json' 'http://127.0.0.1:${ENHANCED_READER_PORT}/https://example.com'
EOF
}

main "$@"
