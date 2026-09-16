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
READER_BACKEND_REGISTRY_FILE="${READER_BACKEND_REGISTRY_FILE:-${PROJECT_DIR}/utils/reader_backend_registry.txt}"
RAW_MARKDOWN_READER_PORT="${RAW_MARKDOWN_READER_PORT:-8003}"
ENHANCED_READER_PORT="${ENHANCED_READER_PORT:-8004}"
ENHANCED_READER_WORKERS="${ENHANCED_READER_WORKERS:-4}"

READER_LOG="${READER_LOG:-${READER_DIR}/reader_log}"
READER_WORKER_LOG_DIR="${READER_WORKER_LOG_DIR:-${PROJECT_DIR}/reader_workers}"
READER_WORKER_PID_DIR="${READER_WORKER_PID_DIR:-${PROJECT_DIR}/reader_worker_pids}"
READER_DISPATCHER_LOG="${READER_DISPATCHER_LOG:-${PROJECT_DIR}/reader_dispatcher_log}"
ENHANCED_READER_LOG="${ENHANCED_READER_LOG:-/tmp/enhanced_reader_log}"
ENHANCED_READER_TIMING_LOG="${ENHANCED_READER_TIMING_LOG:-/tmp/enhanced_reader_timing_log}"
ENHANCED_READER_TIMING_LOG_MAX_BYTES="${ENHANCED_READER_TIMING_LOG_MAX_BYTES:-67108864}"
ENHANCED_READER_TIMING_LOG_BACKUP_COUNT="${ENHANCED_READER_TIMING_LOG_BACKUP_COUNT:-2}"
ENHANCED_READER_TIMING_ENABLED="${ENHANCED_READER_TIMING_ENABLED:-1}"
ENHANCED_READER_TIMING_STDERR="${ENHANCED_READER_TIMING_STDERR:-1}"
RAW_MARKDOWN_READER_LOG="${RAW_MARKDOWN_READER_LOG:-${PROJECT_DIR}/raw_markdown_reader_log}"

# These variables are retained only for compatibility with an externally
# deployed ReaderLM. This script never starts a ReaderLM/vLLM process.
READERLM_API_BASE="${READERLM_API_BASE:-http://127.0.0.1:8005/v1}"
READERLM_API_BASES="${READERLM_API_BASES:-}"
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
READER_DISPATCHER_TRUST_ENV="${READER_DISPATCHER_TRUST_ENV:-0}"
READER_DISPATCHER_MAX_INFLIGHT_PER_BACKEND="${READER_DISPATCHER_MAX_INFLIGHT_PER_BACKEND:-0}"
READER_BIND_HOST="${READER_BIND_HOST:-0.0.0.0}"
IPV6="${IPV6:-0}"

case "${IPV6,,}" in
  1|true|yes|on)
    READER_BIND_HOST="::"
    ;;
esac

# Uvicorn is intentionally started in one address family at a time.  Its
# internal clients must use that same family: on hosts with
# net.ipv6.bindv6only=1, a process bound to :: cannot be reached through
# 127.0.0.1.  Users may still override RAW_READER_URL or
# RAW_MARKDOWN_READER_URL for an external deployment.
if [[ -z "${READER_LOCAL_HOST:-}" ]]; then
  case "${READER_BIND_HOST}" in
    *:*) READER_LOCAL_HOST="::1" ;;
    *) READER_LOCAL_HOST="127.0.0.1" ;;
  esac
fi

local_http_url() {
  local host="$1"
  local port="$2"
  if [[ "${host}" == \[*\] ]]; then
    printf 'http://%s:%s' "${host}" "${port}"
  elif [[ "${host}" == *:* ]]; then
    printf 'http://[%s]:%s' "${host}" "${port}"
  else
    printf 'http://%s:%s' "${host}" "${port}"
  fi
}

RAW_READER_URL="${RAW_READER_URL:-$(local_http_url "${READER_LOCAL_HOST}" "${RAW_READER_PORT}")}"
RAW_MARKDOWN_READER_URL="${RAW_MARKDOWN_READER_URL:-$(local_http_url "${READER_LOCAL_HOST}" "${RAW_MARKDOWN_READER_PORT}")}"
ENHANCED_READER_URL="${ENHANCED_READER_URL:-$(local_http_url "${READER_LOCAL_HOST}" "${ENHANCED_READER_PORT}")}"
READER_DISPATCHER_HEALTH_URL="${RAW_READER_URL%/}/__reader_dispatcher/health"
RAW_MARKDOWN_READER_HEALTH_URL="${RAW_MARKDOWN_READER_URL%/}/__enhanced_reader/health"
ENHANCED_READER_HEALTH_URL="${ENHANCED_READER_URL%/}/__enhanced_reader/health"

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

wait_for_port() {
  local port="$1"
  local attempts="${2:-40}"
  local delay_s="${3:-0.25}"
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if port_in_use "${port}"; then
      return 0
    fi
    sleep "${delay_s}"
  done
  return 1
}

http_ready() {
  local url="$1"
  curl --noproxy '*' -gfsS --connect-timeout 2 --max-time 4 "${url}" >/dev/null 2>&1
}

wait_for_http_ready() {
  local url="$1"
  local label="$2"
  local attempts="${3:-80}"
  local delay_s="${4:-0.25}"
  for ((attempt = 1; attempt <= attempts; attempt++)); do
    if http_ready "${url}"; then
      return 0
    fi
    sleep "${delay_s}"
  done
  echo "${label} did not become healthy at ${url}" >&2
  return 1
}

require_healthy_existing_service() {
  local port="$1"
  local health_url="$2"
  local label="$3"
  if ! port_in_use "${port}"; then
    return 1
  fi
  if ! http_ready "${health_url}"; then
    echo "${label} port ${port} is occupied, but ${health_url} is not its healthy endpoint. Stop the stale/wrong process before retrying." >&2
    return 2
  fi
  echo "${label} is already healthy at ${health_url}; reusing it."
  return 0
}

validate_positive_int() {
  local value="$1"
  local label="$2"
  if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -le 0 ]]; then
    echo "${label} must be a positive integer, got: ${value}" >&2
    exit 1
  fi
}

trim_whitespace() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

load_registered_remote_backends() {
  REGISTERED_REMOTE_BACKENDS=()
  if [[ ! -f "${READER_BACKEND_REGISTRY_FILE}" ]]; then
    echo "Reader backend registry does not exist: ${READER_BACKEND_REGISTRY_FILE}" >&2
    return 1
  fi

  local line backend
  while IFS= read -r line || [[ -n "${line}" ]]; do
    # Permit blank lines, comment-only lines, and comments after a URL.
    line="${line%%#*}"
    backend="$(trim_whitespace "${line}")"
    [[ -z "${backend}" ]] && continue
    case "${backend}" in
      http://*|https://*) ;;
      *)
        echo "Invalid Reader backend registry entry: ${backend}. Expected an http(s) URL." >&2
        return 1
        ;;
    esac
    REGISTERED_REMOTE_BACKENDS+=("${backend%/}")
  done < "${READER_BACKEND_REGISTRY_FILE}"
}

append_unique_backend() {
  local candidate="$1"
  local existing
  for existing in "${REGISTERED_BACKENDS[@]}"; do
    if [[ "${existing}" == "${candidate}" ]]; then
      echo "Ignoring duplicate Reader backend registration: ${candidate}" >&2
      return 0
    fi
  done
  REGISTERED_BACKENDS+=("${candidate}")
}

local_reader_backends_listening() {
  local i backend_http_port
  for ((i=0; i<READER_BACKEND_COUNT; i++)); do
    backend_http_port=$((READER_BACKEND_BASE_PORT + i * READER_BACKEND_PORT_STRIDE + 1))
    if ! port_in_use "${backend_http_port}"; then
      echo "Reader dispatcher is reachable but local backend port ${backend_http_port} is not listening; refusing to reuse a degraded dispatcher." >&2
      return 1
    fi
  done
  return 0
}

start_reader() {
  require_dir "${READER_DIR}" "Reader directory"
  if require_healthy_existing_service "${RAW_READER_PORT}" "${READER_DISPATCHER_HEALTH_URL}" "Reader dispatcher"; then
    if local_reader_backends_listening; then
      return 0
    fi
    echo "Stop the stale Reader stack with utils/stop_enhanced_reader_stack.sh, then start it again." >&2
    return 1
  else
    local service_status=$?
    if [[ "${service_status}" -eq 2 ]]; then
      return 1
    fi
  fi

  validate_positive_int "${READER_BACKEND_COUNT}" "READER_BACKEND_COUNT"
  validate_positive_int "${READER_BACKEND_PORT_STRIDE}" "READER_BACKEND_PORT_STRIDE"
  load_registered_remote_backends
  mkdir -p "${READER_WORKER_LOG_DIR}" "${READER_WORKER_PID_DIR}"

  REGISTERED_BACKENDS=()
  for ((i=0; i<READER_BACKEND_COUNT; i++)); do
    local backend_base_port=$((READER_BACKEND_BASE_PORT + i * READER_BACKEND_PORT_STRIDE))
    local backend_http_port=$((backend_base_port + 1))
    if port_in_use "${backend_base_port}" || port_in_use "${backend_http_port}"; then
      echo "Reader backend ports ${backend_base_port}/${backend_http_port} are already in use; refusing to start." >&2
      return 1
    fi
    append_unique_backend "http://127.0.0.1:${backend_http_port}"
  done

  echo "Starting ${READER_BACKEND_COUNT} Jina Reader backends from ${READER_DIR} ..."
  for ((i=0; i<READER_BACKEND_COUNT; i++)); do
    local backend_base_port=$((READER_BACKEND_BASE_PORT + i * READER_BACKEND_PORT_STRIDE))
    local backend_http_port=$((backend_base_port + 1))
    local worker_log="${READER_WORKER_LOG_DIR}/reader_worker_${i}_${backend_http_port}.log"
    local worker_pid="${READER_WORKER_PID_DIR}/reader_worker_${i}.pid"
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

  local backend
  for backend in "${REGISTERED_REMOTE_BACKENDS[@]}"; do
    append_unique_backend "${backend}"
  done

  local backend_urls_csv
  backend_urls_csv="$(IFS=,; echo "${REGISTERED_BACKENDS[*]}")"
  echo "Starting Reader dispatcher on ${RAW_READER_PORT} ..."
  (
    cd "${PROJECT_DIR}"
    nohup env \
      READER_BACKEND_URLS="${backend_urls_csv}" \
      READER_DISPATCHER_MAX_CONNECTIONS="${READER_DISPATCHER_MAX_CONNECTIONS}" \
      READER_DISPATCHER_MAX_KEEPALIVE_CONNECTIONS="${READER_DISPATCHER_MAX_KEEPALIVE_CONNECTIONS}" \
      READER_DISPATCHER_READ_TIMEOUT="${READER_DISPATCHER_READ_TIMEOUT}" \
      READER_DISPATCHER_CONNECT_TIMEOUT="${READER_DISPATCHER_CONNECT_TIMEOUT}" \
      READER_DISPATCHER_TRUST_ENV="${READER_DISPATCHER_TRUST_ENV}" \
      READER_DISPATCHER_MAX_INFLIGHT_PER_BACKEND="${READER_DISPATCHER_MAX_INFLIGHT_PER_BACKEND}" \
      uvicorn utils.reader_dispatcher:app \
        --host "${READER_BIND_HOST}" \
        --port "${RAW_READER_PORT}" \
        --workers 1 \
      > "${READER_DISPATCHER_LOG}" 2>&1 &
    echo $! > "${PROJECT_DIR}/reader_dispatcher.pid"
  )
  echo "Reader dispatcher log: ${READER_DISPATCHER_LOG}"
  echo "Reader backend registry: ${READER_BACKEND_REGISTRY_FILE}"
  echo "Reader backends: local=${READER_BACKEND_COUNT} external=${#REGISTERED_REMOTE_BACKENDS[@]} total=${#REGISTERED_BACKENDS[@]}"
  echo "Reader backend URLs: ${backend_urls_csv}"
  if ! wait_for_http_ready "${READER_DISPATCHER_HEALTH_URL}" "Reader dispatcher"; then
    tail -n 80 "${READER_DISPATCHER_LOG}" >&2 || true
    return 1
  fi
  echo "Reader dispatcher is healthy at ${READER_DISPATCHER_HEALTH_URL}."
}

start_enhanced_reader() {
  require_dir "${PROJECT_DIR}" "OpenSearch-VL project directory"
  if require_healthy_existing_service "${ENHANCED_READER_PORT}" "${ENHANCED_READER_HEALTH_URL}" "Enhanced Reader"; then
    return
  else
    local service_status=$?
    if [[ "${service_status}" -eq 2 ]]; then
      return 1
    fi
  fi

  local uvicorn_bin
  uvicorn_bin="$(command -v uvicorn || true)"
  if [[ -z "${uvicorn_bin}" ]]; then
    {
      echo "[enhanced_reader_start_error] uvicorn was not found in PATH=${PATH}"
      command -v python || true
    } >> "${ENHANCED_READER_LOG}"
    echo "Cannot start Enhanced Reader: uvicorn was not found; see ${ENHANCED_READER_LOG}" >&2
    return 1
  fi

  echo "Starting Enhanced Reader from ${PROJECT_DIR} ..."
  (
    cd "${PROJECT_DIR}"
    {
      echo "[enhanced_reader_start] date=$(date -Is)"
      echo "[enhanced_reader_start] uid=$(id -u) gid=$(id -g)"
      echo "[enhanced_reader_start] python=$(command -v python || true)"
      echo "[enhanced_reader_start] uvicorn=${uvicorn_bin}"
      echo "[enhanced_reader_start] host=${READER_BIND_HOST} port=${ENHANCED_READER_PORT} workers=${ENHANCED_READER_WORKERS}"
      echo "[enhanced_reader_start] strategy=${ENHANCED_READER_FETCH_STRATEGY} mode=full"
      echo "[enhanced_reader_start] timing_log=${ENHANCED_READER_TIMING_LOG}"
    } >> "${ENHANCED_READER_LOG}"
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
      ENHANCED_READER_TIMING_LOG="${ENHANCED_READER_TIMING_LOG}" \
      ENHANCED_READER_TIMING_LOG_MAX_BYTES="${ENHANCED_READER_TIMING_LOG_MAX_BYTES}" \
      ENHANCED_READER_TIMING_LOG_BACKUP_COUNT="${ENHANCED_READER_TIMING_LOG_BACKUP_COUNT}" \
      ENHANCED_READER_TIMING_ENABLED="${ENHANCED_READER_TIMING_ENABLED}" \
      ENHANCED_READER_TIMING_STDERR="${ENHANCED_READER_TIMING_STDERR}" \
      "${uvicorn_bin}" utils.enhanced_reader:app \
        --host "${READER_BIND_HOST}" \
        --port "${ENHANCED_READER_PORT}" \
        --workers "${ENHANCED_READER_WORKERS}" \
      >> "${ENHANCED_READER_LOG}" 2>&1 &
    enhanced_reader_pid=$!
    echo "${enhanced_reader_pid}" > "${PROJECT_DIR}/enhanced_reader.pid"
    echo "[enhanced_reader_start] pid=${enhanced_reader_pid}" >> "${ENHANCED_READER_LOG}"
  )
  echo "Enhanced Reader log: ${ENHANCED_READER_LOG}"
  echo "Enhanced Reader workers: ${ENHANCED_READER_WORKERS}"
  echo "Enhanced Reader fetch strategy: ${ENHANCED_READER_FETCH_STRATEGY}"

  # The Uvicorn parent can announce its listener before all worker processes
  # are ready, and the `ss`-based port probe can race that transition.  The
  # health endpoint is the authoritative readiness condition and also covers
  # the non-listening case, so avoid rejecting a successfully spawned server
  # on an intermediate port-probe result.
  if ! wait_for_http_ready "${ENHANCED_READER_HEALTH_URL}" "Enhanced Reader" 120 0.25; then
    tail -n 80 "${ENHANCED_READER_LOG}" >&2 || true
    return 1
  fi
  echo "Enhanced Reader is healthy at ${ENHANCED_READER_HEALTH_URL}."
}

start_raw_markdown_reader() {
  require_dir "${PROJECT_DIR}" "OpenSearch-VL project directory"
  if require_healthy_existing_service "${RAW_MARKDOWN_READER_PORT}" "${RAW_MARKDOWN_READER_HEALTH_URL}" "Raw Markdown Reader"; then
    return
  else
    local service_status=$?
    if [[ "${service_status}" -eq 2 ]]; then
      return 1
    fi
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
      ENHANCED_READER_TIMING_LOG="${ENHANCED_READER_TIMING_LOG}" \
      ENHANCED_READER_TIMING_LOG_MAX_BYTES="${ENHANCED_READER_TIMING_LOG_MAX_BYTES}" \
      ENHANCED_READER_TIMING_LOG_BACKUP_COUNT="${ENHANCED_READER_TIMING_LOG_BACKUP_COUNT}" \
      ENHANCED_READER_TIMING_ENABLED="${ENHANCED_READER_TIMING_ENABLED}" \
      ENHANCED_READER_TIMING_STDERR="${ENHANCED_READER_TIMING_STDERR}" \
      uvicorn utils.enhanced_reader:app \
        --host "${READER_BIND_HOST}" \
        --port "${RAW_MARKDOWN_READER_PORT}" \
        --workers 1 \
      > "${RAW_MARKDOWN_READER_LOG}" 2>&1 &
    echo $! > "${PROJECT_DIR}/raw_markdown_reader.pid"
  )
  echo "Raw Markdown Reader log: ${RAW_MARKDOWN_READER_LOG}"
  if ! wait_for_http_ready "${RAW_MARKDOWN_READER_HEALTH_URL}" "Raw Markdown Reader"; then
    tail -n 80 "${RAW_MARKDOWN_READER_LOG}" >&2 || true
    return 1
  fi
  echo "Raw Markdown Reader is healthy at ${RAW_MARKDOWN_READER_HEALTH_URL}."
}

main() {
  require_dir "${PROJECT_DIR}" "OpenSearch-VL project directory"

  if ! start_reader; then
    echo "Reader dispatcher/backend startup failed; not starting dependent 8003/8004 services." >&2
    return 1
  fi
  echo "ReaderLM deployment disabled: this script will not start vLLM/ReaderLM."
  if [[ "${ENHANCED_READER_FETCH_STRATEGY}" != "markdown_clean" ]]; then
    echo "Warning: fetch strategy '${ENHANCED_READER_FETCH_STRATEGY}' uses ReaderLM; configure an external ReaderLM API or use ENHANCED_READER_FETCH_STRATEGY=markdown_clean." >&2
  fi
  if ! start_raw_markdown_reader; then
    echo "Raw Markdown Reader startup failed; not starting dependent 8004 service." >&2
    return 1
  fi
  if ! start_enhanced_reader; then
    echo "Enhanced Reader startup failed." >&2
    return 1
  fi

  cat <<EOF

Startup commands have been issued.

Endpoints:
  Raw Reader HTML endpoint: ${RAW_READER_URL}
  Raw Reader cache endpoint: ${RAW_MARKDOWN_READER_URL}
  Enhanced Reader endpoint: ${ENHANCED_READER_URL}
  Uvicorn bind host:        ${READER_BIND_HOST}
  Internal local host:      ${READER_LOCAL_HOST}
  Reader backend count:     ${READER_BACKEND_COUNT}
  Reader backend base port: ${READER_BACKEND_BASE_PORT}
  Raw Reader Node heap:     ${READER_NODE_MAX_OLD_SPACE_MB} MB each
  Reader auto-restart:      ${READER_AUTO_RESTART}
  ReaderLM deployment:      disabled
  External ReaderLM API:    ${READERLM_API_BASES:-${READERLM_API_BASE} (only if configured/needed)}
  Enhanced Reader workers:  ${ENHANCED_READER_WORKERS}

Use this for OpenSearch-VL:
  export JINA_READER_URL="${ENHANCED_READER_URL}"

Logs:
  Reader dispatcher: ${READER_DISPATCHER_LOG}
  Reader workers:    ${READER_WORKER_LOG_DIR}
  Reader worker PIDs: ${READER_WORKER_PID_DIR}
  Raw Markdown:    ${RAW_MARKDOWN_READER_LOG}
  Enhanced Reader: ${ENHANCED_READER_LOG}
  Timing log:       ${ENHANCED_READER_TIMING_LOG} (rotate at ${ENHANCED_READER_TIMING_LOG_MAX_BYTES} bytes, keep ${ENHANCED_READER_TIMING_LOG_BACKUP_COUNT} backups)

Quick test after services finish loading:
  curl --noproxy '*' -g -H 'Accept: application/json' '${ENHANCED_READER_URL}/https://example.com'
EOF
}

main "$@"
