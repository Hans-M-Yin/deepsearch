#!/usr/bin/env bash
set -euo pipefail

# Paths and ports can be overridden by exporting env vars before running.
PROJECT_DIR="${PROJECT_DIR:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL}"

OPENSERP_BIN="${OPENSERP_BIN:-openserp}"
OPENSERP_HOST="${OPENSERP_HOST:-0.0.0.0}"
OPENSERP_PORT="${OPENSERP_PORT:-7000}"
OPENSERP_BASE="${OPENSERP_BASE:-http://127.0.0.1:${OPENSERP_PORT}}"
OPENSERP_BROWSER_PATH="${OPENSERP_BROWSER_PATH:-${OPENSERP_APP_BROWSER_PATH:-/usr/bin/google-chrome-stable}}"
OPENSERP_PROXY="${OPENSERP_PROXY:-}"
OPENSERP_TEXT_ENGINE="${OPENSERP_TEXT_ENGINE:-google}"
OPENSERP_IMAGE_ENGINE="${OPENSERP_IMAGE_ENGINE:-bing}"
OPENSERP_USE_MEGA="${OPENSERP_USE_MEGA:-false}"
OPENSERP_MEGA_ENGINES="${OPENSERP_MEGA_ENGINES:-google,bing}"

SERPER_ADAPTER_HOST="${SERPER_ADAPTER_HOST:-0.0.0.0}"
SERPER_ADAPTER_PORT="${SERPER_ADAPTER_PORT:-7001}"
SERPER_ADAPTER_TIMEOUT="${SERPER_ADAPTER_TIMEOUT:-60}"
SERPER_FALLBACK_ON_ERROR="${SERPER_FALLBACK_ON_ERROR:-false}"
SERPER_FALLBACK_API_KEY="${SERPER_FALLBACK_API_KEY:-}"

OPENSERP_LOG="${OPENSERP_LOG:-${PROJECT_DIR}/openserp_log}"
SERPER_ADAPTER_LOG="${SERPER_ADAPTER_LOG:-${PROJECT_DIR}/serper_openserp_adapter_log}"

require_dir() {
  local path="$1"
  local label="$2"
  if [[ ! -d "${path}" ]]; then
    echo "Missing ${label}: ${path}" >&2
    exit 1
  fi
}

require_command() {
  local command_name="$1"
  local hint="$2"
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "Missing command: ${command_name}. ${hint}" >&2
    exit 1
  fi
}

require_python_module() {
  local module_name="$1"
  local hint="$2"
  if ! python -c "import ${module_name}" >/dev/null 2>&1; then
    echo "Missing Python module: ${module_name}. ${hint}" >&2
    exit 1
  fi
}

port_in_use() {
  local port="$1"
  ss -lnt 2>/dev/null | awk '{print $4}' | grep -Eq "[:.]${port}$"
}

start_openserp() {
  require_command "${OPENSERP_BIN}" "Build or install OpenSERP first, then set OPENSERP_BIN if needed."

  if port_in_use "${OPENSERP_PORT}"; then
    echo "OpenSERP appears to already be listening on ${OPENSERP_PORT}; skipping."
    return
  fi

  local -a command_args
  command_args=(serve -a "${OPENSERP_HOST}" -p "${OPENSERP_PORT}")
  if [[ -n "${OPENSERP_PROXY}" ]]; then
    command_args+=(--proxy "${OPENSERP_PROXY}")
  fi

  echo "Starting OpenSERP on ${OPENSERP_HOST}:${OPENSERP_PORT} ..."
  (
    cd "${PROJECT_DIR}"
    nohup env \
      OPENSERP_APP_BROWSER_PATH="${OPENSERP_BROWSER_PATH}" \
      OPENSERP_SERVER_HOST="${OPENSERP_HOST}" \
      OPENSERP_SERVER_PORT="${OPENSERP_PORT}" \
      "${OPENSERP_BIN}" "${command_args[@]}" \
      > "${OPENSERP_LOG}" 2>&1 &
    echo $! > "${PROJECT_DIR}/openserp.pid"
  )
  echo "OpenSERP log: ${OPENSERP_LOG}"
}

start_adapter() {
  require_command uvicorn "Install FastAPI/Uvicorn in this environment first."
  require_python_module fastapi "Install it with: pip install fastapi uvicorn httpx"
  require_python_module httpx "Install it with: pip install fastapi uvicorn httpx"

  if port_in_use "${SERPER_ADAPTER_PORT}"; then
    echo "Serper adapter appears to already be listening on ${SERPER_ADAPTER_PORT}; skipping."
    return
  fi

  echo "Starting Serper-compatible OpenSERP adapter on ${SERPER_ADAPTER_HOST}:${SERPER_ADAPTER_PORT} ..."
  (
    cd "${PROJECT_DIR}"
    nohup env \
      OPENSERP_BASE="${OPENSERP_BASE}" \
      OPENSERP_TEXT_ENGINE="${OPENSERP_TEXT_ENGINE}" \
      OPENSERP_IMAGE_ENGINE="${OPENSERP_IMAGE_ENGINE}" \
      OPENSERP_USE_MEGA="${OPENSERP_USE_MEGA}" \
      OPENSERP_MEGA_ENGINES="${OPENSERP_MEGA_ENGINES}" \
      SERPER_ADAPTER_TIMEOUT="${SERPER_ADAPTER_TIMEOUT}" \
      SERPER_FALLBACK_ON_ERROR="${SERPER_FALLBACK_ON_ERROR}" \
      SERPER_FALLBACK_API_KEY="${SERPER_FALLBACK_API_KEY}" \
      uvicorn utils.serper_openserp_adapter:app \
        --host "${SERPER_ADAPTER_HOST}" \
        --port "${SERPER_ADAPTER_PORT}" \
      > "${SERPER_ADAPTER_LOG}" 2>&1 &
    echo $! > "${PROJECT_DIR}/serper_openserp_adapter.pid"
  )
  echo "Serper adapter log: ${SERPER_ADAPTER_LOG}"
}

main() {
  require_dir "${PROJECT_DIR}" "OpenSearch-VL project directory"

  start_openserp
  start_adapter

  cat <<EOF

Startup commands have been issued.

Endpoints:
  OpenSERP:        ${OPENSERP_BASE}
  Serper adapter: http://127.0.0.1:${SERPER_ADAPTER_PORT}

Use this for OpenSearch-VL direct Serper replacement:
  export SERPER_API_KEY="local-openserp"
  export SERPER_SEARCH_URL="http://127.0.0.1:${SERPER_ADAPTER_PORT}/search"

Optional text-to-image search replacement:
  export SERPER_IMAGES_URL="http://127.0.0.1:${SERPER_ADAPTER_PORT}/images"

Note:
  OpenSERP supports text search and text-to-image search. It does not support
  Serper Lens reverse-image search unless SERPER_FALLBACK_API_KEY is configured.

Logs:
  OpenSERP:        ${OPENSERP_LOG}
  Serper adapter: ${SERPER_ADAPTER_LOG}

Quick tests after services finish loading:
  curl "http://127.0.0.1:${OPENSERP_PORT}/health"
  curl "http://127.0.0.1:${SERPER_ADAPTER_PORT}/health"
  curl -X POST "http://127.0.0.1:${SERPER_ADAPTER_PORT}/search" \\
    -H 'Content-Type: application/json' \\
    -H 'X-API-KEY: local-openserp' \\
    -d '{"q":"OpenAI","num":5,"hl":"en","gl":"us"}'
EOF
}

main "$@"
