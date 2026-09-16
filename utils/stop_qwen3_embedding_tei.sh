#!/usr/bin/env bash
# Stop a native Qwen3 TEI service started by start_qwen3_embedding_tei.sh.
set -euo pipefail

MODEL_SIZE="${EMBEDDING_MODEL_SIZE:-0.6B}"
case "${MODEL_SIZE,,}" in
  0.6b|0p6b) DEFAULT_SERVICE_NAME="qwen3-embedding-0p6b" ;;
  4b) DEFAULT_SERVICE_NAME="qwen3-embedding-4b" ;;
  *)
    echo "EMBEDDING_MODEL_SIZE must be 0.6B or 4B, got: ${MODEL_SIZE}" >&2
    exit 1
    ;;
esac
SERVICE_NAME="${EMBEDDING_SERVICE_NAME:-${DEFAULT_SERVICE_NAME}}"
PID_FILE="${EMBEDDING_PID_FILE:-/tmp/${SERVICE_NAME}.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
  echo "[embedding] no PID file: ${PID_FILE}"
  exit 0
fi

service_pid="$(<"${PID_FILE}")"
if ! kill -0 "${service_pid}" 2>/dev/null; then
  rm -f "${PID_FILE}"
  echo "[embedding] stale PID file removed: ${PID_FILE}"
  exit 0
fi
service_cmd="$(ps -p "${service_pid}" -o args= 2>/dev/null || true)"
if [[ "${service_cmd}" != *"text-embeddings-router"* ]]; then
  echo "Refusing to kill pid=${service_pid}: it is not text-embeddings-router." >&2
  exit 1
fi

kill "${service_pid}"
for _ in {1..30}; do
  if ! kill -0 "${service_pid}" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "${service_pid}" 2>/dev/null; then
  echo "[embedding] native service did not exit after 30s: pid=${service_pid}" >&2
  exit 1
fi
rm -f "${PID_FILE}"
echo "[embedding] stopped: pid=${service_pid}"
