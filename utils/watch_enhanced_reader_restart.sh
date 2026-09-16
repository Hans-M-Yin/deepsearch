#!/usr/bin/env bash
# Restart the local Enhanced Reader stack when a trigger file contains an exact
# ``RESTART`` line.  Start this watcher as root because the managed reader
# processes may themselves be owned by root.
#
# Example:
#   : > /mnt/hdfs/byte_ai_sales/user/user/yinzhihan/watcher.log
#   sudo env READER_RESTART_TRIGGER_FILE=/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/watcher.log \
#     bash utils/watch_enhanced_reader_restart.sh
#
# Trigger a restart from another shell:
#   printf 'RESTART\n' > /mnt/hdfs/byte_ai_sales/user/user/yinzhihan/watcher.log

set -u -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR%/utils}}"
TRIGGER_FILE="${READER_RESTART_TRIGGER_FILE:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/watcher.log}"
# Keep the advisory lock on local disk: the trigger file may live on an HDFS
# mount, where flock support is not guaranteed.
LOCK_FILE="${READER_RESTART_LOCK_FILE:-/tmp/enhanced_reader_restart_watch.lock}"
POLL_INTERVAL_S="${READER_RESTART_POLL_INTERVAL_S:-60}"
FAILURE_BACKOFF_S="${READER_RESTART_FAILURE_BACKOFF_S:-30}"

log() {
  printf '[reader-restart-watch] %s %s\n' "$(date -Is)" "$*"
}

if [[ "${EUID}" -ne 0 ]]; then
  echo "Run this watcher with sudo so it can stop and start the Reader stack." >&2
  exit 1
fi

if [[ ! -e "${TRIGGER_FILE}" ]]; then
  echo "Trigger file does not exist: ${TRIGGER_FILE}. Create it as the user who will write RESTART, then retry." >&2
  exit 1
fi

if [[ ! -f "${TRIGGER_FILE}" ]]; then
  echo "Trigger path is not a regular file: ${TRIGGER_FILE}" >&2
  exit 1
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required but was not found in PATH." >&2
  exit 1
fi

restart_requested() {
  grep -Fxq 'RESTART' "${TRIGGER_FILE}" 2>/dev/null
}

restart_stack() {
  log "trigger received; stopping Reader stack"
  if ! bash "${PROJECT_DIR}/utils/stop_enhanced_reader_stack.sh"; then
    log "stop failed"
    return 1
  fi

  log "starting Reader stack"
  if ! env \
    ENHANCED_READER_TIMING_ENABLED="${ENHANCED_READER_TIMING_ENABLED:-0}" \
    ENHANCED_READER_TIMING_STDERR="${ENHANCED_READER_TIMING_STDERR:-0}" \
    ENHANCED_READER_FETCH_STRATEGY="${ENHANCED_READER_FETCH_STRATEGY:-markdown_clean}" \
    READER_BACKEND_COUNT="${READER_BACKEND_COUNT:-8}" \
    READER_NODE_MAX_OLD_SPACE_MB="${READER_NODE_MAX_OLD_SPACE_MB:-65536}" \
    bash "${PROJECT_DIR}/utils/start_enhanced_reader_stack.sh"; then
    log "start failed"
    return 1
  fi

  log "restart completed"
  return 0
}

poll_once() {
  local request_status recheck_status

  if [[ ! -f "${TRIGGER_FILE}" || ! -r "${TRIGGER_FILE}" ]]; then
    log "trigger file is unavailable; skipping this scan"
    return 0
  fi

  restart_requested
  request_status=$?
  if [[ "${request_status}" -eq 1 ]]; then
    return 0
  fi
  if [[ "${request_status}" -ne 0 ]]; then
    log "failed to inspect trigger file; skipping this scan"
    return 0
  fi

  # A non-blocking lock prevents duplicate restart loops if the watcher is
  # accidentally launched twice.  Empty the file before restart, so a new
  # RESTART written while startup is in progress is retained for a later run.
  if ! (
    flock -n 9 || exit 0
    if restart_requested; then
      :
    else
      recheck_status=$?
      if [[ "${recheck_status}" -eq 1 ]]; then
        exit 0
      fi
      log "failed to recheck trigger file; skipping this scan"
      exit 0
    fi

    if ! : > "${TRIGGER_FILE}"; then
      log "failed to clear trigger file; skipping this scan"
      exit 0
    fi

    if ! restart_stack; then
      # Restore the consumed request only if a newer request has not already
      # appeared while stop/start was running.
      if [[ ! -s "${TRIGGER_FILE}" ]]; then
        printf 'RESTART\n' > "${TRIGGER_FILE}"
      fi
      sleep "${FAILURE_BACKOFF_S}"
    fi
  ) 9>"${LOCK_FILE}"; then
    log "restart handler failed unexpectedly; skipping this scan"
  fi
}

log "watching trigger=${TRIGGER_FILE} poll_interval_s=${POLL_INTERVAL_S}"
while true; do
  poll_once || log "poll error; skipping this scan"
  sleep "${POLL_INTERVAL_S}"
done
