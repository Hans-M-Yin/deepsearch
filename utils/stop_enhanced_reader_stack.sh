#!/usr/bin/env bash

# Stop the local Enhanced Reader stack and clean up its Reader-owned
# Node/Chrome process groups.  This deliberately does not use
# `pkill -f chrome`, because other jobs on the machine may use Chrome too.

set -u
shopt -s nullglob

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-${SCRIPT_DIR%/utils}}"
READER_DIR="${READER_DIR:-/workspace/reader}"

READER_BACKEND_BASE_PORT="${READER_BACKEND_BASE_PORT:-8101}"
READER_BACKEND_COUNT="${READER_BACKEND_COUNT:-8}"
READER_BACKEND_PORT_STRIDE="${READER_BACKEND_PORT_STRIDE:-10}"

log() {
  printf '[reader-stop] %s\n' "$*"
}

warn() {
  printf '[reader-stop][warning] %s\n' "$*" >&2
}

is_pid() {
  [[ "${1:-}" =~ ^[0-9]+$ ]] && (( 1 < 10#${1} ))
}

is_alive() {
  local pid="${1:-}"
  is_pid "${pid}" && kill -0 "${pid}" 2>/dev/null
}

CURRENT_PGID="$(ps -o pgid= -p "$$" 2>/dev/null | tr -d ' ' || true)"
declare -A KILL_PGIDS=()
declare -A SEEN_TREE_PIDS=()

READER_PORTS=()
if [[ "${READER_BACKEND_COUNT}" =~ ^[0-9]+$ ]] &&
   [[ "${READER_BACKEND_BASE_PORT}" =~ ^[0-9]+$ ]] &&
   [[ "${READER_BACKEND_PORT_STRIDE}" =~ ^[0-9]+$ ]]; then
  for ((i = 0; i < READER_BACKEND_COUNT; i++)); do
    backend_base_port="$((READER_BACKEND_BASE_PORT + i * READER_BACKEND_PORT_STRIDE))"
    # Jina Reader exposes h2c on PORT and HTTP/1.1 on PORT+1.
    READER_PORTS+=("${backend_base_port}" "$((backend_base_port + 1))")
  done
else
  warn "Invalid Reader backend port configuration; using no environment-based orphan scan."
fi

add_process_group() {
  local pid="${1:-}"
  local pgid
  is_pid "${pid}" || return 0
  pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
  if [[ "${pgid}" =~ ^[0-9]+$ ]] &&
     (( 1 < 10#${pgid} )) &&
     [[ "${pgid}" != "${CURRENT_PGID}" ]]; then
    KILL_PGIDS["${pgid}"]=1
  fi
}

collect_process_tree() {
  local parent="${1:-}"
  local child
  is_pid "${parent}" || return 0

  while read -r child; do
    is_pid "${child}" || continue
    [[ "${SEEN_TREE_PIDS[${child}]+present}" == present ]] && continue
    SEEN_TREE_PIDS["${child}"]=1
    add_process_group "${child}"
    collect_process_tree "${child}"
  done < <(pgrep -P "${parent}" 2>/dev/null || true)
}

wait_for_exit() {
  local pid="${1:-}"
  local attempts="${2:-50}"
  local i
  for ((i = 0; i < attempts; i++)); do
    is_alive "${pid}" || return 0
    sleep 0.1
  done
  return 1
}

stop_checked_pid() {
  local pidfile="$1"
  local label="$2"
  local expected_fragment="$3"
  local pid cmd

  [[ -f "${pidfile}" ]] || return 0
  pid="$(tr -d '[:space:]' < "${pidfile}" 2>/dev/null || true)"
  if ! is_pid "${pid}"; then
    warn "Ignoring invalid PID file ${pidfile}: ${pid@Q}"
    return 0
  fi
  if ! is_alive "${pid}"; then
    log "${label} PID ${pid} is not running."
    return 0
  fi

  cmd="$(ps -o args= -p "${pid}" 2>/dev/null || true)"
  if [[ "${cmd}" != *"${expected_fragment}"* ]]; then
    warn "PID file ${pidfile} points to an unexpected process; skipping: pid=${pid} cmd=${cmd@Q}"
    return 0
  fi

  log "Stopping ${label}: pid=${pid} cmd=${cmd@Q}"
  kill -TERM "${pid}" 2>/dev/null || true
  if ! wait_for_exit "${pid}" 50; then
    warn "${label} did not exit after 5s; sending SIGKILL to pid=${pid}."
    kill -KILL "${pid}" 2>/dev/null || true
  fi
}

capture_worker_supervisor() {
  local pidfile="$1"
  local pid cmd
  [[ -f "${pidfile}" ]] || return 0
  pid="$(tr -d '[:space:]' < "${pidfile}" 2>/dev/null || true)"
  is_pid "${pid}" || return 0
  is_alive "${pid}" || return 0
  cmd="$(ps -o args= -p "${pid}" 2>/dev/null || true)"
  if [[ "${cmd}" != *reader_supervisor.sh* ]]; then
    warn "Worker PID file ${pidfile} does not point to reader_supervisor.sh; skipping capture."
    return 0
  fi
  SEEN_TREE_PIDS["${pid}"]=1
  collect_process_tree "${pid}"
}

reader_supervisor_pids() {
  local proc pid args
  for proc in /proc/[0-9]*; do
    pid="${proc#/proc/}"
    is_pid "${pid}" || continue
    [[ "${pid}" == "$$" ]] && continue
    [[ -r "/proc/${pid}/cmdline" ]] || continue
    args="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    [[ "${args}" == *reader_supervisor.sh* ]] || continue
    # Only stop supervisors that manage this Reader directory.  This avoids
    # touching an unrelated Reader stack on the same host.
    [[ "${args}" == *" ${READER_DIR} "* || "${args}" == *" ${READER_DIR}"* ]] || continue
    printf '%s\n' "${pid}"
  done | sort -nu
}

stop_all_reader_supervisors() {
  local pid cmd
  local supervisors=()

  while read -r pid; do
    is_pid "${pid}" || continue
    supervisors+=("${pid}")
  done < <(reader_supervisor_pids)

  if ((${#supervisors[@]} == 0)); then
    log "No Reader supervisors found for ${READER_DIR}."
    return 0
  fi

  # Stop every supervisor before cleaning backend listeners.  Otherwise a
  # supervisor can observe its Node child exiting and recreate 8101/8102.
  for pid in "${supervisors[@]}"; do
    cmd="$(ps -o args= -p "${pid}" 2>/dev/null || true)"
    log "Stopping Reader supervisor: pid=${pid} cmd=${cmd@Q}"
    kill -TERM "${pid}" 2>/dev/null || true
  done

  for pid in "${supervisors[@]}"; do
    if ! wait_for_exit "${pid}" 50; then
      warn "Reader supervisor pid=${pid} did not exit after 5s; sending SIGKILL."
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}

reader_process_has_backend_port() {
  local pid="$1"
  local env_text port
  [[ -r "/proc/${pid}/environ" ]] || return 1
  env_text="$(tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null || true)"
  for port in "${READER_PORTS[@]}"; do
    if printf '%s\n' "${env_text}" | grep -Fqx "PORT=${port}"; then
      return 0
    fi
  done
  return 1
}

# Find the actual processes listening on both sides of every Jina backend
# port pair.  The HTTP/1.1 port (for example 8102) is not necessarily present
# as PORT=8102 in the inherited environment, so environment-only detection
# cannot reliably find it.
reader_listener_pids() {
  local state receive send local_endpoint remote_endpoint process_info port

  if ! command -v ss >/dev/null 2>&1; then
    warn "Cannot scan Reader listener ports because 'ss' is unavailable."
    return 0
  fi

  while read -r state receive send local_endpoint remote_endpoint process_info; do
    [[ "${state}" == "LISTEN" ]] || continue
    for port in "${READER_PORTS[@]}"; do
      [[ "${local_endpoint}" == *":${port}" ]] || continue
      printf '%s\n' "${process_info}" \
        | grep -oE 'pid=[0-9]+' \
        | sed 's/^pid=//'
    done
  done < <(ss -H -ltnp 2>/dev/null || true) | sort -nu
}

reader_process_pids() {
  local proc pid comm
  for proc in /proc/[0-9]*; do
    pid="${proc#/proc/}"
    is_pid "${pid}" || continue
    [[ "${pid}" == "$$" ]] && continue
    comm="$(cat "/proc/${pid}/comm" 2>/dev/null || true)"
    case "${comm}" in
      node|npm|chrome*|chromium*|reader*|setsid)
        ;;
      *)
        continue
        ;;
    esac
    reader_process_has_backend_port "${pid}" || continue
    printf '%s\n' "${pid}"
  done
}

capture_port_owned_groups() {
  local pid
  while read -r pid; do
    is_pid "${pid}" || continue
    add_process_group "${pid}"
    SEEN_TREE_PIDS["${pid}"]=1
    collect_process_tree "${pid}"
  done < <(
    {
      reader_process_pids
      reader_listener_pids
    } | sort -nu
  )
}

kill_captured_groups() {
  local pgid
  for pgid in "${!KILL_PGIDS[@]}"; do
    log "Stopping Reader process group pgid=${pgid}"
    kill -TERM -- "-${pgid}" 2>/dev/null || true
  done
  sleep 2
  for pgid in "${!KILL_PGIDS[@]}"; do
    if kill -0 -- "-${pgid}" 2>/dev/null; then
      warn "Reader process group pgid=${pgid} still exists; sending SIGKILL."
      kill -KILL -- "-${pgid}" 2>/dev/null || true
    fi
  done
}

kill_remaining_port_owned_processes() {
  local pid
  local remaining=()
  while read -r pid; do
    is_pid "${pid}" || continue
    remaining+=("${pid}")
  done < <(
    {
      reader_process_pids
      reader_listener_pids
    } | sort -nu
  )

  if ((${#remaining[@]} == 0)); then
    return 0
  fi

  warn "Found Reader-owned processes after process-group cleanup: ${remaining[*]}"
  for pid in "${remaining[@]}"; do
    add_process_group "${pid}"
  done
  kill_captured_groups
  sleep 1
  for pid in "${remaining[@]}"; do
    if is_alive "${pid}"; then
      warn "Reader-owned process pid=${pid} still exists; sending SIGKILL."
      kill -KILL "${pid}" 2>/dev/null || true
    fi
  done
}

final_reader_processes=""

log "Project directory: ${PROJECT_DIR}"
log "Reader directory: ${READER_DIR}"
log "Backend ports used for cleanup (h2c + HTTP/1.1): ${READER_PORTS[*]:-none}"

# Capture dedicated worker process groups before stopping supervisors.  The
# supervisor itself owns the cleanup path for its setsid-created child group,
# but capturing the groups first also handles leaked descendants.
worker_pidfiles=("${READER_DIR}"/reader_worker_*.pid)
for pidfile in "${worker_pidfiles[@]}"; do
  capture_worker_supervisor "${pidfile}"
done
while read -r supervisor_pid; do
  is_pid "${supervisor_pid}" || continue
  SEEN_TREE_PIDS["${supervisor_pid}"]=1
  collect_process_tree "${supervisor_pid}"
done < <(reader_supervisor_pids)
capture_port_owned_groups

# Stop every Reader supervisor first.  This prevents auto-restart from
# recreating backend listeners while the rest of the stack is being cleaned.
stop_all_reader_supervisors

# Now stop public entrypoints so no new requests are admitted while the
# backend listeners and their descendants are being cleaned.  The expected
# port fragment prevents stale PID files from terminating an unrelated uvicorn
# process.
stop_checked_pid "${PROJECT_DIR}/enhanced_reader.pid" "Enhanced Reader (8004)" "--port 8004"
stop_checked_pid "${PROJECT_DIR}/raw_markdown_reader.pid" "Raw Markdown Reader (8003)" "--port 8003"
stop_checked_pid "${PROJECT_DIR}/reader_dispatcher.pid" "Reader dispatcher (8002)" "--port 8002"

# Supervisors have already attempted to stop their setsid process groups.
# Force-kill any captured groups that did not exit cleanly.
kill_captured_groups

# Catch Chrome/Node processes that became orphans but retained the Reader
# worker's PORT environment variable.
kill_remaining_port_owned_processes
sleep 1

final_reader_processes="$({ reader_process_pids; reader_listener_pids; } | sort -nu || true)"
if [[ -n "${final_reader_processes}" ]]; then
  warn "Reader-owned processes still remain after cleanup: ${final_reader_processes//$'\n'/ }"
  exit 2
fi

log "Reader stack and Reader-owned Node/Chrome processes have been stopped."
