#!/usr/bin/env bash
set -u

reader_dir="${1:?reader directory is required}"
reader_port="${2:?reader port is required}"
node_heap_mb="${3:?node heap size is required}"
auto_restart="${4:-1}"
restart_delay_s="${5:-5}"

cd "${reader_dir}" || exit 1

child_pid=""
stopping=0

kill_child_group() {
  local pid="${1:-}"
  [[ -n "${pid}" ]] || return 0
  kill -TERM -- "-${pid}" 2>/dev/null || kill -TERM "${pid}" 2>/dev/null || true
  sleep 1
  kill -KILL -- "-${pid}" 2>/dev/null || kill -KILL "${pid}" 2>/dev/null || true
}

stop() {
  stopping=1
  kill_child_group "${child_pid}"
  exit 143
}

trap stop TERM INT HUP

while true; do
  # Put npm, Node, and Chromium descendants in their own process group so a
  # crashed/restarted Reader cannot leave its browser children behind.
  setsid env \
    PORT="${reader_port}" \
    NODE_OPTIONS="--max-old-space-size=${node_heap_mb}" \
    npm run start &
  child_pid=$!

  wait "${child_pid}"
  status=$?
  old_child_pid="${child_pid}"
  child_pid=""
  kill_child_group "${old_child_pid}"

  if [[ "${stopping}" == "1" || "${auto_restart,,}" != "1" && "${auto_restart,,}" != "true" && "${auto_restart,,}" != "yes" && "${auto_restart,,}" != "on" ]]; then
    exit "${status}"
  fi

  printf '[reader-supervisor] Reader exited status=%s; restarting in %ss\n' "${status}" "${restart_delay_s}"
  sleep "${restart_delay_s}"
done
