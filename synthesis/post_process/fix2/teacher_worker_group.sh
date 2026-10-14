#!/usr/bin/env bash
# Keep the Python teacher group as a child so ordinary crashes and forwarded
# termination signals leave an explicit exit record in teacher_workers.log.

set -uo pipefail

[[ $# -gt 0 ]] || {
    echo "[fix2-teacher-exit] error=no_child_command" >&2
    exit 2
}

child_pid=""

record_exit() {
    local status="$1"
    local cause="exit"
    if ((status >= 128)); then
        cause="signal_$((status - 128))"
    fi
    printf '[fix2-teacher-exit] utc=%s child_pid=%s exit_code=%s cause=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${child_pid:-unknown}" "$status" "$cause" >&2
}

forward_signal() {
    local signal="$1"
    if [[ -n "$child_pid" ]] && kill -0 "$child_pid" 2>/dev/null; then
        kill -"$signal" "$child_pid" 2>/dev/null || true
    fi
    if [[ -n "$child_pid" ]]; then
        wait "$child_pid"
        local status=$?
        record_exit "$status"
        exit "$status"
    fi
    record_exit 128
    exit 128
}

trap 'forward_signal TERM' TERM
trap 'forward_signal INT' INT
trap 'forward_signal HUP' HUP

"$@" &
child_pid=$!
wait "$child_pid"
status=$?
record_exit "$status"
exit "$status"
