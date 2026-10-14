#!/usr/bin/env bash
# Read-only diagnosis for a stopped FVQA teacher group. It checks the durable
# teacher log and the local kernel ring buffer for OOM/process-kill evidence.

set -Eeuo pipefail

die() {
    echo "[fix2-teacher-diagnose] error: $*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "usage: $0"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_TAG="${RUN_TAG:-train_scale_v1}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_fvqa_${RUN_TAG}}"
RUN_DIR="$REPO_ROOT/runs/fix2_fvqa_${RUN_TAG}"
HDFS_LOG_FILE="$RUN_DIR/logs/teacher_workers.log"
LOCAL_LOG_FILE="$LOCAL_STATE_DIR/logs/teacher_workers.log"
LOG_FILE="$HDFS_LOG_FILE"
[[ -f "$LOCAL_LOG_FILE" ]] && LOG_FILE="$LOCAL_LOG_FILE"
PID_FILE="$RUN_DIR/pids/teacher_workers.pid"
[[ -f "$LOG_FILE" ]] || die "missing teacher log: $LOG_FILE"

echo "[fix2-teacher-diagnose] run: $RUN_DIR"
echo "[fix2-teacher-diagnose] log: $LOG_FILE"
echo "[fix2-teacher-diagnose] process IDs observed in the teacher log:"
{
    [[ -s "$PID_FILE" ]] && cat "$PID_FILE"
    grep -oE 'worker=[^:[:space:]]+:[0-9]+:' "$LOG_FILE" | sed -E 's/.*:([0-9]+):/\1/'
} | sort -u | tr '\n' ' '
printf '\n'

echo "[fix2-teacher-diagnose] Python-level failures / recorded exits:"
if ! grep -Ein 'traceback|exception|fatal python|memoryerror|\[fix2-teacher-exit\]|\[fix2-teacher\].*status=(pending|failed)' "$LOG_FILE" | tail -n 40; then
    echo "  none found in teacher_workers.log"
fi

echo "[fix2-teacher-diagnose] kernel OOM / kill evidence (if permitted):"
kernel_output="$(dmesg -T 2>&1 || true)"
if [[ "$kernel_output" == *"Operation not permitted" || "$kernel_output" == *"Permission denied" ]]; then
    echo "  kernel ring buffer is not readable by this account"
elif ! printf '%s\n' "$kernel_output" | grep -Ei 'out of memory|oom-kill|killed process|kill process|invoked oom' | tail -n 40; then
    echo "  no OOM/process-kill record found in readable kernel history"
fi

echo "[fix2-teacher-diagnose] final teacher log lines:"
tail -n 30 "$LOG_FILE"
