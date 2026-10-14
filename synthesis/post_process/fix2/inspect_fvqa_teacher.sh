#!/usr/bin/env bash
# Read-only status check for the durable FVQA teacher queue. Run on the same
# coordinator that owns its /tmp SQLite state.

set -Eeuo pipefail

die() {
    echo "[fix2-teacher-status] error: $*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "usage: $0"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_TAG="${RUN_TAG:-train_scale_v1}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_fvqa_${RUN_TAG}}"
QUEUE_NAME="${QUEUE_NAME:-teacher_collect_responses}"
RUN_DIR="$REPO_ROOT/runs/fix2_fvqa_${RUN_TAG}"
PID_FILE="$RUN_DIR/pids/teacher_workers.pid"
QUEUE_DB="$LOCAL_STATE_DIR/teacher_queue.sqlite"
HDFS_LOG_FILE="$RUN_DIR/logs/teacher_workers.log"
LOCAL_LOG_FILE="$LOCAL_STATE_DIR/logs/teacher_workers.log"
LOG_FILE="$HDFS_LOG_FILE"
[[ -f "$LOCAL_LOG_FILE" ]] && LOG_FILE="$LOCAL_LOG_FILE"

[[ -f "$QUEUE_DB" ]] || die "missing local teacher queue: $QUEUE_DB (run this on the original coordinator)"

echo "[fix2-teacher-status] run:   $RUN_DIR"
echo "[fix2-teacher-status] state: $LOCAL_STATE_DIR"
echo "[fix2-teacher-status] log:   $LOG_FILE"
if [[ -s "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
    if [[ "$pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$pid" 2>/dev/null; then
        echo "[fix2-teacher-status] worker process: running (pid=$pid)"
    else
        echo "[fix2-teacher-status] worker process: not running (pid=${pid:-unknown})"
    fi
else
    echo "[fix2-teacher-status] worker process: no PID record"
fi

echo "[fix2-teacher-status] queue:"
# Sixteen workers can briefly hold SQLite's write lock.  Do not let a
# read-only diagnostic appear to freeze behind the queue's 30-second SQLite
# busy timeout: make the contention explicit and still show the recent log.
set +e
queue_output="$(timeout 6s python -m synthesis.post_process.fix2.run queue-stats \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" 2>&1)"
queue_status=$?
set -e
if [[ $queue_status -eq 0 ]]; then
    printf '%s\n' "$queue_output"
elif [[ $queue_status -eq 124 ]]; then
    echo "  unavailable: queue database remained locked for 6 seconds; retry shortly"
else
    echo "  unavailable: queue-stats exited $queue_status"
    [[ -n "$queue_output" ]] && printf '%s\n' "$queue_output"
fi

if [[ -f "$LOG_FILE" ]]; then
    echo "[fix2-teacher-status] recent GPT-5.4 QPM samples:"
    qpm_lines="$(grep -F '[llm-usage] alias=gpt54_2_internal_azure_responses' "$LOG_FILE" | tail -n 5 || true)"
    if [[ -n "$qpm_lines" ]]; then
        printf '%s\n' "$qpm_lines"
    else
        echo "  no completed GPT-5.4 call has been logged yet"
    fi
    echo "[fix2-teacher-status] recent log:"
    tail -n 12 "$LOG_FILE"
fi
