#!/usr/bin/env bash
# Safely pause the direct LiveVQA teacher collection and return active leases.

set -Eeuo pipefail

die() {
    echo "[fix2-livevqa-pause] error: $*" >&2
    exit 1
}

[[ $# -eq 0 ]] || die "usage: $0"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_TAG="${RUN_TAG:-gpt55_loopfix_scale2400}"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_livevqa_${RUN_TAG}}"
RUN_DIR="$REPO_ROOT/runs/fix2_livevqa_${RUN_TAG}"
PID_FILE="$RUN_DIR/pids/teacher_workers.pid"
QUEUE_DB="$LOCAL_STATE_DIR/teacher_queue.sqlite"
QUEUE_NAME="${QUEUE_NAME:-teacher_collect_livevqa_responses}"

[[ -f "$QUEUE_DB" ]] || die "missing local teacher queue: $QUEUE_DB (run on the original coordinator)"
if [[ -s "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
    [[ "$pid" =~ ^[1-9][0-9]*$ ]] || die "malformed PID record: $PID_FILE"
    if kill -0 "$pid" 2>/dev/null; then
        command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
        [[ "$command_line" == *"synthesis.post_process.fix2.run teacher-workers"* ]] || die "refusing pid=$pid; command does not match LiveVQA teacher_workers"
        kill -TERM "$pid"
        echo "[fix2-livevqa-pause] sent TERM to teacher_workers pid=$pid"
        for _ in {1..30}; do
            kill -0 "$pid" 2>/dev/null || break
            sleep 1
        done
        kill -0 "$pid" 2>/dev/null && die "teacher_workers pid=$pid did not stop within 30 seconds; queue was not changed"
    else
        echo "[fix2-livevqa-pause] teacher_workers already stopped (pid=$pid)"
    fi
else
    echo "[fix2-livevqa-pause] no teacher PID record"
fi

python -m synthesis.post_process.fix2.run requeue-leased \
    --queue-db "$QUEUE_DB" \
    --queue-name "$QUEUE_NAME" \
    --reason "controlled_livevqa_pause"

echo "[fix2-livevqa-pause] complete; raw results and completed queue rows were preserved"
