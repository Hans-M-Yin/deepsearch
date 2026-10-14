#!/usr/bin/env bash
# Gracefully stop the four processes launched by launch_fvqa_pipeline.sh.
# With no argument, only the most recently modified fix2_fvqa_* run is used.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage:
  bash synthesis/post_process/fix2/stop_fvqa_pipeline.sh [RUN_DIR]

No RUN_DIR is needed for the usual case: the script selects the most recently
modified runs/fix2_fvqa_* directory. It only stops live PIDs recorded in that
run's pids/ directory after checking their command line belongs to fix2 or
opensearch_vl/run_infer.py. It never scans or kills unrelated Python jobs.
EOF
}

die() {
    echo "[fix2-stop] error: $*" >&2
    exit 1
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
    usage
    exit 0
fi
[[ $# -le 1 ]] || die "expected at most one RUN_DIR argument"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [[ $# -eq 1 ]]; then
    RUN_DIR="$1"
else
    shopt -s nullglob
    pid_dirs=("$REPO_ROOT"/runs/fix2_fvqa_*/pids)
    ((${#pid_dirs[@]})) || die "no runs/fix2_fvqa_*/pids directory was found; pass RUN_DIR explicitly"
    newest=""
    newest_mtime=-1
    for pid_dir in "${pid_dirs[@]}"; do
        mtime="$(stat -c '%Y' "$pid_dir")"
        if ((mtime > newest_mtime)); then
            newest="$pid_dir"
            newest_mtime="$mtime"
        fi
    done
    RUN_DIR="$(dirname "$newest")"
fi

PID_DIR="$RUN_DIR/pids"
[[ -d "$PID_DIR" ]] || die "PID directory does not exist: $PID_DIR"
echo "[fix2-stop] selected run: $RUN_DIR"

stopped=0
already_stopped=0
skipped=0
for name in base_infer base_judge failure_router teacher_workers; do
    pid_file="$PID_DIR/${name}.pid"
    if [[ ! -s "$pid_file" ]]; then
        echo "[fix2-stop] ${name}: no PID file"
        continue
    fi
    pid="$(<"$pid_file")"
    if ! [[ "$pid" =~ ^[1-9][0-9]*$ ]] || ! kill -0 "$pid" 2>/dev/null; then
        echo "[fix2-stop] ${name}: already stopped (pid=${pid})"
        ((already_stopped += 1))
        continue
    fi
    command_line="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
    case "$command_line" in
        *"synthesis.post_process.fix2.run"*|*"opensearch_vl/run_infer.py"*) ;;
        *)
            echo "[fix2-stop] ${name}: refusing pid=${pid}; PID was reused or command is not a fix2 process"
            ((skipped += 1))
            continue
            ;;
    esac
    kill "$pid"
    echo "[fix2-stop] ${name}: sent TERM to pid=${pid}"
    ((stopped += 1))
done

echo "[fix2-stop] summary: stopped=${stopped} already_stopped=${already_stopped} skipped=${skipped}"
