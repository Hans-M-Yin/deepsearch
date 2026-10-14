#!/usr/bin/env bash
# Resume the paused full FVQA collection from its existing local SQLite state.

set -Eeuo pipefail

[[ $# -eq 0 ]] || {
    echo "[fix2-fvqa-resume] usage: $0" >&2
    exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

# The router reads already-completed base evaluation details and replenishes
# only the teacher queue. A zero cap means all remaining eligible candidates.
bash "$REPO_ROOT/synthesis/post_process/fix2/resume_fvqa_router.sh" 0
bash "$REPO_ROOT/synthesis/post_process/fix2/restart_fvqa_teacher.sh"

echo "[fix2-fvqa-resume] complete; base inference/judging and finished teacher results were not rerun"
