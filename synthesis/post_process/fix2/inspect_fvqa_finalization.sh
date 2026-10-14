#!/usr/bin/env bash
# One-command inspection for a completed local FVQA finalization.

set -euo pipefail

[[ $# -ge 1 && $# -le 2 ]] || {
    echo "usage: $0 RUN_DIR [LOCAL_STATE_DIR]" >&2
    exit 1
}

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
RUN_NAME="$(basename "$1")"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${2:-${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/${RUN_NAME}}}"
STATE_DIR="${FINAL_STATE_DIR:-$LOCAL_STATE_DIR/finalization}"

[[ -d "$STATE_DIR" ]] || {
    echo "[fix2-inspect] finalization state directory does not exist: $STATE_DIR" >&2
    exit 1
}

python -m synthesis.post_process.fix2.inspect_finalization --state-dir "$STATE_DIR"
