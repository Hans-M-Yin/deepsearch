#!/usr/bin/env bash
# Finalize an existing evaluation-only FVQA test expert probe without rerunning
# sampling or teacher inference.
#
# Example:
#   bash synthesis/post_process/fix2/finalize_fvqa_test_expert_probe.sh 20260918_030036

set -Eeuo pipefail

die() {
    echo "[fix2-test-probe-finalize] error: $*" >&2
    exit 1
}

[[ $# -le 1 ]] || die "usage: $0 [RUN_TAG]"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_TAG="${1:-${RUN_TAG:-}}"
[[ -n "$RUN_TAG" ]] || die "provide RUN_TAG, e.g. 20260918_030036"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/runs/fix2_fvqa_test_expert_probe_${RUN_TAG}}"
LOCAL_STATE_DIR="${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/fix2_fvqa_test_expert_probe_${RUN_TAG}}"
QUEUE_NAME="${QUEUE_NAME:-fvqa_test_expert_probe_responses}"
PRIVATE_CASES_MANIFEST="$RUN_DIR/test_probe_cases_private.jsonl"

[[ -d "$RUN_DIR" ]] || die "RUN_DIR does not exist: $RUN_DIR"
[[ -f "$PRIVATE_CASES_MANIFEST" ]] || die "missing probe private case manifest: $PRIVATE_CASES_MANIFEST"
[[ -d "$LOCAL_STATE_DIR" ]] || die "LOCAL_STATE_DIR does not exist: $LOCAL_STATE_DIR"

echo "[fix2-test-probe-finalize] EVALUATION ONLY — reusing existing teacher trajectories."
echo "  run directory: $RUN_DIR"
echo "  local state:   $LOCAL_STATE_DIR"

PRIVATE_CASES_MANIFEST="$PRIVATE_CASES_MANIFEST" \
QUEUE_NAME="$QUEUE_NAME" \
LOCAL_STATE_DIR="$LOCAL_STATE_DIR" \
EXPORT_SHAREGPT=0 \
bash synthesis/post_process/fix2/finalize_fvqa_pipeline.sh "$RUN_DIR" "$LOCAL_STATE_DIR"
