#!/usr/bin/env bash
# Export a completed, train-only FVQA finalization to a self-contained
# LlamaFactory ShareGPT dataset, then run the existing causality/leakage audit.

set -Eeuo pipefail

die() {
    echo "[fix2-sharegpt-export] error: $*" >&2
    exit 1
}

[[ $# -ge 1 && $# -le 2 ]] || die "usage: $0 RUN_DIR [LOCAL_STATE_DIR]"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

RUN_DIR="$1"
[[ -d "$RUN_DIR" ]] || die "RUN_DIR does not exist: $RUN_DIR"
RUN_DIR="$(cd "$RUN_DIR" && pwd)"
RUN_NAME="$(basename "$RUN_DIR")"
LOCAL_USER="${USER:-$(id -un 2>/dev/null || printf 'unknown')}"
LOCAL_STATE_DIR="${2:-${LOCAL_STATE_DIR:-/tmp/${LOCAL_USER}/${RUN_NAME}}}"
[[ -d "$LOCAL_STATE_DIR" ]] || die "LOCAL_STATE_DIR does not exist: $LOCAL_STATE_DIR"
LOCAL_STATE_DIR="$(cd "$LOCAL_STATE_DIR" && pwd)"

FINAL_STATE_DIR="${FINAL_STATE_DIR:-$LOCAL_STATE_DIR/finalization}"
ACCEPTED="$FINAL_STATE_DIR/teacher_accepted_raw.jsonl"
[[ -f "$ACCEPTED" ]] || die "accepted raw records are missing: $ACCEPTED"
if [[ ! -s "$ACCEPTED" ]]; then
    echo "[fix2-sharegpt-export] no accepted trajectories; no ShareGPT dataset was written"
    exit 0
fi

# Keep immutable export artifacts with the shared run so the resulting dataset
# can be consumed from a different node.  Resumable judge checkpoints stay on
# the coordinator-local disk with the other finalization state.
EXPORT_DIR="${EXPORT_DIR:-$RUN_DIR/sharegpt_export}"
PRE_LEAKAGE="$EXPORT_DIR/trajectories_sharegpt_pre_leakage.json"
FINAL_SHAREGPT="$EXPORT_DIR/trajectories_sharegpt.json"
LEAKAGE_STAGE="$EXPORT_DIR/.metadata/trajectories_sharegpt_leakage_stage.json"
LEAKAGE_AUDIT="$EXPORT_DIR/.metadata/information_leakage_audit.jsonl"
LEAKAGE_STATE_DIR="${LEAKAGE_STATE_DIR:-$FINAL_STATE_DIR/sharegpt_leakage_state}"
LEAKAGE_JUDGE_MODEL="${LEAKAGE_JUDGE_MODEL:-gemini38flash_internal_azure}"
LEAKAGE_JUDGE_WORKERS="${LEAKAGE_JUDGE_WORKERS:-8}"
LEAKAGE_JUDGE_MAX_TOKENS="${LEAKAGE_JUDGE_MAX_TOKENS:-1200}"
CONVERT_WORKERS="${CONVERT_WORKERS:-4}"

for value in "$LEAKAGE_JUDGE_WORKERS" "$LEAKAGE_JUDGE_MAX_TOKENS" "$CONVERT_WORKERS"; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || die "worker/token settings must be positive integers, got: $value"
done

if [[ -f "$FINAL_SHAREGPT" ]]; then
    echo "[fix2-sharegpt-export] reusing completed export: $FINAL_SHAREGPT"
    exit 0
fi

if [[ ! -f "$PRE_LEAKAGE" ]]; then
    [[ ! -e "$EXPORT_DIR" ]] || die "incomplete export directory without pre-leakage data: $EXPORT_DIR"
    echo "[fix2-sharegpt-export] converting accepted raw trajectories to ShareGPT"
    python scripts/convert_debug_vqa_jsonl_to_sharegpt.py \
        --input-jsonl "$ACCEPTED" \
        --output-dir "$EXPORT_DIR" \
        --workers "$CONVERT_WORKERS"
    mv "$EXPORT_DIR/trajectories_sharegpt.json" "$PRE_LEAKAGE"
else
    [[ -d "$EXPORT_DIR/images" ]] || die "pre-leakage export has no images directory: $EXPORT_DIR/images"
    echo "[fix2-sharegpt-export] reusing converted pre-leakage data: $PRE_LEAKAGE"
fi

echo "[fix2-sharegpt-export] auditing converted ShareGPT trajectories for information leakage"
leakage_audit() {
    python synthesis/post_process/filter_trajectory_information_leakage.py \
        --input "$PRE_LEAKAGE" \
        --output "$LEAKAGE_STAGE" \
        --audit-jsonl "$LEAKAGE_AUDIT" \
        --state-dir "$LEAKAGE_STATE_DIR" \
        --model-alias "$LEAKAGE_JUDGE_MODEL" \
        --max-tokens "$LEAKAGE_JUDGE_MAX_TOKENS" \
        --workers "$LEAKAGE_JUDGE_WORKERS" \
        "$@"
}

# The leakage tool requires an existing run_config.json before it accepts
# --resume.  The first export attempt must therefore initialize a fresh state;
# later attempts reuse only its durable checkpointed keep/filter verdicts.
if [[ -f "$LEAKAGE_STATE_DIR/run_config.json" ]]; then
    echo "[fix2-sharegpt-export] resuming leakage audit checkpoint: $LEAKAGE_STATE_DIR"
    leakage_audit --resume
else
    echo "[fix2-sharegpt-export] starting leakage audit checkpoint: $LEAKAGE_STATE_DIR"
    leakage_audit
fi

mv "$LEAKAGE_STAGE" "$FINAL_SHAREGPT"
echo "[fix2-sharegpt-export] complete"
echo "  ShareGPT dataset: $FINAL_SHAREGPT"
echo "  pre-leakage copy: $PRE_LEAKAGE"
echo "  image directory:  $EXPORT_DIR/images"
echo "  leakage audit:    $LEAKAGE_AUDIT"
