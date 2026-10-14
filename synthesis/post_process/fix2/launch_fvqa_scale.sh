#!/usr/bin/env bash
# Stage a train-only FVQA collection at 512, then extend the exact same
# durable run to a larger queue cap (for example 2,048) without recomputing
# completed base inference, base judgements, or teacher results.

set -Eeuo pipefail

die() {
    echo "[fix2-scale-launch] error: $*" >&2
    exit 1
}

[[ $# -le 1 ]] || die "usage: $0 [TARGET_CASES]"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
TARGET_CASES="${1:-${TARGET_CASES:-512}}"
[[ "$TARGET_CASES" =~ ^[1-9][0-9]*$ ]] || die "TARGET_CASES must be a positive integer"
BASE_LIMIT="${BASE_LIMIT:-0}"
[[ "$BASE_LIMIT" =~ ^[0-9]+$ ]] || die "BASE_LIMIT must be a nonnegative integer (0 means all FVQA train rows)"

# Keep this tag stable between the 512 and 2,048 invocations.  The underlying
# launcher derives both RUN_DIR and its coordinator-local SQLite state from it.
RUN_TAG="${RUN_TAG:-train_scale_v1}"
LEGACY_SHAREGPT="${LEGACY_SHAREGPT:-$REPO_ROOT/data/sharegpt_dataset_refined_gpt54_only_17k_final/trajectories_sharegpt.json}"
STYLE_EXAMPLES_FILE="${STYLE_EXAMPLES_FILE:-$REPO_ROOT/synthesis/post_process/fix2/style_examples.production.legacy.json}"
if [[ ! -f "$STYLE_EXAMPLES_FILE" ]]; then
    [[ -f "$LEGACY_SHAREGPT" ]] || die "legacy ShareGPT source does not exist: $LEGACY_SHAREGPT"
    echo "[fix2-scale-launch] materializing bounded legacy-SFT style excerpts: $STYLE_EXAMPLES_FILE"
    python -m synthesis.post_process.fix2.run build-legacy-style-examples \
        --input-sharegpt "$LEGACY_SHAREGPT" \
        --output-examples "$STYLE_EXAMPLES_FILE"
fi

echo "[fix2-scale-launch] stage target: $TARGET_CASES teacher tasks"
echo "[fix2-scale-launch] durable run tag: $RUN_TAG"
echo "[fix2-scale-launch] base source limit: $BASE_LIMIT (0 means all FVQA train rows)"
echo "[fix2-scale-launch] completed records resume automatically; existing manifests expand through BASE_LIMIT"
echo "[fix2-scale-launch] legacy style anchors: $STYLE_EXAMPLES_FILE"
echo "[fix2-scale-launch] do not finalize this run until the final expansion is complete"

RUN_TAG="$RUN_TAG" \
BASE_LIMIT="$BASE_LIMIT" \
EXPAND_SOURCE_MANIFEST=1 \
HARD_CAP="$TARGET_CASES" \
MIN_CORRECT_TOOL_CALLS=6 \
STYLE_EXAMPLES_FILE="$STYLE_EXAMPLES_FILE" \
REQUIRE_STYLE_EXAMPLES=1 \
bash "$REPO_ROOT/synthesis/post_process/fix2/launch_fvqa_pipeline.sh"
