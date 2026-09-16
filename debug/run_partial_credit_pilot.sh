#!/usr/bin/env bash
# Diagnostic-only partial-credit audit over saved rollout JSON.  It does not
# modify training, model weights, rollouts, or the source dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

set -a
source RL/.env_remote
set +a

export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/RL/rllm${PYTHONPATH:+:${PYTHONPATH}}"
export PARTIAL_CREDIT_JUDGE_TIMEOUT_S="${PARTIAL_CREDIT_JUDGE_TIMEOUT_S:-120}"
export PARTIAL_CREDIT_TRACE_CHUNK_STEPS="${PARTIAL_CREDIT_TRACE_CHUNK_STEPS:-15}"
# Keep this diagnostic independent from the global JUDGE_MODEL used by other
# workflows. Callers may still override it for an A/B comparison.
export PARTIAL_CREDIT_JUDGE_MODEL="${PARTIAL_CREDIT_JUDGE_MODEL:-gpt55_internal_azure}"

exec python -u debug/score_saved_partial_credit.py \
  --rollout-dir "${PARTIAL_CREDIT_PILOT_ROLLOUT_DIR:-RL/rollouts_saved/open_mm_search_8b_4node_vllm/train_step_30_epoch_1}" \
  --dataset-parquet "${PARTIAL_CREDIT_PILOT_DATASET:-runs/0712_multi_seed_visual_test_8192_8/vqa/rl_quality_filtered_benchmark_medium_hard_augmented/train_partial_credit_v3.parquet}" \
  --samples "${PARTIAL_CREDIT_PILOT_SAMPLES:-6}" \
  --statement-pairs "${PARTIAL_CREDIT_PILOT_STATEMENT_PAIRS:-0}" \
  --seed "${PARTIAL_CREDIT_PILOT_SEED:-20260916}" \
  --mode "${PARTIAL_CREDIT_PILOT_MODE:-reward}" \
  --output "${PARTIAL_CREDIT_PILOT_OUTPUT:-/tmp/partial_credit_pilot_step30_v2.json}"
