#!/usr/bin/env bash
set -euo pipefail

# Outcome-only RLOO control for the v3-0930 PPO-epochs=2 experiment.
# The existing baseline launcher pins PPO_EPOCHS=2, partial credit off,
# advantage shaping off, checkpoint-252, and the same 6010-row v3 dataset.
# Match v3-0930's concurrency and newly adopted one-hour workflow timeout.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORKFLOW_TIMEOUT_SECONDS=3600
export N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-768}"
export N_PARALLEL_TOOLS="${N_PARALLEL_TOOLS:-768}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-open_mm_searcher_8b_4node_vllm_rloo_outcome_baseline_v3_ppo_epoch2}"
export EXPERIMENT_DESCRIPTION="${EXPERIMENT_DESCRIPTION:-Outcome-only RLOO v3 control: PPO epochs=2; partial reward and credit-turn advantage shaping off; 0.9 accuracy + 0.1 format; checkpoint-252; 6010-row v3 dataset; TITO/TIS; 768 rollout/tool slots; one-hour workflow timeout; redundancy shadow only.}"

exec bash "${SCRIPT_DIR}/qwen3-vl-8b-4node-vllm-rloo-baseline.sh" "$@"
