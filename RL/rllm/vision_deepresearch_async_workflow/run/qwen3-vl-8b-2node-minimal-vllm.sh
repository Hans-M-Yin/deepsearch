#!/usr/bin/env bash
set -euo pipefail

# Minimal 2-node Qwen3-VL-8B DeepResearch run using vLLM.
#
# The formal configuration remains the single source of truth. This wrapper
# only reduces the registered training dataset and training batch so that the
# rollout -> reward -> update path can be checked with six responses per
# prompt and exactly three training updates.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROLLOUT_NAME="vllm"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-open_mm_searcher_8b_2node_minimal_vllm}"
export RL_LOCAL_IO="${RL_LOCAL_IO:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
export DATASET_NAME="${DATASET_NAME:-Vision-DeepResearch-QA-smoke-128}"
export TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-1}"
export N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-6}"
export TRAIN_PROMPT_MINI_BSZ="${TRAIN_PROMPT_MINI_BSZ:-1}"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-3}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# Keep the smoke test on the same FlashAttention backend as the formal vLLM
# run; vLLM selects the implementation for the current platform.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"

exec bash "${SCRIPT_DIR}/qwen3-vl-8b-2node.sh" "$@"
