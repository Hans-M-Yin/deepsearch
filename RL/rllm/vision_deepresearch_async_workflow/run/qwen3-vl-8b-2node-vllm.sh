#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-8B DeepResearch RL configuration for 2 nodes x 8 GPUs using vLLM.
# Keep all training, data, workflow, parallelism, checkpoint, and logging
# settings in qwen3-vl-8b-2node.sh so the SGLang and vLLM experiments cannot
# drift apart accidentally. This wrapper only selects the rollout backend.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROLLOUT_NAME="vllm"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-open_mm_searcher_8b_2node_vllm}"
export RL_LOCAL_IO="${RL_LOCAL_IO:-1}"
export TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-auto}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.78}"
export N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-384}"
export N_PARALLEL_TOOLS="${N_PARALLEL_TOOLS:-384}"
export ROLLOUT_TAIL_DROP_GROUPS="${ROLLOUT_TAIL_DROP_GROUPS:-2}"

# The async rollout implementation uses vLLM's V1 AsyncLLM engine.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

# Use vLLM's FlashAttention backend and let vLLM select the compatible
# FlashAttention implementation for the current platform.

exec bash "${SCRIPT_DIR}/qwen3-vl-8b-2node.sh" "$@"
