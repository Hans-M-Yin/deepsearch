#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-8B DeepResearch RL configuration for 2 nodes x 8 GPUs using
# vLLM with one rollout GPU per replica.  The training parallelism remains
# unchanged; only the rollout TP and vLLM batching knobs differ from the
# formal TP2 vLLM configuration.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROLLOUT_NAME="vllm"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-open_mm_searcher_8b_2node_vllm_tp1}"
export RL_LOCAL_IO="${RL_LOCAL_IO:-1}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.75}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-16384}"
export N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-288}"
export N_PARALLEL_TOOLS="${N_PARALLEL_TOOLS:-288}"

# The async rollout implementation uses vLLM's V1 AsyncLLM engine.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

exec bash "${SCRIPT_DIR}/qwen3-vl-8b-2node.sh" "$@"
