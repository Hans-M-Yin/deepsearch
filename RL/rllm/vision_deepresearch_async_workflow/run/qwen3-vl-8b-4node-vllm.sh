#!/usr/bin/env bash
set -euo pipefail

# Qwen3-VL-8B DeepResearch RL configuration for 4 nodes x 8 GPUs using vLLM.
# The training/data/workflow/checkpoint settings are inherited from the common
# 2-node configuration.  TP1 is used for rollout so the 32 GPUs can provide
# more independent vLLM rollout replicas than the 2-node TP2 setup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROLLOUT_NAME="vllm"
export NNODES="4"
export N_GPUS_PER_NODE="8"
export SEARCH_REDUNDANCY_MODE="reward"
export SEARCH_REDUNDANCY_EMBEDDING_URL="http://10.124.136.240:8087/embed"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-open_mm_searcher_8b_4node_vllm_dataset2}"
if [[ "${EXPERIMENT_NAME}" != *_add_redundancy ]]; then
  export EXPERIMENT_NAME="${EXPERIMENT_NAME}_add_redundancy"
fi
export EXPERIMENT_DESCRIPTION="${EXPERIMENT_DESCRIPTION:-Add a redundancy penalty, remove the max-turn limit, and tune the reward parameters for another run. The previous max-turn setting only reduced the number of turns, while severe repetition remained.}"
# v3 retains the same 6,010 rows as the formal dataset and adds graph-derived
# source/target anchors for deterministic partial-credit evidence gating.
export DATASET_NAME="${DATASET_NAME:-Vision-DeepResearch-QA-rl-quality-filtered-benchmark-6010-partial-credit-v3}"
# Use token-level loss aggregation for the formal 4-node vLLM run.  The
# shared 2-node launcher keeps its original seq-mean-token-mean default.
export LOSS_AGG_MODE="token-mean"
# Audit statement-level partial credit in W&B without changing PPO/GRPO
# rewards. The auxiliary judge is deliberately kept independent from the
# training reward until its precision has been fully validated.
export PARTIAL_CREDIT_MODE="${PARTIAL_CREDIT_MODE:-shadow}"
export PARTIAL_CREDIT_JUDGE_MODEL="${PARTIAL_CREDIT_JUDGE_MODEL:-multimodal_process_global_2}"
export PARTIAL_CREDIT_TRACE_CHUNK_STEPS="${PARTIAL_CREDIT_TRACE_CHUNK_STEPS:-15}"
# Explicitly off by default: when enabled, judge-validated action tokens get
# local advantage shaping (negative -> 0, positive -> 2x). Outcome-incorrect
# positive-advantage rows also mask assistant tokens after their last verified
# credit turn, preventing a wrong suffix from being reinforced.
export PARTIAL_CREDIT_ADVANTAGE_SHAPING="${PARTIAL_CREDIT_ADVANTAGE_SHAPING:-0}"
# Compute policy entropy for W&B monitoring without adding it to PPO loss.
export LOG_ENTROPY="${LOG_ENTROPY:-True}"
export RL_LOCAL_IO="${RL_LOCAL_IO:-1}"
# Always resume from the latest complete checkpoint published by the trainer.
# Pin this for the formal 4-node vLLM run so a stale caller environment cannot
# accidentally disable resume.
export TRAINER_RESUME_MODE="auto"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
export TEST_FREQ="${TEST_FREQ:-5}"
export RLLM_PRE_UPDATE_WATCHER_PATH="/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/watcher.log"

# Keep headroom for multimodal peak allocations and CUDA graph/private-pool
# overhead.  This is intentionally lower than the 2-node TP2 vLLM default.
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"

# Match the current formal 2-node vLLM configuration.  More GPUs come from
# the additional nodes and TP1 rollout replicas, not from an unbounded request
# fan-out that could increase queueing and memory pressure.
export N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-768}"
export N_PARALLEL_TOOLS="${N_PARALLEL_TOOLS:-768}"
# Keep two incomplete long-tail groups out of the training step while
# preserving complete n=6 groups for RLOO/GRPO advantage computation.
export ROLLOUT_TAIL_DROP_GROUPS="${ROLLOUT_TAIL_DROP_GROUPS:-2}"

# The async rollout implementation uses vLLM's V1 AsyncLLM engine.
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

exec bash "${SCRIPT_DIR}/qwen3-vl-8b-2node.sh" "$@"
