#!/usr/bin/env bash
set -euo pipefail

export PARTIAL_CREDIT_JUDGE_CONCURRENCY=64
# With 192 prompts and PPO epochs=1, split each rollout batch into two updates.
export TRAIN_PROMPT_MINI_BSZ=96

# Qwen3-VL-8B DeepResearch RL: 4 nodes x 8 GPUs with vLLM rollout.
# This is the partial-credit experiment variant of qwen3-vl-8b-4node-vllm.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROLLOUT_NAME="vllm"
export NNODES="4"
export N_GPUS_PER_NODE="8"
# Record search redundancy for analysis, but do not include it in the reward.
export SEARCH_REDUNDANCY_MODE="shadow"
export SEARCH_REDUNDANCY_EMBEDDING_URL="http://10.124.136.240:8087/embed"

# Keep this experiment isolated from the outcome-only/redundancy baseline.
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-open_mm_searcher_8b_4node_vllm_partial_reward_reshaping}"
export EXPERIMENT_DESCRIPTION="${EXPERIMENT_DESCRIPTION:-4-node vLLM Agentic RL with bounded statement-level partial reward and turn-level advantage reshaping, alongside the search-redundancy penalty.}"

# Same 6,010-row formal dataset, augmented with graph-derived intermediate
# statements and source/target anchors under task.extra_info.partial_credit.
export DATASET_NAME="Vision-DeepResearch-QA-rl-quality-filtered-benchmark-6010-partial-credit-v3"

# Use token-level loss aggregation as in the formal 4-node vLLM baseline.
export LOSS_AGG_MODE="token-mean"
export TOTAL_EPOCHS="4"

# Enable the validated statement-level reward and post-RLOO local credit
# assignment. For a final-answer-incorrect trajectory with positive RLOO
# advantage, shaping also zeroes assistant-token advantage after its last
# verified credit turn. Do not inherit a caller's shadow/off setting here.
export PARTIAL_CREDIT_MODE="reward"
export PARTIAL_CREDIT_JUDGE_MODEL="gemini38flash_internal_azure"
export PARTIAL_CREDIT_TRACE_CHUNK_STEPS="15"
# Each judge-verified intermediate statement contributes +0.1, capped at +0.3
# per trajectory.
export PARTIAL_CREDIT_STATEMENT_REWARD="0.1"
export PARTIAL_CREDIT_TRAJECTORY_CAP="0.3"
export PARTIAL_CREDIT_ADVANTAGE_SHAPING="1"

# Compute policy entropy for W&B monitoring without adding it to PPO loss.
export LOG_ENTROPY="${LOG_ENTROPY:-True}"
export RL_LOCAL_IO="${RL_LOCAL_IO:-1}"
export TRAINER_RESUME_MODE="auto"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
export TEST_FREQ="${TEST_FREQ:-5}"
export RLLM_PRE_UPDATE_WATCHER_PATH="/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/watcher.log"

# Keep the rollout engine settings aligned with the 4-node vLLM baseline.
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
export N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-768}"
export N_PARALLEL_TOOLS="${N_PARALLEL_TOOLS:-768}"
export ROLLOUT_TAIL_DROP_GROUPS="${ROLLOUT_TAIL_DROP_GROUPS:-2}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

exec bash "${SCRIPT_DIR}/qwen3-vl-8b-2node.sh" "$@"
