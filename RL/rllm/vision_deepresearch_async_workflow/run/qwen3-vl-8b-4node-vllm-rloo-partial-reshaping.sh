#!/usr/bin/env bash
set -euo pipefail

# Paired experiment copied from the RLOO outcome baseline. Keep the same
# initial SFT model, data and training settings; enable partial credit/shaping.
export TRAIN_PROMPT_MINI_BSZ=96
export RLLM_MM_TITO="${RLLM_MM_TITO:-1}"
export RLLM_MM_CONSISTENCY_DEBUG=0
export RLLM_MM_TIS="${RLLM_MM_TIS:-1}"
export RLLM_MM_TIS_THRESHOLD="${RLLM_MM_TIS_THRESHOLD:-2.0}"
export PARTIAL_CREDIT_MODE="reward"
export PARTIAL_CREDIT_ADVANTAGE_SHAPING="1"
export PARTIAL_CREDIT_JUDGE_MODEL="gemini38flash_internal_azure"
export PARTIAL_CREDIT_JUDGE_CONCURRENCY="${N_PARALLEL_TASKS:-768}"
export PARTIAL_CREDIT_TRACE_CHUNK_STEPS=15
export PARTIAL_CREDIT_STATEMENT_REWARD="0.1"
export PARTIAL_CREDIT_TRAJECTORY_CAP="0.4"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROLLOUT_NAME="vllm"
export MODEL_PATH="/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL/saves/qwen3_vl_8b/full/searchweave_fvqa_livevqa_16004_2node16gpu_lr_2e_5_cosine_to_zero_epoch6/checkpoint-252"
export NNODES="4"
export N_GPUS_PER_NODE="8"
# Retain the existing redundancy monitor only; it contributes no reward.
export SEARCH_REDUNDANCY_MODE="shadow"
export SEARCH_REDUNDANCY_EMBEDDING_URL="http://10.124.136.240:8087/embed"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-open_mm_searcher_8b_4node_vllm_rloo_partial_reward_reshaping}"
export EXPERIMENT_DESCRIPTION="${EXPERIMENT_DESCRIPTION:-RLOO: outcome reward plus failure-only furthest-statement progress credit (capped at 0.4), turn advantage reshaping, TITO and TIS; search redundancy shadow only.}"
export DATASET_NAME="Vision-DeepResearch-QA-rl-quality-filtered-benchmark-6010-partial-credit-v3"
export LOSS_AGG_MODE="token-mean"
export TOTAL_EPOCHS="4"
export LOG_ENTROPY="${LOG_ENTROPY:-True}"
export RL_LOCAL_IO="${RL_LOCAL_IO:-1}"
# Auto-resume applies to this experiment's separate checkpoint directory.
export TRAINER_RESUME_MODE="auto"
export VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-256}"
export TEST_FREQ=10
export SAVE_FREQ=10
export RLLM_PRE_UPDATE_WATCHER_PATH="/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/watcher.log"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.80}"
export ROLLOUT_TP="${ROLLOUT_TP:-1}"
export ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
export N_PARALLEL_TASKS="${N_PARALLEL_TASKS:-768}"
export N_PARALLEL_TOOLS="${N_PARALLEL_TOOLS:-768}"
export ROLLOUT_TAIL_DROP_GROUPS="${ROLLOUT_TAIL_DROP_GROUPS:-2}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

exec bash "${SCRIPT_DIR}/qwen3-vl-8b-2node.sh" "$@"
