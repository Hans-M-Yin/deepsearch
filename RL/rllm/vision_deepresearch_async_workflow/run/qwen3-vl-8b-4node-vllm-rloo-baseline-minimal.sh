#!/usr/bin/env bash
set -euo pipefail

# Outcome-only control for the partial-credit experiment. Same dataset,
# optimizer and rollout settings; no partial judge requests or reshaping.
export TRAIN_PROMPT_MINI_BSZ=96
# smoke-128 is split into 96 training + 32 test examples.
export TRAIN_PROMPT_BSZ=96
export N_RESP_PER_PROMPT=2
export RLLM_MM_TIS="${RLLM_MM_TIS:-1}"
export RLLM_MM_TIS_THRESHOLD="${RLLM_MM_TIS_THRESHOLD:-2.0}"
export PARTIAL_CREDIT_MODE="off"
export PARTIAL_CREDIT_ADVANTAGE_SHAPING="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROLLOUT_NAME="vllm"
export MODEL_PATH="/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL/saves/qwen3_vl_8b/full/searchweave_fvqa_livevqa_16004_2node16gpu_lr_2e_5_cosine_to_zero_epoch6/checkpoint-126"
export NNODES="4"
export N_GPUS_PER_NODE="8"
# Retain the existing redundancy monitor only; it contributes no reward.
export SEARCH_REDUNDANCY_MODE="shadow"
export SEARCH_REDUNDANCY_EMBEDDING_URL="http://10.124.136.240:8087/embed"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-open_mm_searcher_8b_4node_vllm_rloo_outcome_baseline}"
export EXPERIMENT_DESCRIPTION="${EXPERIMENT_DESCRIPTION:-Outcome-only RLOO: 0.9 accuracy + 0.1 format, partial credit off, no turn reshaping, rollout/trainer logprob diagnostics.}"
export DATASET_NAME="Vision-DeepResearch-QA-smoke-128"
export LOSS_AGG_MODE="token-mean"
export TOTAL_EPOCHS="4"
export LOG_ENTROPY="${LOG_ENTROPY:-True}"
export RL_LOCAL_IO="${RL_LOCAL_IO:-1}"
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
