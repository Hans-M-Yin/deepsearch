#!/usr/bin/env bash
set -xeuo pipefail

# Qwen3-VL-8B DeepResearch RL configuration for 2 nodes x 8 GPUs.
# This is the 8B multi-node configuration adapted to a 16-GPU world:
# train_tp x train_pp x train_cp = 4 x 2 x 2 = 16.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${RLLM_ROOT}:${RLLM_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"
cd "${RLLM_ROOT}"

# Load RL credentials when present.  Serper/Reader variables can also be
# exported before launching, or loaded from synthesis/.env by the caller.
if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

export WANDB_BASE_URL=${WANDB_BASE_URL:-https://api.wandb.ai}
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_ENGINE_ITERATION_TIMEOUT_S=${VLLM_ENGINE_ITERATION_TIMEOUT_S:-100000000000}
export CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export HYDRA_FULL_ERROR=1

# Use the validated multimodal sequence path for production training.  Debug
# events are disabled here so Ray workers do not concurrently append to a
# shared HDFS/FUSE log file.
export RLLM_MM_SEQUENCE_MODE=cumulative
export RLLM_MM_DEBUG=0

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
"${PYTHON_BIN}" -c "import transformer_engine.pytorch as te; print('transformer_engine: ok')"

# Ray workers on remote nodes receive only ``runtime_env.env_vars``.  Check
# the exact map passed to ray.init before an expensive rollout job starts so a
# missing reward-augmentation prefix cannot silently fail open for an entire
# run.  Never print the endpoint itself: it may be deployment-specific.
"${PYTHON_BIN}" -c '
import os
from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

keys = ("SEARCH_REDUNDANCY_MODE", "SEARCH_REDUNDANCY_EMBEDDING_URL")
driver = {key: os.environ.get(key, "") for key in keys}
worker = get_ppo_ray_runtime_env()["env_vars"]
missing = [key for key in keys if worker.get(key, "") != driver[key]]
if missing:
    raise RuntimeError(
        "Search redundancy Ray runtime-env forwarding failed for: "
        + ", ".join(missing)
    )
mode = driver["SEARCH_REDUNDANCY_MODE"] or "off"
embedding_url_configured = bool(driver["SEARCH_REDUNDANCY_EMBEDDING_URL"])
print(
    "##############3 [SearchRedundancy][RayEnvPreflight] "
    f"mode={mode} embedding_url_configured={embedding_url_configured}"
)
'

# ========= user-overridable paths / data =========
MODEL_PATH=${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/agent/OpenSearch-VL/SFT/saves/qwen3_vl_8b/full/data_redundancy_compressed_gpt54_640_tagclean_2node16gpu_lr_2e_5_cosine_to_zero_epoch5}
DATASET_NAME=${DATASET_NAME:-Vision-DeepResearch-QA-rl-combined-5871}
EVAL_DATASET_NAME=${EVAL_DATASET_NAME:-Vision-DeepResearch-QA-benchmark-eval-200}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-test}
PROJECT_NAME=${PROJECT_NAME:-vision-deepresearch}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-open_mm_searcher_8b_2node}
EXPERIMENT_DESCRIPTION=${EXPERIMENT_DESCRIPTION:-}
# Hydra parses override syntax after the shell has removed its quoting. Keep
# explicit quotes in the argument and escape user-provided JSON-style chars so
# descriptions containing spaces, punctuation, or quotes remain valid.
HYDRA_EXPERIMENT_DESCRIPTION="${EXPERIMENT_DESCRIPTION//\\/\\\\}"
HYDRA_EXPERIMENT_DESCRIPTION="${HYDRA_EXPERIMENT_DESCRIPTION//\"/\\\"}"
# Checkpoints must live on a shared absolute path. Ray workers run with
# node-local working directories, so a relative path would scatter shards
# across nodes and make resume fail when another worker cannot see them.
CKPTS_DIR=${CKPTS_DIR:-${PROJECT_ROOT}/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# Optional node-local I/O mode for rollout diagnostics. The vLLM wrappers
# enable this by default; the base SGLang launcher remains HDFS-backed unless
# RL_LOCAL_IO=1 is explicitly supplied.
if [[ "${RL_LOCAL_IO:-0}" == "1" ]]; then
    LOCAL_IO_ROOT="${LOCAL_IO_ROOT:-/tmp/open_search_vl_rl_io/${EXPERIMENT_NAME}}"
    export CACHE_DB_PATH="${CACHE_DB_PATH:-${LOCAL_IO_ROOT}/cache/deepresearch_cache.db}"
    export RLLM_MM_DEBUG_LOG_FILE="${RLLM_MM_DEBUG_LOG_FILE:-${LOCAL_IO_ROOT}/debug/multimodal_debug.jsonl}"
    export RLLM_MM_DEBUG_LOG_DIR="${RLLM_MM_DEBUG_LOG_DIR:-${LOCAL_IO_ROOT}/debug}"
    EPISODE_LOG_DIR="${EPISODE_LOG_DIR:-${LOCAL_IO_ROOT}/episodes}"
    echo "[local-io] root=${LOCAL_IO_ROOT}"
    echo "[local-io] cache_db=${CACHE_DB_PATH}"
    echo "[local-io] episode_log_dir=${EPISODE_LOG_DIR}"
    echo "[local-io] multimodal_debug_log=${RLLM_MM_DEBUG_LOG_FILE}"
fi

# ========= rollout / algorithm =========
rollout_mode="async"
rollout_name="${ROLLOUT_NAME:-sglang}"
return_raw_chat="True"
dtype="bfloat16"
adv_estimator="rloo"

kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001
clip_ratio_high=0.28

# ========= batch / workflow =========
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-192}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-6}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-256}"
n_parallel_tasks="${N_PARALLEL_TASKS:-256}"
n_parallel_tools="${N_PARALLEL_TOOLS:-256}"
max_turns=40

# ========= length =========
max_prompt_length=4096
max_response_length=56000
use_dynamic_bsz=True
infer_ppo_max_token_len_per_gpu=81920
actor_ppo_max_token_len_per_gpu=57344

# ========= offload / parallelism =========
offload=False
gen_tp="${ROLLOUT_TP:-2}"
rollout_max_num_batched_tokens="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-8192}"
train_tp=2
train_pp=1
train_cp=2

# ========= sampling =========
temperature=0.7
top_p=1.0
top_k=-1
val_top_p=0.95
loss_agg_mode="${LOSS_AGG_MODE:-seq-mean-token-mean}"
log_entropy="${LOG_ENTROPY:-False}"

# ========= cluster =========
NNODES="${NNODES:-2}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"

# Optional cap used by the minimal smoke configuration.  The formal run does
# not set it and therefore keeps the epoch-derived training length.
training_steps_override=()
if [ -n "${TOTAL_TRAINING_STEPS:-}" ]; then
    training_steps_override=("trainer.total_training_steps=${TOTAL_TRAINING_STEPS}")
fi

# vLLM's CUDA-graph path is kept in PIECEWISE mode for the formal vLLM
# rollout.  This is ignored by SGLang because it is nested under
# actor_rollout_ref.rollout.engine_kwargs.vllm.
vllm_engine_overrides=()
if [ "${rollout_name}" = "vllm" ]; then
    vllm_engine_overrides=(
        "+actor_rollout_ref.rollout.engine_kwargs.vllm.compilation_config.cudagraph_mode=PIECEWISE"
    )
fi

"${PYTHON_BIN}" -m vision_deepresearch_async_workflow.train_deepresearch_workflow_megatron \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    \
    data.dataset_name="${DATASET_NAME}" \
    data.val_dataset_name="${EVAL_DATASET_NAME}" \
    data.train_split="${TRAIN_SPLIT}" \
    data.val_split="${VAL_SPLIT}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size="${VAL_BATCH_SIZE:-64}" \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.return_raw_chat=${return_raw_chat} \
    data.seed=3407 \
    \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.dtype=${dtype} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${rollout_max_num_batched_tokens} \
    actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.68}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.ref.megatron.dtype=${dtype} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.use_mbridge=True \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    \
    actor_rollout_ref.actor.megatron.dtype=${dtype} \
    actor_rollout_ref.actor.megatron.param_offload=True \
    actor_rollout_ref.actor.megatron.optimizer_offload=True \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.checkpoint.async_save=False \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.log_entropy=${log_entropy} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    rllm.workflow.use_workflow=True \
    rllm.workflow.workflow_args.timeout=36000 \
    +rllm.workflow.workflow_args.max_llm_calls=${max_turns} \
    rllm.workflow.n_parallel_tasks=${n_parallel_tasks} \
    rllm.workflow.n_parallel_tools=${n_parallel_tools} \
    rllm.workflow.rollout_tail_drop_groups="${ROLLOUT_TAIL_DROP_GROUPS:-0}" \
    rllm.compact_filtering.enable=True \
    rllm.compact_filtering.mask_max_turns_exceeded=False \
    rllm.compact_filtering.mask_unknown=True \
    rllm.compact_filtering.mask_error=True \
    rllm.rejection_sample.enable=False \
    rllm.rejection_sample.multiplier=1.0 \
    rllm.stepwise_advantage.enable=False \
    +ray_init.include_dashboard=False \
    \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.description="\"${HYDRA_EXPERIMENT_DESCRIPTION}\"" \
    trainer.log_episodes=True \
    +trainer.episode_log_max_per_batch=50 \
    trainer.episode_log_dir="${EPISODE_LOG_DIR:-${PROJECT_ROOT}/RL/rollouts/${EXPERIMENT_NAME}}" \
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=5 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.test_freq="${TEST_FREQ:-10}" \
    trainer.resume_mode="${TRAINER_RESUME_MODE:-disable}" \
    trainer.total_epochs=2 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    "${vllm_engine_overrides[@]}" \
    "${training_steps_override[@]}"
