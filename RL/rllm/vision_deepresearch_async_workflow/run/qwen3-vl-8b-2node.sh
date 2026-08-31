#!/usr/bin/env bash
set -xeuo pipefail

# Qwen3-VL-8B DeepResearch RL configuration for 2 nodes x 8 GPUs.
# This is the 8B multi-node configuration adapted to a 16-GPU world:
# train_tp x train_pp x train_cp = 4 x 2 x 2 = 16.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${RLLM_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
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

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
echo "Using Python: ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import torch; print('torch:', torch.__version__, 'cuda:', torch.version.cuda)"
"${PYTHON_BIN}" -c "import transformer_engine.pytorch as te; print('transformer_engine: ok')"

# ========= user-overridable paths / data =========
MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-VL-8B-Instruct}
DATASET_NAME=${DATASET_NAME:-Vision-DeepResearch-QA}
TRAIN_SPLIT=${TRAIN_SPLIT:-train}
VAL_SPLIT=${VAL_SPLIT:-test}
PROJECT_NAME=${PROJECT_NAME:-vision-deepresearch}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-open_mm_searcher_8b_2node}
CKPTS_DIR=${CKPTS_DIR:-checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}

# ========= rollout / algorithm =========
rollout_mode="async"
rollout_name="sglang"
return_raw_chat="True"
dtype="bfloat16"
adv_estimator="rloo"

kl_coef=0.001
use_kl_loss=False
kl_loss_coef=0.001
clip_ratio_high=0.28

# ========= batch / workflow =========
train_prompt_bsz=256
n_resp_per_prompt=8
train_prompt_mini_bsz=64
n_parallel_tasks=256
n_parallel_tools=2048

# ========= length =========
max_prompt_length=4096
max_response_length=70000
use_dynamic_bsz=True
actor_ppo_max_token_len_per_gpu=74576
infer_ppo_max_token_len_per_gpu=74576

# ========= offload / parallelism =========
offload=True
gen_tp=4
train_tp=4
train_pp=2
train_cp=2

# ========= sampling =========
temperature=0.7
top_p=1.0
top_k=-1
val_top_p=0.95
loss_agg_mode="seq-mean-token-sum"

# ========= cluster =========
NNODES=2
N_GPUS_PER_NODE=8

"${PYTHON_BIN}" -m vision_deepresearch_async_workflow.train_deepresearch_workflow_megatron \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    \
    data.dataset_name="${DATASET_NAME}" \
    data.train_split="${TRAIN_SPLIT}" \
    data.val_split="${VAL_SPLIT}" \
    data.train_batch_size=${train_prompt_bsz} \
    data.val_batch_size=64 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enforce_eager=True \
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
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=1 \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=1 \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.use_mbridge=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len_per_gpu} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1 \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=${offload} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=False \
    \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    \
    rllm.workflow.use_workflow=True \
    rllm.workflow.n_parallel_tasks=${n_parallel_tasks} \
    rllm.workflow.n_parallel_tools=${n_parallel_tools} \
    rllm.compact_filtering.enable=True \
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
    trainer.val_before_train=False \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.total_epochs=100 \
    trainer.default_local_dir="${CKPTS_DIR}"
