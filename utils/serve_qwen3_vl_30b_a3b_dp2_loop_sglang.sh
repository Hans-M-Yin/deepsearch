#!/usr/bin/env bash

# Serve Qwen3-VL-30B-A3B on a single-node Ray cluster with two independent
# SGLang TP4 workers (data parallel size 2), fronted by the SGLang gateway.
#
# Default topology for a single 8-GPU node:
#
#   node 0: replica 0 (TP4, GPU 0-3), replica 1 (TP4, GPU 4-7)
#   router:  http://<router-host>:6658
#
# Usage:
#   RAY_ADDRESS=auto \
#     bash utils/serve_qwen3_vl_30b_a3b_dp2_loop_sglang.sh
#
# Optional RAY_NODE_IPS can pin the single node explicitly, for example:
#   RAY_NODE_IPS=10.0.0.1 \
#     bash utils/serve_qwen3_vl_30b_a3b_dp2_loop_sglang.sh

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_PATH="${MODEL_PATH:-/mnt/hdfs/byte_ai_sales/user/user/yinzhihan/models/Qwen3-VL-30B-A3B}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-VL-32B}"

# One node x two TP4 replicas = two replicas / 8 GPUs.
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-2}"
export RAY_NODE_COUNT="${RAY_NODE_COUNT:-1}"
export REPLICAS_PER_NODE="${REPLICAS_PER_NODE:-2}"

SGLANG_PYTHON="${SGLANG_PYTHON:-python}"
export SGLANG_PYTHON

exec "${SGLANG_PYTHON}" "${SCRIPT_DIR}/sglang_ray_dp_launcher.py"
