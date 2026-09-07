#!/usr/bin/env bash
# Serve Qwen3-VL-32B on a 3-node Ray cluster with six independent SGLang
# TP4 workers (data parallel size 6), fronted by the SGLang Model Gateway.
#
# Default topology for a 3-node x 8-GPU cluster:
#
#   node 0: replica 0 (TP4, GPU 0-3), replica 1 (TP4, GPU 4-7)
#   node 1: replica 2 (TP4, GPU 0-3), replica 3 (TP4, GPU 4-7)
#   node 2: replica 4 (TP4, GPU 0-3), replica 5 (TP4, GPU 4-7)
#   router:  http://<router-host>:6658
#
# The workers are independently managed by Ray and the gateway balances
# requests across them. Each worker uses four GPUs on one node, so no TP group
# crosses a node boundary.
#
# Usage (run once from a Ray head node):
#   RAY_ADDRESS=auto \
#     bash utils/serve_qwen3_vl_32b_dp6_loop_sglang.sh
#
# Optional RAY_NODE_IPS can pin the three nodes explicitly, for example:
#   RAY_NODE_IPS=10.0.0.1,10.0.0.2,10.0.0.3 \
#     bash utils/serve_qwen3_vl_32b_dp6_loop_sglang.sh
#
# The launcher restarts the complete worker/router fleet if a worker or router
# exits. Existing environment variables take precedence over these defaults.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Three nodes x two TP4 replicas per node = six replicas / 24 GPUs.
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-6}"
export RAY_NODE_COUNT="${RAY_NODE_COUNT:-3}"
export REPLICAS_PER_NODE="${REPLICAS_PER_NODE:-2}"

SGLANG_PYTHON="${SGLANG_PYTHON:-python}"
export SGLANG_PYTHON

exec "${SGLANG_PYTHON}" "${SCRIPT_DIR}/sglang_ray_dp_launcher.py"
