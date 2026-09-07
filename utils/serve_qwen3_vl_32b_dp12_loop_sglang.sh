#!/usr/bin/env bash
# Serve Qwen3-VL-32B on a 6-node Ray cluster with twelve independent SGLang
# TP4 workers (data parallel size 12), fronted by the SGLang Model Gateway.
#
# Default topology for a 6-node x 8-GPU cluster:
#
#   node 0: replica 0 (TP4, GPU 0-3), replica 1 (TP4, GPU 4-7)
#   node 1: replica 2 (TP4, GPU 0-3), replica 3 (TP4, GPU 4-7)
#   node 2: replica 4 (TP4, GPU 0-3), replica 5 (TP4, GPU 4-7)
#   node 3: replica 6 (TP4, GPU 0-3), replica 7 (TP4, GPU 4-7)
#   node 4: replica 8 (TP4, GPU 0-3), replica 9 (TP4, GPU 4-7)
#   node 5: replica 10 (TP4, GPU 0-3), replica 11 (TP4, GPU 4-7)
#   router:  http://<router-host>:6658
#
# The workers are independently managed by Ray and the gateway balances
# requests across them. Each worker uses four GPUs on one node, so no TP group
# crosses a node boundary.
#
# Usage (run once from a Ray head node):
#   RAY_ADDRESS=auto \
#     bash utils/serve_qwen3_vl_32b_dp12_loop_sglang.sh
#
# Optional RAY_NODE_IPS can pin the six nodes explicitly, for example:
#   RAY_NODE_IPS=10.0.0.1,10.0.0.2,10.0.0.3,10.0.0.4,10.0.0.5,10.0.0.6 \
#     bash utils/serve_qwen3_vl_32b_dp12_loop_sglang.sh
#
# The launcher restarts the complete worker/router fleet if a worker or router
# exits. Existing environment variables take precedence over these defaults.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# Six nodes x two TP4 replicas per node = twelve replicas / 48 GPUs.
export TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-4}"
export DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-12}"
export RAY_NODE_COUNT="${RAY_NODE_COUNT:-6}"
export REPLICAS_PER_NODE="${REPLICAS_PER_NODE:-2}"

SGLANG_PYTHON="${SGLANG_PYTHON:-python}"
export SGLANG_PYTHON

exec "${SGLANG_PYTHON}" "${SCRIPT_DIR}/sglang_ray_dp_launcher.py"
