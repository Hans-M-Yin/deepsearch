#!/usr/bin/env bash
# Serve Qwen3-VL-32B on a Ray cluster as independent SGLang workers.
#
# Default topology for a 2-node x 8-GPU cluster:
#
#   node 0: replica 0 (TP4, GPU 0-3), replica 1 (TP4, GPU 4-7)
#   node 1: replica 2 (TP4, GPU 0-3), replica 3 (TP4, GPU 4-7)
#   router:  http://<router-host>:6658
#
# This is intentionally separate-worker mode.  Each worker is TP-only and the
# SGLang Model Gateway balances requests across the four workers.  Ray places
# exactly two actors on each selected node; it is not used as the HTTP router.
#
# Usage (run once from a Ray head node):
#   RAY_ADDRESS=auto \
#   bash utils/serve_qwen3_vl_32b_dp4_loop_sglang.sh
#
# The launcher restarts the complete worker/router fleet if a worker or router
# exits.  Useful overrides:
#   TENSOR_PARALLEL_SIZE=2 DATA_PARALLEL_SIZE=8 \
#     bash utils/serve_qwen3_vl_32b_dp4_loop_sglang.sh
# TP2 mode defaults to four replicas per node and is generally the first mode
# to benchmark for aggregate throughput.  TP4 mode is the safer default for
# per-request latency and lower per-GPU weight/KV pressure.
#
# Optional RAY_NODE_IPS is a comma-separated list of the exact nodes to use,
# for example: RAY_NODE_IPS=10.0.0.1,10.0.0.2
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SGLANG_PYTHON="${SGLANG_PYTHON:-python}"
export SGLANG_PYTHON

exec "${SGLANG_PYTHON}" "${SCRIPT_DIR}/sglang_ray_dp_launcher.py"
