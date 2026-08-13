#!/usr/bin/env bash

# Reproducible Ray environment and dashboard health check.
#
# By default this script does not stop an existing Ray cluster.  Pass
# --stop-ray when running an isolated diagnostic and a clean local Ray start
# is required.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RLLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3 || true)}"
RAY_TEST_TIMEOUT_S="${RAY_TEST_TIMEOUT_S:-90}"
STOP_RAY=0
CORE_ONLY=0
STATUS=0

usage() {
    cat <<'EOF'
Usage: check_ray_environment.sh [--stop-ray] [--core-only]

Options:
  --stop-ray    Stop an existing Ray cluster before the local startup test.
                Use this only when no other Ray job is running on the node.
  --core-only   Skip Dashboard module checks and test Ray with
                include_dashboard=False.

Environment variables:
  PYTHON_BIN          Python executable used for every check (default: python3)
  RAY_TEST_TIMEOUT_S  Timeout for the Ray startup test (default: 90)
EOF
}

for arg in "$@"; do
    case "$arg" in
        --stop-ray)
            STOP_RAY=1
            ;;
        --core-only)
            CORE_ONLY=1
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $arg" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN:-<empty>}" >&2
    exit 2
fi

echo "===== Ray environment check ====="
echo "project root: ${PROJECT_ROOT}"
echo "rllm root:    ${RLLM_ROOT}"
echo "python:       ${PYTHON_BIN}"
"${PYTHON_BIN}" --version

echo
echo "===== Python package and import information ====="
if ! "${PYTHON_BIN}" - <<'PY'
import importlib.metadata as metadata
import sys

print("sys.executable:", sys.executable)

for package in ("ray", "bytedray", "verl", "vllm"):
    try:
        distribution = metadata.distribution(package)
        print(f"{package} distribution: {distribution.version}")
        print(f"{package} location: {distribution.locate_file('')}")
    except Exception as exc:
        print(f"{package} distribution: NOT FOUND ({exc})")

try:
    import ray
    import ray._private.utils as ray_utils

    print("ray.__version__:", getattr(ray, "__version__", None))
    print("ray.__file__:", ray.__file__)
    print("ray._private.utils.__file__:", ray_utils.__file__)
    print(
        "has validate_socket_filepath:",
        hasattr(ray_utils, "validate_socket_filepath"),
    )
except Exception as exc:
    print("Ray import failed:", repr(exc))
    raise
PY
then
    echo "[FAIL] Ray package/import check"
    STATUS=1
else
    echo "[PASS] Ray package/import check"
fi

echo
echo "===== Ray Dashboard import check ====="
if [[ "${CORE_ONLY}" == "1" ]]; then
    echo "[SKIP] Dashboard check disabled by --core-only"
else
    # Import every dashboard module, matching Ray's own dashboard startup
    # path.  Importing only one helper module can miss mixed-version failures
    # in modules such as dashboard.modules.job.git.
    if ! "${PYTHON_BIN}" - <<'PY'
from ray.dashboard.utils import DashboardHeadModule, get_all_modules

modules = get_all_modules(DashboardHeadModule)
print("Dashboard modules imported:", len(modules))
for module in modules:
    print("  ", module.__module__, module.__name__)
PY
    then
        echo "[FAIL] Ray Dashboard import check"
        STATUS=1
    else
        echo "[PASS] Ray Dashboard import check"
    fi
fi

echo
echo "===== Installed package metadata ====="
"${PYTHON_BIN}" -m pip show ray bytedray verl vllm || true

echo
echo "===== pip consistency check ====="
if ! "${PYTHON_BIN}" -m pip check; then
    echo "[WARN] pip check reported dependency conflicts"
else
    echo "[PASS] pip check"
fi

if [[ "${STOP_RAY}" == "1" ]]; then
    echo
    echo "===== Stopping existing Ray cluster ====="
    if command -v ray >/dev/null 2>&1; then
        ray stop --force || true
    else
        echo "ray executable is not on PATH; skipping ray stop"
    fi
fi

echo
echo "===== Ray local startup test ====="
if [[ "${CORE_ONLY}" == "1" ]]; then
    export RAY_CHECK_INCLUDE_DASHBOARD=0
else
    export RAY_CHECK_INCLUDE_DASHBOARD=1
fi
if command -v timeout >/dev/null 2>&1; then
    timeout "${RAY_TEST_TIMEOUT_S}" "${PYTHON_BIN}" - <<'PY'
import ray
import os

include_dashboard = os.environ.get("RAY_CHECK_INCLUDE_DASHBOARD") == "1"
print(f"Starting Ray with include_dashboard={include_dashboard} ...")
ray.init(
    num_cpus=1,
    include_dashboard=include_dashboard,
    ignore_reinit_error=True,
)

print("Ray started successfully")
print("cluster resources:", ray.cluster_resources())
ray.shutdown()
print("Ray shutdown successfully")
PY
    RAY_TEST_STATUS=$?
else
    "${PYTHON_BIN}" - <<'PY'
import ray
import os

include_dashboard = os.environ.get("RAY_CHECK_INCLUDE_DASHBOARD") == "1"
print(f"Starting Ray with include_dashboard={include_dashboard} ...")
ray.init(
    num_cpus=1,
    include_dashboard=include_dashboard,
    ignore_reinit_error=True,
)

print("Ray started successfully")
print("cluster resources:", ray.cluster_resources())
ray.shutdown()
print("Ray shutdown successfully")
PY
    RAY_TEST_STATUS=$?
fi

if [[ "${RAY_TEST_STATUS}" -ne 0 ]]; then
    echo "[FAIL] Ray local startup test (exit code ${RAY_TEST_STATUS})"
    STATUS=1
else
    if [[ "${CORE_ONLY}" == "1" ]]; then
        echo "[PASS] Ray core startup test"
        echo "[WARN] Dashboard errors may still appear because Ray starts a minimal Dashboard process even with include_dashboard=False"
    else
        echo "[PASS] Ray local startup test"
    fi
fi

echo
if [[ "${STATUS}" -eq 0 ]]; then
    echo "RESULT: Ray environment checks passed"
else
    echo "RESULT: Ray environment checks failed; inspect the first [FAIL] section"
fi

exit "${STATUS}"
