"""Verify W&B environment propagation into a Ray worker.

This starts no model and performs no training.  It launches one tiny Ray
remote function with the same runtime environment used by the RL trainer and
prints only safe W&B values (API-key values are never printed).
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RLLM_ROOT = PROJECT_ROOT / "RL" / "rllm"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(RLLM_ROOT) not in sys.path:
    sys.path.insert(0, str(RLLM_ROOT))

CHECK_NAMES = (
    "WANDB_MODE",
    "WANDB_DISABLED",
    "WANDB_BASE_URL",
    "WANDB_DIR",
    "WANDB_PROJECT",
    "WANDB_ENTITY",
    "WANDB_API_KEY",
    "RLLM_WANDB_DEBUG",
    "RLLM_WANDB_DEBUG_EVERY",
)


def _safe_environment() -> dict[str, object]:
    result: dict[str, object] = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python": sys.executable,
        "pythonpath": os.environ.get("PYTHONPATH", ""),
    }
    for name in CHECK_NAMES:
        value = os.environ.get(name)
        if name == "WANDB_API_KEY":
            result[name] = {"present": bool(value), "length": len(value) if value else 0}
        else:
            result[name] = value if value is not None else "<unset>"
    try:
        import wandb

        result["wandb_module"] = getattr(wandb, "__file__", "<unknown>")
        result["wandb_version"] = getattr(wandb, "__version__", "<unknown>")
    except Exception as exc:
        result["wandb_import"] = f"<error:{type(exc).__name__}:{exc}>"
    return result


def inspect_worker_environment() -> dict[str, object]:
    return _safe_environment()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--address",
        default=None,
        help="Ray address, e.g. auto. Omit to start a local Ray instance.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # Defer Ray import until after argument parsing so --help works even on a
    # host with a stale/broken Ray daemon or native library.
    import ray

    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

    runtime_env = get_ppo_ray_runtime_env()
    forwarded = runtime_env.get("env_vars", {})
    driver_values = _safe_environment()

    print("[RAY_WANDB_ENV] driver_safe_values:")
    for key, value in driver_values.items():
        if key != "pythonpath":
            print(f"  {key}={value}")

    print("[RAY_WANDB_ENV] runtime_env_forwarded:")
    for name in CHECK_NAMES:
        if name == "WANDB_API_KEY":
            value = forwarded.get(name)
            print(f"  {name}={{'present': {bool(value)}, 'length': {len(value) if value else 0}}}")
        else:
            print(f"  {name}={forwarded.get(name, '<not-forwarded>')}")

    ray_init_kwargs = {"runtime_env": runtime_env}
    if args.address:
        ray_init_kwargs["address"] = args.address
    ray.init(**ray_init_kwargs)
    try:
        remote_inspector = ray.remote(inspect_worker_environment)
        worker_values = ray.get(remote_inspector.options(runtime_env=runtime_env).remote())
        print("[RAY_WANDB_ENV] remote_worker_safe_values:")
        for key, value in worker_values.items():
            if key != "pythonpath":
                print(f"  {key}={value}")

        expected_names = [
            name
            for name in CHECK_NAMES
            if driver_values.get(name) not in (None, "<unset>")
        ]
        missing = [
            name for name in expected_names if worker_values.get(name) == "<unset>"
        ]
        mismatched = [
            name
            for name in expected_names
            if name != "WANDB_API_KEY"
            and worker_values.get(name) != driver_values.get(name)
        ]
        if missing or mismatched:
            if missing:
                print(f"[RAY_WANDB_ENV] FAIL missing_in_worker={','.join(missing)}")
            if mismatched:
                print(f"[RAY_WANDB_ENV] FAIL value_mismatch={','.join(mismatched)}")
            raise SystemExit(1)
        print("[RAY_WANDB_ENV] PASS driver W&B variables reached the Ray worker")
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
