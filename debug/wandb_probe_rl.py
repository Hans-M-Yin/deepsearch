"""Training-free probes for the RL/verl W&B path.

The default mode follows ``verl.utils.tracking.Tracking`` in the current
process.  ``--ray-worker`` additionally runs the same probe inside a Ray
worker, which is the execution context used by the real RL trainer.

No model, Ray GPU worker, rollout, or training update is started.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VERL_ROOT = PROJECT_ROOT / "RL" / "rllm" / "verl"
RLLM_ROOT = PROJECT_ROOT / "RL" / "rllm"
for path in (PROJECT_ROOT, RLLM_ROOT, VERL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from wandb_probe_common import (  # noqa: E402
    add_common_args,
    configure_environment,
    print_snapshot,
    run_probe_loop,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_project="vision-deepresearch")
    parser.add_argument(
        "--ray-worker",
        action="store_true",
        help="Run the probe inside a Ray worker, matching the formal RL context.",
    )
    parser.add_argument(
        "--address",
        default=None,
        help="Ray address, e.g. auto. Omit to start a local Ray instance.",
    )
    parser.add_argument(
        "--formal-lifecycle",
        action="store_true",
        help=(
            "In Ray mode, imitate formal RL: wait without calling wandb.log "
            "until one logical training step ends, and do not explicitly call "
            "wandb.finish()."
        ),
    )
    return parser


def _build_rl_config(project: str, run_name: str, include_console: bool) -> dict[str, Any]:
    return {
        "trainer": {
            "project_name": project,
            "experiment_name": run_name,
            "logger": ["console", "wandb"] if include_console else ["wandb"],
            "total_epochs": 1,
        },
        "data": {"train_batch_size": 1, "max_prompt_length": 128},
        "algorithm": {"adv_estimator": "rloo"},
        "actor_rollout_ref": {"rollout": {"name": "sglang", "n": 1}},
        # This resembles the long profiler key that previously caused the
        # internal tracking API to reject the whole run config.  The local RL
        # Tracking sanitizer should remove only the overlong flattened leaf.
        "global_profiler": {
            "global_tool_config": {
                "nsys": {
                    "worker_nsight_options": {"cuda-graph-trace": "graph"}
                }
            }
        },
        "probe": {
            "kind": "rl_tracking",
            "training_started": False,
            "execution_context": "ray_worker" if include_console else "local",
        },
    }


def _run_tracking_probe(
    project: str,
    run_name: str,
    duration_seconds: float,
    interval_seconds: float,
    post_finish_seconds: float,
    formal_lifecycle: bool = False,
) -> None:
    # Import the exact Tracking implementation used by
    # rllm/trainer/verl/agent_workflow_trainer.py.
    from verl.utils.tracking import Tracking

    include_console = formal_lifecycle
    config = _build_rl_config(project, run_name, include_console=include_console)
    tracker = Tracking(
        project_name=project,
        experiment_name=run_name,
        default_backend=["console", "wandb"] if include_console else ["wandb"],
        config=config,
    )
    wandb_module = tracker.logger["wandb"]
    run = wandb_module.run
    print(
        "[WANDB_PROBE] rl_tracking_started "
        f"context={'ray_worker' if include_console else 'local'} "
        f"pid={os.getpid()} host={socket.gethostname()} "
        f"wandb_module={getattr(wandb_module, '__file__', None)} "
        f"wandb_version={getattr(wandb_module, '__version__', None)} "
        f"url={getattr(run, 'url', None)}",
        flush=True,
    )

    if formal_lifecycle:
        # Formal RL does not call Tracking.log during generate_trajectories.
        # It logs after the complete logical step (rollout, reward, old log
        # probability, advantage and actor update) has finished.  Keep the
        # system monitor alive during this wait but do not create history data.
        print_snapshot(run, event="init")
        started = time.monotonic()
        while time.monotonic() - started < max(0.0, duration_seconds):
            sleep_seconds = min(
                max(0.1, interval_seconds),
                max(0.0, duration_seconds - (time.monotonic() - started)),
            )
            time.sleep(sleep_seconds)
            print_snapshot(run, event="formal_rollout_wait")

        tracker.log(
            data={
                "probe/formal_step": 1,
                "training/global_step": 1,
                "probe/python_pid": os.getpid(),
            },
            step=1,
        )
        print_snapshot(run, event="formal_step_log", step=1)
        print_snapshot(run, event="pre_finish", step=1)
        # This intentionally mirrors Tracking's current formal lifecycle:
        # finish is only reached through Tracking.__del__ at teardown.
        print(
            "[WANDB_PROBE] formal_lifecycle_no_explicit_finish=true",
            flush=True,
        )
        if post_finish_seconds > 0:
            time.sleep(post_finish_seconds)
        print_snapshot(run, event="post_finish", step=1)
        return

    def log_fn(data: dict[str, Any], step: int) -> None:
        # This is the same interface used by the RL trainer:
        # Tracking.log(data=..., step=...).
        tracker.log(data=data, step=step)

    def finish_fn() -> None:
        # The original local probe finishes explicitly so it can inspect the
        # post-finish state.  The formal-lifecycle mode above intentionally
        # skips this call.
        wandb_module.finish(exit_code=0)
        tracker.logger.pop("wandb", None)

    try:
        run_probe_loop(
            run=run,
            log_fn=log_fn,
            duration_seconds=duration_seconds,
            interval_seconds=interval_seconds,
            post_finish_seconds=post_finish_seconds,
            finish_fn=finish_fn,
        )
    finally:
        if "wandb" in tracker.logger:
            wandb_module.finish(exit_code=1)
            tracker.logger.pop("wandb", None)


class _RayWandbProbeActor:
    """Actor-shaped wrapper matching the formal ``TaskRunner`` context."""

    def run(self, arguments: dict[str, Any]) -> dict[str, str]:
        _run_tracking_probe(
            project=arguments["project"],
            run_name=arguments["run_name"],
            duration_seconds=arguments["duration_seconds"],
            interval_seconds=arguments["interval_seconds"],
            post_finish_seconds=arguments["post_finish_seconds"],
            formal_lifecycle=arguments["formal_lifecycle"],
        )
        return {"hostname": socket.gethostname(), "pid": str(os.getpid())}


def main() -> None:
    args = build_parser().parse_args()
    configure_environment(args)
    run_name = args.run_name or f"rl-wandb-probe-{time.strftime('%Y%m%d-%H%M%S')}"

    if not args.ray_worker:
        _run_tracking_probe(
            project=args.project,
            run_name=run_name,
            duration_seconds=args.duration_seconds,
            interval_seconds=args.interval_seconds,
            post_finish_seconds=args.post_finish_seconds,
        )
        return

    import ray

    from rllm.trainer.verl.ray_runtime_env import get_ppo_ray_runtime_env

    runtime_env = get_ppo_ray_runtime_env()
    ray_init_kwargs: dict[str, Any] = {"runtime_env": runtime_env}
    if args.address:
        ray_init_kwargs["address"] = args.address
    ray.init(**ray_init_kwargs)
    try:
        arguments = {
            "project": args.project,
            "run_name": run_name,
            "duration_seconds": args.duration_seconds,
            "interval_seconds": args.interval_seconds,
            "post_finish_seconds": args.post_finish_seconds,
            "formal_lifecycle": args.formal_lifecycle,
        }
        remote_actor = ray.remote(num_cpus=1)(_RayWandbProbeActor)
        actor = remote_actor.options(runtime_env=runtime_env).remote()
        result = ray.get(actor.run.remote(arguments))
        print(f"[WANDB_PROBE] ray_worker_completed {result}", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
