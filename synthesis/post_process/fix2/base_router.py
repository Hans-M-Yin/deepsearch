"""Continuously judge newly written base trajectories for the async router."""

from __future__ import annotations

from pathlib import Path
import json
import time
from typing import Any


def watch_base_evaluations(
    *,
    trajectory_dir: str | Path,
    answer_file: str | Path,
    output_path: str | Path,
    judge_model_alias: str,
    judge_max_tokens: int = 1024,
    max_workers: int = 8,
    prompt_style: str = "local",
    poll_s: float = 10.0,
    once: bool = False,
    debug: bool = False,
) -> dict[str, Any]:
    """Run the existing resumable evaluator repeatedly as base files arrive.

    This is a router-side worker group. It can run while run_infer is still
    producing per-case trajectory files. New wrong-answer rows land in the
    evaluator details JSONL, where enqueue-failures --follow picks them up.
    """

    if poll_s <= 0:
        raise ValueError("poll_s must be positive")
    from opensearch_vl.eval_infer_with_llm import run_eval

    trajectory_root = Path(trajectory_dir)
    last: dict[str, Any] = {"status": "waiting_for_base_trajectories", "trajectory_count": 0}
    previous_trajectory_count: int | None = None
    last_evaluated_trajectory_count: int | None = None
    while True:
        trajectory_count = len(list(trajectory_root.glob("*_trajectory.json")))
        if debug and trajectory_count != previous_trajectory_count:
            print(
                f"[fix2-base-judge] discovered {trajectory_count} base trajectory file(s) in {trajectory_root}",
                flush=True,
            )
            previous_trajectory_count = trajectory_count
        # run_eval rewrites its canonical detail JSONL after a successful
        # pass. Re-running it every poll when no new trajectory arrived both
        # wastes judge work and exposes a transient partial file to the queue
        # router. A trajectory file is immutable once run_infer publishes it,
        # so count changes are the correct trigger.
        if trajectory_count and trajectory_count != last_evaluated_trajectory_count:
            last = run_eval(
                str(trajectory_root),
                answer_file=str(answer_file),
                output_path=str(output_path),
                max_workers=max_workers,
                judge_model_alias=judge_model_alias,
                judge_max_tokens=judge_max_tokens,
                prompt_style=prompt_style,
                rerun=False,
            )
            last["trajectory_count"] = trajectory_count
            last_evaluated_trajectory_count = trajectory_count
            if debug:
                print(f"[fix2-base-judge] {json.dumps(last, ensure_ascii=False, sort_keys=True)}", flush=True)
        if once:
            return last
        time.sleep(poll_s)
