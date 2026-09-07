"""Training-free W&B probe that follows the SFT reporting path.

This mirrors the relevant SFT behavior: ``wandb.init`` is created by the
Trainer integration, ``ReporterCallback`` updates ``wandb.config``, and
training logs are sent with ``wandb.log``.  No model or training process is
started here.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

from wandb_probe_common import add_common_args, configure_environment, run_probe_loop


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    add_common_args(parser, default_project="mmdeepsearch")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_environment(args)
    run_name = args.run_name or f"sft-wandb-probe-{time.strftime('%Y%m%d-%H%M%S')}"

    # Import after argument/environment setup so ``--help`` remains usable
    # even on a node whose W&B installation has an unrelated import problem.
    import wandb

    # The project/name arguments and environment handling follow the SFT
    # ReporterCallback + Transformers WandbCallback path.
    run = wandb.init(
        project=args.project,
        name=run_name,
        config={
            "probe": {"kind": "sft_reporting", "training_started": False},
        },
    )
    if run is None:
        raise RuntimeError("wandb.init returned None; check WANDB_MODE and W&B setup")

    print(f"[WANDB_PROBE] sft_reporting_started url={getattr(run, 'url', None)}", flush=True)

    # This is the shape used by SFT's ReporterCallback.on_train_begin.
    run.config.update(
        {
            "model_args": {
                "model_name_or_path": "wandb-probe-model",
                "image_max_pixels": 1048576,
            },
            "data_args": {
                "template": "qwen3_vl",
                "cutoff_len": 32768,
            },
            "finetuning_args": {"stage": "sft", "finetuning_type": "full"},
            "generating_args": {"max_new_tokens": 1024},
        },
        allow_val_change=True,
    )

    def log_fn(data: dict[str, Any], step: int) -> None:
        # This is the direct logging primitive used underneath the SFT
        # Transformers integration.
        wandb.log(data, step=step)

    def finish_fn() -> None:
        wandb.finish(exit_code=0)

    run_probe_loop(
        run=run,
        log_fn=log_fn,
        duration_seconds=args.duration_seconds,
        interval_seconds=args.interval_seconds,
        post_finish_seconds=args.post_finish_seconds,
        finish_fn=finish_fn,
    )


if __name__ == "__main__":
    main()
