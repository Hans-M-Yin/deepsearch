#!/usr/bin/env python3
"""Unified entrypoint for OpenSearch-VL inference.

Usage examples
--------------

Run the dense 8B model on the FVQA training split using two GPUs::

    python run_infer.py --model 8b --gpus 0,1 \
        --data-path /data/fvqa_train.parquet \
        --output-dir ./outputs/fvqa_train_8b

Use Claude Opus 4.5 (no local GPUs needed; requires CLAUDE_API_* env vars)::

    python run_infer.py --model claude \
        --data-path /data/fvqa_test.parquet \
        --output-dir ./outputs/fvqa_test_claude

Switch to the MoE 30B-A3B variant::

    python run_infer.py --model 30b-a3b --gpus 0,1,2,3 \
        --checkpoint /models/Qwen3-VL-30B-A3B-Instruct \
        --data-path /data/fvqa_train.parquet \
        --output-dir ./outputs/fvqa_train_30b_a3b

The model registry is defined in ``opensearch_infer/config.py``. All
default values pull from environment variables so the codebase is safe
to open-source.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import sys
import traceback
from typing import Any, Optional

from tqdm.auto import tqdm

from opensearch_infer import config


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "OpenSearch-VL Visual Investigation Agent. Supported tools: "
            "web_search, text_search, image_search, crop, layout_parsing, "
            "perspective_correct, super_resolution, sharpen."
        )
    )
    parser.add_argument(
        "--model",
        choices=sorted(config.MODEL_REGISTRY.keys()),
        default="8b",
        help="Model variant to run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Override the checkpoint path or HuggingFace id. Defaults to "
            "the per-model env variable, then the registry default."
        ),
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help=(
            "Comma-separated CUDA device ids for Qwen3-VL. Single id uses "
            "single-GPU placement; multiple ids enable model parallelism "
            'via device_map="auto".'
        ),
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Floating point dtype for Qwen3-VL weights.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="local",
        choices=["local", "api"],
        help=(
            "Inference backend. 'local' loads HF weights in-process; 'api' calls "
            "an OpenAI-compatible endpoint such as vLLM serve."
        ),
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="OpenAI-compatible base URL for --backend api, e.g. http://localhost:8001/v1.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for --backend api. Defaults to AGENT_API_KEY or EMPTY.",
    )
    parser.add_argument(
        "--served-model-name",
        type=str,
        default=None,
        help=(
            "Model name sent to the API server. Defaults to --checkpoint, then "
            "the registry display name."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Parquet file with FVQA-style rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory where per-case trajectories and artifacts are written.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Tag stored in trajectories; controls intermediate filename prefix.",
    )
    parser.add_argument(
        "--start", type=int, default=0, help="First row index (inclusive)."
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="Last row index (exclusive). Defaults to len(df).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N rows starting from --start.",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Restrict to rows whose 'category' column equals this value.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16384,
        help="Per-turn generation cap.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help=(
            "Number of benchmark examples to run concurrently. Each trajectory "
            "remains turn-by-turn serial."
        ),
    )
    parser.add_argument(
        "--api-timeout",
        type=int,
        default=600,
        help="HTTP timeout in seconds for --backend api.",
    )
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=3,
        help="HTTP retry count for --backend api.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (Qwen3-VL VL default: 0.7; 0 disables sampling).",
    )
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--presence-penalty", type=float, default=1.5)
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Root logger level.",
    )
    return parser


def _configure_logging(level: str) -> None:
    logging.basicConfig(
        format="[%(asctime)s][%(levelname)5s][%(name)s] %(message)s",
        level=getattr(logging, level),
    )


def _row_to_case_id(row: Any, idx: int) -> str:
    if hasattr(row, "to_dict"):
        try:
            row_dict = row.to_dict()
        except Exception:
            row_dict = {}
    elif isinstance(row, dict):
        row_dict = row
    else:
        row_dict = {}

    if isinstance(row_dict, dict):
        for key in ("data_id", "id", "idx", "_id"):
            value = row_dict.get(key)
            if value is not None:
                return str(value)
    return f"case_{idx}"


def _completed_case_ids(output_dir: str) -> set[str]:
    completed: set[str] = set()
    if not os.path.isdir(output_dir):
        return completed
    suffix = "_trajectory.json"
    for name in os.listdir(output_dir):
        if name.endswith(suffix):
            completed.add(name[: -len(suffix)])
    return completed


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_infer")

    from opensearch_infer import pipeline
    from opensearch_infer.runners import InferenceConfig, build_runner

    runner = build_runner(
        model_name=args.model,
        checkpoint=args.checkpoint,
        gpus=args.gpus,
        dtype=args.dtype,
        backend=args.backend,
        base_url=args.base_url,
        api_key=args.api_key,
        served_model_name=args.served_model_name,
        timeout=args.api_timeout,
        max_retries=args.api_max_retries,
    )
    logger.info("Model: %s (backend=%s)", runner.display_name, args.backend)

    try:
        runner.load()
    except Exception as exc:
        logger.error("Failed to initialize runner: %s", exc, exc_info=True)
        return 1

    if not os.path.exists(args.data_path):
        logger.error("Data file not found: %s", args.data_path)
        return 1

    import pandas as pd

    df = pd.read_parquet(args.data_path)
    logger.info("Loaded %d rows from %s", len(df), args.data_path)

    if args.category:
        df = df[df["category"] == args.category]
        logger.info("Filtered to category=%s: %d rows", args.category, len(df))

    start = max(0, int(args.start))
    if args.end is not None:
        end = min(int(args.end), len(df))
    elif args.limit is not None:
        end = min(start + int(args.limit), len(df))
    else:
        end = len(df)

    os.makedirs(args.output_dir, exist_ok=True)
    logger.info("Processing rows [%d, %d) -> %s", start, end, args.output_dir)

    inference_cfg = InferenceConfig(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
    )

    workers = max(1, int(args.parallel_workers))
    if args.backend == "local" and workers > 1:
        logger.warning(
            "--parallel-workers=%d with --backend local shares one in-process model; "
            "API/vLLM backend is recommended for parallel evaluation.",
            workers,
        )

    def _run_one(idx: int) -> tuple[int, bool, Optional[str]]:
        row = df.iloc[idx]
        case_id = _row_to_case_id(row, idx)
        timing = pipeline.InferenceTiming() if logger.isEnabledFor(logging.DEBUG) else None
        status = "failed"
        try:
            pipeline.process_single_case(
                row=row,
                runner=runner,
                output_dir=args.output_dir,
                case_idx=idx,
                dataset_type=args.dataset,
                inference_cfg=inference_cfg,
                timing=timing,
            )
            status = "completed"
            return idx, True, None
        except Exception as exc:
            logger.error("Case %d failed: %s", idx, exc)
            traceback.print_exc()
            return idx, False, str(exc)
        finally:
            if timing is not None:
                timing.emit(case_id=case_id, case_idx=idx, status=status)

    success, failure, skipped = 0, 0, 0
    completed_case_ids = _completed_case_ids(args.output_dir)
    all_indices = list(range(start, end))
    indices = []
    for idx in all_indices:
        case_id = _row_to_case_id(df.iloc[idx], idx)
        if case_id in completed_case_ids:
            skipped += 1
            continue
        indices.append(idx)

    if skipped:
        logger.info(
            "Resume detected: skipped %d completed case(s) already present in %s",
            skipped,
            args.output_dir,
        )

    if workers == 1:
        for idx in tqdm(indices, desc="Inference", unit="case"):
            _, ok, _ = _run_one(idx)
            success += int(ok)
            failure += int(not ok)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, idx): idx for idx in indices}
            with tqdm(total=len(futures), desc="Inference", unit="case") as progress:
                for future in concurrent.futures.as_completed(futures):
                    idx, ok, error = future.result()
                    success += int(ok)
                    failure += int(not ok)
                    progress.update(1)
                    if ok:
                        logger.info(
                            "Case %d completed (%d/%d)",
                            idx,
                            success + failure,
                            len(indices),
                        )
                    else:
                        logger.error(
                            "Case %d failed (%d/%d): %s",
                            idx,
                            success + failure,
                            len(indices),
                            error,
                        )

    logger.info(
        "Done. success=%d failure=%d skipped=%d output=%s",
        success,
        failure,
        skipped,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
