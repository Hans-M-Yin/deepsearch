#!/usr/bin/env python3
"""Run an OpenSearch-VL evaluation set through an API-served agent.

This wrapper mirrors the Vision-DeepResearch eval driver shape: accept one
parquet or a directory of parquets, run examples concurrently, and write one
trajectory directory per benchmark file. It intentionally preserves benchmark
schemas. The per-row adapter lives in ``pipeline.process_single_case`` and
accepts existing OpenSearch fields (``data_id`` / ``prompt``) as well as
reference eval fields (``id`` / ``question`` / ``answer`` / ``images``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

from run_infer import main as run_infer_main


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run OpenSearch-VL inference over one parquet or a directory."
    )
    parser.add_argument(
        "--data-path",
        "--parquet",
        dest="data_path",
        required=True,
        help="A parquet file or a directory containing *.parquet files.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Output root. Each parquet writes to <output-dir>/<parquet-stem>/.",
    )
    parser.add_argument("--model", default=os.getenv("MODEL", "30b-a3b"))
    parser.add_argument("--backend", choices=["api", "local"], default="api")
    parser.add_argument(
        "--base-url",
        default=(
            os.getenv("AGENT_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        ),
        help="OpenAI-compatible endpoint, e.g. http://localhost:8001/v1.",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("AGENT_API_KEY", "EMPTY"),
        help="API key for the OpenAI-compatible endpoint.",
    )
    parser.add_argument(
        "--served-model-name",
        default=os.getenv("AGENT_MODEL_NAME"),
        help="Model name sent to the API server.",
    )
    parser.add_argument("--checkpoint", default=os.getenv("CHECKPOINT"))
    parser.add_argument("--gpus", default=os.getenv("GPUS", "0"))
    parser.add_argument("--dtype", default=os.getenv("DTYPE", "bfloat16"))
    parser.add_argument("--dataset", default=os.getenv("DATASET", "test"))
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument(
        "--parallel-tasks",
        "--parallel-workers",
        dest="parallel_workers",
        type=int,
        default=int(os.getenv("PARALLEL_TASKS", "10")),
        help="Concurrent examples. Each example's trajectory remains serial.",
    )
    parser.add_argument("--api-timeout", type=int, default=600)
    parser.add_argument("--api-max-retries", type=int, default=3)
    parser.add_argument("--log-level", default=os.getenv("LOG_LEVEL", "INFO"))
    return parser


def _resolve_parquets(path: Path) -> List[Path]:
    if path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No .parquet files found under {path}")
        return files
    if path.is_file():
        return [path]
    raise FileNotFoundError(f"Data path does not exist: {path}")


def _append_optional(argv: List[str], flag: str, value: Optional[object]) -> None:
    if value is not None and value != "":
        argv.extend([flag, str(value)])


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    parquets = _resolve_parquets(Path(args.data_path).expanduser())
    output_root = Path(args.output_dir).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    limit = args.limit if args.limit is not None else args.max_samples

    failures = 0
    for parquet in parquets:
        sub_output = output_root / parquet.stem
        infer_argv: List[str] = [
            "--model",
            args.model,
            "--backend",
            args.backend,
            "--gpus",
            args.gpus,
            "--dtype",
            args.dtype,
            "--data-path",
            str(parquet),
            "--output-dir",
            str(sub_output),
            "--dataset",
            args.dataset,
            "--start",
            str(args.start),
            "--temperature",
            str(args.temperature),
            "--max-tokens",
            str(args.max_tokens),
            "--parallel-workers",
            str(args.parallel_workers),
            "--api-timeout",
            str(args.api_timeout),
            "--api-max-retries",
            str(args.api_max_retries),
            "--log-level",
            args.log_level,
        ]
        _append_optional(infer_argv, "--limit", limit)
        _append_optional(infer_argv, "--category", args.category)
        _append_optional(infer_argv, "--checkpoint", args.checkpoint)
        _append_optional(infer_argv, "--base-url", args.base_url)
        _append_optional(infer_argv, "--api-key", args.api_key)
        _append_optional(infer_argv, "--served-model-name", args.served_model_name)

        print("=" * 80)
        print(f"[EVAL-SET] parquet = {parquet}")
        print(f"           output  = {sub_output}")
        print(f"           workers = {args.parallel_workers}")
        print("=" * 80)
        rc = run_infer_main(infer_argv)
        failures += int(rc != 0)

    if failures:
        print(f"Evaluation set finished with {failures} failed parquet file(s).")
        return 1
    print(f"Evaluation set complete. Results under: {output_root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
