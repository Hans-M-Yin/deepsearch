#!/usr/bin/env python3
"""Unified entrypoint for OpenSearch-VL no-tools inference.

This mirrors ``run_infer.py`` but uses a no-tools single-turn baseline.
"""

from __future__ import annotations

import concurrent.futures
import logging
import os
import sys
import traceback
from typing import Optional

from run_infer import _build_arg_parser, _configure_logging


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.log_level)
    logger = logging.getLogger("run_infer_no_tools")

    from opensearch_infer import no_tools_pipeline
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
    logger.info("Model: %s (backend=%s, no-tools)", runner.display_name, args.backend)

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
    )

    workers = max(1, int(args.parallel_workers))
    if args.backend == "local" and workers > 1:
        logger.warning(
            "--parallel-workers=%d with --backend local shares one in-process model; "
            "API/vLLM backend is recommended for parallel evaluation.",
            workers,
        )

    def _run_one(idx: int) -> tuple[int, bool, Optional[str]]:
        try:
            row = df.iloc[idx]
            no_tools_pipeline.process_single_case(
                row=row,
                runner=runner,
                output_dir=args.output_dir,
                case_idx=idx,
                dataset_type=args.dataset,
                inference_cfg=inference_cfg,
            )
            return idx, True, None
        except Exception as exc:
            logger.error("Case %d failed: %s", idx, exc)
            traceback.print_exc()
            return idx, False, str(exc)

    success, failure = 0, 0
    indices = list(range(start, end))
    if workers == 1:
        for idx in indices:
            _, ok, _ = _run_one(idx)
            success += int(ok)
            failure += int(not ok)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(_run_one, idx): idx for idx in indices}
            for future in concurrent.futures.as_completed(futures):
                idx, ok, error = future.result()
                success += int(ok)
                failure += int(not ok)
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
        "Done. success=%d failure=%d output=%s",
        success,
        failure,
        args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
