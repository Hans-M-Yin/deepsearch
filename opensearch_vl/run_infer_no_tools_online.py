#!/usr/bin/env python3
"""Alias-backed, single-turn online baseline with the original no-tools prompt.

Reuses the online benchmark CLI and the exact no_tools_pipeline used by
run_infer_no_tools.py: images, sampling, parallelism, retries, timing, resume
and reproducibility config are shared rather than independently implemented.
No tool schemas are sent and no tools are executed.

Example::

    python opensearch_vl/run_infer_no_tools_online.py \\
        --model-alias claude_4_6_sonnet \\
        --data-path ../data/benchmarks/searchweave/annotation_benchmark_300.parquet \\
        --output-dir opensearch_infer/outputs/Claude-4.6-Sonnet/searchweave_no_tools \\
        --dataset test --limit 300 --parallel-workers 6 --max-tokens 32768

The alias supplies the model, credentials and OpenAI/Azure Chat Completions
route. Sampling parameters are sent only when explicitly provided on the CLI.
"""

from __future__ import annotations

import sys
from typing import Optional

from run_infer_react_online import main as _run_benchmark


def main(argv: Optional[list[str]] = None) -> int:
    return _run_benchmark(argv, no_tools=True, entrypoint=__file__)


if __name__ == "__main__":
    sys.exit(main())
