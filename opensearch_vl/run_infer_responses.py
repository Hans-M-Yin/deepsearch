#!/usr/bin/env python3
"""Benchmark inference using native Responses API function calls.

Reuse a synthesis/models.json alias (including Azure/internal gateways)::

    python opensearch_vl/run_infer_responses.py \
        --model-alias gpt56_terra_internal_azure \
        --data-path ../data/benchmarks/mmsearch/end2end/data.parquet \
        --output-dir opensearch_infer/outputs/GPT56/mmsearch_responses \
        --dataset test --parallel-workers 16

Or call an OpenAI-compatible Responses endpoint directly::

    OPENAI_API_KEY=... python opensearch_vl/run_infer_responses.py \
        --model YOUR_MODEL --base-url https://api.openai.com/v1 \
        --data-path DATA.parquet --output-dir OUTPUT --dataset test

Models use native function_call items, not XML tool tags. Final answers may
be plain text; the saved evaluation field is wrapped in <answer> locally.
Native output and assistant phase values are retained. Encrypted reasoning is
replayed intact in memory, but omitted from saved trajectories.
An invalid_encrypted_content error defaults to a logged, case-local fallback:
drop reasoning state and replay public history plus native tool pairs once.
Use --encrypted-reasoning-fallback fail to forbid lossy recovery, or
--no-encrypted-reasoning to avoid client-side reasoning replay from the outset.
"""

from __future__ import annotations

import sys

try:
    from .run_infer import main as _shared_main
except ImportError:
    from run_infer import main as _shared_main


def main(argv: list[str] | None = None) -> int:
    return _shared_main(argv, responses_api=True, entrypoint=__file__)


if __name__ == "__main__":
    sys.exit(main())
