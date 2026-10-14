#!/usr/bin/env python3
"""Single-turn baseline with the same sampling, resume and logging as run_infer.

Use the same CLI flags as ``run_infer.py``. Only the inference pipeline and
system prompt change: the model answers once without executing tools.
"""

from __future__ import annotations

import sys
from typing import Optional

from run_infer import main as _run_benchmark


def main(argv: Optional[list[str]] = None) -> int:
    return _run_benchmark(argv, no_tools=True, entrypoint=__file__)


if __name__ == "__main__":
    sys.exit(main())
