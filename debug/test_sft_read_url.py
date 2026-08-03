#!/usr/bin/env python3
"""Run the current SFT ``read_url`` tool against one URL.

Example:
    python debug/test_sft_read_url.py \
        --url 'https://example.com/article' \
        --goal 'Find the publication date and the main claim.'
"""

from __future__ import annotations

import argparse
import json

from synthesis.sft.tools import read_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="URL passed directly to read_url.")
    parser.add_argument(
        "--goal",
        default="",
        help="Evidence-extraction objective. Supplying this invokes the Qwen summarizer branch.",
    )
    parser.add_argument(
        "--assistant-output",
        default="",
        help="Optional current agent reasoning supplied to the summarizer.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = read_url(
        url=args.url,
        goal=args.goal,
        assistant_output=args.assistant_output,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
