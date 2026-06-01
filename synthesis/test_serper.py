"""Minimal Serper image-search smoke test.

Usage:
    python synthesis/test_serper.py --query "Kobe Bryant final game 2016"

It loads `synthesis/.env` by default, then calls the official Serper images
endpoint and prints a compact result list.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = Path(__file__).with_name(".env")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Image search query.")
    parser.add_argument("--limit", type=int, default=5, help="Maximum image results to print.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing env vars.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from synthesis.run_min_graph import load_env_file
    from synthesis.search_client import SerperSearchClient

    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    load_env_file(env_path, override=args.override_env)

    client = SerperSearchClient()
    response = client.search_image(args.query, limit=args.limit)

    print(f"engine={response.engine}")
    print(f"status={response.status_code}")
    print(f"query={response.query!r}")
    print(f"result_count={len(response.results)}")
    print()

    for index, item in enumerate(response.results[: args.limit], start=1):
        print(f"[{index}] title={item.title or ''}")
        print(f"    image_url={item.image_url or ''}")
        print(f"    source_page_url={item.source_page_url or ''}")
        print(f"    thumbnail_url={item.thumbnail_url or ''}")
        print(f"    source={item.source or ''}")
        print(f"    size={item.width or '?'}x{item.height or '?'}")
        if item.snippet:
            print(f"    snippet={item.snippet}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
