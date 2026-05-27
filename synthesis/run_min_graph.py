"""Run a small end-to-end synthesis graph expansion.

This is the early-development entrypoint for checking whether the graph
construction stack can produce a tiny mixed text/image graph from Wikipedia.
Run it from the repository root:

    python synthesis/run_min_graph.py \
      --seed-url https://en.wikipedia.org/wiki/Kobe_Bryant \
      --store-dir synthesis/runs/kobe_min_graph
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import sys
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = Path(__file__).with_name(".env")
DEFAULT_STORE_DIR = Path(__file__).with_name("runs") / "min_graph"
DEFAULT_SEED_URL = "https://en.wikipedia.org/wiki/Kobe_Bryant"


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    """Load simple `export KEY=value` shell env files without extra deps."""

    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_no} is not KEY=value syntax: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{line_no} has empty env key.")
        parsed = shlex.split(value, comments=False, posix=True)
        env_value = parsed[0] if parsed else ""
        if override or key not in os.environ:
            os.environ[key] = env_value
        loaded[key] = env_value
    return loaded


def check_python_version() -> None:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            "The synthesis pipeline requires Python 3.10+ because the data "
            "objects use dataclass(slots=True). Current interpreter: "
            f"{sys.version.split()[0]}"
        )


def check_reader_service(base_url: str, *, timeout_s: float = 5.0) -> tuple[bool, str]:
    """Check that the Enhanced Reader HTTP service is reachable."""

    request = Request(base_url.rstrip("/") + "/", headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return True, f"reachable, status={response.getcode()}"
    except HTTPError as exc:
        if exc.code < 500:
            return True, f"reachable, status={exc.code}"
        return False, f"server error, status={exc.code}"
    except URLError as exc:
        return False, f"not reachable: {exc.reason}"
    except TimeoutError:
        return False, f"not reachable: timed out after {timeout_s}s"


def has_serpapi_credentials() -> bool:
    return bool(
        os.environ.get("SERPAPI_AK")
        or os.environ.get("AIDP_SERP_AK")
        or os.environ.get("SERPAPI_API_KEY")
        or os.environ.get("SERP_API_KEY")
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing env vars.")
    parser.add_argument("--seed-url", default=DEFAULT_SEED_URL, help="Seed Wikipedia URL.")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Output JSONL graph store directory.")
    parser.add_argument("--reader-base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--skip-reader-check", action="store_true", help="Skip preflight reader reachability check.")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum text pages to expand.")
    parser.add_argument("--max-nodes", type=int, default=10, help="Stop after this many graph nodes.")
    parser.add_argument("--max-depth", type=int, default=1, help="Maximum text-neighbor BFS depth.")
    parser.add_argument("--max-neighbors", type=int, default=2, help="Text neighbors queued per text node.")
    parser.add_argument("--max-links", type=int, default=20, help="Wiki links extracted per page before queue slicing.")
    parser.add_argument("--per-query-image-limit", type=int, default=3, help="Image search results per visual query.")
    parser.add_argument("--max-images-per-plan", type=int, default=1, help="Accepted images per visual plan.")
    parser.add_argument("--no-images", action="store_true", help="Disable visual planning and image discovery.")
    parser.add_argument("--skip-attributes", action="store_true", help="Do not call LLM attribute extraction.")
    parser.add_argument("--fatal-attribute-errors", action="store_true", help="Fail the task if attribute extraction fails.")
    parser.add_argument("--no-serp-fallback", action="store_true", help="Do not use SerpApi fallback after Commons.")
    parser.add_argument("--run-id", default=None, help="Optional stable run id.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing runner checkpoint state.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    check_python_version()

    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    loaded_env = load_env_file(env_path, override=args.override_env)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from synthesis.graph_expansion import GraphExpansionConfig, GraphExpansionStrategy
    from synthesis.graph_runner import GraphRunner, GraphRunnerConfig
    from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
    from synthesis.model_worker import LLM_WORKER
    from synthesis.search_client import CommonsImageSearchClient, SerpApiSearchClient
    from synthesis.store import JsonlGraphStore
    from synthesis.visual_planner import LLMVisualSearchPlanner
    from synthesis.wiki_text_builder import EnhancedReaderClient, WikiTextBuilder

    if not args.skip_reader_check:
        ok, message = check_reader_service(args.reader_base_url)
        if not ok:
            print(f"[preflight] Enhanced Reader is unavailable at {args.reader_base_url}: {message}", file=sys.stderr)
            print("[preflight] Start the reader stack or rerun with --skip-reader-check.", file=sys.stderr)
            return 2
        print(f"[preflight] Enhanced Reader {message}")

    store_dir = Path(args.store_dir)
    if not store_dir.is_absolute():
        store_dir = PROJECT_ROOT / store_dir
    store = JsonlGraphStore(store_dir)

    reader = EnhancedReaderClient(base_url=args.reader_base_url)
    wiki_builder = WikiTextBuilder(
        reader=reader,
        store=store,
        model_client=LLM_WORKER,
        max_links=args.max_links,
    )

    visual_planner = None
    image_builder = None
    if not args.no_images:
        visual_planner = LLMVisualSearchPlanner(model_client=LLM_WORKER)
        commons_client = CommonsImageSearchClient()
        fallback_client = None
        if not args.no_serp_fallback and has_serpapi_credentials():
            fallback_client = SerpApiSearchClient()
        image_builder = ImageDiscoveryBuilder(
            store=store,
            commons_client=commons_client,
            fallback_client=fallback_client,
            config=ImageDiscoveryConfig(
                per_query_limit=args.per_query_image_limit,
                max_images_per_plan=args.max_images_per_plan,
            ),
            model_client=LLM_WORKER,
        )

    strategy = GraphExpansionStrategy(
        store=store,
        wiki_builder=wiki_builder,
        visual_planner=visual_planner,
        image_builder=image_builder,
        config=GraphExpansionConfig(
            max_depth=args.max_depth,
            max_new_text_neighbors=args.max_neighbors,
            extract_attributes=not args.skip_attributes,
            attribute_errors_fatal=args.fatal_attribute_errors,
            enable_image_expansion=not args.no_images,
            persist=True,
        ),
    )
    runner = GraphRunner(
        strategy=strategy,
        store=store,
        config=GraphRunnerConfig(
            max_steps=args.max_steps,
            max_nodes=args.max_nodes,
            checkpoint_every=1,
            stop_on_error=False,
        ),
        run_id=args.run_id,
        resume=not args.fresh,
    )
    if runner.strategy.queue_size() == 0:
        runner.add_seed(args.seed_url)

    result = runner.run()
    print("=== min graph run ===")
    print(f"env_file: {env_path} ({len(loaded_env)} vars loaded)")
    print(f"store_dir: {store_dir}")
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"steps: {result.steps}")
    print(f"queue_size: {result.queue_size}")
    print(f"completed: {result.completed_count}")
    print(f"failed: {result.failed_count}")
    print(f"store_stats: {result.store_stats}")
    if result.last_error:
        print(f"last_error: {result.last_error}")
    return 0 if result.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
