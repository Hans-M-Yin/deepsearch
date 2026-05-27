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
import time
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


def check_reader_service(
    base_url: str,
    *,
    test_url: str,
    timeout_s: float = 60.0,
) -> tuple[bool, str]:
    """Check that the Enhanced Reader can read an actual target URL."""

    target = test_url if test_url.startswith(("http://", "https://")) else f"https://{test_url}"
    request_url = f"{base_url.rstrip('/')}/{target}"
    request = Request(request_url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            return True, f"reachable, status={response.getcode()}, test_url={target}"
    except HTTPError as exc:
        if exc.code < 500:
            return True, f"reachable, status={exc.code}, test_url={target}"
        return False, f"server error, status={exc.code}, test_url={target}"
    except URLError as exc:
        return False, f"not reachable: {exc.reason}, test_url={target}"
    except TimeoutError:
        return False, f"not reachable: timed out after {timeout_s}s, test_url={target}"


def has_serpapi_credentials() -> bool:
    return bool(
        os.environ.get("SERPAPI_AK")
        or os.environ.get("AIDP_SERP_AK")
        or os.environ.get("SERPAPI_API_KEY")
        or os.environ.get("SERP_API_KEY")
    )


def directory_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def format_bytes(size: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{size} B"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, remainder = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {remainder:.2f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {remainder:.2f}s"


def graph_density_metrics(store_stats: dict[str, int]) -> dict[str, float | int]:
    node_count = int(store_stats.get("nodes", 0))
    edge_count = int(store_stats.get("edges", 0))
    directed_possible_edges = node_count * (node_count - 1)
    undirected_possible_edges = node_count * (node_count - 1) / 2
    return {
        "nodes": node_count,
        "edges": edge_count,
        "avg_out_degree": edge_count / node_count if node_count else 0.0,
        "avg_total_degree": (2 * edge_count) / node_count if node_count else 0.0,
        "directed_density": edge_count / directed_possible_edges if directed_possible_edges else 0.0,
        "undirected_density_upper_bound": edge_count / undirected_possible_edges if undirected_possible_edges else 0.0,
    }


def print_timing_summary(summary: dict[str, Any]) -> None:
    metrics = summary.get("metrics") if isinstance(summary, dict) else None
    if not isinstance(metrics, dict) or not metrics:
        return
    print("timing_summary:")
    print(f"  steps_with_timing: {summary.get('steps_with_timing')}")
    for key in sorted(metrics):
        item = metrics[key]
        if not isinstance(item, dict):
            continue
        print(
            "  "
            f"{key}: "
            f"total={item.get('total_s', 0.0):.2f}s "
            f"avg={item.get('avg_s', 0.0):.2f}s "
            f"p50={item.get('p50_s', 0.0):.2f}s "
            f"max={item.get('max_s', 0.0):.2f}s"
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing env vars.")
    parser.add_argument("--seed-url", default=DEFAULT_SEED_URL, help="Seed Wikipedia URL.")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Output JSONL graph store directory.")
    parser.add_argument("--reader-base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--reader-check-timeout", type=float, default=60.0, help="Enhanced Reader preflight timeout in seconds.")
    parser.add_argument("--skip-reader-check", action="store_true", help="Skip preflight reader reachability check.")
    parser.add_argument("--max-steps", type=int, default=5, help="Maximum text pages to expand.")
    parser.add_argument("--max-nodes", type=int, default=10, help="Stop after this many graph nodes.")
    parser.add_argument("--parallel-workers", type=int, default=1, help="Number of text-node expansion workers.")
    parser.add_argument("--batch-size", type=int, default=None, help="Tasks popped from the queue per parallel expansion round.")
    parser.add_argument("--max-depth", type=int, default=1, help="Maximum text-neighbor BFS depth.")
    parser.add_argument("--max-neighbors", type=int, default=2, help="Text neighbors queued per text node.")
    parser.add_argument("--max-links", type=int, default=20, help="Wiki links extracted per page before queue slicing.")
    parser.add_argument("--link-window-size", type=int, default=1200, help="Character window size for wiki-link diversity.")
    parser.add_argument("--max-links-per-window", type=int, default=2, help="Maximum selected wiki links per character window.")
    parser.add_argument("--min-link-char-distance", type=int, default=500, help="Minimum character distance between selected wiki links.")
    parser.add_argument("--lead-chars", type=int, default=3000, help="Leading page characters that receive a looser link quota.")
    parser.add_argument("--lead-max-links-per-window", type=int, default=4, help="Maximum selected links per window in the leading page region.")
    parser.add_argument("--max-content-chars", type=int, default=50000, help="Max cleaned markdown chars stored in each text node/evidence. <=0 disables truncation.")
    parser.add_argument("--max-link-markdown-chars", type=int, default=80000, help="Max raw markdown chars used for wiki-link extraction. <=0 disables truncation.")
    parser.add_argument("--max-llm-neighbor-candidates", type=int, default=40, help="Maximum rule-recalled wiki links sent to WIKI_NEIGHBOR_MODEL per page.")
    parser.add_argument("--per-query-image-limit", type=int, default=3, help="Image search results per visual query.")
    parser.add_argument("--max-images-per-plan", type=int, default=1, help="Accepted images per visual plan.")
    parser.add_argument("--no-images", action="store_true", help="Disable visual planning and image discovery.")
    parser.add_argument("--skip-attributes", action="store_true", help="Do not call LLM attribute extraction.")
    parser.add_argument("--fatal-attribute-errors", action="store_true", help="Fail the task if attribute extraction fails.")
    parser.add_argument("--persist-snapshots", action="store_true", help="Persist verbose SearchSnapshot records for debugging.")
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
        ok, message = check_reader_service(
            args.reader_base_url,
            test_url=args.seed_url,
            timeout_s=args.reader_check_timeout,
        )
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
        persist_snapshots=args.persist_snapshots,
        diversity_window_size=args.link_window_size,
        max_links_per_window=args.max_links_per_window,
        min_link_char_distance=args.min_link_char_distance,
        lead_chars=args.lead_chars,
        lead_max_links_per_window=args.lead_max_links_per_window,
        max_content_chars=args.max_content_chars if args.max_content_chars > 0 else None,
        max_link_markdown_chars=args.max_link_markdown_chars if args.max_link_markdown_chars > 0 else None,
        max_llm_neighbor_candidates=args.max_llm_neighbor_candidates,
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
                persist_search_snapshots=args.persist_snapshots,
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
            parallel_workers=args.parallel_workers,
            batch_size=args.batch_size,
        ),
        run_id=args.run_id,
        resume=not args.fresh,
    )
    if runner.strategy.queue_size() == 0:
        runner.add_seed(args.seed_url)

    started_at = time.perf_counter()
    result = runner.run()
    elapsed_s = time.perf_counter() - started_at
    store_size = directory_size_bytes(store_dir)
    print("=== min graph run ===")
    print(f"env_file: {env_path} ({len(loaded_env)} vars loaded)")
    print(f"store_dir: {store_dir}")
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"steps: {result.steps}")
    print(f"queue_size: {result.queue_size}")
    print(f"completed: {result.completed_count}")
    print(f"failed: {result.failed_count}")
    print(f"skipped: {result.skipped_count}")
    print(f"store_stats: {result.store_stats}")
    graph_metrics = graph_density_metrics(result.store_stats)
    print(
        "graph_density: "
        f"directed={graph_metrics['directed_density']:.6f} "
        f"undirected_upper_bound={graph_metrics['undirected_density_upper_bound']:.6f} "
        f"avg_out_degree={graph_metrics['avg_out_degree']:.2f} "
        f"avg_total_degree={graph_metrics['avg_total_degree']:.2f}"
    )
    print(f"elapsed_s: {elapsed_s:.2f}")
    print(f"elapsed: {format_duration(elapsed_s)}")
    print(f"store_size_bytes: {store_size}")
    print(f"store_size: {format_bytes(store_size)}")
    print_timing_summary(result.timing_summary)
    if result.last_error:
        print(f"last_error: {result.last_error}")
    return 0 if result.failed_count == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
