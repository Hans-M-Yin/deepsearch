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
import json
import os
from pathlib import Path
import shlex
import subprocess
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


def load_seed_urls_file(path: Path) -> list[str]:
    seeds: list[str] = []
    if not path.exists():
        raise FileNotFoundError(f"Seed URL file does not exist: {path}")
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(("http://", "https://")):
            raise ValueError(f"{path}:{line_no} seed URL must start with http:// or https://: {raw_line!r}")
        seeds.append(line)
    return seeds


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


def has_serper_credentials() -> bool:
    keys_file = Path("synthesis/serper_keys.txt.example").resolve()
    return keys_file.exists()


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


def _redact_env_value(name: str, value: str | None) -> str:
    if not value:
        return "<unset>"
    secret_markers = ("KEY", "TOKEN", "SECRET", "PASSWORD", "AK")
    if any(marker in name.upper() for marker in secret_markers):
        if len(value) <= 8:
            return "<set>"
        return f"{value[:4]}...{value[-4:]}"
    return value


def print_startup_config(
    *,
    args: argparse.Namespace,
    env_path: Path,
    loaded_env: dict[str, str],
    store_dir: Path,
    seed_urls: list[str],
    runner_queue_size: int,
    text_queue_size: int,
    image_queue_size: int,
) -> None:
    relevant_env_vars = [
        "SYNTHESIS_MODEL_CONFIG",
        "OPENAI_API_KEY",
        "SERPAPI_AK",
        "AIDP_SERP_AK",
        "SERPAPI_API_KEY",
        "SERP_API_KEY",
        "SERPER_API_KEY",
        "SERPER_API_KEYS",
        "SERPER_API_KEYS_FILE",
        "IMAGE_GROUND_MODEL",
        "IMAGE_ENTITY_RESOLVE_MODEL",
        "IMAGE_QUERY_ENTITY_FILTER_MODEL",
        "SYNTHESIS_TRACE_TIMING",
        "VQA_WRITER_MODEL",
    ]
    print("=== min graph startup ===")
    print(f"env_file: {env_path} ({len(loaded_env)} vars loaded)")
    print(f"store_dir: {store_dir}")
    print(f"seed_count: {len(seed_urls)}")
    print(f"seed_preview: {seed_urls[:5]}")
    print(
        "runner_config: "
        f"max_steps={args.max_steps} "
        f"max_nodes={args.max_nodes} "
        f"parallel_workers={args.parallel_workers} "
        f"batch_size={args.batch_size} "
        f"max_depth={args.max_depth} "
        f"max_neighbors={args.max_neighbors} "
        f"images_enabled={not args.no_images} "
        f"image_backend={args.image_backend} "
        f"store_flush_record_threshold={args.store_flush_record_threshold} "
        f"store_flush_interval_s={args.store_flush_interval_s}"
    )
    print(
        "queue_state: "
        f"queue={runner_queue_size} "
        f"text_queue={text_queue_size} "
        f"image_queue={image_queue_size}"
    )
    print("environment:")
    for name in relevant_env_vars:
        print(f"  {name}={_redact_env_value(name, os.environ.get(name))}")


def load_failed_task_preview(store_dir: Path, *, limit: int = 1) -> list[dict[str, Any]]:
    state_path = store_dir / "graph_runner_state.json"
    if not state_path.exists():
        return []
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    failed = payload.get("failed_tasks")
    if not isinstance(failed, list):
        return []
    return [item for item in failed[-limit:] if isinstance(item, dict)]


def current_git_metadata(project_root: Path) -> dict[str, Any]:
    def _run_git(*args: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", *args],
                cwd=project_root,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return None
        return completed.stdout.strip()

    commit = _run_git("rev-parse", "HEAD")
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    status_output = _run_git("status", "--porcelain")
    dirty = bool(status_output) if status_output is not None else None
    return {
        "commit": commit,
        "branch": branch,
        "dirty": dirty,
    }


def save_run_config(
    store_dir: Path,
    *,
    args: argparse.Namespace,
    env_path: Path,
    loaded_env: dict[str, str],
    git_metadata: dict[str, Any],
) -> Path:
    payload = {
        "project_root": str(PROJECT_ROOT),
        "store_dir": str(store_dir),
        "command": " ".join(shlex.quote(part) for part in sys.argv),
        "args": vars(args),
        "env_file": str(env_path),
        "loaded_env_count": len(loaded_env),
        "loaded_env_keys": sorted(loaded_env.keys()),
        "git": git_metadata,
    }
    path = store_dir / "run_config.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing env vars.")
    parser.add_argument("--seed-url", default=None, help="Seed Wikipedia URL. If omitted, defaults to Kobe Bryant unless --seed-urls-file is provided.")
    parser.add_argument("--seed-urls-file", default=None, help="Text file containing one seed Wikipedia URL per line. Blank lines and # comments are ignored.")
    parser.add_argument("--store-dir", default=str(DEFAULT_STORE_DIR), help="Output JSONL graph store directory.")
    parser.add_argument("--reader-base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--reader-check-timeout", type=float, default=60.0, help="Enhanced Reader preflight timeout in seconds.")
    parser.add_argument("--skip-reader-check", action="store_true", help="Skip preflight reader reachability check.")
    parser.add_argument(
        "--max-steps",
        type=int,
        default=5,
        help="Maximum total expansion tasks, including text and image tasks.",
    )
    parser.add_argument(
        "--max-nodes",
        type=int,
        default=10,
        help="Maximum number of text nodes to expand. Pending image tasks continue after this limit is reached.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="Number of workers shared by text and image expansion tasks.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Tasks popped from the queue per parallel expansion round.")
    parser.add_argument(
        "--max-inflight-text",
        type=int,
        default=None,
        help="When image tasks are queued, cap concurrently running text-expansion tasks so workers are left for images. <=0 disables the cap.",
    )
    parser.add_argument("--max-depth", type=int, default=1, help="Maximum text-neighbor BFS depth.")
    parser.add_argument(
        "--queue-pop-strategy",
        choices=("fifo", "random"),
        default="fifo",
        help="How to select the next eligible expansion task from the queue.",
    )
    parser.add_argument(
        "--queue-pop-random-seed",
        default="graph_expansion_queue_v1",
        help="Stable seed used when --queue-pop-strategy=random.",
    )
    parser.add_argument("--max-neighbors", type=int, default=5, help="Text neighbors queued per text node.")
    parser.add_argument("--max-links", type=int, default=60, help="Wiki links extracted per page before queue slicing.")
    parser.add_argument("--link-window-size", type=int, default=1200, help="Character window size for wiki-link diversity.")
    parser.add_argument("--max-links-per-window", type=int, default=2, help="Maximum selected wiki links per character window.")
    parser.add_argument("--min-link-char-distance", type=int, default=500, help="Minimum character distance between selected wiki links.")
    parser.add_argument("--lead-chars", type=int, default=3000, help="Leading page characters that receive a looser link quota.")
    parser.add_argument("--lead-max-links-per-window", type=int, default=4, help="Maximum selected links per window in the leading page region.")
    parser.add_argument("--max-content-chars", type=int, default=70000, help="Max cleaned markdown chars stored in each text node/evidence. <=0 disables truncation.")
    parser.add_argument("--max-link-markdown-chars", type=int, default=100000, help="Max raw markdown chars used for wiki-link extraction. <=0 disables truncation.")
    parser.add_argument("--max-llm-neighbor-candidates", type=int, default=60, help="Maximum rule-recalled wiki links sent to WIKI_NEIGHBOR_MODEL per page.")
    parser.add_argument("--max-qa-neighbor-candidates", type=int, default=0, help="Maximum reranked wiki links sent through neighbor familiarity QA penalty per page. <=0 means use all kept neighbors.")
    parser.add_argument("--per-query-image-limit", type=int, default=3, help="Image search results per visual query.")
    parser.add_argument("--max-images-per-plan", type=int, default=5, help="Accepted images per visual plan.")
    parser.add_argument(
        "--image-budget-chars",
        type=int,
        default=8000,
        help="Legacy visual-planner context budget setting. Pages below the internal minimum threshold produce no visual plans; eligible pages may produce up to five plans.",
    )
    parser.add_argument("--no-images", action="store_true", help="Disable visual planning and image discovery.")
    parser.add_argument(
        "--force-accept-images",
        action="store_true",
        help="Debug mode: bypass semantic image rejection for every resolvable image candidate.",
    )
    parser.add_argument("--skip-attributes", action="store_true", help="Do not call LLM attribute extraction.")
    parser.add_argument("--fatal-attribute-errors", action="store_true", help="Fail the task if attribute extraction fails.")
    parser.add_argument("--persist-snapshots", action="store_true", help="Persist verbose SearchSnapshot records for debugging.")
    parser.add_argument(
        "--image-backend",
        choices=("commons", "serpapi", "serper", "openserp", "serper_adapter", "commons_serpapi"),
        default="commons_serpapi",
        help="Image search backend.",
    )
    parser.add_argument("--run-id", default=None, help="Optional stable run id.")
    parser.add_argument("--fresh", action="store_true", help="Ignore existing runner checkpoint state.")
    parser.add_argument(
        "--store-flush-record-threshold",
        type=int,
        default=100,
        help="Flush graph tables after at least this many record upserts have accumulated.",
    )
    parser.add_argument(
        "--store-flush-interval-s",
        type=float,
        default=30.0,
        help="Flush graph tables if this many seconds pass since the last flush, even if the record threshold is not reached.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    check_python_version()

    seed_urls: list[str] = []
    if args.seed_urls_file:
        seed_file_path = Path(args.seed_urls_file)
        if not seed_file_path.is_absolute():
            seed_file_path = PROJECT_ROOT / seed_file_path
        seed_urls.extend(load_seed_urls_file(seed_file_path))
    if args.seed_url:
        seed_urls.append(args.seed_url)
    if not seed_urls:
        seed_urls = [DEFAULT_SEED_URL]
    # Preserve order while deduplicating.
    seed_urls = list(dict.fromkeys(seed_urls))

    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    loaded_env = load_env_file(env_path, override=args.override_env)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from synthesis.graph_expansion import ExpansionTaskType, GraphExpansionConfig, GraphExpansionStrategy
    from synthesis.graph_runner import GraphRunner, GraphRunnerConfig
    from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
    from synthesis.model_worker import LLM_WORKER
    from synthesis.search_client import (
        CommonsImageSearchClient,
        CommonsSerpApiSearchClient,
        OpenSerpSearchClient,
        SerperAdapterSearchClient,
        SerperSearchClient,
        SerpApiSearchClient,
    )
    from synthesis.store import JsonlGraphStore
    from synthesis.visual_planner import LLMVisualSearchPlanner
    from synthesis.wiki_text_builder import EnhancedReaderClient, WikiTextBuilder

    if not args.skip_reader_check:
        ok, message = check_reader_service(
            args.reader_base_url,
            test_url=seed_urls[0],
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
    store = JsonlGraphStore(
        store_dir,
        flush_record_threshold=args.store_flush_record_threshold,
        flush_interval_s=args.store_flush_interval_s,
    )
    git_metadata = current_git_metadata(PROJECT_ROOT)

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
        max_qa_neighbor_candidates=args.max_qa_neighbor_candidates,
    )

    visual_planner = None
    image_builder = None
    if not args.no_images:
        visual_planner = LLMVisualSearchPlanner(
            model_client=LLM_WORKER,
            target_chars_per_budget=args.image_budget_chars,
        )
        backend_builders = {
            "commons": CommonsImageSearchClient,
            "commons_serpapi": CommonsSerpApiSearchClient,
            "serpapi": SerpApiSearchClient,
            "serper": SerperSearchClient,
            "openserp": OpenSerpSearchClient,
            "serper_adapter": SerperAdapterSearchClient,
        }
        credential_checks = {
            "commons": lambda: True,
            "commons_serpapi": has_serpapi_credentials,
            "serpapi": has_serpapi_credentials,
            "serper": has_serper_credentials,
            "openserp": lambda: True,
            "serper_adapter": lambda: True,
        }

        def build_backend(name: str):
            if not credential_checks[name]():
                raise ValueError(
                    f"Image backend {name!r} is selected but required credentials/service configuration is missing."
                )
            return backend_builders[name]()

        image_builder = ImageDiscoveryBuilder(
            store=store,
            search_client=build_backend(args.image_backend),
            config=ImageDiscoveryConfig(
                per_query_limit=args.per_query_image_limit,
                max_images_per_plan=args.max_images_per_plan,
                persist_search_snapshots=args.persist_snapshots,
                force_accept_images=args.force_accept_images,
                image_grounding_reader_base_url=args.reader_base_url,
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
            queue_pop_strategy=args.queue_pop_strategy,
            queue_pop_random_seed=args.queue_pop_random_seed,
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
            max_inflight_text=args.max_inflight_text if args.max_inflight_text and args.max_inflight_text > 0 else None,
        ),
        run_id=args.run_id,
        resume=not args.fresh,
    )
    run_config_path = save_run_config(
        store_dir,
        args=args,
        env_path=env_path,
        loaded_env=loaded_env,
        git_metadata=git_metadata,
    )
    if runner.strategy.queue_size() == 0:
        runner.add_seeds(seed_urls)
    print_startup_config(
        args=args,
        env_path=env_path,
        loaded_env=loaded_env,
        store_dir=store_dir,
        seed_urls=seed_urls,
        runner_queue_size=runner.strategy.queue_size(),
        text_queue_size=runner.strategy.queue_size(ExpansionTaskType.TEXT_EXPAND),
        image_queue_size=runner.strategy.queue_size(ExpansionTaskType.IMAGE_EXPAND),
    )

    started_at = time.perf_counter()
    result = runner.run()
    elapsed_s = time.perf_counter() - started_at
    store_size = directory_size_bytes(store_dir)
    print("=== min graph run ===")
    print(f"env_file: {env_path} ({len(loaded_env)} vars loaded)")
    print(f"run_config: {run_config_path}")
    if git_metadata.get("commit"):
        print(
            "git: "
            f"commit={git_metadata.get('commit')} "
            f"branch={git_metadata.get('branch')} "
            f"dirty={git_metadata.get('dirty')}"
        )
    print(f"store_dir: {store_dir}")
    print(f"seed_count: {len(seed_urls)}")
    print(f"seed_preview: {seed_urls[:5]}")
    print(f"run_id: {result.run_id}")
    print(f"status: {result.status}")
    print(f"steps: {result.steps}")
    print(f"queue_size: {result.queue_size}")
    print(f"text_queue_size: {runner.strategy.queue_size(ExpansionTaskType.TEXT_EXPAND)}")
    print(f"image_queue_size: {runner.strategy.queue_size(ExpansionTaskType.IMAGE_EXPAND)}")
    print(f"completed: {result.completed_count}")
    print(f"failed: {result.failed_count}")
    print(f"skipped: {result.skipped_count}")
    print(f"store_stats: {result.store_stats}")
    if result.last_error:
        print(f"last_error: {result.last_error}")
    failed_preview = load_failed_task_preview(store_dir, limit=1)
    if failed_preview:
        record = failed_preview[0]
        task = record.get("task") if isinstance(record.get("task"), dict) else {}
        print("last_failed_task:")
        print(f"  url: {task.get('url')}")
        print(f"  title: {task.get('title')}")
        print(f"  depth: {task.get('depth')}")
        print(f"  error: {record.get('error')}")
    graph_metrics = graph_density_metrics(result.store_stats)
    print(
        "graph_density: "
        f"directed={graph_metrics['directed_density']:.6f} "
        f"undirected_upper_bound={graph_metrics['undirected_density_upper_bound']:.6f} "
        f"avg_out_degree={graph_metrics['avg_out_degree']:.2f} "
        f"avg_total_degree={graph_metrics['avg_total_degree']:.2f}"
    )
    image_summary = result.image_summary or {}
    print(
        "image_summary: "
        f"returned={int(image_summary.get('returned') or 0)} "
        f"accepted={int(image_summary.get('accepted') or 0)} "
        f"rejected={int(image_summary.get('rejected') or 0)} "
        f"fetch_failed={int(image_summary.get('fetch_failed') or 0)}"
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
