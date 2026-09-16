#!/usr/bin/env python3
"""Benchmark the Enhanced Reader and the Qwen summarizer as separate stages.

The benchmark has three independent phases:

1. Collect a diverse URL set with real Serper text searches. Each search asks
   for ten results; the default is 100 searches.
2. Fetch every URL through the Enhanced Reader (8004 by default). Requests
   use ``debug=1&refresh=1`` so the 8004 response cache is bypassed. The
   Enhanced Reader still fetches its source through the local 8002 service,
   whose source-page request is not cached.
3. Run ``synthesis.sft.tools.summarize_with_qwen`` on the saved reader
   contents with independent worker levels.

The reader and summarizer stages are deliberately run separately. This keeps
network/ReaderLM latency from being mixed with summarizer LLM latency. Reader
results are written as one JSONL per worker level, so the summarizer stage can
be rerun without spending Serper or reader requests again.

Examples::

    python debug/benchmark_enhanced_reader_pipeline.py \
      --stages all --search-count 100 --results-per-search 10 \
      --workers 1,4,8,16,32 \
      --output-dir /tmp/enhanced_reader_pipeline

    # Reuse an existing reader JSONL and only measure summarization.
    python debug/benchmark_enhanced_reader_pipeline.py \
      --stages summarizer \
      --reader-documents /tmp/enhanced_reader_pipeline/reader_results_w16.jsonl \
      --workers 1,2,4,8 \
      --output-dir /tmp/enhanced_reader_pipeline

The summarizer requires the normal ``SYNTHESIS_MODEL_CONFIG`` and a
registered ``SFT_QWEN_MODEL_ALIAS`` (or the usual fallback alias).
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import random
import re
import statistics
import sys
import time
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QUESTION_LIST = (
    PROJECT_ROOT
    / "runs/0712_multi_seed_visual_test_8192_6/vqa/0803_batch_1/questions.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("/tmp/enhanced_reader_pipeline")
DEFAULT_READER_URL = os.environ.get("ENHANCED_READER_URL") or "http://127.0.0.1:8004"
DEFAULT_ENV_FILE = PROJECT_ROOT / "synthesis" / ".env"
DEFAULT_READER_TIMEOUT_S = 180.0
DEFAULT_WORKERS = (1, 4, 8, 16, 32)
DEFAULT_SEARCH_COUNT = 100
DEFAULT_RESULTS_PER_SEARCH = 10
DEFAULT_SEARCH_WORKERS = 8

_DEFAULT_EXCLUDED_DOMAINS = (
    "wikipedia.org",
    "wikimedia.org",
    "wikidata.org",
)


def _short_error(exc: BaseException) -> str:
    text = str(exc).strip().replace("\n", " ")
    return text[:1200] if text else exc.__class__.__name__


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("worker/concurrency values must be positive")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one worker level is required")
    return values


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                print(f"[pipeline] skipping invalid JSONL line {path}:{line_number}", file=sys.stderr)
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def _load_question_texts(path: Path) -> list[str]:
    values: list[str] = []
    for record in _load_jsonl(path):
        value = (
            record.get("final_question")
            or record.get("question")
            or record.get("enhanced_question")
            or record.get("polished_question")
        )
        value = re.sub(r"\s+", " ", str(value or "")).strip()
        if value:
            values.append(value)
    return values


def _fallback_queries() -> list[str]:
    """Diverse non-Wikipedia queries used only when the question file is absent."""

    topics = [
        "renewable energy storage",
        "ocean microplastics research",
        "ancient trade routes",
        "public health policy",
        "modern architecture materials",
        "astronomy exoplanet discovery",
        "classical music performance history",
        "wildlife conservation project",
        "computer security vulnerability",
        "volcanic eruption monitoring",
        "archaeological excavation report",
        "space telescope instrument",
        "urban transportation planning",
        "agricultural drought resistance",
        "constitutional court decision",
        "marine biology field study",
        "museum painting conservation",
        "mathematical theorem history",
        "international climate agreement",
        "engineering bridge inspection",
    ]
    modifiers = [
        "history",
        "research report",
        "official data",
        "timeline",
        "case study",
    ]
    return [f"{topic} {modifier}" for topic in topics for modifier in modifiers]


def _build_queries(
    *,
    question_list: Path | None,
    count: int,
    seed: int,
) -> list[str]:
    rng = random.Random(seed)
    candidates: list[str] = []
    if question_list is not None and question_list.exists():
        candidates.extend(_load_question_texts(question_list))
    if not candidates:
        candidates.extend(_fallback_queries())
    if not candidates:
        raise RuntimeError("No search queries are available.")

    # Sampling questions directly gives a much more varied domain mix than a
    # hard-coded Wikipedia-oriented query list. Keep queries bounded so a
    # single long VQA question does not dominate the Serper request.
    rng.shuffle(candidates)
    queries: list[str] = []
    for index in range(count):
        source = candidates[index % len(candidates)]
        words = source.split()
        if len(words) > 14:
            width = min(10, len(words))
            start = rng.randrange(0, len(words) - width + 1)
            source = " ".join(words[start : start + width])
        queries.append(source[:240].strip())
    return queries


def _excluded(url: str, domains: tuple[str, ...]) -> bool:
    host = (urlsplit(url).hostname or "").casefold().rstrip(".")
    return any(host == domain or host.endswith("." + domain) for domain in domains)


def _search_one(query: str, *, results_per_search: int, timeout_s: float) -> dict[str, Any]:
    from synthesis.search_client import SerperSearchClient

    started = time.perf_counter()
    try:
        client = SerperSearchClient(timeout_s=timeout_s)
        response = client.search_text(query, limit=results_per_search, hl="en")
        results: list[dict[str, Any]] = []
        for rank, item in enumerate(response.results, 1):
            url = str(getattr(item, "url", "") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            results.append(
                {
                    "url": url,
                    "title": str(getattr(item, "title", "") or ""),
                    "snippet": str(getattr(item, "snippet", "") or ""),
                    "rank": rank,
                }
            )
        return {
            "query": query,
            "ok": True,
            "elapsed_s": time.perf_counter() - started,
            "results": results,
        }
    except Exception as exc:  # pragma: no cover - live network dependent
        return {
            "query": query,
            "ok": False,
            "elapsed_s": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": _short_error(exc),
            "results": [],
        }


def _collect_urls(args: argparse.Namespace, output_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries = _build_queries(
        question_list=args.question_list,
        count=args.search_count,
        seed=args.seed,
    )
    print(
        f"[search] collecting queries={len(queries)} results_per_search={args.results_per_search} "
        f"workers={args.search_workers}",
        file=sys.stderr,
        flush=True,
    )
    started = time.perf_counter()
    search_results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.search_workers, thread_name_prefix="serper-url-seed") as executor:
        futures = [
            executor.submit(
                _search_one,
                query,
                results_per_search=args.results_per_search,
                timeout_s=args.serper_timeout_s,
            )
            for query in queries
        ]
        for future in as_completed(futures):
            search_results.append(future.result())

    excluded_domains = tuple(args.exclude_domain)
    urls: list[dict[str, Any]] = []
    seen: set[str] = set()
    for search_result in search_results:
        for item in search_result.get("results") or []:
            url = str(item.get("url") or "").strip()
            if not url or url in seen or _excluded(url, excluded_domains):
                continue
            seen.add(url)
            urls.append(
                {
                    "url": url,
                    "query": search_result.get("query", ""),
                    "title": item.get("title", ""),
                    "snippet": item.get("snippet", ""),
                    "rank": item.get("rank"),
                }
            )
            if len(urls) >= args.max_urls:
                break
        if len(urls) >= args.max_urls:
            break

    output_dir.mkdir(parents=True, exist_ok=True)
    search_path = output_dir / "serper_search_results.jsonl"
    with search_path.open("w", encoding="utf-8") as handle:
        for item in search_results:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    url_path = output_dir / "urls.jsonl"
    with url_path.open("w", encoding="utf-8") as handle:
        for item in urls:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    summary = {
        "queries": len(queries),
        "search_successes": sum(bool(item.get("ok")) for item in search_results),
        "search_failures": sum(not bool(item.get("ok")) for item in search_results),
        "raw_results": sum(len(item.get("results") or []) for item in search_results),
        "unique_urls_after_filter": len(urls),
        "excluded_domains": list(excluded_domains),
        "elapsed_s": time.perf_counter() - started,
        "search_results_jsonl": str(search_path),
        "urls_jsonl": str(url_path),
    }
    print(
        f"[search] unique_urls={len(urls)} search_failures={summary['search_failures']} "
        f"elapsed_s={summary['elapsed_s']:.3f} output={url_path}",
        file=sys.stderr,
        flush=True,
    )
    failed_searches = [item for item in search_results if not item.get("ok")]
    if failed_searches:
        sample = failed_searches[0]
        print(
            "[search-error] "
            f"error_type={sample.get('error_type', 'unknown')} "
            f"error={sample.get('error', '')!r}",
            file=sys.stderr,
            flush=True,
        )
    return urls, summary


def _reader_endpoint(base_url: str, url: str, *, refresh: bool) -> str:
    # Encode the target URL as a path segment so its own '?'/'&' query string
    # is not mistaken for the local Enhanced Reader query string.
    endpoint = f"{base_url.rstrip('/')}/{quote(url, safe='')}"
    port = urlsplit(base_url).port
    # The raw 8002 dispatcher does not have the 8004 cache controls. It is
    # already uncached, so leave its request untouched.
    if refresh and port != 8002:
        endpoint += "?" + urlencode({"debug": "1", "refresh": "1"})
    return endpoint


def _fetch_reader_one(
    item: dict[str, Any],
    *,
    reader_url: str,
    timeout_s: float,
    refresh: bool,
) -> dict[str, Any]:
    url = str(item.get("url") or "")
    started = time.perf_counter()
    endpoint = _reader_endpoint(reader_url, url, refresh=refresh)
    request = Request(endpoint, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_s) as response:
            status_code = int(response.getcode() or 200)
            body = response.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            payload = None
        if isinstance(payload, dict):
            data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
            content = str(data.get("content") or data.get("markdown") or "")
            debug_timing = data.get("debug_timing") or payload.get("debug_timing") or {}
            content_source = data.get("content_source")
            content_quality = data.get("content_quality")
        else:
            content = body
            debug_timing = {}
            content_source = "raw_response"
            content_quality = "unknown"
        if not content.strip():
            raise RuntimeError("reader returned empty content")
        return {
            **item,
            "ok": True,
            "status_code": status_code,
            "content": content,
            "content_chars": len(content),
            "content_source": content_source,
            "content_quality": content_quality,
            "server_debug_timing": debug_timing if isinstance(debug_timing, dict) else {},
            "elapsed_s": time.perf_counter() - started,
        }
    except HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[:1000]
        except Exception:
            error_body = ""
        return {
            **item,
            "ok": False,
            "status_code": int(exc.code),
            "elapsed_s": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": f"{_short_error(exc)} body={error_body}",
        }
    except Exception as exc:  # pragma: no cover - live network dependent
        return {
            **item,
            "ok": False,
            "elapsed_s": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": _short_error(exc),
        }


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _latency_summary(results: list[dict[str, Any]], *, wall_s: float, workers: int) -> dict[str, Any]:
    latencies = [float(item.get("elapsed_s") or 0.0) for item in results]
    successes = sum(bool(item.get("ok")) for item in results)
    errors = Counter(
        f"{item.get('error_type')}: {item.get('error', '')}".strip()
        for item in results
        if not item.get("ok")
    )
    return {
        "requests": len(results),
        "successes": successes,
        "failures": len(results) - successes,
        "success_rate": successes / len(results) if results else 0.0,
        "workers": workers,
        "wall_s": wall_s,
        "throughput_rps": len(results) / wall_s if wall_s > 0 else 0.0,
        "latency_mean_s": statistics.fmean(latencies) if latencies else None,
        "latency_p50_s": _percentile(latencies, 0.50),
        "latency_p95_s": _percentile(latencies, 0.95),
        "latency_p99_s": _percentile(latencies, 0.99),
        "latency_max_s": max(latencies) if latencies else None,
        "content_chars_mean": (
            statistics.fmean(float(item.get("content_chars") or 0) for item in results if item.get("ok"))
            if successes
            else None
        ),
        "top_errors": [
            {"error": error, "count": count} for error, count in errors.most_common(10)
        ],
    }


def _run_reader_level(
    urls: list[dict[str, Any]],
    *,
    workers: int,
    reader_url: str,
    timeout_s: float,
    refresh: bool,
    output_path: Path,
) -> dict[str, Any]:
    print(
        f"[reader] workers={workers} requests={len(urls)} reader_url={reader_url}",
        file=sys.stderr,
        flush=True,
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"reader-{workers}") as executor:
            futures: list[Future[dict[str, Any]]] = [
                executor.submit(
                    _fetch_reader_one,
                    item,
                    reader_url=reader_url,
                    timeout_s=timeout_s,
                    refresh=refresh,
                )
                for item in urls
            ]
            for future in as_completed(futures):
                result = future.result()
                if not result.get("ok"):
                    print(
                        "[reader-error] "
                        f"workers={workers} "
                        f"url={result.get('url', '')!r} "
                        f"status_code={result.get('status_code')!r} "
                        f"error_type={result.get('error_type', '')!r} "
                        f"error={result.get('error', '')!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                results.append({key: value for key, value in result.items() if key != "content"})
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
    wall_s = time.perf_counter() - started
    summary = _latency_summary(results, wall_s=wall_s, workers=workers)
    summary["reader_documents_jsonl"] = str(output_path)
    print(
        f"[reader] workers={workers} success={summary['successes']}/{summary['requests']} "
        f"wall_s={wall_s:.3f} rps={summary['throughput_rps']:.2f} "
        f"mean_s={summary['latency_mean_s']:.3f} p95_s={summary['latency_p95_s']:.3f}",
        file=sys.stderr,
        flush=True,
    )
    return summary


def _summarize_one(item: dict[str, Any], *, goal: str) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        from synthesis.sft.tools import summarize_with_qwen

        content = str(item.get("content") or "")
        if not content.strip():
            raise ValueError("empty reader content")
        summary = summarize_with_qwen(content=content, goal=goal, assistant_output="")
        return {
            "url": item.get("url", ""),
            "query": item.get("query", ""),
            "ok": bool(str(summary or "").strip()),
            "input_chars": len(content),
            "output_chars": len(str(summary or "")),
            "elapsed_s": time.perf_counter() - started,
            "error": None if str(summary or "").strip() else "empty summarizer output",
        }
    except Exception as exc:  # pragma: no cover - live model dependent
        return {
            "url": item.get("url", ""),
            "query": item.get("query", ""),
            "ok": False,
            "input_chars": len(str(item.get("content") or "")),
            "output_chars": 0,
            "elapsed_s": time.perf_counter() - started,
            "error_type": type(exc).__name__,
            "error": _short_error(exc),
        }


def _run_summarizer_level(
    documents_path: Path,
    *,
    workers: int,
    goal: str,
    output_path: Path,
) -> dict[str, Any]:
    documents = [item for item in _load_jsonl(documents_path) if item.get("ok") and item.get("content")]
    print(
        f"[summarizer] workers={workers} requests={len(documents)} input={documents_path}",
        file=sys.stderr,
        flush=True,
    )
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as output:
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"summarizer-{workers}") as executor:
            futures = [executor.submit(_summarize_one, item, goal=goal) for item in documents]
            for future in as_completed(futures):
                result = future.result()
                if not result.get("ok"):
                    print(
                        "[summarizer-error] "
                        f"workers={workers} "
                        f"url={result.get('url', '')!r} "
                        f"error_type={result.get('error_type', '')!r} "
                        f"error={result.get('error', '')!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                results.append(result)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
    wall_s = time.perf_counter() - started
    summary = _latency_summary(results, wall_s=wall_s, workers=workers)
    summary["summarizer_results_jsonl"] = str(output_path)
    print(
        f"[summarizer] workers={workers} success={summary['successes']}/{summary['requests']} "
        f"wall_s={wall_s:.3f} rps={summary['throughput_rps']:.2f} "
        f"mean_s={summary['latency_mean_s']:.3f} p95_s={summary['latency_p95_s']:.3f}",
        file=sys.stderr,
        flush=True,
    )
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", default="all", help="Comma-separated: search,reader,summarizer,all")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Environment file loaded before Serper/reader calls (default: synthesis/.env).",
    )
    parser.add_argument(
        "--override-env",
        action="store_true",
        help="Let --env-file override variables already present in the shell.",
    )
    parser.add_argument("--question-list", type=Path, default=DEFAULT_QUESTION_LIST)
    parser.add_argument("--url-list", type=Path, help="Existing urls.jsonl; skips Serper collection.")
    parser.add_argument("--reader-documents", type=Path, help="Existing reader result JSONL for summarizer-only mode.")
    parser.add_argument("--search-count", type=int, default=DEFAULT_SEARCH_COUNT)
    parser.add_argument("--results-per-search", type=int, default=DEFAULT_RESULTS_PER_SEARCH)
    parser.add_argument("--search-workers", type=int, default=DEFAULT_SEARCH_WORKERS)
    parser.add_argument("--serper-timeout-s", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=52)
    parser.add_argument("--max-urls", type=int, default=1000)
    parser.add_argument("--exclude-domain", action="append", default=list(_DEFAULT_EXCLUDED_DOMAINS))
    parser.add_argument("--workers", default=",".join(str(value) for value in DEFAULT_WORKERS))
    parser.add_argument("--reader-url", default=DEFAULT_READER_URL)
    parser.add_argument("--reader-timeout-s", type=float, default=DEFAULT_READER_TIMEOUT_S)
    parser.add_argument("--no-reader-refresh", action="store_true", help="Do not add refresh=1 to 8004 requests.")
    parser.add_argument(
        "--summarizer-goal",
        default="Extract the main factual content of this webpage faithfully and remove navigation noise.",
    )
    parser.add_argument("--model-config", help="Set SYNTHESIS_MODEL_CONFIG before importing the model worker.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    env_path = args.env_file.expanduser()
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    try:
        from synthesis.run_min_graph import load_env_file

        loaded_env = load_env_file(env_path, override=args.override_env)
    except Exception as exc:
        parser.error(f"failed to load --env-file {env_path}: {exc}")
    try:
        workers = _parse_int_list(args.workers)
    except ValueError as exc:
        parser.error(str(exc))
    if args.model_config:
        os.environ["SYNTHESIS_MODEL_CONFIG"] = str(args.model_config)
    if args.search_count <= 0 or args.results_per_search <= 0 or args.max_urls <= 0:
        parser.error("search-count, results-per-search, and max-urls must be positive")
    if args.search_workers <= 0 or args.reader_timeout_s <= 0 or args.serper_timeout_s <= 0:
        parser.error("worker counts and timeouts must be positive")

    requested = {item.strip().lower() for item in args.stages.split(",") if item.strip()}
    if "all" in requested:
        requested = {"search", "reader", "summarizer"}
    if not requested.issubset({"search", "reader", "summarizer"}):
        parser.error("--stages must contain search, reader, summarizer, or all")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "config": {
            "stages": sorted(requested),
            "workers": workers,
            "reader_url": args.reader_url,
            "reader_refresh": not args.no_reader_refresh,
            "env_file": str(env_path),
            "loaded_env_count": len(loaded_env),
            "question_list": str(args.question_list),
            "max_urls": args.max_urls,
        },
        "search": None,
        "reader": [],
        "summarizer": [],
    }

    urls: list[dict[str, Any]] = []
    if args.url_list and args.url_list.exists():
        urls = _load_jsonl(args.url_list)
        urls = [item for item in urls if str(item.get("url") or "").startswith(("http://", "https://"))][: args.max_urls]
        report["search"] = {"skipped": True, "urls_jsonl": str(args.url_list), "unique_urls_after_filter": len(urls)}
    elif "search" in requested:
        urls, report["search"] = _collect_urls(args, args.output_dir)
    elif "reader" in requested:
        default_urls = args.output_dir / "urls.jsonl"
        if not default_urls.exists():
            parser.error("reader stage requires --url-list or an existing output-dir/urls.jsonl")
        urls = _load_jsonl(default_urls)[: args.max_urls]
    elif "summarizer" not in requested:
        parser.error("no stage to run")

    selected_documents: Path | None = args.reader_documents
    if "reader" in requested:
        if not urls:
            parser.error("reader stage has no URLs")
        for worker_count in workers:
            path = args.output_dir / f"reader_results_w{worker_count}.jsonl"
            report["reader"].append(
                _run_reader_level(
                    urls,
                    workers=worker_count,
                    reader_url=args.reader_url,
                    timeout_s=args.reader_timeout_s,
                    refresh=not args.no_reader_refresh,
                    output_path=path,
                )
            )
        selected_documents = args.output_dir / f"reader_results_w{max(workers)}.jsonl"

    if "summarizer" in requested:
        if selected_documents is None or not selected_documents.exists():
            parser.error("summarizer stage requires --reader-documents or a reader stage")
        # Fail before launching a large benchmark if the summarizer alias was
        # not registered.  ``summarize_with_qwen`` intentionally has a cheap
        # content-truncation fallback for normal tool execution, but using it
        # here would produce a misleading latency benchmark with no LLM calls.
        from synthesis.sft import tools as sft_tools

        summarizer_alias = sft_tools._summarizer_model_alias()
        if not summarizer_alias:
            parser.error(
                "No registered summarizer model alias. Set SFT_QWEN_MODEL_ALIAS "
                "and SYNTHESIS_MODEL_CONFIG before running the summarizer stage."
            )
        report["config"]["summarizer_model_alias"] = summarizer_alias
        # The summarizer should run against one fixed reader snapshot; only
        # the summarizer worker level changes between benchmark rows.
        for worker_count in workers:
            path = args.output_dir / f"summarizer_results_w{worker_count}.jsonl"
            report["summarizer"].append(
                _run_summarizer_level(
                    selected_documents,
                    workers=worker_count,
                    goal=args.summarizer_goal,
                    output_path=path,
                )
            )
        report["config"]["summarizer_input"] = str(selected_documents)

    report_path = args.output_dir / "benchmark_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[pipeline] report={report_path}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
