"""Compare Enhanced Reader and Firecrawl on the same URL set.

The benchmark measures the backend fetch itself.  It does not run the
``summarize_with_qwen`` step used by ``sft.tools.read_url`` so that backend
latency and availability are not mixed with LLM latency.

Add URLs to ``URLS`` below, or provide them with ``--url-file``/positional
arguments.  A URL file contains one URL per line; blank lines and ``#``
comments are ignored.

Examples:

    python -m synthesis.vqa.debug.benchmark_read_url_backends
    python -m synthesis.vqa.debug.benchmark_read_url_backends \
      --url-file urls.txt --workers 1,2,4,8 --output reader_benchmark.json
    python -m synthesis.vqa.debug.benchmark_read_url_backends \
      https://example.com/a https://example.com/b --backends firecrawl
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time
from typing import Any, Callable, Iterable


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "synthesis.vqa.debug"

from synthesis.firecrawl_client import FirecrawlClient
from synthesis.wiki_text_builder import EnhancedReaderClient


# Fill this list later if you prefer not to pass --url-file or positional URLs.
URLS: list[str] = []

DEFAULT_WORKER_LEVELS = (1, 2, 4, 8)
DEFAULT_READER_TIMEOUT_S = 180.0
DEFAULT_FIRECRAWL_MAX_AGE_MS = 172800000


@dataclass
class ProbeResult:
    backend: str
    url: str
    ok: bool
    elapsed_s: float
    content_chars: int = 0
    title_chars: int = 0
    status_code: int | None = None
    credits_used: int | None = None
    error_type: str | None = None
    error: str | None = None


class InFlightTracker:
    """Track actual active probes, not only the configured worker count."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active = 0
        self._max_active = 0

    def enter(self) -> None:
        with self._lock:
            self._active += 1
            self._max_active = max(self._max_active, self._active)

    def leave(self) -> None:
        with self._lock:
            self._active = max(0, self._active - 1)

    @property
    def max_active(self) -> int:
        with self._lock:
            return self._max_active


def _short_error(exc: BaseException) -> str:
    message = str(exc).strip().replace("\n", " ")
    return message[:1000] if message else exc.__class__.__name__


def _status_code(value: Any) -> int | None:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if 100 <= value <= 599 else None


def _nested_payload(response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data")
    return data if isinstance(data, dict) else response


def _firecrawl_content(response: dict[str, Any]) -> tuple[str, str, int | None, int | None]:
    payload = _nested_payload(response)
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    content = str(payload.get("markdown") or payload.get("content") or "")
    title = str(metadata.get("title") or payload.get("title") or "")
    status_code = _status_code(
        metadata.get("statusCode", metadata.get("status_code", response.get("statusCode")))
    )
    credits_used: int | None
    try:
        credits_used = int(metadata.get("creditsUsed", metadata.get("credits_used")))
    except (TypeError, ValueError):
        credits_used = None
    return content, title, status_code, credits_used


def _probe_enhanced_reader(
    url: str,
    reader: EnhancedReaderClient,
    tracker: InFlightTracker,
) -> ProbeResult:
    started_at = time.perf_counter()
    tracker.enter()
    try:
        document = reader.read(url)
        content = str(document.content or "")
        title = str(document.title or "")
        return ProbeResult(
            backend="enhanced_reader",
            url=url,
            ok=True,
            elapsed_s=time.perf_counter() - started_at,
            content_chars=len(content),
            title_chars=len(title),
        )
    except Exception as exc:  # pragma: no cover - network dependent
        return ProbeResult(
            backend="enhanced_reader",
            url=url,
            ok=False,
            elapsed_s=time.perf_counter() - started_at,
            error_type=exc.__class__.__name__,
            error=_short_error(exc),
        )
    finally:
        tracker.leave()


def _probe_firecrawl(
    url: str,
    client: FirecrawlClient,
    tracker: InFlightTracker,
    max_age_ms: int | None,
) -> ProbeResult:
    started_at = time.perf_counter()
    tracker.enter()
    try:
        response = client.scrape(
            url,
            only_main_content=True,
            max_age=max_age_ms,
            parsers=["pdf"],
            formats=["markdown"],
        )
        if not isinstance(response, dict):
            raise TypeError(f"unexpected Firecrawl response type: {type(response).__name__}")
        content, title, status_code, credits_used = _firecrawl_content(response)
        error = str(response.get("error") or "").strip()
        if error:
            return ProbeResult(
                backend="firecrawl",
                url=url,
                ok=False,
                elapsed_s=time.perf_counter() - started_at,
                status_code=status_code,
                credits_used=credits_used,
                error_type="FirecrawlResponseError",
                error=error[:1000],
            )
        if response.get("success") is False or not content.strip():
            return ProbeResult(
                backend="firecrawl",
                url=url,
                ok=False,
                elapsed_s=time.perf_counter() - started_at,
                content_chars=len(content),
                title_chars=len(title),
                status_code=status_code,
                credits_used=credits_used,
                error_type="FirecrawlEmptyResponse",
                error="Firecrawl returned no markdown content.",
            )
        return ProbeResult(
            backend="firecrawl",
            url=url,
            ok=True,
            elapsed_s=time.perf_counter() - started_at,
            content_chars=len(content),
            title_chars=len(title),
            status_code=status_code,
            credits_used=credits_used,
        )
    except Exception as exc:  # pragma: no cover - network dependent
        return ProbeResult(
            backend="firecrawl",
            url=url,
            ok=False,
            elapsed_s=time.perf_counter() - started_at,
            error_type=exc.__class__.__name__,
            error=_short_error(exc),
        )
    finally:
        tracker.leave()


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
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _summarize_results(
    results: list[ProbeResult],
    *,
    configured_workers: int,
    max_active: int,
    wall_time_s: float,
) -> dict[str, Any]:
    latencies = [result.elapsed_s for result in results]
    successful_latencies = [result.elapsed_s for result in results if result.ok]
    successes = sum(result.ok for result in results)
    failures = len(results) - successes
    errors = Counter(
        f"{result.error_type}: {result.error or ''}".strip()
        for result in results
        if not result.ok
    )
    return {
        "requests": len(results),
        "successes": successes,
        "failures": failures,
        "success_rate": successes / len(results) if results else 0.0,
        "configured_workers": configured_workers,
        "max_in_flight": max_active,
        "wall_time_s": wall_time_s,
        "throughput_requests_per_s": len(results) / wall_time_s if wall_time_s > 0 else 0.0,
        "successful_throughput_per_s": successes / wall_time_s if wall_time_s > 0 else 0.0,
        "avg_latency_s": statistics.fmean(latencies) if latencies else None,
        "successful_avg_latency_s": statistics.fmean(successful_latencies) if successful_latencies else None,
        "p50_latency_s": _percentile(latencies, 0.50),
        "p90_latency_s": _percentile(latencies, 0.90),
        "p95_latency_s": _percentile(latencies, 0.95),
        "avg_content_chars": (
            statistics.fmean(result.content_chars for result in results if result.ok)
            if successful_latencies
            else None
        ),
        "total_credits_used": sum(result.credits_used or 0 for result in results),
        "top_errors": [
            {"error": error, "count": count}
            for error, count in errors.most_common(10)
        ],
    }


def _run_batch(
    *,
    backend: str,
    urls: list[str],
    workers: int,
    reader_base_url: str,
    reader_timeout_s: float,
    firecrawl_max_age_ms: int | None,
) -> dict[str, Any]:
    tracker = InFlightTracker()
    if backend == "enhanced_reader":
        reader = EnhancedReaderClient(
            base_url=reader_base_url,
            timeout_s=reader_timeout_s,
        )
        probe: Callable[[str], ProbeResult] = lambda url: _probe_enhanced_reader(url, reader, tracker)
    elif backend == "firecrawl":
        client = FirecrawlClient()
        probe = lambda url: _probe_firecrawl(url, client, tracker, firecrawl_max_age_ms)
    else:  # pragma: no cover - guarded by argparse
        raise ValueError(f"unknown backend: {backend}")

    started_at = time.perf_counter()
    results: list[ProbeResult] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"benchmark-{backend}") as executor:
        futures: dict[Future[ProbeResult], str] = {
            executor.submit(probe, url): url
            for url in urls
        }
        for future in as_completed(futures):
            results.append(future.result())
    wall_time_s = time.perf_counter() - started_at
    results.sort(key=lambda result: urls.index(result.url))
    return {
        "backend": backend,
        "workers": workers,
        "summary": _summarize_results(
            results,
            configured_workers=workers,
            max_active=tracker.max_active,
            wall_time_s=wall_time_s,
        ),
        "results": [asdict(result) for result in results],
    }


def _parse_worker_levels(raw: str) -> list[int]:
    values: list[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value <= 0:
            raise ValueError("worker counts must be positive integers")
        if value not in values:
            values.append(value)
    if not values:
        raise ValueError("at least one worker count is required")
    return values


def _load_urls(args: argparse.Namespace) -> list[str]:
    candidates: list[str] = []
    if args.url_file:
        candidates.extend(
            line.strip()
            for line in Path(args.url_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    candidates.extend(args.urls)
    if not candidates:
        candidates.extend(URLS)

    deduplicated: list[str] = []
    seen: set[str] = set()
    for url in candidates:
        normalized = str(url).strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            deduplicated.append(normalized)
    return deduplicated


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "urls",
        nargs="*",
        help="URLs to benchmark; otherwise use --url-file or the URLS list in this script.",
    )
    parser.add_argument(
        "--url-file",
        help="Text file containing one URL per line. Blank lines and # comments are ignored.",
    )
    parser.add_argument(
        "--backends",
        default="enhanced_reader,firecrawl",
        help="Comma-separated backends to run: enhanced_reader, firecrawl, or both.",
    )
    parser.add_argument(
        "--workers",
        default=",".join(str(value) for value in DEFAULT_WORKER_LEVELS),
        help="Comma-separated concurrency levels, e.g. 1,2,4,8.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Repeat the complete URL set at every backend/concurrency level.",
    )
    parser.add_argument(
        "--reader-base-url",
        default=os.environ.get("ENHANCED_READER_URL") or "http://127.0.0.1:8004",
        help="Enhanced Reader base URL; defaults to ENHANCED_READER_URL or http://127.0.0.1:8004.",
    )
    parser.add_argument(
        "--reader-timeout-s",
        type=float,
        default=DEFAULT_READER_TIMEOUT_S,
        help="Enhanced Reader request timeout in seconds.",
    )
    parser.add_argument(
        "--firecrawl-max-age-ms",
        type=int,
        default=DEFAULT_FIRECRAWL_MAX_AGE_MS,
        help="Firecrawl cache max_age in milliseconds; use --firecrawl-no-cache to disable it.",
    )
    parser.add_argument(
        "--firecrawl-no-cache",
        action="store_true",
        help="Pass max_age=None to Firecrawl instead of using the normal 48-hour cache.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path. Without it, the report is printed to stdout.",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    try:
        workers = _parse_worker_levels(args.workers)
    except ValueError as exc:
        print(f"argument error: {exc}", file=sys.stderr)
        return 2
    if args.repeats <= 0:
        print("argument error: --repeats must be positive", file=sys.stderr)
        return 2
    if args.reader_timeout_s <= 0:
        print("argument error: --reader-timeout-s must be positive", file=sys.stderr)
        return 2

    requested_backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    allowed_backends = {"enhanced_reader", "firecrawl"}
    if not requested_backends or any(item not in allowed_backends for item in requested_backends):
        print(
            "argument error: --backends must contain only enhanced_reader and/or firecrawl",
            file=sys.stderr,
        )
        return 2
    requested_backends = list(dict.fromkeys(requested_backends))

    urls = _load_urls(args)
    report: dict[str, Any] = {
        "config": {
            "url_count": len(urls),
            "worker_levels": workers,
            "repeats": args.repeats,
            "reader_base_url": args.reader_base_url,
            "reader_timeout_s": args.reader_timeout_s,
            "firecrawl_max_age_ms": None if args.firecrawl_no_cache else args.firecrawl_max_age_ms,
            "backends": requested_backends,
        },
        "benchmarks": [],
    }

    if not urls:
        print(
            "No URLs configured. Add URLs to URLS in this script, pass --url-file, or provide positional URLs.",
            file=sys.stderr,
        )
    else:
        for backend in requested_backends:
            for worker_count in workers:
                for repeat in range(1, args.repeats + 1):
                    print(
                        f"[benchmark] backend={backend} workers={worker_count} "
                        f"repeat={repeat}/{args.repeats} urls={len(urls)}",
                        file=sys.stderr,
                        flush=True,
                    )
                    report["benchmarks"].append(
                        _run_batch(
                            backend=backend,
                            urls=urls,
                            workers=worker_count,
                            reader_base_url=args.reader_base_url,
                            reader_timeout_s=args.reader_timeout_s,
                            firecrawl_max_age_ms=(
                                None if args.firecrawl_no_cache else args.firecrawl_max_age_ms
                            ),
                        )
                        | {"repeat": repeat}
                    )

        for backend in requested_backends:
            for repeat in range(1, args.repeats + 1):
                matching = [
                    item
                    for item in report["benchmarks"]
                    if item["backend"] == backend and item["repeat"] == repeat
                ]
                baseline = next(
                    (
                        item["summary"]["wall_time_s"]
                        for item in matching
                        if item["workers"] == 1 and item["summary"]["wall_time_s"] > 0
                    ),
                    None,
                )
                for item in matching:
                    if baseline is None:
                        item["summary"]["speedup_vs_workers_1"] = None
                        item["summary"]["scaling_efficiency_vs_workers_1"] = None
                        continue
                    speedup = baseline / item["summary"]["wall_time_s"]
                    item["summary"]["speedup_vs_workers_1"] = speedup
                    item["summary"]["scaling_efficiency_vs_workers_1"] = speedup / item["workers"]

    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"[benchmark] wrote {args.output}", file=sys.stderr)
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
