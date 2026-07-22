#!/usr/bin/env python3
"""Concurrent load test for the local Enhanced Reader service on port 8004.

Example:
  python3 utils/benchmark_enhanced_reader.py \
    --base-url http://127.0.0.1:8004 \
    --count 100 \
    --concurrency 100

# Exercise forced-cache-miss, same-page cache-hit, and mixed traffic:
  python3 utils/benchmark_enhanced_reader.py --scenario mixed \
    --count 20 --hot-requests 40 --mixed-cold-count 20 --concurrency 16
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx


# A curated pool of relatively long Wikipedia pages. We sample from this pool by seed.
LONG_WIKIPEDIA_URLS: list[str] = [
    "https://en.wikipedia.org/wiki/World_War_II",
    "https://en.wikipedia.org/wiki/World_War_I",
    "https://en.wikipedia.org/wiki/United_States",
    "https://en.wikipedia.org/wiki/China",
    "https://en.wikipedia.org/wiki/India",
    "https://en.wikipedia.org/wiki/Russia",
    "https://en.wikipedia.org/wiki/Japan",
    "https://en.wikipedia.org/wiki/Germany",
    "https://en.wikipedia.org/wiki/United_Kingdom",
    "https://en.wikipedia.org/wiki/France",
    "https://en.wikipedia.org/wiki/History_of_the_United_States",
    "https://en.wikipedia.org/wiki/History_of_China",
    "https://en.wikipedia.org/wiki/History_of_India",
    "https://en.wikipedia.org/wiki/Roman_Empire",
    "https://en.wikipedia.org/wiki/Byzantine_Empire",
    "https://en.wikipedia.org/wiki/Mongol_Empire",
    "https://en.wikipedia.org/wiki/Ottoman_Empire",
    "https://en.wikipedia.org/wiki/Qing_dynasty",
    "https://en.wikipedia.org/wiki/Han_dynasty",
    "https://en.wikipedia.org/wiki/Tang_dynasty",
    "https://en.wikipedia.org/wiki/Three_Kingdoms",
    "https://en.wikipedia.org/wiki/Chinese_Civil_War",
    "https://en.wikipedia.org/wiki/Cold_War",
    "https://en.wikipedia.org/wiki/French_Revolution",
    "https://en.wikipedia.org/wiki/Industrial_Revolution",
    "https://en.wikipedia.org/wiki/Renaissance",
    "https://en.wikipedia.org/wiki/Ancient_Egypt",
    "https://en.wikipedia.org/wiki/Ancient_Greece",
    "https://en.wikipedia.org/wiki/Russia%E2%80%93Ukraine_war",
    "https://en.wikipedia.org/wiki/Israeli%E2%80%93Palestinian_conflict",
    "https://en.wikipedia.org/wiki/Islam",
    "https://en.wikipedia.org/wiki/Christianity",
    "https://en.wikipedia.org/wiki/Buddhism",
    "https://en.wikipedia.org/wiki/Hinduism",
    "https://en.wikipedia.org/wiki/Catholic_Church",
    "https://en.wikipedia.org/wiki/Bible",
    "https://en.wikipedia.org/wiki/Quran",
    "https://en.wikipedia.org/wiki/United_Nations",
    "https://en.wikipedia.org/wiki/European_Union",
    "https://en.wikipedia.org/wiki/NATO",
    "https://en.wikipedia.org/wiki/European_Union",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Computer_science",
    "https://en.wikipedia.org/wiki/Physics",
    "https://en.wikipedia.org/wiki/Chemistry",
    "https://en.wikipedia.org/wiki/Biology",
    "https://en.wikipedia.org/wiki/Mathematics",
    "https://en.wikipedia.org/wiki/Quantum_mechanics",
    "https://en.wikipedia.org/wiki/General_relativity",
    "https://en.wikipedia.org/wiki/Evolution",
    "https://en.wikipedia.org/wiki/DNA",
    "https://en.wikipedia.org/wiki/Solar_System",
    "https://en.wikipedia.org/wiki/Earth",
    "https://en.wikipedia.org/wiki/Moon",
    "https://en.wikipedia.org/wiki/Mars",
    "https://en.wikipedia.org/wiki/Black_hole",
    "https://en.wikipedia.org/wiki/Universe",
    "https://en.wikipedia.org/wiki/Climate_change",
    "https://en.wikipedia.org/wiki/Global_warming",
    "https://en.wikipedia.org/wiki/Weather",
    "https://en.wikipedia.org/wiki/Economics",
    "https://en.wikipedia.org/wiki/Capitalism",
    "https://en.wikipedia.org/wiki/Socialism",
    "https://en.wikipedia.org/wiki/Democracy",
    "https://en.wikipedia.org/wiki/Communism",
    "https://en.wikipedia.org/wiki/Philosophy",
    "https://en.wikipedia.org/wiki/Psychology",
    "https://en.wikipedia.org/wiki/Sociology",
    "https://en.wikipedia.org/wiki/English_language",
    "https://en.wikipedia.org/wiki/Chinese_language",
    "https://en.wikipedia.org/wiki/Spanish_language",
    "https://en.wikipedia.org/wiki/History_of_English",
    "https://en.wikipedia.org/wiki/Literature",
    "https://en.wikipedia.org/wiki/Music",
    "https://en.wikipedia.org/wiki/Film",
    "https://en.wikipedia.org/wiki/Television",
    "https://en.wikipedia.org/wiki/Video_game",
    "https://en.wikipedia.org/wiki/Internet",
    "https://en.wikipedia.org/wiki/World_Wide_Web",
    "https://en.wikipedia.org/wiki/Google",
    "https://en.wikipedia.org/wiki/Microsoft",
    "https://en.wikipedia.org/wiki/Apple_Inc.",
    "https://en.wikipedia.org/wiki/Amazon_(company)",
    "https://en.wikipedia.org/wiki/Meta_Platforms",
    "https://en.wikipedia.org/wiki/OpenAI",
    "https://en.wikipedia.org/wiki/New_York_City",
    "https://en.wikipedia.org/wiki/Beijing",
    "https://en.wikipedia.org/wiki/Shanghai",
    "https://en.wikipedia.org/wiki/Tokyo",
    "https://en.wikipedia.org/wiki/London",
    "https://en.wikipedia.org/wiki/Paris",
    "https://en.wikipedia.org/wiki/Los_Angeles",
    "https://en.wikipedia.org/wiki/Chicago",
    "https://en.wikipedia.org/wiki/Hong_Kong",
    "https://en.wikipedia.org/wiki/Kobe_Bryant",
    "https://en.wikipedia.org/wiki/Michael_Jordan",
    "https://en.wikipedia.org/wiki/LeBron_James",
    "https://en.wikipedia.org/wiki/Lionel_Messi",
    "https://en.wikipedia.org/wiki/Cristiano_Ronaldo",
    "https://en.wikipedia.org/wiki/Taylor_Swift",
    "https://en.wikipedia.org/wiki/Beyonc%C3%A9",
    "https://en.wikipedia.org/wiki/Barack_Obama",
    "https://en.wikipedia.org/wiki/Donald_Trump",
    "https://en.wikipedia.org/wiki/Joe_Biden",
    "https://en.wikipedia.org/wiki/Xi_Jinping",
    "https://en.wikipedia.org/wiki/Vladimir_Putin",
    "https://en.wikipedia.org/wiki/Albert_Einstein",
    "https://en.wikipedia.org/wiki/Isaac_Newton",
    "https://en.wikipedia.org/wiki/Charles_Darwin",
    "https://en.wikipedia.org/wiki/William_Shakespeare",
    "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
    "https://en.wikipedia.org/wiki/Napoleon",
    "https://en.wikipedia.org/wiki/Julius_Caesar",
    "https://en.wikipedia.org/wiki/Abraham_Lincoln",
    "https://en.wikipedia.org/wiki/Adolf_Hitler",
    "https://en.wikipedia.org/wiki/Mahatma_Gandhi",
    "https://en.wikipedia.org/wiki/Mao_Zedong",
    "https://en.wikipedia.org/wiki/Elon_Musk",
]


@dataclass(slots=True)
class RequestResult:
    phase: str
    url: str
    ok: bool
    status_code: int | None
    elapsed_s: float
    content_chars: int = 0
    error: str | None = None
    server_error_message: str | None = None
    response_preview: str | None = None
    debug_total_s: float | None = None
    debug_fetch_parallel_s: float | None = None
    debug_readerlm_s: float | None = None
    debug_payload: dict[str, Any] | None = None
    failure_stage: str | None = None
    failure_exception_type: str | None = None
    failure_upstream_status_code: int | None = None
    failure_upstream_url: str | None = None
    cache_hit: bool | None = None
    content_source: str | None = None
    content_quality: str | None = None
    raw_markdown_chars: int | None = None
    anomaly: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Enhanced Reader service with concurrent Wikipedia reads.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--count", type=int, default=100, help="Number of Wikipedia pages to request.")
    parser.add_argument("--concurrency", type=int, default=100, help="Maximum concurrent requests.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to sample pages.")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write full results JSON.")
    parser.add_argument(
        "--scenario",
        choices=("unique", "mixed"),
        default="unique",
        help="unique: concurrent distinct pages; mixed: forced-cache-miss, hot same-page, then mixed traffic.",
    )
    parser.add_argument(
        "--hot-url",
        default="https://en.wikipedia.org/wiki/ByteDance",
        help="Page repeated by the hot-cache phase when --scenario=mixed.",
    )
    parser.add_argument(
        "--hot-requests",
        type=int,
        default=20,
        help="Number of concurrent same-page requests in the hot phase.",
    )
    parser.add_argument(
        "--mixed-cold-count",
        type=int,
        default=20,
        help="Number of forced-cache-miss pages included alongside hot requests in the mixed phase.",
    )
    parser.add_argument(
        "--urls-file",
        type=str,
        default="",
        help="Optional text file with one URL per line. If set, uses this pool instead of built-in URLs.",
    )
    return parser.parse_args()


def load_url_pool(args: argparse.Namespace) -> list[str]:
    if not args.urls_file:
        return list(LONG_WIKIPEDIA_URLS)
    path = Path(args.urls_file)
    urls: list[str] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    if not urls:
        raise ValueError(f"No URLs found in {path}")
    return urls


def choose_urls(pool: list[str], count: int, seed: int) -> list[str]:
    if count <= 0:
        raise ValueError("--count must be positive")
    if len(pool) < count:
        raise ValueError(f"URL pool only has {len(pool)} items, but --count={count}")
    rng = random.Random(seed)
    urls = list(pool)
    rng.shuffle(urls)
    return urls[:count]


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


async def fetch_one(
    client: httpx.AsyncClient,
    base_url: str,
    url: str,
    timeout_s: float,
    *,
    phase: str,
    refresh: bool = False,
) -> RequestResult:
    started = time.perf_counter()
    try:
        response = await client.get(
            f"{base_url.rstrip('/')}/{url}",
            headers={"Accept": "application/json"},
            params={"refresh": "1"} if refresh else None,
            timeout=timeout_s,
        )
        elapsed_s = time.perf_counter() - started
        payload: dict[str, Any] | None = None
        response_preview: str | None = None
        try:
            payload = response.json()
        except Exception:
            text = response.text
            response_preview = text[:1000] if text else None
        if response.is_error:
            debug = {}
            if isinstance(payload, dict):
                debug = payload.get("debug_timing") or {}
            return RequestResult(
                phase=phase,
                url=url,
                ok=False,
                status_code=response.status_code,
                elapsed_s=elapsed_s,
                error=f"HTTPStatusError: Server error '{response.status_code} {response.reason_phrase}' for url '{response.url}'",
                server_error_message=payload.get("message") if isinstance(payload, dict) else None,
                response_preview=response_preview,
                debug_total_s=float(debug["total_s"]) if debug.get("total_s") is not None else None,
                debug_fetch_parallel_s=(
                    float(debug["fetch_markdown_html_parallel_s"])
                    if debug.get("fetch_markdown_html_parallel_s") is not None
                    else None
                ),
                debug_readerlm_s=float(debug["readerlm_s"]) if debug.get("readerlm_s") is not None else None,
                debug_payload=debug if isinstance(debug, dict) and debug else None,
                failure_stage=debug.get("failure_stage") if isinstance(debug, dict) else None,
                failure_exception_type=debug.get("failure_exception_type") if isinstance(debug, dict) else None,
                failure_upstream_status_code=(
                    int(debug["failure_upstream_status_code"])
                    if isinstance(debug, dict) and debug.get("failure_upstream_status_code") is not None
                    else None
                ),
                failure_upstream_url=debug.get("failure_upstream_url") if isinstance(debug, dict) else None,
            )
        assert isinstance(payload, dict)
        data = payload.get("data") or {}
        debug = payload.get("debug_timing") or data.get("debug_timing") or {}
        content = data.get("content") or ""
        raw_markdown_chars = debug.get("raw_markdown_chars")
        anomaly = None
        if not content.strip():
            anomaly = "empty_content"
        elif raw_markdown_chars == 0 and str(data.get("content_source") or "").endswith("readerlm_content"):
            anomaly = "readerlm_output_with_empty_raw_input"
        return RequestResult(
            phase=phase,
            url=url,
            ok=True,
            status_code=response.status_code,
            elapsed_s=elapsed_s,
            content_chars=len(content),
            debug_total_s=float(debug["total_s"]) if debug.get("total_s") is not None else None,
            debug_fetch_parallel_s=(
                float(debug["fetch_markdown_html_parallel_s"])
                if debug.get("fetch_markdown_html_parallel_s") is not None
                else None
            ),
            debug_readerlm_s=float(debug["readerlm_s"]) if debug.get("readerlm_s") is not None else None,
            debug_payload=debug if isinstance(debug, dict) and debug else None,
            cache_hit=bool(debug.get("cache_hit")) if isinstance(debug, dict) and "cache_hit" in debug else None,
            content_source=data.get("content_source"),
            content_quality=data.get("content_quality"),
            raw_markdown_chars=int(raw_markdown_chars) if raw_markdown_chars is not None else None,
            anomaly=anomaly,
        )
    except Exception as exc:
        return RequestResult(
            phase=phase,
            url=url,
            ok=False,
            status_code=getattr(getattr(exc, "response", None), "status_code", None),
            elapsed_s=time.perf_counter() - started,
            error=f"{exc.__class__.__name__}: {exc}",
        )


async def run_benchmark(
    *,
    base_url: str,
    requests: list[tuple[str, bool]],
    concurrency: int,
    timeout_s: float,
    phase: str,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[RequestResult] = []
    done_count = 0
    started_at = time.perf_counter()

    async with httpx.AsyncClient() as client:
        async def guarded(url: str, refresh: bool) -> RequestResult:
            async with semaphore:
                return await fetch_one(client, base_url, url, timeout_s, phase=phase, refresh=refresh)

        tasks = [asyncio.create_task(guarded(url, refresh)) for url, refresh in requests]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(requests):
                elapsed = time.perf_counter() - started_at
                print(
                    f"[reader-bench] phase={phase} completed={done_count}/{len(requests)} "
                    f"elapsed_s={elapsed:.2f} "
                    f"avg_req_per_s={(done_count / elapsed) if elapsed > 0 else 0.0:.2f}",
                    flush=True,
                )
    return results


def print_summary(results: list[RequestResult], wall_s: float, concurrency: int, *, phase: str) -> None:
    ok_results = [item for item in results if item.ok]
    failed_results = [item for item in results if not item.ok]
    elapsed_values = [item.elapsed_s for item in ok_results]
    readerlm_values = [item.debug_readerlm_s for item in ok_results if item.debug_readerlm_s is not None]
    fetch_values = [item.debug_fetch_parallel_s for item in ok_results if item.debug_fetch_parallel_s is not None]
    total_values = [item.debug_total_s for item in ok_results if item.debug_total_s is not None]
    content_values = [item.content_chars for item in ok_results]

    print(f"=== enhanced reader benchmark summary: {phase} ===")
    print(f"requests: {len(results)}")
    print(f"concurrency: {concurrency}")
    print(f"success: {len(ok_results)}")
    print(f"failed: {len(failed_results)}")
    print(f"wall_time_s: {wall_s:.3f}")
    print(f"throughput_req_per_s: {(len(results) / wall_s) if wall_s > 0 else 0.0:.3f}")
    if elapsed_values:
        print(
            "client_latency_s: "
            f"avg={statistics.mean(elapsed_values):.3f} "
            f"p50={_percentile(elapsed_values, 0.50):.3f} "
            f"p95={_percentile(elapsed_values, 0.95):.3f} "
            f"max={max(elapsed_values):.3f}"
        )
    if total_values:
        print(
            "server_total_s: "
            f"avg={statistics.mean(total_values):.3f} "
            f"p50={_percentile(total_values, 0.50):.3f} "
            f"p95={_percentile(total_values, 0.95):.3f} "
            f"max={max(total_values):.3f}"
        )
    if fetch_values:
        print(
            "server_fetch_markdown_html_parallel_s: "
            f"avg={statistics.mean(fetch_values):.3f} "
            f"p50={_percentile(fetch_values, 0.50):.3f} "
            f"p95={_percentile(fetch_values, 0.95):.3f} "
            f"max={max(fetch_values):.3f}"
        )
    if readerlm_values:
        print(
            "server_readerlm_s: "
            f"avg={statistics.mean(readerlm_values):.3f} "
            f"p50={_percentile(readerlm_values, 0.50):.3f} "
            f"p95={_percentile(readerlm_values, 0.95):.3f} "
            f"max={max(readerlm_values):.3f}"
        )
    if content_values:
        print(
            "content_chars: "
            f"avg={statistics.mean(content_values):.1f} "
            f"p50={_percentile([float(v) for v in content_values], 0.50):.0f} "
            f"p95={_percentile([float(v) for v in content_values], 0.95):.0f} "
            f"max={max(content_values)}"
        )
    cache_values = [item.cache_hit for item in ok_results if item.cache_hit is not None]
    if cache_values:
        print(f"cache_hits: {sum(cache_values)}/{len(cache_values)}")
    source_counts = Counter(item.content_source or "unknown" for item in ok_results)
    if source_counts:
        print(f"content_source_counts: {dict(source_counts)}")
    anomaly_counts = Counter(item.anomaly for item in results if item.anomaly)
    if anomaly_counts:
        print(f"anomaly_counts: {dict(anomaly_counts)}")
        print("sample_anomalies:")
        for item in [value for value in results if value.anomaly][:10]:
            print(
                f"  phase={item.phase} anomaly={item.anomaly} status={item.status_code} "
                f"content_chars={item.content_chars} raw_markdown_chars={item.raw_markdown_chars} url={item.url}"
            )
    if failed_results:
        stage_counts = Counter(item.failure_stage or "unknown" for item in failed_results)
        exception_counts = Counter(item.failure_exception_type or "unknown" for item in failed_results)
        print(f"failure_stage_counts: {dict(stage_counts)}")
        print(f"failure_exception_type_counts: {dict(exception_counts)}")
        print("sample_failures:")
        for item in failed_results[:10]:
            print(
                f"  status={item.status_code!r} elapsed_s={item.elapsed_s:.3f} "
                f"url={item.url} error={item.error}"
            )
            if item.failure_stage or item.failure_exception_type:
                print(
                    "    failure_meta="
                    f"stage={item.failure_stage!r} "
                    f"exception_type={item.failure_exception_type!r} "
                    f"upstream_status_code={item.failure_upstream_status_code!r} "
                    f"upstream_url={item.failure_upstream_url!r}"
                )
            if item.server_error_message:
                print(f"    message={item.server_error_message}")
            if item.debug_total_s is not None or item.debug_fetch_parallel_s is not None or item.debug_readerlm_s is not None:
                print(
                    "    debug_timing="
                    f"total_s={item.debug_total_s!r} "
                    f"fetch_parallel_s={item.debug_fetch_parallel_s!r} "
                    f"readerlm_s={item.debug_readerlm_s!r}"
                )
            if item.debug_payload:
                print(f"    debug_payload={json.dumps(item.debug_payload, ensure_ascii=False)}")
            if item.response_preview:
                print(f"    response_preview={item.response_preview!r}")


def maybe_write_json(path: str, results: list[RequestResult], wall_s: float, args: argparse.Namespace) -> None:
    if not path:
        return
    output = {
        "base_url": args.base_url,
        "count": args.count,
        "concurrency": args.concurrency,
        "timeout": args.timeout,
        "seed": args.seed,
        "wall_time_s": wall_s,
        "results": [
            {
                "url": item.url,
                "phase": item.phase,
                "ok": item.ok,
                "status_code": item.status_code,
                "elapsed_s": item.elapsed_s,
                "content_chars": item.content_chars,
                "error": item.error,
                "server_error_message": item.server_error_message,
                "response_preview": item.response_preview,
                "debug_total_s": item.debug_total_s,
                "debug_fetch_parallel_s": item.debug_fetch_parallel_s,
                "debug_readerlm_s": item.debug_readerlm_s,
                "debug_payload": item.debug_payload,
                "failure_stage": item.failure_stage,
                "failure_exception_type": item.failure_exception_type,
                "failure_upstream_status_code": item.failure_upstream_status_code,
                "failure_upstream_url": item.failure_upstream_url,
                "cache_hit": item.cache_hit,
                "content_source": item.content_source,
                "content_quality": item.content_quality,
                "raw_markdown_chars": item.raw_markdown_chars,
                "anomaly": item.anomaly,
            }
            for item in results
        ],
    }
    Path(path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reader-bench] wrote_json={path}")


def main() -> int:
    args = parse_args()
    pool = load_url_pool(args)
    required_urls = args.count + (args.mixed_cold_count if args.scenario == "mixed" else 0)
    urls = choose_urls(pool, required_urls, args.seed)

    print("=== enhanced reader benchmark ===")
    print(f"base_url: {args.base_url}")
    print(f"pool_size: {len(pool)}")
    print(f"scenario: {args.scenario}")
    print(f"cold_page_count: {args.count}")
    print(f"concurrency: {args.concurrency}")
    print(f"timeout_s: {args.timeout}")
    print(f"seed: {args.seed}")
    print("sample_urls:")
    for item in urls[:10]:
        print(f"  - {item}")

    phase_specs: list[tuple[str, list[tuple[str, bool]]]]
    if args.scenario == "unique":
        phase_specs = [("unique", [(url, False) for url in urls])]
    else:
        cold_urls = urls[: args.count]
        mixed_cold_urls = urls[args.count :]
        phase_specs = [
            ("forced_cold", [(url, True) for url in cold_urls]),
            ("prime_hot", [(args.hot_url, True)]),
            ("hot_same_page", [(args.hot_url, False) for _ in range(args.hot_requests)]),
            (
                "mixed_hot_and_cold",
                [(args.hot_url, False) for _ in range(args.hot_requests)] + [(url, True) for url in mixed_cold_urls],
            ),
        ]

    all_results: list[RequestResult] = []
    total_wall_s = 0.0
    for phase, requests in phase_specs:
        if not requests:
            continue
        started = time.perf_counter()
        results = asyncio.run(
            run_benchmark(
                base_url=args.base_url,
                requests=requests,
                concurrency=args.concurrency,
                timeout_s=args.timeout,
                phase=phase,
            )
        )
        wall_s = time.perf_counter() - started
        total_wall_s += wall_s
        print_summary(results, wall_s, args.concurrency, phase=phase)
        all_results.extend(results)
    maybe_write_json(args.output_json, all_results, total_wall_s, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
