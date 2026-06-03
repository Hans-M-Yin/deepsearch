#!/usr/bin/env python3
"""Small load test for the local Enhanced Reader service on port 8004.

Example:
  python3 utils/benchmark_enhanced_reader.py \
    --base-url http://127.0.0.1:8004 \
    --count 100 \
    --concurrency 100
"""

from __future__ import annotations

import argparse
import asyncio
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
    url: str
    ok: bool
    status_code: int | None
    elapsed_s: float
    content_chars: int = 0
    error: str | None = None
    debug_total_s: float | None = None
    debug_fetch_parallel_s: float | None = None
    debug_readerlm_s: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the Enhanced Reader service with concurrent Wikipedia reads.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL.")
    parser.add_argument("--count", type=int, default=100, help="Number of Wikipedia pages to request.")
    parser.add_argument("--concurrency", type=int, default=100, help="Maximum concurrent requests.")
    parser.add_argument("--timeout", type=float, default=300.0, help="Per-request timeout in seconds.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used to sample pages.")
    parser.add_argument("--output-json", type=str, default="", help="Optional path to write full results JSON.")
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
) -> RequestResult:
    started = time.perf_counter()
    try:
        response = await client.get(
            f"{base_url.rstrip('/')}/{url}",
            headers={"Accept": "application/json"},
            timeout=timeout_s,
        )
        elapsed_s = time.perf_counter() - started
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        data = payload.get("data") or {}
        debug = payload.get("debug_timing") or data.get("debug_timing") or {}
        content = data.get("content") or ""
        return RequestResult(
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
        )
    except Exception as exc:
        return RequestResult(
            url=url,
            ok=False,
            status_code=getattr(getattr(exc, "response", None), "status_code", None),
            elapsed_s=time.perf_counter() - started,
            error=f"{exc.__class__.__name__}: {exc}",
        )


async def run_benchmark(
    *,
    base_url: str,
    urls: list[str],
    concurrency: int,
    timeout_s: float,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    results: list[RequestResult] = []
    done_count = 0
    started_at = time.perf_counter()

    async with httpx.AsyncClient() as client:
        async def guarded(url: str) -> RequestResult:
            async with semaphore:
                return await fetch_one(client, base_url, url, timeout_s)

        tasks = [asyncio.create_task(guarded(url)) for url in urls]
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            done_count += 1
            if done_count % 10 == 0 or done_count == len(urls):
                elapsed = time.perf_counter() - started_at
                print(
                    f"[reader-bench] completed={done_count}/{len(urls)} "
                    f"elapsed_s={elapsed:.2f} "
                    f"avg_req_per_s={(done_count / elapsed) if elapsed > 0 else 0.0:.2f}",
                    flush=True,
                )
    return results


def print_summary(results: list[RequestResult], wall_s: float, concurrency: int) -> None:
    ok_results = [item for item in results if item.ok]
    failed_results = [item for item in results if not item.ok]
    elapsed_values = [item.elapsed_s for item in ok_results]
    readerlm_values = [item.debug_readerlm_s for item in ok_results if item.debug_readerlm_s is not None]
    fetch_values = [item.debug_fetch_parallel_s for item in ok_results if item.debug_fetch_parallel_s is not None]
    total_values = [item.debug_total_s for item in ok_results if item.debug_total_s is not None]
    content_values = [item.content_chars for item in ok_results]

    print("=== enhanced reader benchmark summary ===")
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
    if failed_results:
        print("sample_failures:")
        for item in failed_results[:10]:
            print(
                f"  status={item.status_code!r} elapsed_s={item.elapsed_s:.3f} "
                f"url={item.url} error={item.error}"
            )


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
                "ok": item.ok,
                "status_code": item.status_code,
                "elapsed_s": item.elapsed_s,
                "content_chars": item.content_chars,
                "error": item.error,
                "debug_total_s": item.debug_total_s,
                "debug_fetch_parallel_s": item.debug_fetch_parallel_s,
                "debug_readerlm_s": item.debug_readerlm_s,
            }
            for item in results
        ],
    }
    Path(path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[reader-bench] wrote_json={path}")


def main() -> int:
    args = parse_args()
    pool = load_url_pool(args)
    urls = choose_urls(pool, args.count, args.seed)

    print("=== enhanced reader benchmark ===")
    print(f"base_url: {args.base_url}")
    print(f"pool_size: {len(pool)}")
    print(f"request_count: {len(urls)}")
    print(f"concurrency: {args.concurrency}")
    print(f"timeout_s: {args.timeout}")
    print(f"seed: {args.seed}")
    print("sample_urls:")
    for item in urls[:10]:
        print(f"  - {item}")

    started = time.perf_counter()
    results = asyncio.run(
        run_benchmark(
            base_url=args.base_url,
            urls=urls,
            concurrency=args.concurrency,
            timeout_s=args.timeout,
        )
    )
    wall_s = time.perf_counter() - started
    print_summary(results, wall_s, args.concurrency)
    maybe_write_json(args.output_json, results, wall_s, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
