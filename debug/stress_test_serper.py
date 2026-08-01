#!/usr/bin/env python3
"""Run concurrent direct-Serper text searches and report transport failures.

By default this launches 32 distinct text searches concurrently against the
configured Serper endpoint. It uses ``SerperSearchClient`` directly, so it
exercises the same request path used by ``synthesis.sft.tools.t2t_search``.

Examples:
  python debug/stress_test_serper.py
  python debug/stress_test_serper.py --requests 16 --workers 16 --timeout-s 30
  SERPER_SEARCH_URL=https://google.serper.dev/search python debug/stress_test_serper.py --output /tmp/serper_stress.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from synthesis.search_client import SerperSearchClient


DEFAULT_QUERIES = [
    "1960 Rome Olympics women's 100 metres winner",
    "Apollo 11 lunar module pilot biography",
    "Machu Picchu UNESCO World Heritage Site history",
    "Marie Curie Nobel Prize chemistry 1911",
    "Great Barrier Reef marine biodiversity facts",
    "Shakespeare Hamlet first performance date",
    "Amazon rainforest annual rainfall climate",
    "Eiffel Tower construction completed year",
    "Ada Lovelace analytical engine notes",
    "Hubble Space Telescope launch mission",
    "Japanese tea ceremony historical origins",
    "Nelson Mandela Rivonia Trial sentence",
    "Mount Everest first successful ascent",
    "French Revolution Tennis Court Oath location",
    "Beethoven Symphony No. 9 premiere",
    "Galapagos Islands Charles Darwin voyage",
    "International Space Station first module launch",
    "Ancient Egyptian pyramids Giza construction",
    "World War II D-Day Normandy landing date",
    "Leonardo da Vinci Mona Lisa museum location",
    "CERN Large Hadron Collider first collisions",
    "Suez Canal opening ceremony 1869",
    "Roman Colosseum gladiatorial games history",
    "Nikola Tesla alternating current inventions",
    "Pacific Ocean deepest point Mariana Trench",
    "United Nations founding conference San Francisco",
    "Antarctica research stations climate data",
    "Kilimanjaro highest peak Africa elevation",
    "Impressionism art movement Claude Monet",
    "DNA double helix Watson Crick publication",
    "Silk Road trade routes ancient China",
    "Statue of Liberty dedication ceremony 1886",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--requests",
        type=int,
        default=32,
        help="Number of distinct predefined searches to launch concurrently (default: 32).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Thread-pool size. Defaults to --requests, producing one concurrent request per query.",
    )
    parser.add_argument("--timeout-s", type=float, default=60.0, help="Timeout for each Serper request.")
    parser.add_argument("--limit", type=int, default=5, help="Requested result count for each search.")
    parser.add_argument("--lang", default="en", help="Serper hl parameter.")
    parser.add_argument(
        "--search-url",
        default=None,
        help="Override SERPER_SEARCH_URL for this run.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path for the complete JSON report.",
    )
    return parser


def _run_one(
    *,
    index: int,
    query: str,
    client: SerperSearchClient,
    limit: int,
    lang: str,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    try:
        response = client.search_text(query, limit=limit, hl=lang)
    except Exception as exc:  # Remote service failures are the purpose of the test.
        return {
            "index": index,
            "query": query,
            "ok": False,
            "elapsed_s": round(time.perf_counter() - started_at, 3),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
    return {
        "index": index,
        "query": query,
        "ok": True,
        "elapsed_s": round(time.perf_counter() - started_at, 3),
        "status_code": response.status_code,
        "result_count": len(response.results),
        "engine": response.engine,
        "key_pool": response.metadata.get("serper_key_pool"),
    }


def run_stress_test(
    *,
    queries: list[str],
    workers: int,
    timeout_s: float,
    limit: int,
    lang: str,
    search_url: str | None,
) -> dict[str, Any]:
    """Run one direct Serper request for every supplied query."""

    client = SerperSearchClient(search_url=search_url, timeout_s=timeout_s)
    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="serper-stress") as executor:
        futures = {
            executor.submit(
                _run_one,
                index=index,
                query=query,
                client=client,
                limit=limit,
                lang=lang,
            ): index
            for index, query in enumerate(queries, start=1)
        }
        for future in as_completed(futures):
            results.append(future.result())

    results.sort(key=lambda item: int(item["index"]))
    failures = [item for item in results if not item["ok"]]
    elapsed_values = [float(item["elapsed_s"]) for item in results]
    return {
        "request_count": len(results),
        "workers": workers,
        "timeout_s": timeout_s,
        "limit": limit,
        "lang": lang,
        "search_url": client.search_url,
        "elapsed_s": round(time.perf_counter() - started_at, 3),
        "success_count": len(results) - len(failures),
        "failure_count": len(failures),
        "error_types": dict(Counter(str(item["error_type"]) for item in failures)),
        "request_elapsed_s": {
            "min": round(min(elapsed_values), 3) if elapsed_values else 0.0,
            "max": round(max(elapsed_values), 3) if elapsed_values else 0.0,
            "mean": round(sum(elapsed_values) / len(elapsed_values), 3) if elapsed_values else 0.0,
        },
        "results": results,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("=" * 88)
    print("Serper concurrent stress-test report")
    print(f"endpoint: {report['search_url']}")
    print(
        "requests={request_count} workers={workers} success={success_count} failure={failure_count} "
        "wall_elapsed_s={elapsed_s}".format(**report)
    )
    elapsed = report["request_elapsed_s"]
    print(f"request_elapsed_s: min={elapsed['min']} mean={elapsed['mean']} max={elapsed['max']}")
    if report["error_types"]:
        print(f"error_types: {json.dumps(report['error_types'], ensure_ascii=False, sort_keys=True)}")
    print("-" * 88)
    for item in report["results"]:
        if item["ok"]:
            print(
                f"[{item['index']:02d}] OK   elapsed_s={item['elapsed_s']:<7} "
                f"status={item['status_code']} results={item['result_count']} query={item['query']}"
            )
        else:
            print(
                f"[{item['index']:02d}] FAIL elapsed_s={item['elapsed_s']:<7} "
                f"type={item['error_type']} query={item['query']}\n"
                f"     {item['error']}"
            )
    print("=" * 88)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.requests <= 0:
        raise SystemExit("--requests must be positive")
    if args.requests > len(DEFAULT_QUERIES):
        raise SystemExit(f"--requests cannot exceed {len(DEFAULT_QUERIES)} predefined distinct queries")
    if args.limit <= 0:
        raise SystemExit("--limit must be positive")
    if args.timeout_s <= 0:
        raise SystemExit("--timeout-s must be positive")

    workers = args.workers if args.workers is not None else args.requests
    if workers <= 0:
        raise SystemExit("--workers must be positive")

    report = run_stress_test(
        queries=DEFAULT_QUERIES[: args.requests],
        workers=workers,
        timeout_s=args.timeout_s,
        limit=args.limit,
        lang=args.lang,
        search_url=args.search_url,
    )
    _print_report(report)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Detailed JSON report: {args.output}")
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
