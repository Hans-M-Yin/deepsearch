#!/usr/bin/env python3
"""Run concurrent Firecrawl URL scrapes through the local key-pool backend.

The default workload submits 32 URL scrapes with 32 workers.  It includes the
URLs supplied for Firecrawl validation and fills the remaining slots with public
Wikipedia pages.  This script performs real, credit-consuming Firecrawl calls.

Examples:
  python debug/test_firecrawl.py
  python debug/test_firecrawl.py --workers 8 --requests 12
  python debug/test_firecrawl.py --failed-read-url-jsonl runs/.../0804_test_2_0_to_500.jsonl --requests 61
  python debug/test_firecrawl.py --output /tmp/firecrawl_report.json
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

from synthesis.firecrawl_client import FirecrawlClient


DEFAULT_URLS = [
    "https://www.reddit.com/r/todayilearned/comments/187sbo9/til_that_at_the_height_of_the_napoleonic_wars_the/",
    "https://www.jstor.org/stable/10.1525/phr.2008.77.4.553",
    "https://whitmanarchive.org/item/loc.05013",
    "https://whitmanarchive.org/item/loc.01278",
    "https://sanmames.athletic-club.eus/en/blog/why-san-mames-called-that/",
    "https://warfarehistorynetwork.com/article/derailing-case-blue/",
    "https://www.academia.edu/37548402/Operation_Kreml_German_Strategic_Deception_on_the_Eastern_Front_in_1942_in_Christopher_M_Rein_ed_Weaving_the_Tangled_Web_Military_Deception_in_Large_Scale_Combat_Operations",
    "https://en.wikipedia.org/wiki/World_War_II",
    "https://en.wikipedia.org/wiki/Cold_War",
    "https://en.wikipedia.org/wiki/Operation_Barbarossa",
    "https://en.wikipedia.org/wiki/Battle_of_Stalingrad",
    "https://en.wikipedia.org/wiki/History_of_the_United_Kingdom",
    "https://en.wikipedia.org/wiki/History_of_the_United_States",
    "https://en.wikipedia.org/wiki/History_of_China",
    "https://en.wikipedia.org/wiki/Roman_Empire",
    "https://en.wikipedia.org/wiki/Byzantine_Empire",
    "https://en.wikipedia.org/wiki/French_Revolution",
    "https://en.wikipedia.org/wiki/Industrial_Revolution",
    "https://en.wikipedia.org/wiki/Renaissance",
    "https://en.wikipedia.org/wiki/Ancient_Greece",
    "https://en.wikipedia.org/wiki/Ancient_Egypt",
    "https://en.wikipedia.org/wiki/United_Nations",
    "https://en.wikipedia.org/wiki/European_Union",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/Quantum_mechanics",
    "https://en.wikipedia.org/wiki/Climate_change",
    "https://en.wikipedia.org/wiki/Internet",
    "https://en.wikipedia.org/wiki/World_Wide_Web",
    "https://en.wikipedia.org/wiki/William_Shakespeare",
    "https://en.wikipedia.org/wiki/Leonardo_da_Vinci",
    "https://en.wikipedia.org/wiki/Marie_Curie",
    "https://en.wikipedia.org/wiki/Ada_Lovelace",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requests", type=int, default=32, help="Number of URLs to scrape (default: 32).")
    parser.add_argument("--workers", type=int, default=32, help="Concurrent Firecrawl requests (default: 32).")
    parser.add_argument("--max-age", type=int, default=172800000, help="Firecrawl max_age in milliseconds.")
    parser.add_argument("--no-main-content", action="store_true", help="Disable Firecrawl only_main_content.")
    parser.add_argument("--parsers", nargs="*", default=["pdf"], help="Firecrawl parsers (default: pdf).")
    parser.add_argument("--formats", nargs="*", default=["markdown"], help="Firecrawl formats (default: markdown).")
    parser.add_argument(
        "--show-raw",
        action="store_true",
        help="Print the complete raw Firecrawl response for every completed request.",
    )
    parser.add_argument(
        "--failed-read-url-jsonl",
        type=Path,
        help=(
            "Extract unique failed text-page URLs from a debug_vqa_batch JSONL file, "
            "using its saved runtime.url_resources."
        ),
    )
    parser.add_argument(
        "--include-failed-image-resources",
        action="store_true",
        help=(
            "Also test failed image resources from --failed-read-url-jsonl. Disabled by default because "
            "read_url downloads images directly and does not use the Firecrawl fallback for them."
        ),
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report path.")
    return parser


def _failed_read_urls_from_jsonl(path: Path, *, include_images: bool = False) -> list[str]:
    """Recover original URLs for failed read_url calls from persisted runtime provenance."""

    urls: list[str] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"[failed-read-url extraction] skip malformed JSONL at {path}:{line_number}: {exc}", file=sys.stderr)
                continue
            resources = {
                str(item.get("resource_id") or ""): item
                for item in ((record.get("runtime") or {}).get("url_resources") or [])
                if isinstance(item, dict)
            }
            for message in record.get("raw_messages") or []:
                if not isinstance(message, dict) or message.get("role") != "tool" or message.get("name") != "read_url":
                    continue
                try:
                    observation = json.loads(message.get("content") or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                if not isinstance(observation, dict) or observation.get("content") != "Unable to read the requested page.":
                    continue
                resource = resources.get(str(observation.get("page_id") or "")) or {}
                if not include_images and str(resource.get("kind") or "") == "image":
                    continue
                url = str(resource.get("primary_url") or "").strip()
                if url and url not in seen:
                    seen.add(url)
                    urls.append(url)
    return urls


def _metadata(response: dict[str, Any]) -> dict[str, Any]:
    data = response.get("data")
    if isinstance(data, dict) and isinstance(data.get("metadata"), dict):
        return data["metadata"]
    return response.get("metadata") if isinstance(response.get("metadata"), dict) else {}


def _metadata_value(metadata: dict[str, Any], camel_case: str, snake_case: str) -> Any:
    return metadata.get(camel_case, metadata.get(snake_case))


def _run_one(index: int, url: str, client: FirecrawlClient, args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        response = client.scrape(
            url,
            only_main_content=not args.no_main_content,
            max_age=args.max_age,
            parsers=args.parsers or None,
            formats=args.formats or None,
        )
    except Exception as exc:  # Provider/network failures are the purpose of this test.
        return {
            "index": index,
            "url": url,
            "ok": False,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    metadata = _metadata(response)
    error = response.get("error")
    content = response.get("data") if isinstance(response.get("data"), dict) else response
    markdown = content.get("markdown")
    return {
        "index": index,
        "url": url,
        "ok": not bool(error),
        "elapsed_s": round(time.perf_counter() - started, 3),
        "status_code": _metadata_value(metadata, "statusCode", "status_code"),
        "credits_used": _metadata_value(metadata, "creditsUsed", "credits_used"),
        "markdown_chars": len(markdown) if isinstance(markdown, str) else 0,
        "error": str(error) if error else None,
        "raw_response": response,
    }


def run_test(urls: list[str], args: argparse.Namespace) -> dict[str, Any]:
    client = FirecrawlClient()
    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="firecrawl-test") as executor:
        futures = [executor.submit(_run_one, index, url, client, args) for index, url in enumerate(urls, start=1)]
        for future in as_completed(futures):
            results.append(future.result())
    results.sort(key=lambda result: int(result["index"]))
    failures = [result for result in results if not result["ok"]]
    return {
        "request_count": len(results),
        "workers": args.workers,
        "wall_elapsed_s": round(time.perf_counter() - started, 3),
        "success_count": len(results) - len(failures),
        "failure_count": len(failures),
        "credits_used_total": sum(int(result.get("credits_used") or 0) for result in results),
        "error_types": dict(Counter(result.get("error_type", "FirecrawlError") for result in failures)),
        "results": results,
        "pool_status": client.key_pool.status(),
        "show_raw": args.show_raw,
    }


def _print_report(report: dict[str, Any]) -> None:
    print("Firecrawl concurrent test report")
    print(
        "requests={request_count} workers={workers} success={success_count} failure={failure_count} "
        "credits_used={credits_used_total} wall_elapsed_s={wall_elapsed_s}".format(**report)
    )
    for result in report["results"]:
        prefix = "OK" if result["ok"] else "FAIL"
        print(
            f"[{result['index']:02d}] {prefix:<4} elapsed_s={result['elapsed_s']:<7} "
            f"status={result.get('status_code')} credits={result.get('credits_used')} url={result['url']}"
        )
        if result.get("error"):
            print(f"     {result['error']}")
        if report["show_raw"] or result.get("status_code") is None or result.get("credits_used") is None:
            print("     raw_response:")
            print(json.dumps(result.get("raw_response"), ensure_ascii=False, indent=2, default=str))


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    urls = DEFAULT_URLS
    if args.failed_read_url_jsonl:
        urls = _failed_read_urls_from_jsonl(
            args.failed_read_url_jsonl,
            include_images=args.include_failed_image_resources,
        )
        if not urls:
            raise SystemExit(f"No failed read_url URLs found in {args.failed_read_url_jsonl}")
        print(f"[failed-read-url extraction] extracted_unique_urls={len(urls)} source={args.failed_read_url_jsonl}")
    if not 1 <= args.requests <= len(urls):
        raise SystemExit(f"--requests must be between 1 and {len(urls)}")
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    if args.max_age < 0:
        raise SystemExit("--max-age must be non-negative")

    report = run_test(urls[: args.requests], args)
    _print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 1 if report["failure_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
