#!/usr/bin/env python3
"""Exercise the synthesis image resolver on image-search results from gated sites.

The goal is to test the same resolver strategy used by the visual-plan image
pipeline before wiring it into SFT inference tools:

1. Search images with Serper for several domains that often block direct image
   downloads or redirect to a post/page.
2. For matching image-search results, call ImageDiscoveryBuilder._resolve_image_asset.
3. Report whether the resolver succeeded directly or via source-page recovery
   (og:image/twitter:image/display_url/image_url extraction).

Examples:
  python debug/test_image_download_resolver.py --max-results-per-case 3
  python debug/test_image_download_resolver.py --case 'instagram|site:instagram.com NASA photo|instagram.com,lookaside.instagram.com,cdninstagram.com'
  python debug/test_image_download_resolver.py --search-backend serper_adapter --pretty
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any
from urllib.parse import urlparse

# Allow running from the repository root as `python debug/...`.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig, ResolvedImageAsset
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.search_client import ImageSearchResult, SearchResponse, SerperAdapterSearchClient, SerperSearchClient


@dataclass(frozen=True)
class TestCase:
    name: str
    query: str
    domains: tuple[str, ...]


DEFAULT_CASES: tuple[TestCase, ...] = (
    TestCase(
        name="instagram",
        query="site:instagram.com NASA photograph",
        domains=("instagram.com", "www.instagram.com", "lookaside.instagram.com", "cdninstagram.com"),
    ),
    TestCase(
        name="facebook",
        query="site:facebook.com NASA photo",
        domains=("facebook.com", "www.facebook.com", "m.facebook.com", "lookaside.fbsbx.com", "fbsbx.com", "fbcdn.net"),
    ),
    TestCase(
        name="tiktok",
        query="site:tiktok.com official video thumbnail",
        domains=("tiktok.com", "www.tiktok.com", "tiktokcdn.com", "p16-sign.tiktokcdn-us.com"),
    ),
    TestCase(
        name="pinterest",
        query="site:pinterest.com vintage poster image",
        domains=("pinterest.com", "www.pinterest.com", "pinimg.com", "i.pinimg.com"),
    ),
    TestCase(
        name="x_twitter",
        query="site:x.com NASA photo OR site:twitter.com NASA photo",
        domains=("x.com", "www.x.com", "twitter.com", "www.twitter.com", "pbs.twimg.com", "twimg.com"),
    ),
)


class _UnusedSearchClient:
    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:  # pragma: no cover
        raise NotImplementedError("not used by this resolver test")

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:  # pragma: no cover
        raise NotImplementedError("not used by this resolver test")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _short(text: str | None, limit: int = 180) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= limit:
        return normalized
    return normalized[: max(0, limit - 3)].rstrip() + "..."


def _hostname(url: str | None) -> str:
    try:
        return (urlparse(str(url or "").strip()).hostname or "").lower()
    except Exception:
        return ""


def _host_matches(hostname: str, domains: tuple[str, ...]) -> bool:
    host = hostname.lower().strip()
    if not host:
        return False
    for domain in domains:
        domain = domain.lower().strip()
        if host == domain or host.endswith(f".{domain}"):
            return True
    return False


def _result_matches_domains(result: ImageSearchResult, domains: tuple[str, ...]) -> bool:
    return any(
        _host_matches(_hostname(url), domains)
        for url in (result.image_url, result.source_page_url, result.thumbnail_url)
    )


def _parse_case(raw: str) -> TestCase:
    parts = raw.split("|", 2)
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--case must use the format 'name|query|domain1,domain2'"
        )
    name, query, domains_text = (part.strip() for part in parts)
    domains = tuple(item.strip().lower() for item in domains_text.split(",") if item.strip())
    if not name or not query or not domains:
        raise argparse.ArgumentTypeError(
            "--case requires a non-empty name, query, and at least one domain"
        )
    return TestCase(name=name, query=query, domains=domains)


def _asset_record(asset: ResolvedImageAsset | None) -> dict[str, Any] | None:
    if asset is None:
        return None
    return {
        "strategy": asset.strategy,
        "original_url": asset.original_url,
        "resolved_url": asset.resolved_url,
        "source_page_url": asset.source_page_url,
        "asset_uri": asset.asset_uri,
        "cache_path": asset.cache_path,
        "content_type": asset.content_type,
        "width": asset.width,
        "height": asset.height,
        "model_url_is_data_url": asset.model_url.startswith("data:"),
        "model_url_preview": _short(asset.model_url, 96),
    }


def _result_record(result: ImageSearchResult, *, case: TestCase, selected_by_domain_match: bool) -> dict[str, Any]:
    return {
        "title": result.title,
        "image_url": result.image_url,
        "source_page_url": result.source_page_url,
        "thumbnail_url": result.thumbnail_url,
        "snippet": result.snippet,
        "source": result.source,
        "width": result.width,
        "height": result.height,
        "rank": result.rank,
        "image_host": _hostname(result.image_url),
        "source_page_host": _hostname(result.source_page_url),
        "thumbnail_host": _hostname(result.thumbnail_url),
        "domain_match": _result_matches_domains(result, case.domains),
        "selected_by_domain_match": selected_by_domain_match,
    }


def _build_search_client(name: str) -> Any:
    if name == "serper":
        return SerperSearchClient()
    if name == "serper_adapter":
        return SerperAdapterSearchClient()
    raise ValueError(f"Unknown search backend: {name}")


def _select_candidates(
    results: list[ImageSearchResult],
    *,
    case: TestCase,
    max_results: int,
    include_nonmatching_fallback: bool,
) -> list[tuple[ImageSearchResult, bool]]:
    matched = [(item, True) for item in results if _result_matches_domains(item, case.domains)]
    if matched:
        return matched[:max_results]
    if include_nonmatching_fallback:
        return [(item, False) for item in results[:max_results]]
    return []


def _run_case(
    *,
    case: TestCase,
    search_client: Any,
    resolver: ImageDiscoveryBuilder,
    search_limit: int,
    max_results_per_case: int,
    include_nonmatching_fallback: bool,
    persist_asset: bool,
) -> dict[str, Any]:
    print(f"\n=== Case: {case.name} ===")
    print(f"query: {case.query}")
    print(f"target_domains: {', '.join(case.domains)}")

    try:
        response = search_client.search_image(case.query, limit=search_limit)
    except Exception as exc:  # noqa: BLE001 - debug script should keep going.
        print(f"search_failed: {exc.__class__.__name__}: {exc}")
        return {
            "case": asdict(case),
            "search_ok": False,
            "search_error": f"{exc.__class__.__name__}: {exc}",
            "search_result_count": 0,
            "attempts": [],
        }

    image_results = [item for item in response.results if isinstance(item, ImageSearchResult)]
    selected = _select_candidates(
        image_results,
        case=case,
        max_results=max_results_per_case,
        include_nonmatching_fallback=include_nonmatching_fallback,
    )
    print(f"search_returned: {len(image_results)} | selected_for_resolve: {len(selected)}")

    attempts: list[dict[str, Any]] = []
    for index, (search_result, selected_by_domain_match) in enumerate(selected, start=1):
        print(f"\n[{index}] rank={search_result.rank} title={_short(search_result.title, 120)}")
        print(f"    image_url: {_short(search_result.image_url, 220)}")
        print(f"    source_page_url: {_short(search_result.source_page_url, 220)}")
        print(f"    thumbnail_url: {_short(search_result.thumbnail_url, 220)}")
        print(f"    domain_match: {_result_matches_domains(search_result, case.domains)}")

        try:
            recovery_candidates = resolver._recover_candidate_image_urls(search_result)
        except Exception as exc:  # noqa: BLE001
            recovery_candidates = []
            recovery_probe_error = f"{exc.__class__.__name__}: {exc}"
        else:
            recovery_probe_error = None

        try:
            asset, error = resolver._resolve_image_asset(
                search_result,
                persist_asset=persist_asset,
                recovery_query=case.query,
            )
        except Exception as exc:  # noqa: BLE001
            asset = None
            error = f"resolver_exception:{exc.__class__.__name__}: {exc}"

        success = asset is not None
        strategy = asset.strategy if asset is not None else ""
        print(f"    resolve_success: {success} strategy={strategy} error={error or ''}")
        if asset is not None:
            print(f"    resolved_url: {_short(asset.resolved_url, 220)}")
            print(f"    content_type: {asset.content_type} size={asset.width}x{asset.height}")
            print(f"    cache_path: {asset.cache_path or ''}")
        if recovery_candidates:
            print(f"    recovery_candidates: {len(recovery_candidates)}")
            for recovered_url in recovery_candidates[:3]:
                print(f"      - {_short(recovered_url, 220)}")
        elif recovery_probe_error:
            print(f"    recovery_probe_error: {recovery_probe_error}")

        attempts.append(
            {
                "case_name": case.name,
                "query": case.query,
                "selected_index": index,
                "search_result": _result_record(
                    search_result,
                    case=case,
                    selected_by_domain_match=selected_by_domain_match,
                ),
                "recovery_probe_error": recovery_probe_error,
                "recovery_candidates": recovery_candidates,
                "resolve_success": success,
                "resolve_error": error,
                "asset": _asset_record(asset),
            }
        )

    success_count = sum(1 for item in attempts if item.get("resolve_success"))
    recovery_success_count = sum(
        1
        for item in attempts
        if item.get("resolve_success") and ((item.get("asset") or {}).get("strategy") == "source_page_recovery")
    )
    print(
        f"\ncase_summary: attempts={len(attempts)} success={success_count} "
        f"source_page_recovery_success={recovery_success_count}"
    )
    return {
        "case": asdict(case),
        "search_ok": True,
        "search_engine": response.engine,
        "search_status_code": response.status_code,
        "search_result_count": len(image_results),
        "selected_count": len(selected),
        "success_count": success_count,
        "source_page_recovery_success_count": recovery_success_count,
        "attempts": attempts,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing env vars.")
    parser.add_argument(
        "--search-backend",
        choices=("serper", "serper_adapter"),
        default="serper",
        help="Image search backend to use.",
    )
    parser.add_argument("--search-limit", type=int, default=10, help="Image results requested per query.")
    parser.add_argument("--max-results-per-case", type=int, default=3, help="Resolver attempts per test case.")
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        help="Custom case as 'name|query|domain1,domain2'. May be repeated. If provided, defaults are replaced unless --include-default-cases is set.",
    )
    parser.add_argument("--include-default-cases", action="store_true", help="Run default cases in addition to --case entries.")
    parser.add_argument(
        "--include-nonmatching-fallback",
        action="store_true",
        help="If a query returns no result whose URL hosts match the case domains, still test top results.",
    )
    parser.add_argument("--precheck-timeout", type=float, default=15.0, help="Single image download timeout in seconds.")
    parser.add_argument("--precheck-retries", type=int, default=3, help="Retry count for image download.")
    parser.add_argument("--source-page-timeout", type=float, default=20.0, help="Source page recovery fetch timeout in seconds.")
    parser.add_argument("--model-image-max-edge", type=int, default=2000, help="Resize max edge before model data URL creation.")
    parser.add_argument("--cache-dir", default="debug/image_download_resolver_cache", help="Cache directory for persisted resolved images.")
    parser.add_argument("--no-persist-asset", action="store_true", help="Do not write successful images to cache files.")
    parser.add_argument("--user-agent", default="", help="Optional user agent override for image/source-page fetches.")
    parser.add_argument("--no-source-page-recovery", action="store_true", help="Disable source-page recovery.")
    parser.add_argument("--output-json", default="debug/image_download_resolver_report.json", help="Path to write JSON report.")
    parser.add_argument("--pretty", action="store_true", help="Print full JSON report after the concise summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    env_path = Path(args.env_file).expanduser().resolve()
    loaded_env = load_env_file(env_path, override=args.override_env)

    custom_cases = tuple(args.case or ())
    if custom_cases and not args.include_default_cases:
        cases = custom_cases
    elif custom_cases:
        cases = (*DEFAULT_CASES, *custom_cases)
    else:
        cases = DEFAULT_CASES

    cache_dir = Path(args.cache_dir).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = REPO_ROOT / cache_dir
    cache_dir.mkdir(parents=True, exist_ok=True)

    search_client = _build_search_client(args.search_backend)
    resolver = ImageDiscoveryBuilder(
        store=None,
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(
            precheck_image_urls=True,
            precheck_timeout_s=args.precheck_timeout,
            precheck_retries=max(1, args.precheck_retries),
            source_page_timeout_s=args.source_page_timeout,
            model_image_max_edge=args.model_image_max_edge,
            cache_dir=str(cache_dir),
            upload_cached_images=False,
            try_source_page_recovery=not args.no_source_page_recovery,
            user_agent=args.user_agent or None,
        ),
    )

    print("=== image download resolver robustness test ===")
    print(f"env_file: {env_path} ({len(loaded_env)} vars loaded)")
    print(f"search_backend: {args.search_backend}")
    print(f"cases: {len(cases)}")
    print(f"cache_dir: {cache_dir}")
    print(f"source_page_recovery: {resolver.config.try_source_page_recovery}")
    print(f"user_agent: {resolver._user_agent()}")

    case_reports = [
        _run_case(
            case=case,
            search_client=search_client,
            resolver=resolver,
            search_limit=max(1, args.search_limit),
            max_results_per_case=max(1, args.max_results_per_case),
            include_nonmatching_fallback=args.include_nonmatching_fallback,
            persist_asset=not args.no_persist_asset,
        )
        for case in cases
    ]

    total_attempts = sum(len(case.get("attempts") or []) for case in case_reports)
    total_success = sum(int(case.get("success_count") or 0) for case in case_reports)
    total_recovery_success = sum(int(case.get("source_page_recovery_success_count") or 0) for case in case_reports)
    report = {
        "created_at": _utc_now(),
        "env_file": str(env_path),
        "loaded_env_count": len(loaded_env),
        "search_backend": args.search_backend,
        "search_limit": args.search_limit,
        "max_results_per_case": args.max_results_per_case,
        "include_nonmatching_fallback": args.include_nonmatching_fallback,
        "persist_asset": not args.no_persist_asset,
        "cache_dir": str(cache_dir),
        "source_page_recovery": resolver.config.try_source_page_recovery,
        "user_agent": resolver._user_agent(),
        "summary": {
            "case_count": len(case_reports),
            "attempt_count": total_attempts,
            "success_count": total_success,
            "source_page_recovery_success_count": total_recovery_success,
        },
        "cases": case_reports,
    }

    output_path = Path(args.output_json).expanduser()
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n=== Overall Summary ===")
    print(f"attempts: {total_attempts}")
    print(f"success: {total_success}")
    print(f"source_page_recovery_success: {total_recovery_success}")
    print(f"report: {output_path}")

    if args.pretty:
        print("\n=== Full JSON Report ===")
        print(json.dumps(report, ensure_ascii=False, indent=2))

    return 0 if total_attempts == 0 or total_success > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
