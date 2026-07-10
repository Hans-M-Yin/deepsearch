"""Debug whether one remote image URL can be fetched for model input.

Examples:
  python synthesis/debug_image_url_access.py \
    --image-url "https://lookaside.instagram.com/seo/google_widget/crawler/?media_id=3551158118295182776"

  python synthesis/debug_image_url_access.py \
    --image-url "https://www.tiktok.com/api/img/?itemId=7640207404520475918&location=0&aid=1988" \
    --source-page-url "https://www.tiktok.com/@example/video/7640207404520475918"
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig, ResolvedImageAsset
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.search_client import ImageSearchResult


class _UnusedSearchClient:
    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any):
        raise NotImplementedError("Not used in debug_image_url_access")

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any):
        raise NotImplementedError("Not used in debug_image_url_access")


def _short(text: str | None, limit: int = 120) -> str:
    raw = " ".join((text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


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
        "model_url_is_data_url": bool(asset.model_url.startswith("data:")),
        "model_url_preview": _short(asset.model_url, 80),
    }


def _attempt_record(
    *,
    step: str,
    url: str,
    asset: ResolvedImageAsset | None,
    error: str | None,
) -> dict[str, Any]:
    return {
        "step": step,
        "url": url,
        "success": asset is not None,
        "error": error,
        "asset": _asset_record(asset),
    }


def _print_attempts(records: list[dict[str, Any]]) -> None:
    print("\n=== Access Attempts ===")
    if not records:
        print("(none)")
        return
    for index, record in enumerate(records, start=1):
        print(f"{index}. step: {record.get('step')}")
        print(f"   url: {record.get('url')}")
        print(f"   success: {record.get('success')}")
        print(f"   error: {record.get('error') or ''}")
        asset = record.get("asset") or {}
        if asset:
            print(f"   strategy: {asset.get('strategy')}")
            print(f"   content_type: {asset.get('content_type')}")
            print(f"   width: {asset.get('width')}")
            print(f"   height: {asset.get('height')}")
            print(f"   model_url_is_data_url: {asset.get('model_url_is_data_url')}")
            print(f"   resolved_url: {asset.get('resolved_url')}")
            print(f"   asset_uri: {asset.get('asset_uri')}")
            print(f"   cache_path: {asset.get('cache_path') or ''}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image-url", required=True, help="Remote image URL to test.")
    parser.add_argument("--source-page-url", default="", help="Optional source page URL used for recovery.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing env vars.")
    parser.add_argument("--precheck-timeout", type=float, default=15.0, help="Single download timeout in seconds.")
    parser.add_argument("--precheck-retries", type=int, default=3, help="Retry count for image download.")
    parser.add_argument("--model-image-max-edge", type=int, default=2000, help="Resize max edge before creating the model data URL.")
    parser.add_argument("--cache-dir", default="", help="Optional cache dir when --persist-asset is enabled.")
    parser.add_argument("--user-agent", default="", help="Optional user agent override for image and source-page fetches.")
    parser.add_argument("--persist-asset", action="store_true", help="Persist successful downloads to a local cache file.")
    parser.add_argument("--no-source-page-recovery", action="store_true", help="Disable source-page recovery when direct fetch fails.")
    parser.add_argument("--pretty", action="store_true", help="Also print the full JSON payload.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    env_path = Path(args.env_file).expanduser().resolve()
    loaded_env = load_env_file(env_path, override=args.override_env)

    with tempfile.TemporaryDirectory(prefix="debug_image_url_access_") as tmpdir:
        cache_dir = Path(args.cache_dir).expanduser().resolve() if args.cache_dir else Path(tmpdir) / "image_cache"
        builder = ImageDiscoveryBuilder(
            store=None,
            search_client=_UnusedSearchClient(),
            config=ImageDiscoveryConfig(
                precheck_timeout_s=args.precheck_timeout,
                precheck_retries=max(1, args.precheck_retries),
                model_image_max_edge=args.model_image_max_edge,
                cache_dir=str(cache_dir),
                upload_cached_images=False,
                try_source_page_recovery=not args.no_source_page_recovery,
                user_agent=args.user_agent or None,
            ),
        )

        search_result = ImageSearchResult(
            title="debug_image_url_access",
            image_url=args.image_url,
            source_page_url=args.source_page_url or None,
            source="debug_image_url_access",
            rank=1,
            raw={"debug_image_url_access": True},
        )
        cache_key = builder._resolved_image_cache_key(search_result.image_url, search_result.source_page_url)

        attempts: list[dict[str, Any]] = []
        recovery_candidates: list[str] = []
        final_asset: ResolvedImageAsset | None = None
        final_error: str | None = None

        direct_asset, direct_error = builder._download_and_prepare_image_asset(
            args.image_url,
            source_page_url=search_result.source_page_url,
            strategy="direct",
            cache_key=cache_key,
            persist_asset=args.persist_asset,
        )
        attempts.append(
            _attempt_record(
                step="direct",
                url=args.image_url,
                asset=direct_asset,
                error=direct_error,
            )
        )

        if direct_asset is not None:
            final_asset = direct_asset
        else:
            final_error = direct_error
            if builder.config.try_source_page_recovery and search_result.source_page_url:
                recovery_candidates = builder._recover_candidate_image_urls(search_result)
                for recovered_url in recovery_candidates:
                    recovered_asset, recovered_error = builder._download_and_prepare_image_asset(
                        recovered_url,
                        source_page_url=search_result.source_page_url,
                        strategy="source_page_recovery",
                        cache_key=cache_key,
                        persist_asset=args.persist_asset,
                    )
                    attempts.append(
                        _attempt_record(
                            step="source_page_recovery",
                            url=recovered_url,
                            asset=recovered_asset,
                            error=recovered_error,
                        )
                    )
                    if recovered_asset is not None:
                        final_asset = recovered_asset
                        final_error = None
                        break
                    final_error = recovered_error

        payload = {
            "env_file": str(env_path),
            "loaded_env_count": len(loaded_env),
            "image_url": args.image_url,
            "source_page_url": search_result.source_page_url,
            "persist_asset": args.persist_asset,
            "try_source_page_recovery": builder.config.try_source_page_recovery,
            "user_agent": builder._user_agent(),
            "recovery_candidates": recovery_candidates,
            "attempts": attempts,
            "result": {
                "success": final_asset is not None,
                "error": final_error,
                "asset": _asset_record(final_asset),
            },
        }

        print("=== image url access debug ===")
        print(f"env_file: {payload['env_file']} ({payload['loaded_env_count']} vars loaded)")
        print(f"image_url: {payload['image_url']}")
        print(f"source_page_url: {payload['source_page_url'] or ''}")
        print(f"persist_asset: {payload['persist_asset']}")
        print(f"try_source_page_recovery: {payload['try_source_page_recovery']}")
        print(f"user_agent: {payload['user_agent']}")

        print("\n=== Recovery Candidates ===")
        if recovery_candidates:
            for index, url in enumerate(recovery_candidates, start=1):
                print(f"{index}. {url}")
        else:
            print("(none)")

        _print_attempts(attempts)

        print("\n=== Final Result ===")
        print(f"success: {'yes' if final_asset is not None else 'no'}")
        print(f"error: {final_error or ''}")
        final_asset_record = _asset_record(final_asset) or {}
        if final_asset_record:
            print(f"strategy: {final_asset_record.get('strategy')}")
            print(f"resolved_url: {final_asset_record.get('resolved_url')}")
            print(f"content_type: {final_asset_record.get('content_type')}")
            print(f"width: {final_asset_record.get('width')}")
            print(f"height: {final_asset_record.get('height')}")
            print(f"model_url_is_data_url: {final_asset_record.get('model_url_is_data_url')}")
            print(f"asset_uri: {final_asset_record.get('asset_uri')}")
            print(f"cache_path: {final_asset_record.get('cache_path') or ''}")

        if args.pretty:
            print("\n=== json ===")
            print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
