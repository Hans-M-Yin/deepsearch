"""Run image_check and image_ground on one image without the full graph pipeline.

Examples:
  python3 synthesis/test_image_grounding.py \
    --image-path /tmp/kobe_shaq.jpg \
    --title "Kobe to Shaq alley-oop" \
    --snippet "Kobe Bryant throws an alley-oop pass to Shaquille O'Neal" \
    --target-text "Kobe Bryant throwing the alley-oop pass to Shaquille O'Neal in Game 7 of the 2000 Western Conference Finals"

  python3 synthesis/test_image_grounding.py \
    --image-url https://cdn.nba.com/manage/2021/08/kobe-to-shaq.jpg \
    --title "Kobe to Shaq alley-oop"
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.evidence import Evidence, EvidenceType
from synthesis.image_discovery import (
    ImageCandidateStatus,
    ImageDiscoveryBuilder,
    ImageDiscoveryConfig,
    ImageValidationResult,
    ResolvedImageAsset,
)
from synthesis.nodes import ImageNode
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.search_client import ImageSearchResult, SearchClient, SearchResponse
from synthesis.visual_planner import SearchQuerySpec, VisualSearchPlan


class _UnusedSearchClient:
    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        raise NotImplementedError("Not used in test_image_grounding")

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        raise NotImplementedError("Not used in test_image_grounding")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run image grounding on a single image.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--image-path", type=str, help="Local image file path.")
    group.add_argument("--image-url", type=str, help="Remote image URL.")
    parser.add_argument("--title", type=str, default="", help="Search result title for disambiguation.")
    parser.add_argument("--snippet", type=str, default="", help="Search result snippet/caption for disambiguation.")
    parser.add_argument("--source-page-url", type=str, default="", help="Optional source page URL.")
    parser.add_argument(
        "--target-text",
        type=str,
        default="",
        help="Visual target text. Defaults to title/snippet if omitted.",
    )
    parser.add_argument(
        "--env-file",
        type=str,
        default=str(DEFAULT_ENV_PATH),
        help="Path to synthesis env file.",
    )
    parser.add_argument(
        "--skip-check",
        action="store_true",
        help="Skip image_check and force image_ground on the provided image.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print the final JSON result.",
    )
    return parser.parse_args()


def _build_search_result(args: argparse.Namespace) -> ImageSearchResult:
    if args.image_path:
        path = Path(args.image_path).expanduser().resolve()
        image_url = path.as_uri()
    else:
        image_url = args.image_url
    return ImageSearchResult(
        title=args.title or None,
        image_url=image_url,
        source_page_url=args.source_page_url or None,
        snippet=args.snippet or None,
        source="manual_test",
        rank=1,
        raw={"manual_test": True},
    )


def _build_plan(args: argparse.Namespace, search_result: ImageSearchResult) -> VisualSearchPlan:
    target_text = (args.target_text or args.snippet or args.title or "Test image grounding").strip()
    target = Evidence.create(
        EvidenceType.WEB_TEXT,
        content=target_text,
        url=search_result.source_page_url,
        extractor="test_image_grounding",
        evidence_key=f"manual_target:{target_text}",
    )
    query = SearchQuerySpec.create(
        target_text,
        target.evidence_id,
        expected_visual=target_text,
        source="manual_test",
    )
    return VisualSearchPlan.create(
        target,
        queries=[query],
        planner="manual_test",
        metadata={"manual_test": True},
    )


def _seed_local_resolved_asset(
    builder: ImageDiscoveryBuilder,
    search_result: ImageSearchResult,
    image_path: Path,
) -> ResolvedImageAsset:
    payload = image_path.read_bytes()
    content_type = mimetypes.guess_type(str(image_path))[0] or builder._sniff_content_type(payload) or "image/jpeg"
    cache_key = builder._resolved_image_cache_key(search_result.image_url, search_result.source_page_url)
    cache_path = builder._write_image_cache_file(cache_key, payload, content_type)
    model_content_type, model_payload = builder._prepare_model_payload(
        payload=payload,
        content_type=content_type,
        max_edge=builder.config.model_image_max_edge,
    )
    asset = ResolvedImageAsset(
        cache_key=cache_key,
        original_url=search_result.image_url,
        resolved_url=search_result.image_url,
        source_page_url=search_result.source_page_url,
        model_url=builder._data_url(model_content_type, model_payload),
        asset_uri=cache_path,
        cache_path=cache_path,
        content_type=content_type,
        strategy="local_file",
    )
    builder._resolved_image_cache[cache_key] = asset
    return asset


def _run_image_check(
    *,
    builder: ImageDiscoveryBuilder,
    plan: VisualSearchPlan,
    search_result: ImageSearchResult,
    local_image_path: Path | None,
) -> ImageValidationResult:
    if local_image_path is None:
        return builder.image_check(
            plan=plan,
            query=plan.queries[0],
            search_result=search_result,
            run_id="manual_test",
        )

    resolved_asset = _seed_local_resolved_asset(builder, search_result, local_image_path)
    result = builder._image_check_with_mllm(
        plan=plan,
        search_result=search_result,
        model_alias=builder.image_check_model_alias or os.environ.get("IMAGE_CHECK_MODEL"),
        run_id="manual_test",
        resolved_asset=resolved_asset,
    )
    result.metadata = dict(result.metadata or {})
    result.metadata["resolved_image_key"] = resolved_asset.cache_key
    result.metadata["resolved_image"] = resolved_asset.to_metadata()
    return result


def _force_accept_validation(builder: ImageDiscoveryBuilder, search_result: ImageSearchResult, local_image_path: Path | None) -> ImageValidationResult:
    metadata: dict[str, Any] = {"check": "manual_force_accept"}
    if local_image_path is not None:
        resolved_asset = _seed_local_resolved_asset(builder, search_result, local_image_path)
        metadata["resolved_image_key"] = resolved_asset.cache_key
        metadata["resolved_image"] = resolved_asset.to_metadata()
    return ImageValidationResult(
        status=ImageCandidateStatus.ACCEPTED,
        confidence=1.0,
        reason="manual_force_accept",
        metadata=metadata,
    )


def main() -> int:
    args = parse_args()
    load_env_file(Path(args.env_file))

    builder = ImageDiscoveryBuilder(
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(
            precheck_image_urls=not bool(args.image_path),
            try_source_page_recovery=False,
        ),
        image_check_model_alias=os.environ.get("IMAGE_CHECK_MODEL"),
    )

    search_result = _build_search_result(args)
    plan = _build_plan(args, search_result)
    image_node = ImageNode.from_url(
        search_result.image_url or "",
        source_page_url=search_result.source_page_url,
        title=search_result.title,
        caption=search_result.snippet or search_result.title,
        run_id="manual_test",
        metadata={"manual_test": True},
    )

    local_image_path = Path(args.image_path).expanduser().resolve() if args.image_path else None

    if args.skip_check:
        validation = _force_accept_validation(builder, search_result, local_image_path)
    else:
        validation = _run_image_check(
            builder=builder,
            plan=plan,
            search_result=search_result,
            local_image_path=local_image_path,
        )

    grounding = builder.image_ground(
        plan=plan,
        search_result=search_result,
        image_node=image_node,
        validation=validation,
        run_id="manual_test",
    )

    output = {
        "image": {
            "image_url": search_result.image_url,
            "title": search_result.title,
            "snippet": search_result.snippet,
            "source_page_url": search_result.source_page_url,
        },
        "validation": validation.to_dict(),
        "grounding": grounding,
        "image_node_metadata": image_node.metadata,
    }
    if args.pretty:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(output, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
