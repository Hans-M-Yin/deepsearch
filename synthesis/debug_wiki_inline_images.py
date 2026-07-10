"""Inspect Wikipedia inline-image filtering for one page URL.

Examples:
  python synthesis/debug_wiki_inline_images.py \
    --url https://en.wikipedia.org/wiki/Kobe_Bryant
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from synthesis.image_discovery import ImageDiscoveryBuilder, ImageDiscoveryConfig
from synthesis.model_worker import LLM_WORKER
from synthesis.run_min_graph import DEFAULT_ENV_PATH, load_env_file
from synthesis.search_client import ImageSearchResult
from synthesis.visual_planner import SearchQuerySpec, VisualSearchPlan
from synthesis.wiki_text_builder import (
    EnhancedReaderClient,
    RawMarkdownReaderClient,
    WikiTextBuilder,
)


class _UnusedSearchClient:
    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any):
        raise NotImplementedError("Not used in debug_wiki_inline_images")

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any):
        raise NotImplementedError("Not used in debug_wiki_inline_images")


def _short(text: str | None, limit: int = 240) -> str:
    raw = " ".join((text or "").split())
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 3)].rstrip() + "..."


def _build_plan(*, text_result: Any, candidate: Any, run_id: str) -> VisualSearchPlan:
    from synthesis.evidence import Evidence, EvidenceType

    caption = str(candidate.caption or "").strip()
    target = Evidence.create(
        EvidenceType.VISUAL_TARGET,
        content=caption,
        node_ids=[text_result.node.node_id],
        url=candidate.image_url,
        extractor="debug_wikipedia_inline_image",
        metadata={
            "source_evidence_ids": [text_result.text_evidence.evidence_id],
            "run_id": run_id,
            "source_page_url": candidate.source_page_url,
            "file_page_url": candidate.file_page_url,
            "image_url": candidate.image_url,
            "thumbnail_url": candidate.thumbnail_url,
            "caption": caption,
            "rank": candidate.rank,
        },
        evidence_key=f"{text_result.node.node_id}:debug_wiki_inline:{candidate.rank}:{candidate.image_url}",
    )
    query = SearchQuerySpec.create(
        caption,
        target.evidence_id,
        expected_visual=caption,
        source="wikipedia_inline_image",
        metadata={
            "source_page_url": candidate.source_page_url,
            "file_page_url": candidate.file_page_url,
            "image_url": candidate.image_url,
            "thumbnail_url": candidate.thumbnail_url,
            "rank": candidate.rank,
        },
    )
    return VisualSearchPlan.create(
        target,
        queries=[query],
        source_node_id=text_result.node.node_id,
        source_evidence_ids=[text_result.text_evidence.evidence_id],
        planner="debug_wikipedia_inline_image",
        metadata={
            "plan_source": "wikipedia_inline_image",
            "image_url": candidate.image_url,
            "thumbnail_url": candidate.thumbnail_url,
            "source_page_url": candidate.source_page_url,
            "file_page_url": candidate.file_page_url,
            "caption": caption,
            "raw_caption": candidate.raw_caption,
            "alt_text": candidate.alt_text,
            "rank": candidate.rank,
        },
    )


def _build_search_result(candidate: Any) -> ImageSearchResult:
    title = candidate.alt_text or candidate.caption or candidate.file_page_url or candidate.image_url
    return ImageSearchResult(
        title=title,
        image_url=candidate.image_url,
        source_page_url=candidate.source_page_url,
        thumbnail_url=candidate.thumbnail_url,
        snippet=candidate.caption,
        source="wikipedia_inline",
        rank=candidate.rank,
        raw={
            "file_page_url": candidate.file_page_url,
            "thumbnail_url": candidate.thumbnail_url,
            "raw_caption": candidate.raw_caption,
            "alt_text": candidate.alt_text,
            "plan_source": "wikipedia_inline_image",
        },
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Wikipedia page URL to inspect.")
    parser.add_argument("--title", default="", help="Optional page title override.")
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing env vars.")
    parser.add_argument("--reader-base-url", default="http://127.0.0.1:8004", help="Enhanced Reader base URL for building the source text node.")
    parser.add_argument("--raw-reader-base-url", default="http://127.0.0.1:8002", help="Raw markdown Reader base URL for extracting inline images.")
    parser.add_argument("--max-links", type=int, default=60, help="Maximum Wikipedia links extracted while building the source text node.")
    parser.add_argument("--limit", type=int, default=0, help="Optional max number of inline images to inspect. <=0 means all.")
    parser.add_argument("--run-id", default="debug_wiki_inline_images", help="Run id recorded in debug evidence.")
    parser.add_argument("--pretty", action="store_true", help="Also print the full JSON payload.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    env_path = Path(args.env_file).expanduser().resolve()
    loaded_env = load_env_file(env_path, override=args.override_env)

    reader = EnhancedReaderClient(base_url=args.reader_base_url)
    raw_reader = RawMarkdownReaderClient(base_url=args.raw_reader_base_url)
    wiki_builder = WikiTextBuilder(
        reader=reader,
        store=None,
        model_client=LLM_WORKER,
        max_links=args.max_links,
    )
    image_builder = ImageDiscoveryBuilder(
        search_client=_UnusedSearchClient(),
        store=None,
        model_client=LLM_WORKER,
        config=ImageDiscoveryConfig(),
    )

    text_result = wiki_builder.build_from_url(
        args.url,
        title=args.title or None,
        run_id=args.run_id,
        persist=False,
    )
    document = raw_reader.read(args.url)
    markdown = document.raw_markdown or document.content or ""
    candidates = WikiTextBuilder.extract_wiki_inline_images(markdown, source_url=args.url)
    if args.limit and args.limit > 0:
        candidates = candidates[: args.limit]

    results: list[dict[str, Any]] = []
    for index, candidate in enumerate(candidates, start=1):
        search_result = _build_search_result(candidate)
        plan = _build_plan(text_result=text_result, candidate=candidate, run_id=args.run_id)
        validation = image_builder._wiki_inline_image_check(
            plan=plan,
            search_result=search_result,
            run_id=args.run_id,
        )
        keep = (
            validation.status.value == "accepted"
            and not validation.drop_candidate
        )
        metadata = validation.metadata or {}
        results.append(
            {
                "index": index,
                "keep": keep,
                "status": validation.status.value,
                "drop_candidate": validation.drop_candidate,
                "reason": validation.reason,
                "image_url": candidate.image_url,
                "thumbnail_url": candidate.thumbnail_url,
                "file_page_url": candidate.file_page_url,
                "caption": candidate.caption,
                "raw_caption": candidate.raw_caption,
                "alt_text": candidate.alt_text,
                "question": metadata.get("question"),
                "expected_answer": metadata.get("expected_answer"),
                "model_answer": metadata.get("answer"),
                "judge_reason": metadata.get("judge_reason"),
                "question_raw_output": metadata.get("question_raw_output"),
                "answer_raw_output": metadata.get("answer_raw_output"),
                "judge_raw_output": metadata.get("judge_raw_output"),
                "resolved_image": metadata.get("resolved_image"),
            }
        )

    print("=== debug wiki inline images ===")
    print(f"env_file: {env_path} ({len(loaded_env)} vars loaded)")
    print(f"url: {args.url}")
    print(f"title: {text_result.node.title or ''}")
    print(f"node_id: {text_result.node.node_id}")
    print(f"inline_image_count: {len(results)}")
    for item in results:
        print(f"\n=== Inline Image {item['index']} ===")
        print(f"keep: {'yes' if item['keep'] else 'no'}")
        print(f"reason: {item.get('reason') or '-'}")
        print(f"image_url: {item.get('image_url') or ''}")
        print(f"thumbnail_url: {item.get('thumbnail_url') or ''}")
        print(f"file_page_url: {item.get('file_page_url') or ''}")
        print(f"caption: {_short(item.get('caption'), 400)}")
        print(f"alt_text: {_short(item.get('alt_text'), 240)}")
        print(f"question: {_short(item.get('question'), 400)}")
        print(f"expected_answer: {_short(item.get('expected_answer'), 240)}")
        print(f"model_answer: {_short(item.get('model_answer'), 240)}")
        print(f"judge_reason: {_short(item.get('judge_reason'), 400)}")
        print(f"question_raw_output: {_short(item.get('question_raw_output'), 400)}")
        print(f"answer_raw_output: {_short(item.get('answer_raw_output'), 240)}")
        print(f"judge_raw_output: {_short(item.get('judge_raw_output'), 400)}")

    if args.pretty:
        payload = {
            "url": args.url,
            "title": text_result.node.title,
            "node_id": text_result.node.node_id,
            "inline_images": results,
        }
        print("\n=== json ===")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
