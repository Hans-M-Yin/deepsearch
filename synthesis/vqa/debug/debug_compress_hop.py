"""Debug one ``compress_hop`` call from explicit source/relation/target inputs.

Minimal example:

    python -m synthesis.vqa.debug.debug_compress_hop \
      --hop-type text->image \
      --source "Port Jackson" \
      --relation "Japanese midget submarine recovered from Sydney Harbour after the 31 May 1942 raid" \
      --target "photo of the recovered Japanese midget submarine"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    __package__ = "synthesis.vqa.debug"

from synthesis.model_worker import LLM_WORKER

from synthesis.vqa.question_writer import HopContext, QuestionWriter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--hop-type",
        required=True,
        choices=("text->text", "text->image", "image->text", "image->image"),
        help="Source/target modality pair for this synthetic hop.",
    )
    parser.add_argument("--source", required=True, help="Source title or source clue text.")
    parser.add_argument("--relation", required=True, help="Raw edge relation string.")
    parser.add_argument("--target", required=True, help="Target title or target clue text.")
    parser.add_argument(
        "--model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for compress_hop.",
    )
    parser.add_argument(
        "--source-json",
        default=None,
        help="Optional JSON object overriding the auto-built source node payload.",
    )
    parser.add_argument(
        "--target-json",
        default=None,
        help="Optional JSON object overriding the auto-built target node payload.",
    )
    parser.add_argument(
        "--source-summary",
        default=None,
        help="Optional summary to attach when the source is a text node.",
    )
    parser.add_argument(
        "--source-description",
        default=None,
        help="Optional description to attach when the source is a text node.",
    )
    parser.add_argument(
        "--target-summary",
        default=None,
        help="Optional summary to attach when the target is a text node.",
    )
    parser.add_argument(
        "--target-description",
        default=None,
        help="Optional description to attach when the target is a text node.",
    )
    parser.add_argument(
        "--source-caption",
        default=None,
        help="Optional caption to attach when the source is an image node.",
    )
    parser.add_argument(
        "--target-caption",
        default=None,
        help="Optional caption to attach when the target is an image node.",
    )
    parser.add_argument(
        "--source-search-query",
        default=None,
        help="Optional search_query metadata to attach when the source is an image node.",
    )
    parser.add_argument(
        "--target-search-query",
        default=None,
        help="Optional search_query metadata to attach when the target is an image node.",
    )
    parser.add_argument(
        "--source-visual-fact",
        action="append",
        default=None,
        help="Repeatable visual fact for an image source node.",
    )
    parser.add_argument(
        "--target-visual-fact",
        action="append",
        default=None,
        help="Repeatable visual fact for an image target node.",
    )
    parser.add_argument(
        "--source-ocr-text",
        action="append",
        default=None,
        help="Repeatable OCR snippet for an image source node.",
    )
    parser.add_argument(
        "--target-ocr-text",
        action="append",
        default=None,
        help="Repeatable OCR snippet for an image target node.",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Include the selected system prompt in the output JSON.",
    )
    return parser


def _parse_json_object(raw: str | None, *, flag_name: str) -> dict[str, Any] | None:
    if raw is None:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{flag_name} must be valid JSON") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object")
    return parsed


def _text_node_payload(
    *,
    title: str,
    summary: str | None,
    description: str | None,
    raw_override: dict[str, Any] | None,
) -> dict[str, Any]:
    if raw_override is not None:
        payload = dict(raw_override)
        payload.setdefault("node_type", "text")
        payload.setdefault("title", title)
        return payload
    return {
        "node_type": "text",
        "title": title,
        "summary": summary,
        "description": description,
        "aliases": [],
        "attributes": {},
    }


def _image_node_payload(
    *,
    title: str,
    caption: str | None,
    search_query: str | None,
    visual_facts: list[str] | None,
    ocr_texts: list[str] | None,
    raw_override: dict[str, Any] | None,
) -> dict[str, Any]:
    if raw_override is not None:
        payload = dict(raw_override)
        payload.setdefault("node_type", "image")
        payload.setdefault("title", title)
        return payload
    return {
        "node_type": "image",
        "title": title,
        "caption": caption,
        "search_query": search_query,
        "visual_facts": list(visual_facts or []),
        "ocr_texts": list(ocr_texts or []),
        "grounded_entities": [],
    }


def _node_payload_from_args(
    *,
    modality: str,
    label: str,
    summary: str | None,
    description: str | None,
    caption: str | None,
    search_query: str | None,
    visual_facts: list[str] | None,
    ocr_texts: list[str] | None,
    raw_override: dict[str, Any] | None,
) -> dict[str, Any]:
    if modality == "text":
        return _text_node_payload(
            title=label,
            summary=summary,
            description=description,
            raw_override=raw_override,
        )
    return _image_node_payload(
        title=label,
        caption=caption,
        search_query=search_query,
        visual_facts=visual_facts,
        ocr_texts=ocr_texts,
        raw_override=raw_override,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    src_modality, dst_modality = args.hop_type.split("->", maxsplit=1)

    source_override = _parse_json_object(args.source_json, flag_name="--source-json")
    target_override = _parse_json_object(args.target_json, flag_name="--target-json")

    source_node = _node_payload_from_args(
        modality=src_modality,
        label=args.source,
        summary=args.source_summary,
        description=args.source_description,
        caption=args.source_caption,
        search_query=args.source_search_query,
        visual_facts=args.source_visual_fact,
        ocr_texts=args.source_ocr_text,
        raw_override=source_override,
    )
    target_node = _node_payload_from_args(
        modality=dst_modality,
        label=args.target,
        summary=args.target_summary,
        description=args.target_description,
        caption=args.target_caption,
        search_query=args.target_search_query,
        visual_facts=args.target_visual_fact,
        ocr_texts=args.target_ocr_text,
        raw_override=target_override,
    )

    writer = QuestionWriter(
        model_client=LLM_WORKER if args.model_alias else None,
        model=args.model_alias,
        compress_hop_model_client=LLM_WORKER if args.model_alias else None,
        compress_hop_model=args.model_alias,
    )
    hop = HopContext(
        hop_index=0,
        src_node_id="debug_source",
        dst_node_id="debug_target",
        src_modality=src_modality,
        dst_modality=dst_modality,
        edge_id="debug_edge",
        edge_type="debug_edge",
        relation=args.relation,
        src_content=source_node,
        dst_content=target_node,
    )
    selected_prompt = writer._compress_hop_prompt(hop=hop)
    result = writer.compress_hop(hop=hop)

    output = {
        "hop_type": args.hop_type,
        "runtime_mode": "llm" if args.model_alias else "fallback",
        "model_alias": args.model_alias,
        "prompt_name": _prompt_name_for_hop_type(args.hop_type),
        "input": writer._compress_hop_prompt_payload(hop=hop),
        "source_node": source_node,
        "target_node": target_node,
        "result": result,
    }
    if args.show_prompt:
        output["prompt"] = selected_prompt

    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


def _prompt_name_for_hop_type(hop_type: str) -> str:
    mapping = {
        "text->text": "PROMPT_COMPRESS_HOP_TEXT_TO_TEXT",
        "text->image": "PROMPT_COMPRESS_HOP_TEXT_TO_IMAGE",
        "image->text": "PROMPT_COMPRESS_HOP_IMAGE_TO_TEXT",
    }
    return mapping.get(hop_type, "PROMPT_COMPRESS_HOP_GENERIC")


if __name__ == "__main__":
    raise SystemExit(main())
