#!/usr/bin/env python3
"""Run an independent ReAct verification trajectory for a distractor candidate.

This runner deliberately keeps verification separate from the reasoning
polisher.  It reuses the tool dispatcher and agent implementation from
``synthesis.sft.api_tools`` but uses a verification-specific model alias and
context.  The returned JSON contains the complete verification trajectory and
the structured tool results, so a later refinement stage can decide how to
insert it into the main trajectory.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.model_worker import LLM_WORKER
from synthesis.sft.api_tools import (
    AgentRunResult,
    OpenAIToolAgent,
    OpenAIToolAgentConfig,
    ToolRuntimeContext,
)


# This is intentionally a separate constant from the normal solving and
# reasoning-polishing prompts.  We can refine it independently after testing
# the runner and its evidence format.
VERIFICATION_SYSTEM_PROMPT = """
I will give you a question, the preceding trajectory, the current main-turn draft, and one specific externally searchable candidate. Determine whether that candidate is the referent in the question. You are not being given, and must not assume, an alternative candidate that is already correct. This verification is for named entities, places, artworks, organizations, records, or other independently searchable objects; it is not for directly recognizing an image region, color, hand, animal, count, pose, or left/right position.

Task requirements

1. In the reasoning process you write, you must explicitly show the following logic: before every tool use, first write a detailed and substantial progress update. This update should analyze what exactly has already been established by the most recent tool observation, what uncertainties still remain, and which concrete clues in the question and preceding trajectory can test whether this candidate is the referent. For example, if the question says that the referent has a certain feature or participated in a certain event, then you can search whether the candidate in fact has that feature or took part in that event.
2. In the reasoning you write, every factual statement must be supported by evidence, and the only allowed evidence sources are the question, the image, or tool-returned results. Do not introduce any fact, entity, date, name, or relationship that is not directly supported by the available evidence.
3. This verification task is not for pure visual recognition. Do not use image appearance, an image region, color, hand, animal, count, pose, or left/right position as the fact that decides YES or NO. Use an external, text-verifiable fact about the supplied candidate instead.
4. Pay attention to the pace of reasoning: once you believe the evidence is sufficient, you should give the final judgment.
5. Analyze the candidate carefully and choose the tool most likely to retrieve an external discriminating fact. Prefer `t2t_search` followed by `read_url` of a source page. You may use image search only to locate a named candidate’s source page, but must not use visual similarity or image appearance itself as the deciding evidence.
6. You have at most five tool-use rounds. Before the first tool call, think carefully about the shortest discriminative check that can rule out the distractor; avoid broad exploration and minor query rewrites.
7. Once you can determine whether the referent is or is not the specific object provided to you, then in the final turn output only YES or NO. Output NO only when evidence shows that the candidate fails a concrete question condition; do not turn missing evidence into NO. Output YES only when evidence supports that the candidate can be the referent.

Tool-use tips
1. `t2t_search` returns compact records containing `title`, `snippet`, and `source_page_id`. You should first use `read_url` together with `source_page_id` to inspect a promising page before treating the page content as verified evidence.
2. `t2i_search` and `i2i_search` return compact records containing `image_id` and `source_page_id`. Treat them only as ways to locate an associated source page. Use `read_url` on the source page before making any factual judgment; do not use the image itself as the decisive evidence.
3. Search metadata and URL keyword hints can only help choose a source to inspect. They are not verified evidence by themselves.
4. Use search tools flexibly. If a query cannot find the information you need, try locating it indirectly through related pages. For example, if you cannot find information by searching for the delegation with 108 athletes at a certain Olympic Games, you can first search for delegation-size statistics for all countries at that edition of the Olympics. In that case, the query does not contain the key detail “108,” but once you read the relevant pages, they may be more likely to contain useful clues. When you believe a certain detail is difficult to search directly, you should carefully analyze the problem and plan your search strategy accordingly. Do not keep using the same query repeatedly. If two consecutive searches on the same sub-goal fail to produce useful information, stop making minor wording tweaks; instead, analyze why the direct path failed and pivot to a different angle (an intermediate entity, record keeper, specific date, or raw data).
5. For `i2i_search`, the `region` coordinates use x-first normalized coordinates on a 0–1000 scale, in the order `[x1, y1, x2, y2]`. x increases from left to right, and y increases from top to bottom. If you want to inspect the whole image, use `[0, 0, 1000, 1000]`.
6. For `read_url`, whenever available, you should prioritize using the `source_page_id` or `image_id` returned in search results. `source_page_id` is used to read a webpage, and `image_id` is used to download an image. Old-style raw URLs are supported only for links that are already directly provided in the conversation. This tool cannot see your prior reasoning history, so you must clearly explain in `goal` what evidence you need.

Example

A club that this player in the image once played for later signed a Brazilian winger, who arrived as the record signing from an English club. What is the name of that English club’s home stadium?

Referent question: Could the first club be Paris Saint-Germain?

Suggested starting approach: Search whether PSG ever signed a Brazilian winger who arrived as a record signing from an English club. If its transfer history contradicts that condition, output NO. You are not asked to identify the correct club.
""".strip()


def _image_content_part(source: str) -> dict[str, Any] | None:
    if source.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": source}}
    path = Path(source)
    if path.is_absolute():
        candidates = [path]
    else:
        # ShareGPT image fields are relative to one of the local dataset
        # roots, e.g. ``images/dataset00_sharegpt_dataset_8k/...jpg``.
        bases = [ROOT, ROOT.parent]
        data_root = ROOT / "data"
        if data_root.is_dir():
            bases.append(data_root)
            bases.extend(
                sorted(
                    item
                    for item in data_root.iterdir()
                    if item.is_dir() and item.name.startswith("sharegpt_dataset")
                )
            )
        candidates = [base / path for base in bases]
    resolved = next((candidate for candidate in candidates if candidate.is_file()), None)
    if resolved is None:
        return None
    mime_type = _image_mime_type(resolved)
    if mime_type is None:
        return None
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
    }


def _image_mime_type(path: Path) -> str | None:
    """Return an image MIME type without falling back to octet-stream."""

    guessed = mimetypes.guess_type(str(path))[0]
    if guessed and guessed.startswith("image/"):
        return guessed
    suffix_mimes = {
        ".avif": "image/avif",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    return suffix_mimes.get(path.suffix.casefold())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True, help="Original question.")
    parser.add_argument("--referent", required=True, help="The exact referent in the question being disambiguated.")
    parser.add_argument(
        "--object-to-verify",
        default="",
        help="Specific object/entity to investigate and potentially exclude.",
    )
    parser.add_argument(
        "--distractor-candidate",
        default="",
        help="Backward-compatible alias for --object-to-verify.",
    )
    parser.add_argument(
        "--verification-goal",
        required=True,
        help="Concrete question condition used to test the candidate.",
    )
    parser.add_argument(
        "--context",
        help="Optional JSON file containing preceding reasoning/tool context.",
    )
    parser.add_argument(
        "--context-text",
        default="",
        help="Optional plain-text preceding reasoning/tool context.",
    )
    parser.add_argument(
        "--current-main-turn",
        default="",
        help="Original current assistant turn that triggered verification; it provides direction, not evidence.",
    )
    parser.add_argument(
        "--model-alias",
        default=os.environ.get("VERIFICATION_MODEL_ALIAS") or "",
        help="Dedicated registered Model Worker alias for verification.",
    )
    parser.add_argument(
        "--api-mode",
        choices=("auto", "manual_react", "responses"),
        default="auto",
        help="Tool-calling protocol. auto selects Responses for /responses endpoints.",
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help="Maximum tool-use rounds; the final YES/NO response is allowed after this limit.",
    )
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--workdir", type=Path, default=ROOT / "synthesis/.ignore/verification_runs")
    parser.add_argument("--case-id", default="verification_case")
    parser.add_argument(
        "--image",
        action="append",
        help="Optional local image supplied directly to the verifier.",
    )
    parser.add_argument(
        "--image-url",
        action="append",
        help="Optional remote image supplied directly to the verifier.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON output path.")
    return parser.parse_args()


def _load_context(path: str | None, context_text: str) -> Any:
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return context_text


def _select_api_mode(alias: str, requested: str) -> str:
    if requested != "auto":
        return requested
    config = LLM_WORKER.get_model(alias) or {}
    endpoint = str(config.get("azure_endpoint") or config.get("base_url") or "")
    return "responses" if endpoint.rstrip("/").endswith("/responses") else "manual_react"


def _build_verification_context(
    *,
    question: str,
    distractor_candidate: str,
    preceding_context: Any,
    referent: str,
    verification_goal: str,
    current_main_turn: str = "",
) -> str:
    context_text = (
        json.dumps(preceding_context, ensure_ascii=False, indent=2)
        if not isinstance(preceding_context, str)
        else preceding_context
    )
    return (
        "Verification task:\n"
        f"Original question:\n{question}\n\n"
        f"Referent in the question to resolve:\n{referent or '(not specified)'}\n\n"
        f"Candidate/hypothesis to investigate and potentially rule out:\n{distractor_candidate}\n\n"
        f"Verification goal:\n{verification_goal}\n\n"
        "The candidate is an unselected externally searchable alternative from "
        "the main trajectory. Do not assume any other candidate is correct and do "
        "not try to identify a replacement. Do not solve this by directly "
        "recognizing a hand, color, animal, count, pose, or left/right position "
        "in the image; investigate an external fact about this candidate instead.\n\n"
        "Original current assistant turn that triggered this verification (it gives "
        "the current reasoning direction, but it is not independent evidence):\n"
        f"{current_main_turn or '(not provided)'}\n\n"
        "Preceding reasoning and tool context (use only as evidence already "
        "available before this verification task):\n"
        f"{context_text or '(none)'}\n\n"
        "Investigate this candidate only. Determine whether it fails a concrete "
        "condition from the question; if it does, output NO once the evidence is "
        "sufficient. If it is supported as the referent, output YES."
    )


def _tool_result_to_dict(item: Any) -> dict[str, Any]:
    return {
        "name": item.name,
        "arguments": item.arguments,
        "output": item.output,
        "output_text": item.output_text,
        "new_images": item.new_images,
    }


def _result_to_dict(
    result: AgentRunResult,
    *,
    question: str,
    referent: str,
    distractor_candidate: str,
    verification_goal: str,
    current_main_turn: str,
    model_alias: str,
    api_mode: str,
) -> dict[str, Any]:
    return {
        "task": "distractor_verification",
        "question": question,
        "referent": referent,
        "object_to_verify": distractor_candidate,
        "distractor_candidate": distractor_candidate,
        "verification_goal": verification_goal,
        "current_main_turn": current_main_turn,
        "model_alias": model_alias,
        "api_mode": api_mode,
        "final_text": result.final_text,
        "messages": result.messages,
        "tool_results": [_tool_result_to_dict(item) for item in result.tool_results],
        "raw_responses": result.raw_responses,
        "metadata": result.metadata,
    }


def main() -> int:
    args = _parse_args()
    if not args.model_alias:
        raise SystemExit("--model-alias is required (or set VERIFICATION_MODEL_ALIAS).")
    if args.max_turns < 1:
        raise SystemExit("--max-turns must be at least 1.")

    api_mode = _select_api_mode(args.model_alias, args.api_mode)
    preceding_context = _load_context(args.context, args.context_text)
    distractor_candidate = args.object_to_verify or args.distractor_candidate
    if not distractor_candidate:
        raise SystemExit("one of --object-to-verify or --distractor-candidate is required")
    prompt = _build_verification_context(
        question=args.question,
        referent=args.referent,
        distractor_candidate=distractor_candidate,
        verification_goal=args.verification_goal,
        preceding_context=preceding_context,
        current_main_turn=args.current_main_turn,
    )

    context = ToolRuntimeContext(
        working_dir=str(args.workdir),
        case_id=args.case_id,
    )
    verifier_user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    for image_index, image_source in enumerate(
        [*(args.image or []), *(args.image_url or [])],
        start=1,
    ):
        source = str(image_source)
        context.register_image(source)
        image_part = _image_content_part(source)
        if image_part is not None:
            verifier_user_content.append(
                {
                    "type": "text",
                    "text": f"[Verifier image {image_index}; inspect this image if it is relevant.]",
                }
            )
            verifier_user_content.append(image_part)

    config = OpenAIToolAgentConfig(
        model=args.model_alias,
        api_mode=api_mode,
        max_tokens=args.max_tokens,
        # OpenAIToolAgent counts model responses separately from executed
        # tools.  Reserve a normal final response and a fallback response if
        # the model attempts a tool call after the hard tool budget.
        max_turns=args.max_turns + 2,
        max_tool_calls=args.max_turns,
        timeout_s=args.timeout_s,
        system_prompt=VERIFICATION_SYSTEM_PROMPT,
        print_rounds=True,
    )
    agent = OpenAIToolAgent(config)
    result = agent.run(
        messages=[
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": verifier_user_content},
        ],
        context=context,
    )
    output = _result_to_dict(
        result,
        question=args.question,
        referent=args.referent,
        distractor_candidate=distractor_candidate,
        verification_goal=args.verification_goal,
        current_main_turn=args.current_main_turn,
        model_alias=args.model_alias,
        api_mode=api_mode,
    )

    serialized = json.dumps(output, ensure_ascii=False, indent=2, default=str)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0 if result.metadata.get("generation_complete") else 2


if __name__ == "__main__":
    raise SystemExit(main())
