#!/usr/bin/env python3
"""Repair SFT trajectories that accept an i2i candidate without viewing it.

The script is intentionally separate from ``expand_trajectory_reasoning.py``.
It performs a conservative, local repair:

1. Locate an ``i2i_search`` observation followed by at most three assistant
   turns, or stop at the next non-verification search.
2. Replay that i2i request with a configurable Top-K so private resource URLs
   can be rebuilt.  Historical compact IDs are never treated as URLs.
3. Ask a selector model whether the original reasoning was too decisive and
   which replay candidates are actually grounded in the historical result.
4. Download candidates with ``read_url(image_id)`` and ask a vision model for
   ``YES``, ``NO`` or ``UNCERTAIN`` plus an analysis.  ``NO`` advances to the
   next candidate.
5. If a candidate matches, ask an integration model to write the reasoning
   around the complete verification trace.  Tool calls and answer blocks from
   the original trajectory are preserved programmatically.

The input is never modified.  Use ``--selected-only`` for a small smoke-test
output; otherwise the output contains the complete input array with selected
records repaired.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import os
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

from tqdm import tqdm

try:
    from PIL import Image
    from PIL import ImageOps
except ImportError:  # pragma: no cover - the SFT tool stack normally provides Pillow
    Image = None
    ImageOps = None

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis import model_worker
from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest
from synthesis.post_process.expand_trajectory_reasoning import (
    _image_content_part,
    _record_question,
    _resolve_image_path,
)
from synthesis.sft import tools as sft_tools
from synthesis.sft.api_tools import ToolExecutionResult, ToolRuntimeContext, execute_tool_call


LOGGER = logging.getLogger("repair_i2i_visual_verification")
ASSISTANT_ROLES = {"gpt", "assistant", "model"}
OBSERVATION_ROLES = {"observation", "tool"}
SEARCH_TOOLS = {"t2t_search", "t2i_search", "i2i_search"}


class _ReplayToolRuntimeContext(ToolRuntimeContext):
    """Runtime context for replaying search without extra LLM URL labeling.

    ``ToolRuntimeContext.postprocess_search_output`` normally asks the Qwen
    worker to extract semantic keywords from every returned URL.  That is
    useful during ordinary SFT inference, but it is unrelated to this script's
    conservative candidate alignment (which uses title/snippet/provenance),
    and would turn one replay into ten additional model requests.
    """

    def postprocess_search_output(self, tool_name: str, output: dict[str, Any]) -> dict[str, Any]:
        compact, resources = sft_tools.postprocess_search_output(
            tool_name=tool_name,
            output=output,
            extract_url_keywords=False,
        )
        for resource in resources:
            self.register_url_resource(resource)
        return compact


SELECTOR_SYSTEM_PROMPT = """
You are a conservative trajectory auditor. The trajectory contains a reverse
image search and the next few reasoning turns. Decide whether the original
assistant made a visually unsupported identity claim: it treated an i2i search
candidate's title, source, snippet, or URL keywords as if it had seen the
candidate image, without a successful read_url(image_id).

Use only the supplied question, local trajectory window, and candidate table.
Do not solve the question and do not invent a candidate. A candidate index must
come from the eligible replay candidates. The original image is supplied only
to understand what the target is; it is not proof that a search candidate is
the same object.

Return exactly one JSON object:
{
  "action": "add_verification" | "no_change" | "flag_review",
  "target": "the object/person/entity that must be checked",
  "candidate_indices": [0, 2],
  "reason": "short evidence-based explanation"
}

Important decision rule: hedging language is not enough to avoid verification.
Phrases such as "likely", "very likely", "working identification",
"consistent with", "this locks down", or "I will verify it next" still count
as reliance when the assistant has selected a named candidate and uses that
identity to choose the next search, page read, or reasoning step. Reading the
candidate's source page is not the same as reading the candidate image: a
page_id read does not satisfy visual verification. In these cases choose
add_verification and insert the image check before the dependent next action.

Use no_change only when at least one of these is true: (a) the assistant merely
lists possible candidates without adopting any identity or using one to drive
the next step; (b) the exact candidate image_id was already passed to
read_url and the following observation contains an image; (c) the candidate
was explicitly ruled out and is not used later; or (d) the i2i result is
irrelevant noise and the next action is genuinely independent of it. Do not
use no_change merely because the assistant says the evidence is tentative or
plans to verify the candidate with text. When uncertain between add_verification
and no_change, choose add_verification if a candidate name/title is already
being used operationally.

Use flag_review only when the reasoning is unsupported but the candidate
mapping or causal splice is unsafe. Use add_verification when the assistant
materially relies on an eligible candidate as the identity without viewing its
image. candidate_indices must be ordered from most likely to least likely and
must be selected only from the eligible replay candidates.
""".strip()


VERIFIER_SYSTEM_PROMPT = """
You are a visual consistency verifier. You will receive the original image
(and, when available, the target crop) plus one candidate image downloaded from
an i2i search result. Determine whether the candidate image supports that the
target object/person/entity in the original image is the same object/person/
entity referred to by the verification goal.

Do not use the candidate title, URL, search ranking, or world knowledge as
visual proof. Compare the supplied images. A different photograph of the same
person or object may be YES; a merely similar scene, clothing pattern, object
type, or composition is not enough. If the image is insufficient, return
UNCERTAIN rather than guessing.

Return exactly one JSON object:
{
  "analysis": "concrete comparison of the target and candidate images",
  "verdict": "YES" | "NO" | "UNCERTAIN"
}

YES means the candidate image is visually consistent enough to support the
target identification. NO means it clearly depicts a different object/entity
or contradicts the target. UNCERTAIN means the image cannot decide.
""".strip()


INTEGRATION_SYSTEM_PROMPT = """
You are an integration editor for a multimodal ReAct trajectory. A local
trajectory segment contains an i2i search where the original assistant moved
from search metadata to an object identity without viewing the candidate image.
The verification attempts below contain every candidate read, including failed
NO attempts and the final successful YES attempt.

Rewrite only the reasoning of the original assistant turn immediately after
the i2i observation. The program will insert each verification read_url call
and image observation before that turn and will preserve the original turn's
tool_call and answer blocks byte-for-byte. Do not emit tool_call or answer
blocks in your response.

The rewritten reasoning must:
1. explain why the i2i metadata was only a lead;
2. describe each rejected candidate and why it was rejected, without inventing
   facts beyond the verifier analyses;
3. explain why the accepted candidate is visually consistent;
4. continue naturally toward the original next tool call or answer.

Return exactly one JSON object:
{
  "attempt_reasoning": ["reasoning before read_url for attempt 1", "..."],
  "continuation_reasoning": "reasoning after the final verification, before the original next tool call"
}

The length of attempt_reasoning must equal the number of verification attempts.
Keep the language natural and preserve uncertainty where appropriate.
""".strip()


def _role(message: Any) -> str:
    if not isinstance(message, dict):
        return ""
    return str(message.get("from") or message.get("role") or message.get("speaker") or "").strip().lower()


def _content(message: Any) -> str:
    if not isinstance(message, dict):
        return str(message or "")
    value = message.get("value")
    if value is None:
        value = message.get("content")
    if value is None:
        value = message.get("response_text")
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return str(value or "")


def _extract_tool_call(text: str) -> tuple[str | None, dict[str, Any] | None]:
    for pattern in (
        r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
        r"<action>\s*(\{.*?\})\s*</action>",
    ):
        match = re.search(pattern, str(text or ""), re.DOTALL | re.IGNORECASE)
        if not match:
            continue
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        name = payload.get("name") or payload.get("tool_name")
        arguments = payload.get("arguments")
        if isinstance(name, str) and isinstance(arguments, dict):
            return name, payload
    return None, None


def _json_from_text(text: str) -> dict[str, Any] | None:
    value = str(text or "").strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", value, re.DOTALL | re.IGNORECASE)
    if fenced:
        value = fenced.group(1).strip()
    try:
        payload = json.loads(value)
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        start, end = value.find("{"), value.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            payload = json.loads(value[start : end + 1])
            return payload if isinstance(payload, dict) else None
        except json.JSONDecodeError:
            return None


def _parse_observation(text: str) -> dict[str, Any] | None:
    payload = _json_from_text(text)
    if isinstance(payload, dict):
        return payload
    return None


def _record_id(record: dict[str, Any], index: int) -> str:
    for key in ("id", "question_id", "uid", "record_id", "sample_id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return f"record_{index:05d}"


def _question(record: dict[str, Any]) -> str:
    return _record_question(record)


def _norm(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = re.sub(r"[^\w\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _tokens(value: Any) -> set[str]:
    return set(_norm(value).split())


def _jaccard(left: Any, right: Any) -> float:
    a, b = _tokens(left), _tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _message_image_count(messages: list[Any]) -> int:
    return sum(_content(message).count("<image>") for message in messages)


def _candidate_entries(observation: str) -> list[dict[str, Any]]:
    payload = _parse_observation(observation)
    if not payload:
        return []
    raw = payload.get("matches") or payload.get("results") or []
    if not isinstance(raw, list):
        return []
    return [dict(item) for item in raw if isinstance(item, dict) and item.get("image_id")]


def _find_i2i_windows(
    messages: list[Any],
    *,
    max_assistant_turns: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if _role(message) not in ASSISTANT_ROLES:
            continue
        tool_name, call = _extract_tool_call(_content(message))
        if tool_name != "i2i_search" or not call or index + 1 >= len(messages):
            continue
        if _role(messages[index + 1]) not in OBSERVATION_ROLES:
            continue
        candidates = _candidate_entries(_content(messages[index + 1]))
        if not candidates:
            continue

        candidate_ids = {str(item.get("image_id")) for item in candidates}
        read_candidate_ids: set[str] = set()
        window_indices = [index + 1]
        assistant_turns = 0
        boundary = len(messages)
        for cursor in range(index + 2, len(messages)):
            current = messages[cursor]
            current_role = _role(current)
            if current_role in ASSISTANT_ROLES:
                if assistant_turns >= max_assistant_turns:
                    boundary = cursor
                    break
                assistant_turns += 1
                tool_name_2, call_2 = _extract_tool_call(_content(current))
                if tool_name_2 == "read_url" and call_2:
                    arguments = call_2.get("arguments") or {}
                    resource_id = str(
                        arguments.get("resource_id")
                        or arguments.get("image_id")
                        or arguments.get("page_id")
                        or ""
                    )
                    if resource_id in candidate_ids:
                        next_observation = messages[cursor + 1] if cursor + 1 < len(messages) else None
                        if next_observation is not None and "<image>" in _content(next_observation):
                            read_candidate_ids.add(resource_id)
                window_indices.append(cursor)
                if tool_name_2 in SEARCH_TOOLS:
                    boundary = cursor + 1
                    break
            else:
                window_indices.append(cursor)
        else:
            boundary = len(messages)

        windows.append(
            {
                "i2i_index": index,
                "observation_index": index + 1,
                "first_post_index": index + 2,
                "boundary": boundary,
                "window_indices": window_indices,
                "candidates": candidates,
                "read_candidate_ids": sorted(read_candidate_ids),
                "tool_call": call,
            }
        )
    return windows


def _render_window(messages: list[Any], indices: list[int]) -> str:
    chunks = []
    for turn, index in enumerate(indices):
        chunks.append(f"[LOCAL MESSAGE {turn} / ORIGINAL INDEX {index}][{_role(messages[index])}]\n{_content(messages[index])}")
    return "\n\n".join(chunks)


def _resolve_record_images(record: dict[str, Any], messages: list[Any], before_index: int) -> list[str]:
    # Records keep paths relative to one of the ShareGPT dataset roots.  The
    # tool runtime accepts a local path string only after it has been resolved
    # against the repository/data roots; registering the raw relative path
    # makes read_url/i2i replay fail with ``Unsupported image source``.
    paths: list[str] = []
    for path in (record.get("images") or []):
        raw = str(path)
        resolved = _resolve_image_path(raw)
        paths.append(str(resolved) if resolved is not None else raw)
    count = _message_image_count(messages[:before_index])
    if count <= 0:
        return paths[:1]
    return paths[:count]


def _model_metadata(label: str, record_id: str) -> dict[str, str]:
    return {
        "trace_label": label,
        "session_id": record_id,
        "prompt_cache_key": record_id,
        "user_id": record_id,
        "x_tt_logid": record_id,
    }


def _generate(
    *,
    alias: str,
    system_prompt: str,
    user_content: Any,
    max_tokens: int,
    label: str,
    record_id: str,
) -> str:
    response = LLM_WORKER.generate(
        ModelRequest(
            model=alias,
            messages=[
                ModelMessage(role="system", content=system_prompt),
                ModelMessage(role="user", content=user_content),
            ],
            # GPT-5.4 rejects an explicit temperature=0.0 and only accepts
            # the service default.  Leaving this unset also keeps the script
            # compatible with aliases whose sampling policy is config-driven.
            temperature=None,
            max_tokens=max_tokens,
            metadata=_model_metadata(label, record_id),
        )
    )
    return str(response.content if response else "")


def _image_parts(paths: list[str], labels: list[str]) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    for index, path in enumerate(paths):
        image_part = _image_content_part(_model_image_path(path))
        if image_part is None:
            continue
        label = labels[index] if index < len(labels) else f"Image {index + 1}"
        parts.extend([{"type": "text", "text": label}, image_part])
    return parts


def _image_quality(path: str) -> dict[str, Any]:
    """Check that a downloaded candidate is a real, decodable image.

    Some image hosts return a valid 1x1 transparent/lazy-loading placeholder
    with HTTP 200.  It passes the downloader's format check but is not useful
    evidence and can be rejected by vision endpoints as an unsupported image.
    """

    result: dict[str, Any] = {"path": str(path), "usable": False}
    if Image is None:
        result["error"] = "Pillow is unavailable"
        return result
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            result.update(
                {
                    "format": image.format,
                    "mode": image.mode,
                    "width": width,
                    "height": height,
                    "usable": width >= 2 and height >= 2,
                }
            )
            if width < 2 or height < 2:
                result["error"] = "image is a 1x1-or-smaller placeholder"
    except Exception as exc:
        result["error"] = f"{exc.__class__.__name__}: {exc}"
    return result


def _model_image_path(path: str) -> str:
    """Return a local image path encoded in a format accepted by GPT-5.4."""

    if Image is None or ImageOps is None or str(path).startswith(("http://", "https://")):
        return str(path)
    try:
        with Image.open(path) as image:
            image.load()
            image_format = str(image.format or "").upper()
            if image_format in {"GIF", "JPEG", "JPG", "PNG", "WEBP"}:
                return str(path)
            normalized = ImageOps.exif_transpose(image)
            if normalized.mode not in {"RGB", "RGBA"}:
                normalized = normalized.convert("RGBA" if "transparency" in image.info else "RGB")
            digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()[:20]
            cache_dir = ROOT / "synthesis/.ignore/i2i_visual_verification_image_cache"
            cache_dir.mkdir(parents=True, exist_ok=True)
            converted = cache_dir / f"{digest}.png"
            if not converted.exists():
                normalized.save(converted, format="PNG")
            return str(converted)
    except Exception:
        # The quality check reports the detailed failure; leave the original
        # path here so callers can still record a verifier error if needed.
        return str(path)


def _candidate_signature(item: dict[str, Any]) -> str:
    return " ".join(
        str(item.get(key) or "")
        for key in (
            "title",
            "source",
            "snippet",
            "image_url_keywords",
            "source_page_url_keywords",
        )
    )


def _align_replayed_candidates(
    historical: list[dict[str, Any]],
    replayed: list[dict[str, Any]],
    local_text: str,
) -> list[dict[str, Any]]:
    """Align replayed candidates to historical compact records conservatively."""

    available = set(range(len(replayed)))
    aligned: list[dict[str, Any]] = []
    for historical_index, old in enumerate(historical):
        best: tuple[float, int, float, float] | None = None
        for replay_index in sorted(available):
            new = replayed[replay_index]
            title_score = _jaccard(old.get("title"), new.get("title"))
            overall_score = _jaccard(_candidate_signature(old), _candidate_signature(new))
            exact_title = _norm(old.get("title")) and _norm(old.get("title")) == _norm(new.get("title"))
            score = 1.0 if exact_title else 0.72 * title_score + 0.28 * overall_score
            mentioned = bool(_norm(new.get("title")) and _norm(new.get("title")) in _norm(local_text))
            candidate = (score, replay_index, title_score, overall_score)
            if best is None or candidate > best:
                best = candidate
            if mentioned and score < 0.35:
                # A title explicitly adopted by the original assistant is a
                # useful secondary anchor, but never enough by itself to make
                # an arbitrary replay result eligible.
                best = max(best, (0.35, replay_index, title_score, overall_score))
        if best is None:
            continue
        score, replay_index, title_score, overall_score = best
        if score < 0.35:
            continue
        available.discard(replay_index)
        replayed_item = replayed[replay_index]
        aligned.append(
            {
                "replay_index": replay_index,
                "historical_index": historical_index,
                "match_score": round(score, 4),
                "title_score": round(title_score, 4),
                "overall_score": round(overall_score, 4),
                **replayed_item,
            }
        )
    return aligned


def _resource_audit(resource: Any) -> dict[str, Any]:
    if resource is None:
        return {}
    return {
        "resource_id": resource.resource_id,
        "result_id": resource.result_id,
        "kind": resource.kind,
        "title": resource.title,
        "source_page_url": resource.source_page_url,
        "primary_url": resource.primary_url,
        "image_url": resource.image_url,
        "thumbnail_url": resource.thumbnail_url,
        "rank": resource.rank,
    }


def _replay_i2i(
    *,
    record: dict[str, Any],
    record_index: int,
    messages: list[Any],
    window: dict[str, Any],
    replay_top_k: int,
    workdir: Path,
) -> tuple[ToolRuntimeContext | None, list[dict[str, Any]], dict[str, Any]]:
    rid = _record_id(record, record_index)
    original_images = _resolve_record_images(record, messages, window["i2i_index"])
    context = _ReplayToolRuntimeContext(
        working_dir=str(workdir / rid / f"turn_{window['i2i_index']}"),
        case_id=f"{rid}_turn_{window['i2i_index']}_i2i_replay",
        metadata=dict(record.get("metadata") or {}) if isinstance(record.get("metadata"), dict) else {},
    )
    for image_path in original_images:
        context.register_image(image_path)
    args = copy.deepcopy(window["tool_call"].get("arguments") or {})
    args["top_k"] = replay_top_k
    # Historical trajectories can contain an old internal image reference.
    # Use the current context's latest registered image instead of allowing an
    # obsolete img/image ID to break the replay.
    image_arg = args.get("image") or args.get("url")
    if image_arg and str(image_arg) not in context.image_registry:
        if not str(image_arg).startswith(("http://", "https://")):
            args.pop("image", None)
            args.pop("url", None)
    old_max = sft_tools.MAX_SEARCH_RESULTS
    sft_tools.MAX_SEARCH_RESULTS = max(int(old_max), int(replay_top_k))
    try:
        result = execute_tool_call(
            "i2i_search",
            args,
            context,
            question_text=_question(record),
            assistant_text=_content(messages[window["i2i_index"]]),
        )
    except Exception as exc:
        return None, [], {"status": "replay_failed", "error": repr(exc), "original_images": original_images}
    finally:
        sft_tools.MAX_SEARCH_RESULTS = old_max

    output = result.output if isinstance(result.output, dict) else {}
    raw_candidates = output.get("matches") or output.get("results") or []
    replayed: list[dict[str, Any]] = []
    for replay_index, item in enumerate(raw_candidates if isinstance(raw_candidates, list) else []):
        if not isinstance(item, dict) or not item.get("image_id"):
            continue
        image_id = str(item["image_id"])
        replayed.append(
            {
                "replay_index": replay_index,
                "image_id": image_id,
                "title": item.get("title", ""),
                "source": item.get("source", ""),
                "snippet": item.get("snippet", ""),
                "image_url_keywords": item.get("image_url_keywords", ""),
                "source_page_url_keywords": item.get("source_page_url_keywords", ""),
                "resource": context.resolve_resource_id(image_id),
            }
        )
    audit = {
        "status": "replayed",
        "original_images": original_images,
        "replayed_count": len(replayed),
        "replayed_candidates": [
            {key: value for key, value in item.items() if key != "resource"}
            | {"resource": _resource_audit(item.get("resource"))}
            for item in replayed
        ],
        "tool_output": output,
    }
    return context, replayed, audit


def _selector_decision(
    *,
    record: dict[str, Any],
    record_index: int,
    messages: list[Any],
    window: dict[str, Any],
    aligned: list[dict[str, Any]],
    selector_alias: str,
    max_tokens: int,
) -> tuple[dict[str, Any] | None, str]:
    rid = _record_id(record, record_index)
    local_text = _render_window(messages, window["window_indices"])
    candidate_view = [
        {
            "eligible_index": index,
            "historical_result_index": item["historical_index"],
            "replay_result_index": item["replay_index"],
            "match_score": item["match_score"],
            "title": item.get("title", ""),
            "source": item.get("source", ""),
            "snippet": item.get("snippet", ""),
            "image_url_keywords": item.get("image_url_keywords", ""),
            "source_page_url_keywords": item.get("source_page_url_keywords", ""),
            "image_id": item.get("image_id", ""),
        }
        for index, item in enumerate(aligned)
    ]
    prompt = (
        f"Original question:\n{_question(record)}\n\n"
        f"Local trajectory window:\n{local_text}\n\n"
        f"Already-read historical candidate image IDs (these count as visual checks): {window['read_candidate_ids']}\n\n"
        f"Eligible replay candidates (use only eligible_index values):\n{json.dumps(candidate_view, ensure_ascii=False, indent=2)}\n\n"
        "Decide whether to add an image verification before the first post-i2i action."
    )
    image_paths = _resolve_record_images(record, messages, window["i2i_index"])
    user_content: Any = [{"type": "text", "text": prompt}]
    user_content.extend(_image_parts(image_paths, [f"Original trajectory image {i + 1}" for i in range(len(image_paths))]))
    raw = _generate(
        alias=selector_alias,
        system_prompt=SELECTOR_SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=max_tokens,
        label="i2i_visual_selector",
        record_id=rid,
    )
    payload = _json_from_text(raw)
    if not payload:
        return None, raw
    action = str(payload.get("action") or "").strip().lower()
    if action not in {"add_verification", "no_change", "flag_review"}:
        return None, raw
    try:
        indices = [int(item) for item in payload.get("candidate_indices") or []]
    except (TypeError, ValueError):
        indices = []
    if any(index < 0 or index >= len(aligned) for index in indices):
        indices = [index for index in indices if 0 <= index < len(aligned)]
    return {
        "action": action,
        "target": str(payload.get("target") or "").strip(),
        "candidate_indices": indices,
        "reason": str(payload.get("reason") or "").strip(),
    }, raw


def _verifier_decision(
    *,
    record: dict[str, Any],
    record_index: int,
    original_images: list[str],
    candidate: dict[str, Any],
    read_result: ToolExecutionResult,
    target: str,
    verifier_alias: str,
    max_tokens: int,
) -> tuple[str, str, str]:
    rid = _record_id(record, record_index)
    candidate_paths = [str(path) for path in (read_result.new_images or {}).values()]
    goal = target or _question(record)
    prompt = (
        f"Original question:\n{_question(record)}\n\n"
        f"Verification goal:\n{goal}\n\n"
        f"Candidate result title:\n{candidate.get('title', '')}\n"
        f"Candidate source:\n{candidate.get('source', '')}\n"
        f"Candidate image resource ID requested from read_url:\n{candidate.get('image_id', '')}\n\n"
        "The original image is the reference. The candidate image is the newly downloaded image. "
        "Compare the target content, not just broad scene similarity. "
        f"read_url public observation:\n{read_result.output_text}"
    )
    user_content: Any = [{"type": "text", "text": prompt}]
    user_content.extend(_image_parts(original_images, [f"REFERENCE original image {i + 1}" for i in range(len(original_images))]))
    user_content.extend(_image_parts(candidate_paths, ["CANDIDATE image downloaded by read_url"]))
    raw = _generate(
        alias=verifier_alias,
        system_prompt=VERIFIER_SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=max_tokens,
        label="i2i_visual_verifier",
        record_id=rid,
    )
    payload = _json_from_text(raw) or {}
    analysis = str(payload.get("analysis") or "").strip()
    verdict = str(payload.get("verdict") or "").strip().upper()
    if verdict not in {"YES", "NO", "UNCERTAIN"}:
        matches = re.findall(r"\b(YES|NO|UNCERTAIN)\b", raw.upper())
        verdict = matches[-1] if matches else "UNCERTAIN"
    if not analysis:
        analysis = raw.strip()
    return verdict, analysis, raw


def _extract_block(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}\b[^>]*>.*?</{tag}>", str(text or ""), re.DOTALL | re.IGNORECASE)
    return match.group(0) if match else None


def _thinking_only(text: str) -> str:
    value = str(text or "").strip()
    block = _extract_block(value, "thinking")
    if block:
        return block
    value = re.sub(r"<tool_call>.*?</tool_call>", "", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"<answer>.*?</answer>", "", value, flags=re.DOTALL | re.IGNORECASE)
    return f"<thinking>\n{value.strip()}\n</thinking>"


def _preserve_original_action(original: str, reasoning: str) -> str:
    pieces = [_thinking_only(reasoning)]
    for tag in ("tool_call", "answer"):
        block = _extract_block(original, tag)
        if block:
            pieces.append(block)
    return "\n".join(pieces)


def _replace_message_content(message: Any, value: str) -> dict[str, Any]:
    result = copy.deepcopy(message) if isinstance(message, dict) else {"from": "gpt"}
    if "value" in result:
        result["value"] = value
    elif "content" in result:
        result["content"] = value
    elif "response_text" in result:
        result["response_text"] = value
    else:
        result["value"] = value
    return result


def _integration(
    *,
    record: dict[str, Any],
    record_index: int,
    messages: list[Any],
    window: dict[str, Any],
    attempts: list[dict[str, Any]],
    target: str,
    integration_alias: str,
    max_tokens: int,
) -> tuple[list[str], str, str]:
    rid = _record_id(record, record_index)
    original_index = window["first_post_index"]
    original = _content(messages[original_index]) if original_index < len(messages) else ""
    local_text = _render_window(messages, window["window_indices"])
    attempt_view = []
    candidate_paths: list[str] = []
    for attempt_index, attempt in enumerate(attempts):
        read_result: ToolExecutionResult = attempt["read_result"]
        downloaded_paths = [str(path) for path in (read_result.new_images or {}).values()]
        image_checks = attempt.get("image_checks") or []
        if image_checks:
            candidate_paths.extend(
                check["path"]
                for check in image_checks
                if check.get("usable") and check.get("path")
            )
        else:
            # Backward-compatible fallback for attempts created before the
            # image-quality audit was added.
            candidate_paths.extend(downloaded_paths)
        attempt_view.append(
            {
                "attempt_index": attempt_index,
                "candidate_image_id": attempt["candidate"].get("image_id"),
                "candidate_title": attempt["candidate"].get("title", ""),
                "read_url_observation": read_result.output_text,
                "download_succeeded": bool(read_result.new_images),
                "verdict": attempt.get("verdict"),
                "analysis": attempt.get("analysis", ""),
            }
        )
    prompt = (
        f"Original question:\n{_question(record)}\n\n"
        f"Verification target:\n{target}\n\n"
        f"Local trajectory window:\n{local_text}\n\n"
        f"Original assistant turn to rewrite around the verification:\n{original}\n\n"
        f"Complete verification attempts, in order:\n{json.dumps(attempt_view, ensure_ascii=False, indent=2)}\n\n"
        "Use every attempt in the rewritten reasoning. The original next action remains the same."
    )
    user_content: Any = [{"type": "text", "text": prompt}]
    original_images = _resolve_record_images(record, messages, window["i2i_index"])
    user_content.extend(_image_parts(original_images, [f"REFERENCE original image {i + 1}" for i in range(len(original_images))]))
    user_content.extend(_image_parts(candidate_paths, [f"DOWNLOADED candidate image {i + 1}" for i in range(len(candidate_paths))]))
    raw = _generate(
        alias=integration_alias,
        system_prompt=INTEGRATION_SYSTEM_PROMPT,
        user_content=user_content,
        max_tokens=max_tokens,
        label="i2i_visual_integration",
        record_id=rid,
    )
    payload = _json_from_text(raw) or {}
    attempt_reasoning = payload.get("attempt_reasoning")
    continuation = payload.get("continuation_reasoning")
    if not isinstance(attempt_reasoning, list) or len(attempt_reasoning) != len(attempts) or not isinstance(continuation, str):
        raise ValueError("integration model returned invalid attempt_reasoning/continuation_reasoning")
    if any(not isinstance(item, str) or not item.strip() for item in attempt_reasoning):
        raise ValueError("integration model returned an empty verification reasoning")
    if not continuation.strip():
        raise ValueError("integration model returned an empty continuation reasoning")
    return [str(item).strip() for item in attempt_reasoning], continuation.strip(), raw


def _append_image_placeholders(text: str, image_count: int) -> str:
    if image_count <= 0:
        return text
    return str(text).rstrip() + "\n" + "\n".join("<image>" for _ in range(image_count))


def _apply_repair(
    *,
    record: dict[str, Any],
    record_index: int,
    window: dict[str, Any],
    context: ToolRuntimeContext,
    aligned: list[dict[str, Any]],
    decision: dict[str, Any],
    verifier_alias: str,
    integration_alias: str,
    verifier_max_tokens: int,
    integration_max_tokens: int,
    workdir: Path,
) -> dict[str, Any]:
    key = "conversations" if isinstance(record.get("conversations"), list) else "messages"
    messages = record[key]
    original_images = _resolve_record_images(record, messages, window["i2i_index"])
    target = decision.get("target") or _question(record)
    candidate_indices = decision.get("candidate_indices") or []
    attempts: list[dict[str, Any]] = []
    final_match = False

    for aligned_index in candidate_indices:
        candidate = aligned[aligned_index]
        resource_id = str(candidate.get("image_id") or "")
        goal = f"Visually verify whether the candidate image supports the target: {target}"
        try:
            read_result = execute_tool_call(
                "read_url",
                {"resource_id": resource_id, "goal": goal},
                context,
                question_text=_question(record),
                assistant_text="I will inspect the selected i2i candidate image before accepting its identity.",
                tool_goal=goal,
            )
        except Exception as exc:
            # Preserve the failed read as an attempt, but do not confuse it
            # with a visual NO verdict.
            read_result = ToolExecutionResult(
                name="read_url",
                arguments={"resource_id": resource_id, "goal": goal},
                output={"ok": False, "error": f"{exc.__class__.__name__}: {exc}"},
                output_text=json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False),
            )
        attempt: dict[str, Any] = {"candidate": candidate, "read_result": read_result}
        downloaded_paths = [str(path) for path in (read_result.new_images or {}).values()]
        image_checks = [_image_quality(path) for path in downloaded_paths]
        attempt["image_checks"] = image_checks
        usable_paths = [check["path"] for check in image_checks if check.get("usable")]
        if not downloaded_paths:
            verdict, analysis, raw = "DOWNLOAD_FAILED", "The candidate image was not downloaded; no visual verdict is possible.", ""
        elif not usable_paths:
            verdict, analysis, raw = (
                "INVALID_IMAGE",
                "The download produced no usable image; the candidate will not be treated as visually verified.",
                "",
            )
        else:
            try:
                verdict, analysis, raw = _verifier_decision(
                    record=record,
                    record_index=record_index,
                    original_images=original_images,
                    candidate=candidate,
                    read_result=read_result,
                    target=target,
                    verifier_alias=verifier_alias,
                    max_tokens=verifier_max_tokens,
                )
            except Exception as exc:
                verdict, analysis, raw = (
                    "VERIFIER_FAILED",
                    f"The vision verifier failed for this candidate: {exc.__class__.__name__}: {exc}",
                    "",
                )
        attempt.update({"verdict": verdict, "analysis": analysis, "verifier_raw": raw})
        attempts.append(attempt)
        if verdict == "YES":
            final_match = True
            break

    audit: dict[str, Any] = {
        "target": target,
        "selector": decision,
        "attempts": [
            {
                "candidate": {
                    key: value
                    for key, value in attempt["candidate"].items()
                    if key != "resource"
                },
                "read_url": attempt["read_result"].output,
                "downloaded_image_paths": [str(path) for path in (attempt["read_result"].new_images or {}).values()],
                "image_checks": attempt.get("image_checks", []),
                "verdict": attempt.get("verdict"),
                "analysis": attempt.get("analysis", ""),
            }
            for attempt in attempts
        ],
    }
    if not final_match:
        audit["status"] = "candidates_rejected_or_inconclusive"
        record["_i2i_visual_verification"] = audit
        return audit

    try:
        attempt_reasoning, continuation, integration_raw = _integration(
            record=record,
            record_index=record_index,
            messages=messages,
            window=window,
            attempts=attempts,
            target=target,
            integration_alias=integration_alias,
            max_tokens=integration_max_tokens,
        )
    except Exception as exc:
        audit["status"] = "integration_failed"
        audit["integration_error"] = repr(exc)
        record["_i2i_visual_verification"] = audit
        return audit

    insertion_index = window["first_post_index"]
    if insertion_index >= len(messages) or _role(messages[insertion_index]) not in ASSISTANT_ROLES:
        audit["status"] = "splice_skipped_no_post_i2i_assistant"
        record["_i2i_visual_verification"] = audit
        return audit

    inserted: list[dict[str, Any]] = []
    image_paths_to_insert: list[str] = []
    for reasoning, attempt in zip(attempt_reasoning, attempts):
        candidate = attempt["candidate"]
        tool_call = {
            "name": "read_url",
            "arguments": {
                "resource_id": candidate.get("image_id", ""),
                "goal": f"Visually verify whether the candidate image supports the target: {target}",
            },
        }
        inserted.append(
            {
                "from": "gpt",
                "value": (
                    _thinking_only(reasoning)
                    + "\n<tool_call>\n"
                    + json.dumps(tool_call, ensure_ascii=False)
                    + "\n</tool_call>"
                ),
            }
        )
        read_result: ToolExecutionResult = attempt["read_result"]
        image_paths = [str(path) for path in (read_result.new_images or {}).values()]
        image_paths_to_insert.extend(image_paths)
        inserted.append(
            {
                "from": "observation",
                "value": _append_image_placeholders(read_result.output_text, len(image_paths)),
            }
        )

    original_message = messages[insertion_index]
    rewritten_original = _replace_message_content(
        original_message,
        _preserve_original_action(_content(original_message), continuation),
    )
    messages[insertion_index : insertion_index + 1] = [*inserted, rewritten_original]
    image_insert_index = _message_image_count(messages[:insertion_index])
    image_list = record.setdefault("images", [])
    image_list[image_insert_index:image_insert_index] = image_paths_to_insert
    audit.update(
        {
            "status": "integrated",
            "integrated_before_message_index": insertion_index,
            "inserted_message_count": len(inserted),
            "inserted_image_count": len(image_paths_to_insert),
            "image_insert_index": image_insert_index,
            "integration_raw": integration_raw,
        }
    )
    record["_i2i_visual_verification"] = audit
    return audit


def _process_record(record: dict[str, Any], record_index: int, args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(record)
    key = "conversations" if isinstance(output.get("conversations"), list) else "messages"
    messages = output.get(key) or []
    rid = _record_id(output, record_index)
    windows = _find_i2i_windows(messages, max_assistant_turns=args.max_window_turns)
    audit: dict[str, Any] = {
        "record_index": record_index,
        "record_id": rid,
        "status": "no_eligible_i2i_window",
        "windows": [],
    }
    if not windows:
        output["_i2i_visual_verification"] = audit
        return output, audit

    for window_index, window in enumerate(windows):
        window_audit: dict[str, Any] = {
            "window_index": window_index,
            "i2i_message_index": window["i2i_index"],
            "historical_candidate_count": len(window["candidates"]),
            "already_read_candidate_ids": window["read_candidate_ids"],
        }
        if window["read_candidate_ids"]:
            window_audit["status"] = "already_visually_checked"
            audit["windows"].append(window_audit)
            continue

        context, replayed, replay_audit = _replay_i2i(
            record=output,
            record_index=record_index,
            messages=messages,
            window=window,
            replay_top_k=args.replay_top_k,
            workdir=Path(args.workdir),
        )
        window_audit["replay"] = replay_audit
        if context is None:
            window_audit["status"] = "replay_failed"
            audit["windows"].append(window_audit)
            continue
        local_text = _render_window(messages, window["window_indices"])
        aligned = _align_replayed_candidates(window["candidates"], replayed, local_text)
        window_audit["aligned_candidates"] = [
            {key: value for key, value in item.items() if key != "resource"}
            | {"resource": _resource_audit(item.get("resource"))}
            for item in aligned
        ]
        if not aligned:
            window_audit["status"] = "historical_candidate_unrecoverable"
            audit["windows"].append(window_audit)
            continue

        try:
            decision, selector_raw = _selector_decision(
                record=output,
                record_index=record_index,
                messages=messages,
                window=window,
                aligned=aligned,
                selector_alias=args.selector_model_alias,
                max_tokens=args.selector_max_tokens,
            )
        except Exception as exc:
            window_audit["status"] = "selector_failed"
            window_audit["selector_error"] = repr(exc)
            audit["windows"].append(window_audit)
            continue
        window_audit["selector_raw"] = selector_raw
        window_audit["selector"] = decision
        if not decision:
            window_audit["status"] = "selector_invalid"
            audit["windows"].append(window_audit)
            continue
        if decision["action"] != "add_verification":
            window_audit["status"] = decision["action"]
            audit["windows"].append(window_audit)
            continue
        # Keep the selector's order, remove duplicates, and enforce the
        # command-line safety cap even if the model returns more candidates.
        decision["candidate_indices"] = list(dict.fromkeys(decision["candidate_indices"]))[: args.max_candidates]
        if not decision["candidate_indices"]:
            window_audit["status"] = "selector_requested_verification_without_candidate"
            audit["windows"].append(window_audit)
            continue

        repair_audit = _apply_repair(
            record=output,
            record_index=record_index,
            window=window,
            context=context,
            aligned=aligned,
            decision=decision,
            verifier_alias=args.verifier_model_alias,
            integration_alias=args.integration_model_alias,
            verifier_max_tokens=args.verifier_max_tokens,
            integration_max_tokens=args.integration_max_tokens,
            workdir=Path(args.workdir),
        )
        window_audit["repair"] = repair_audit
        window_audit["status"] = repair_audit.get("status", "unknown")
        audit["windows"].append(window_audit)
        if repair_audit.get("status") == "integrated":
            audit["status"] = "integrated"
            break

    if audit["status"] == "no_eligible_i2i_window":
        audit["status"] = "processed_no_change"
    if audit["status"] == "processed_no_change":
        window_statuses = {str(item.get("status") or "") for item in audit["windows"]}
        for status in ("flag_review", "selector_failed", "replay_failed", "integration_failed"):
            if status in window_statuses:
                audit["status"] = status
                break
    if "_i2i_visual_verification" not in output:
        output["_i2i_visual_verification"] = audit
    return output, audit


def _parse_indices(value: str | None) -> list[int] | None:
    if not value:
        return None
    indices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        index = int(item)
        if index < 0:
            raise ValueError("record indices must be non-negative")
        indices.append(index)
    return sorted(set(indices))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--record-indices", help="Comma-separated source record indices, e.g. 3,20.")
    parser.add_argument("--limit", type=int, help="Process the first N records when --record-indices is omitted.")
    parser.add_argument("--selected-only", action="store_true", help="Write only selected records, useful for smoke tests.")
    parser.add_argument("--replay-top-k", type=int, default=10)
    parser.add_argument("--max-candidates", type=int, default=5, help="Maximum candidates to inspect after selector ordering.")
    parser.add_argument("--max-window-turns", type=int, default=3)
    parser.add_argument(
        "--selector-model-alias",
        default=os.environ.get("I2I_SELECTOR_MODEL_ALIAS", "gpt54_2_internal_azure"),
    )
    parser.add_argument(
        "--verifier-model-alias",
        default=os.environ.get("I2I_VERIFIER_MODEL_ALIAS", "gpt54_2_internal_azure"),
    )
    parser.add_argument(
        "--integration-model-alias",
        default=os.environ.get("I2I_INTEGRATION_MODEL_ALIAS", "gpt54_2_internal_azure"),
    )
    parser.add_argument("--selector-max-tokens", type=int, default=2048)
    parser.add_argument("--verifier-max-tokens", type=int, default=2048)
    parser.add_argument("--integration-max-tokens", type=int, default=4096)
    parser.add_argument(
        "--llm-retry-count",
        type=int,
        default=int(os.environ.get("I2I_LLM_RETRY_COUNT", "2")),
        help="Retries per selector/verifier/integration request; defaults to 2 for fail-fast repair runs.",
    )
    parser.add_argument("--workdir", type=Path, default=ROOT / "synthesis/.ignore/i2i_visual_verification")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.replay_top_k < 1 or args.max_candidates < 1 or args.max_window_turns < 1 or args.llm_retry_count < 0:
        raise SystemExit("replay-top-k, max-candidates and max-window-turns must be positive; llm-retry-count cannot be negative")
    # The shared worker defaults to a large retry budget for long-running SFT
    # jobs.  A repair pass should be fail-fast when its optional model service
    # is down, while still allowing callers to opt into the shared behavior.
    model_worker.LLM_RETRY_COUNT = args.llm_retry_count
    with args.input.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise SystemExit("--input must contain a JSON array")

    requested = _parse_indices(args.record_indices)
    if requested is None:
        requested = list(range(len(data)))
        if args.limit is not None:
            requested = requested[: max(0, args.limit)]
    missing = [index for index in requested if index >= len(data)]
    if missing:
        raise SystemExit(f"record indices out of range: {missing[:10]}")

    selected_set = set(requested)
    output_records = [copy.deepcopy(data[index]) for index in requested] if args.selected_only else copy.deepcopy(data)
    output_position = {index: position for position, index in enumerate(requested)} if args.selected_only else {index: index for index in requested}
    audits: list[dict[str, Any]] = []
    for source_index in tqdm(requested, desc="i2i visual verification", unit="trajectory"):
        processed, audit = _process_record(data[source_index], source_index, args)
        output_records[output_position[source_index]] = processed
        audits.append(audit)
        if args.verbose:
            print(
                f"[{audit['record_id']}] status={audit.get('status')} windows={len(audit.get('windows') or [])}",
                flush=True,
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(output_records, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    with args.audit_jsonl.open("w", encoding="utf-8") as handle:
        for audit in audits:
            handle.write(json.dumps(audit, ensure_ascii=False, default=str) + "\n")
    counts: dict[str, int] = {}
    for audit in audits:
        status = str(audit.get("status") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    print(json.dumps({"processed": len(requested), "output": str(args.output), "audit": str(args.audit_jsonl), "status_counts": counts}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
