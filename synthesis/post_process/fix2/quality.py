"""Independent, label-blind quality review for expert FVQA trajectories."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
from typing import Any

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest

from .io import append_jsonl, iter_json_records, load_completed_ids
from .trajectory import message_text, _content_image_urls
from .visual import _to_model_image_url


TRAJECTORY_QUALITY_SYSTEM_PROMPT = """You are an independent quality reviewer for a visual web-research trajectory.

You receive only the public question, original images when present, images actually
attached in the visible trajectory, the reasoning/tool transcript, and the final answer. You never receive a
reference answer, hidden intermediate statement, or knowledge-graph fact.

Decide whether this trajectory is safe to use as SFT data. Judge evidence, not
whether the final answer happens to be true in the outside world. A claim is
grounded only when the original image or a successful tool observation in this
transcript supports it. Search-result titles/snippets are leads; an image from
i2i_search/t2i_search is not visually inspected until read_url(image_id)
succeeds. Do not supply missing world knowledge yourself.

Filter a trajectory if it has any material unsupported entity/fact jump,
appears to recognize an entity from internal/world knowledge and merely seeks
after-the-fact confirmation, uses tool output inconsistently, identifies the
wrong object, or has a reasoning gap that prevents a reader from following the
chain. An original image, when supplied, must contribute a concrete clue or a justified
reason for the chosen search; an answer that ignores it is not acceptable.

Candidate-image comparison is valuable but is NOT mandatory. Use:
- strong: candidate image(s) were opened and provide a material visual match;
- weak: the image contributes a concrete non-comparison anchor, such as legible
  text, logo, map shape, or distinctive visible structure, and the external
  evidence chain is sufficient;
- none: visual grounding is absent or insufficient.

Images are supplied at their available original resolution, deduplicated. image_evidence_map
maps each image_index to original input positions and/or zero-based transcript message indices.
An attachment becomes agent evidence only at its mapped message, never earlier. Search-result
image URLs alone are not opened-image evidence. Absence of an original image is legitimate for
a text-only question; evaluate its tool evidence normally. Do not reject solely for that absence.

Independently inspect the attached images: the assistant's 'I see', 'matches', or 'verified'
claims are assertions to check, not ground truth. If essential details are unreadable, do not
invent them or approve certainty. Distinguish same entity from same photograph, event, and version.
First extract the question's required entity, time/event, version, spatial referent, attribute,
and exact counting/text criteria where applicable. Then trace each requirement to specific
observations. Related evidence is not sufficient: a different event cannot establish a requested
event's visual detail; a neighboring object's attribute is not the target's; broad category
membership does not prove a particular historical location or variant; a related label does not
satisfy an exact-label count. Check every indispensable link, not only the final supporting source.
Do not silently relax the question or use outside knowledge to fill gaps. Distinguish a proven
contradiction from missing support; either material defect warrants filter, with a concrete reason.
Include the violated requirement, message/image indices, and conflicting or missing evidence in
your evidence/reason. A fluent narrative and plausible answer do not establish a valid chain.
If the transcript is marked truncated and an essential link cannot be assessed, do not assume it.

Return exactly one JSON object and no markdown:
{
  "decision": "keep" | "filter",
  "visual_verification_level": "strong" | "weak" | "none",
  "evidence_sufficient": true | false,
  "internal_knowledge_suspected": true | false,
  "logical_gap": true | false,
  "reason_codes": ["short_machine_readable_reason"],
  "evidence": ["specific observation from the supplied image or transcript"],
  "reason": "concise overall explanation"
}

For decision=keep, evidence_sufficient must be true and both
internal_knowledge_suspected and logical_gap must be false. Do not reject a
trajectory solely because its visual_verification_level is weak or none.
"""

_MAX_TRANSCRIPT_CHARS = 64_000
_MAX_MESSAGE_CHARS = 12_000


def _record_id(record: dict[str, Any]) -> str:
    return str(record.get("case_id") or record.get("question_id") or record.get("sample_id") or "").strip()


def _shorten(value: str, *, limit: int = _MAX_MESSAGE_CHARS) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[truncated for reviewer input]"


def _tool_call_summary(message: dict[str, Any]) -> str:
    calls: list[dict[str, Any]] = []
    for call in message.get("tool_calls") or []:
        if not isinstance(call, dict):
            continue
        function = call.get("function") if isinstance(call.get("function"), dict) else call
        calls.append(
            {
                "name": str(function.get("name") or ""),
                "arguments": function.get("arguments") or {},
            }
        )
    return json.dumps(calls, ensure_ascii=False, sort_keys=True) if calls else ""


def build_quality_transcript(record: dict[str, Any], *, max_chars: int | None = _MAX_TRANSCRIPT_CHARS, message_max_chars: int | None = _MAX_MESSAGE_CHARS) -> str:
    """Serialize visible evidence without copying input-image base64 payloads."""

    blocks: list[str] = []
    def shorten(text):
        return str(text or "").strip() if message_max_chars is None else _shorten(text, limit=message_max_chars)
    used = 0
    for index, message in enumerate(record.get("raw_messages") or record.get("messages") or []):
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role == "assistant":
            text = message_text(message.get("content"))
            calls = _tool_call_summary(message)
            if calls:
                text = f"{text}\nNative tool calls: {calls}".strip()
            block = f"[assistant step {index}]\n{shorten(text)}"
        elif role == "tool":
            name = str(message.get("name") or "unknown_tool")
            content = message.get("content")
            if isinstance(content, list):
                content = message_text(content)
            elif not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, sort_keys=True)
            block = f"[tool result {index}: {name}]\n{shorten(content)}"
        else:
            # User image attachments can contain large base64 strings. The
            # reviewer receives the original image separately, so retain only
            # natural-language user text (usually the public question).
            text = message_text(message.get("content"))
            if not text:
                continue
            block = f"[{role or 'message'} {index}]\n{shorten(text)}"
        if max_chars is not None and used + len(block) + 2 > max_chars:
            blocks.append("[trajectory transcript truncated after this point]")
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def _original_image_path(record: dict[str, Any]) -> str:
    for item in record.get("input_images") or []:
        if isinstance(item, dict):
            value = str(item.get("image_path") or item.get("image_url") or "").strip()
        else:
            value = str(item or "").strip()
        if value:
            return value
    return ""


def build_quality_reviewer_messages(record: dict[str, Any], *, full_transcript: bool = False) -> list[ModelMessage]:
    """Build the public, gold-free multimodal review request."""

    images: list[str] = []
    image_map: list[dict[str, Any]] = []
    positions: dict[str, int] = {}

    def attach(value: str, occurrence: dict[str, Any]) -> None:
        # Preserve source bytes: no resize or recompression. Missing files raise
        # an execution error rather than silently removing review evidence.
        url = _to_model_image_url(value)
        if url not in positions:
            positions[url] = len(images)
            images.append(url)
            image_map.append({"image_index": len(images) - 1, "occurrences": []})
        image_map[positions[url]]["occurrences"].append(occurrence)

    for index, item in enumerate(record.get("input_images") or []):
        value = item.get("image_path") or item.get("image_url") if isinstance(item, dict) else item
        if isinstance(value, dict):
            value = value.get("url")
        if value:
            attach(str(value), {"kind": "original", "input_index": index})
    for index, message in enumerate(record.get("raw_messages") or record.get("messages") or []):
        if not isinstance(message, dict):
            continue
        for attachment_index, value in enumerate(_content_image_urls(message.get("content"))):
            attach(value, {"kind": "trajectory", "message_index": index, "attachment_index": attachment_index})
    payload = {
        "question": str(record.get("question") or ""),
        "final_answer": str(record.get("extracted_answer") or ""),
        "visible_trajectory": build_quality_transcript(record, max_chars=None, message_max_chars=None) if full_transcript else build_quality_transcript(record),
        "image_evidence_map": image_map,
    }
    content: list[dict[str, Any]] = [
        {"type": "text", "text": json.dumps(payload, ensure_ascii=False)},
    ]
    for index, url in enumerate(images):
        original = any(x["kind"] == "original" for x in image_map[index]["occurrences"])
        content.extend([
            {"type": "text", "text": f"Image {index}" + (" (Original image)" if original else " (trajectory attachment)")},
            {"type": "image_url", "image_url": {"url": url, "detail": "high"}},
        ])
    return [
        ModelMessage(role="system", content=TRAJECTORY_QUALITY_SYSTEM_PROMPT),
        ModelMessage(role="user", content=content),
    ]


def _parse_json_object(text: str) -> dict[str, Any] | None:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = candidate.find("{")
        if start < 0:
            return None
        try:
            value, _end = json.JSONDecoder().raw_decode(candidate[start:])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


def _normalize_review(value: dict[str, Any]) -> tuple[dict[str, Any], str | None]:
    decision = str(value.get("decision") or "").strip().lower()
    visual_level = str(value.get("visual_verification_level") or "").strip().lower()
    required_bools = ("evidence_sufficient", "internal_knowledge_suspected", "logical_gap")
    missing = [name for name in required_bools if not isinstance(value.get(name), bool)]
    if decision not in {"keep", "filter"}:
        missing.append("decision")
    if visual_level not in {"strong", "weak", "none"}:
        missing.append("visual_verification_level")
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not any(str(item).strip() for item in evidence):
        missing.append("evidence")
    reason_codes = value.get("reason_codes")
    if not isinstance(reason_codes, list):
        missing.append("reason_codes")
    normalized = {
        **value,
        "decision": decision,
        "visual_verification_level": visual_level,
        "reason_codes": [str(item) for item in reason_codes] if isinstance(reason_codes, list) else [],
        "evidence": [str(item) for item in evidence] if isinstance(evidence, list) else [],
    }
    return normalized, ("missing_or_invalid_" + ",".join(missing) if missing else None)


def validate_trajectory_quality_review(value: Any) -> tuple[bool, str]:
    """Return a deterministic filter decision for a persisted quality row."""

    if not isinstance(value, dict) or value.get("error") or value.get("status") != "ok":
        return False, "trajectory_quality_review_missing_or_failed"
    if str(value.get("decision") or "").lower() != "keep":
        return False, "trajectory_quality_review_filter"
    if value.get("evidence_sufficient") is not True:
        return False, "trajectory_quality_evidence_insufficient"
    if value.get("internal_knowledge_suspected") is not False:
        return False, "trajectory_quality_internal_knowledge_suspected"
    if value.get("logical_gap") is not False:
        return False, "trajectory_quality_logical_gap"
    if str(value.get("visual_verification_level") or "").lower() not in {"strong", "weak", "none"}:
        return False, "trajectory_quality_review_malformed"
    if not isinstance(value.get("evidence"), list) or not value["evidence"]:
        return False, "trajectory_quality_review_malformed"
    return True, ""


def review_trajectory_quality(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    model_alias: str = "gemini38flash_internal_azure",
    max_tokens: int = 1200,
    resume: bool = False,
    max_workers: int = 1,
) -> dict[str, int]:
    """Review evidence quality after writing, without exposing a gold answer."""

    if Path(output_jsonl).exists() and not resume:
        raise FileExistsError(f"Quality-review output exists; use --resume or choose a new path: {output_jsonl}")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    completed = load_completed_ids(output_jsonl, require_success=True) if resume else set()
    counts = {"seen": 0, "written": 0, "skipped_resume": 0, "errors": 0}

    def review_one(record: dict[str, Any], case_id: str) -> tuple[dict[str, Any], bool]:
        try:
            response = LLM_WORKER.generate(
                ModelRequest(
                    model=model_alias,
                    messages=build_quality_reviewer_messages(record),
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=max_tokens,
                    metadata={"trace_label": "fvqa_fix2_trajectory_quality_review"},
                )
            )
            parsed = _parse_json_object(response.content or "")
            if parsed is None:
                return {
                    "case_id": case_id,
                    "status": "error",
                    "error": "malformed_reviewer_json",
                    "raw_model_output": str(response.content or ""),
                    "model_alias": model_alias,
                }, True
            verdict, invalid = _normalize_review(parsed)
            if invalid:
                verdict = {**verdict, "status": "error", "error": invalid}
            else:
                verdict = {**verdict, "status": "ok"}
            return {
                "case_id": case_id,
                **verdict,
                "model_alias": model_alias,
                "raw_model_output": str(response.content or ""),
            }, bool(invalid)
        except Exception as error:
            return {
                "case_id": case_id,
                "status": "error",
                "error": f"{error.__class__.__name__}: {error}",
                "model_alias": model_alias,
            }, True

    def persist(value: tuple[dict[str, Any], bool]) -> None:
        row, has_error = value
        append_jsonl(output_jsonl, row)
        counts["written"] += 1
        counts["errors"] += int(has_error)

    if max_workers == 1:
        for record in iter_json_records(input_jsonl):
            case_id = _record_id(record)
            if not case_id:
                continue
            counts["seen"] += 1
            if case_id in completed:
                counts["skipped_resume"] += 1
                continue
            persist(review_one(record, case_id))
        return counts

    # Keep no more than two batches in memory.  Worker completion can be out
    # of order, but the coordinator writes only the contiguous input prefix so
    # audit JSONL remains deterministic and is safe to resume after a crash.
    next_index = 0
    next_to_write = 0
    outcomes: dict[int, tuple[dict[str, Any], bool]] = {}
    pending: dict[Any, int] = {}

    def collect_done(*, block: bool) -> None:
        nonlocal next_to_write
        if not pending:
            return
        if block:
            done, _ = wait(pending, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in pending if future.done()}
        for future in done:
            index = pending.pop(future)
            outcomes[index] = future.result()
        while next_to_write in outcomes:
            persist(outcomes.pop(next_to_write))
            next_to_write += 1

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fix2-quality") as pool:
        for record in iter_json_records(input_jsonl):
            case_id = _record_id(record)
            if not case_id:
                continue
            counts["seen"] += 1
            if case_id in completed:
                counts["skipped_resume"] += 1
                continue
            pending[pool.submit(review_one, record, case_id)] = next_index
            next_index += 1
            if len(pending) >= max_workers * 2:
                collect_done(block=True)
        while pending:
            collect_done(block=True)
    return counts
