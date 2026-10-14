"""Blind visual-review protocol and local payload construction."""

from __future__ import annotations

import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import mimetypes
from pathlib import Path
from typing import Any

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest

from .io import append_jsonl, iter_json_records, load_completed_ids
from .schema import BlindVisualRequest
from .trajectory import TrajectoryProvenance, parse_trajectory


BLIND_VISUAL_SYSTEM_PROMPT = """You are an independent visual verifier.
You receive one original image and candidate images. You do not receive any
search title, snippet, URL, trajectory, candidate name, answer, or hidden
label. Judge visual correspondence only.

Return exactly JSON:
{
  "decision": "keep" | "filter",
  "verdict": "CONSISTENT" | "SAME_ENTITY" | "SIMILAR" | "UNRELATED" | "DIFFERENT_PHOTO" | "UNRESOLVED",
  "confidence": 0.0,
  "positive_candidate_indices": [0],
  "evidence": ["concrete visible comparison"],
  "reason": "short"
}
Use SAME_ENTITY only when the images visibly show the same person/object/place.
Use CONSISTENT when the candidate visibly supports the public target description
but exact identity cannot be established. SIMILAR and UNRESOLVED must filter.
Do not infer names or facts from hidden metadata."""


def _to_model_image_url(value: str) -> str:
    """Convert a local collector image into a portable data URL for the VLM."""

    source = str(value or "").strip()
    if source.startswith(("data:image/", "http://", "https://")):
        return source
    path = Path(source)
    if not path.is_file():
        raise ValueError(f"Blind reviewer cannot access image source: {source}")
    mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
    if not mime_type.startswith("image/"):
        raise ValueError(f"Blind reviewer source is not an image: {source}")
    return f"data:{mime_type};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def build_blind_visual_request(record: dict[str, Any], provenance: TrajectoryProvenance | None = None) -> BlindVisualRequest | None:
    provenance = provenance or parse_trajectory(record)
    input_images = record.get("input_images") or []
    original = ""
    for item in input_images:
        if isinstance(item, dict):
            original = str(item.get("image_path") or item.get("image_url") or "").strip()
        elif isinstance(item, str):
            original = item.strip()
        if original:
            break
    candidate_ids = sorted(provenance.opened_candidate_ids)
    runtime_images = {
        str(item.get("resource_id") or ""): str(
            item.get("image_url") or item.get("primary_url") or item.get("thumbnail_url") or ""
        )
        for item in ((record.get("runtime") or {}).get("url_resources") or [])
        if isinstance(item, dict)
    }
    candidate_urls = [
        provenance.candidate_image_urls.get(candidate_id) or runtime_images.get(candidate_id) or ""
        for candidate_id in candidate_ids
    ]
    paired = [(candidate_id, image_url) for candidate_id, image_url in zip(candidate_ids, candidate_urls) if image_url]
    candidate_ids = [candidate_id for candidate_id, _image_url in paired]
    candidate_urls = [image_url for _candidate_id, image_url in paired]
    if not original or not candidate_urls:
        return None
    # The question is public input, so it is safe to use as a target description.
    # A later optional annotator may replace this with an equally public visual phrase.
    description = str(record.get("public_visual_target") or record.get("question") or "").strip()
    return BlindVisualRequest(
        case_id=str(record.get("case_id") or record.get("question_id") or ""),
        original_image=original,
        candidate_images=candidate_urls,
        public_target_description=description,
        candidate_resource_ids=candidate_ids,
    )


def make_blind_reviewer_messages(request: BlindVisualRequest) -> list[dict[str, Any]]:
    """Build a multimodal request with no trajectory/search metadata leakage."""

    content: list[dict[str, Any]] = [
        {"type": "text", "text": "Original image:"},
        {"type": "image_url", "image_url": {"url": _to_model_image_url(request.original_image), "detail": "high"}},
        {
            "type": "text",
            "text": (
                "Public visual target description from the user question:\n"
                f"{request.public_target_description}\n\nCandidate images (numbered in order):"
            ),
        },
    ]
    for index, image in enumerate(request.candidate_images):
        content.extend(
            [
                {"type": "text", "text": f"Candidate {index}:"},
                {"type": "image_url", "image_url": {"url": _to_model_image_url(image), "detail": "high"}},
            ]
        )
    return [{"role": "system", "content": BLIND_VISUAL_SYSTEM_PROMPT}, {"role": "user", "content": content}]


def validate_blind_visual_verdict(value: Any) -> tuple[bool, str]:
    if not isinstance(value, dict):
        return False, "blind_visual_review_missing_or_malformed"
    verdict = str(value.get("verdict") or "").upper()
    decision = str(value.get("decision") or "").lower()
    if decision != "keep":
        return False, f"blind_visual_review_{verdict.lower() or 'filter'}"
    if verdict not in {"CONSISTENT", "SAME_ENTITY"}:
        return False, f"blind_visual_review_{verdict.lower() or 'invalid'}"
    if not value.get("evidence"):
        return False, "blind_visual_review_missing_visible_evidence"
    return True, ""


def load_blind_reviews(path: str | Path) -> dict[str, dict[str, Any]]:
    """Accept JSONL review rows keyed by case_id without tying filtering to a model."""

    reviews: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict) and str(value.get("case_id") or ""):
            reviews[str(value["case_id"])] = value
    return reviews


def _parse_json_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(str(text or "").strip())
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start = str(text or "").find("{")
        if start < 0:
            return None
        try:
            value, _end = json.JSONDecoder().raw_decode(str(text)[start:])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


def review_blind_visual_requests(
    input_jsonl: str | Path,
    output_jsonl: str | Path,
    *,
    model_alias: str,
    max_tokens: int = 800,
    resume: bool = False,
    max_workers: int = 1,
) -> dict[str, int]:
    """Run an independent VLM only on a sanitized blind payload."""

    if Path(output_jsonl).exists() and not resume:
        raise FileExistsError(f"Blind-review output exists; use --resume or choose a new path: {output_jsonl}")
    if max_workers < 1:
        raise ValueError("max_workers must be positive")
    completed = load_completed_ids(output_jsonl, require_success=True) if resume else set()
    counts = {"seen": 0, "written": 0, "skipped_resume": 0, "unreviewable": 0, "errors": 0}

    def review_one(record: dict[str, Any], case_id: str) -> tuple[dict[str, Any], str]:
        request = build_blind_visual_request(record)
        if request is None:
            return {"case_id": case_id, "decision": "filter", "verdict": "UNRESOLVED", "reason": "no_opened_candidate_image"}, "unreviewable"
        messages = make_blind_reviewer_messages(request)
        try:
            response = LLM_WORKER.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=BLIND_VISUAL_SYSTEM_PROMPT),
                        ModelMessage(role="user", content=messages[1]["content"]),
                    ],
                    response_format={"type": "json_object"},
                    temperature=0.0,
                    max_tokens=max_tokens,
                    metadata={"trace_label": "fvqa_fix2_blind_visual_review"},
                )
            )
            verdict = _parse_json_object(response.content or "")
            if verdict is None:
                verdict = {"decision": "filter", "verdict": "UNRESOLVED", "reason": "malformed_reviewer_json"}
            return {
                "case_id": case_id,
                **verdict,
                "reviewed_candidate_resource_ids": request.candidate_resource_ids,
            }, "written"
        except Exception as error:
            return {
                "case_id": case_id,
                "decision": "filter",
                "verdict": "UNRESOLVED",
                "error": f"{error.__class__.__name__}: {error}",
            }, "error"

    def persist(value: tuple[dict[str, Any], str]) -> None:
        row, kind = value
        append_jsonl(output_jsonl, row)
        counts[kind] += 1

    if max_workers == 1:
        for record in iter_json_records(input_jsonl):
            case_id = str(record.get("case_id") or record.get("question_id") or "").strip()
            if not case_id:
                continue
            counts["seen"] += 1
            if case_id in completed:
                counts["skipped_resume"] += 1
                continue
            persist(review_one(record, case_id))
        return counts

    next_index = 0
    next_to_write = 0
    outcomes: dict[int, tuple[dict[str, Any], str]] = {}
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

    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="fix2-blind-visual") as pool:
        for record in iter_json_records(input_jsonl):
            case_id = str(record.get("case_id") or record.get("question_id") or "").strip()
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
