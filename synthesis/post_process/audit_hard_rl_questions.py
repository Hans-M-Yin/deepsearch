#!/usr/bin/env python3
"""Audit the answerability and gold correctness of hard RL questions.

This is a second-stage audit for datasets produced by
``filter_rl_vqa_dataset.py``.  The original quality judge deliberately checks
only whether a question distorts its construction hops.  This script reuses
that result and audits the dimensions it does not cover:

* whether the reference answer is correct and complete;
* whether the wording and any supplied image identify a unique answer;
* whether visual evidence can be uniquely discovered with the available web
  tools when no image is supplied with the question;
* whether the complete item is objectively scoreable.

No policy rollout is performed.  Only records already labelled ``hard`` in a
source manifest are sent to the judges.  A dedicated uniqueness judge runs
first; its structured finding is then independently reviewed by the general
quality judges.  The script is resumable and appends one compact audit record
per hard question.  It produces three dataset variants:

* ``rl_dataset_drop_filtered.jsonl`` removes only high-confidence ``drop``
  decisions;
* ``rl_dataset_quality_filtered.jsonl`` additionally removes hard questions
  whose validated uniqueness decision is ``ambiguous`` and whose aggregate
  quality decision is ``repair`` or ``review`` (the recommended automatic
  cleanup);
* ``rl_dataset_strict.jsonl`` additionally quarantines ``repair`` and
  ``review`` hard questions, retaining only hard questions judged
  ``keep_hard``.

Example:

    python synthesis/post_process/audit_hard_rl_questions.py \
      --dataset-jsonl runs/.../rl_combined_filtered/rl_dataset.jsonl \
      --source eight=runs/.../0825_batch_1_filtered \
      --source six=runs/.../rl_sft_overlap_2k_nonoverlap_2k_filtered \
      --output-dir runs/.../rl_combined_hard_quality_audit \
      --uniqueness-judge-alias gpt54_internal_azure \
      --judge-model-alias gpt54_internal_azure \
      --judge-model-alias gemini37flash_internal_azure \
      --judge-repeats 1 \
      --resume \
      --llm-usage-debug \
      --workers 32
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import logging
import mimetypes
import os
import re
import sys
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps
from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logger = logging.getLogger("audit_hard_rl_questions")


AUDIT_SYSTEM_PROMPT = """You are a strict dataset auditor for multimodal, multi-hop deep-research QA.

Your task is to assess the benchmark item itself, not whether a particular policy model is capable of solving it. The item was previously labelled hard because a tool-using model answered it incorrectly in all evaluation attempts. Difficulty alone is never a quality defect.

You receive the question, reference answer, construction evidence, the result of an earlier relation-consistency audit, deterministic file checks, and any image supplied with the question. This is an agentic visual-research task: when no image is supplied initially, the policy is expected to use web/image-search and page-reading tools to discover the described image or other visual evidence. Absence of an initial image is therefore not a defect by itself. Decide whether the item has one objectively defensible answer and can be scored reliably.

Audit all of the following:
1. Gold correctness: Is the reference answer factually compatible with the supplied evidence? Do not claim the gold is wrong merely because terminal evidence is absent.
2. Gold completeness: Does it answer every entity, date, number, comparison, or sub-question explicitly requested?
3. Unique answerability: Do the wording and any supplied image uniquely identify the intended target? To claim ambiguity, name at least two concrete candidates or readings that produce different answers and explain why both satisfy the wording.
4. Visual-search validity: If an image is supplied, is it readable, relevant, and sufficiently specific? If no image is supplied, do the textual clues identify visual evidence that an agent can reasonably discover online? Do not use ``missing_required_image`` merely because the initial ``images`` field is empty. Use ``unretrievable_visual_target`` only when the task requires a particular unseen image but the wording does not provide enough information to locate or distinguish it.
5. Construction validity: Does any hop reverse subject/object, alter chronology, conflate different versions or objects, or strengthen a weaker source relation?
6. Scoreability: Can a correct answer be judged objectively without relying on an unstated date, edition, image version, or subjective interpretation?

Decision policy:
- keep_hard: The item is valid, uniquely answerable, and the gold is correct and complete. It may remain very difficult.
- repair: There is a concrete defect, but a small change to the wording, image, evidence, or gold would make the item valid. Provide the exact repair.
- drop: There is a demonstrated objective defect that makes the current item unreliable or impossible to score, and no small unambiguous repair is evident.
- review: The packet is insufficient to establish validity or a defect. Missing terminal evidence is review, not drop.

Allowed issue categories:
wrong_gold, incomplete_gold, non_unique_referent, missing_required_image, unretrievable_visual_target, corrupt_image, wrong_image, visual_ambiguity, broken_relation, temporal_error, unsupported_strengthening, unscorable_question, conflicting_duplicate, other.

For drop, ``objective_failure`` must be true, at least one allowed category must be supplied, and the evidence must identify the exact contradiction, missing requirement, or two concrete alternative answers. Never drop merely because the item is obscure, requires many hops, has a long answer, or was answered incorrectly by the rollout model.

Return one JSON object only:
{
  "decision": "keep_hard" | "repair" | "drop" | "review",
  "objective_failure": true | false,
  "issue_categories": ["..."],
  "reason": "concise evidence-based explanation",
  "gold_assessment": "correct" | "incorrect" | "incomplete" | "uncertain",
  "unique_answerability": "unique" | "ambiguous" | "uncertain",
  "alternative_answers": [{"answer": "...", "why_it_fits": "..."}],
  "evidence": ["specific supporting or contradicting fact"],
  "repair_suggestion": "exact minimal repair, or empty string"
}
"""


UNIQUENESS_SYSTEM_PROMPT = """You are the dedicated uniqueness auditor for multimodal, multi-hop deep-research QA data.

Assess the benchmark item, not model capability. This is an agentic visual-research task: the solver may use web search, image search, and page-reading tools to locate visual evidence described by the question. An empty initial image list is expected for many valid examples and is not ambiguity or missing data by itself. Decide whether the question, discoverable web evidence, and any supplied image determine one objectively scoreable answer. The reference answer and construction evidence reveal the intended path, but they are not extra constraints silently available to the solver and must not be used to manufacture uniqueness.

Treat an item as ambiguous only when at least two concrete interpretations or referents satisfy every explicit constraint in the question yet lead to materially different answers. Check both visual and knowledge ambiguity:
- visual_referent: multiple visible objects or visually plausible entities satisfy the image wording;
- visual_retrievability: the question requires one particular unseen image, but its textual clues cannot locate or distinguish that image through reasonable web/image search;
- undefined_knowledge_relation: terms such as closest, first, main, related, influential, or similar lack a named metric, candidate set, or authoritative definition;
- unspecified_temporal_scope: the answer depends on an unstated year, date, edition, version, or historical period;
- unspecified_statistical_measure: the answer depends on whether the statistic means native speakers, home language, self-identification, population, etc.;
- entity_collision: multiple people, places, works, organizations, or objects share the description or name;
- scope_or_quantifier: geography, population, inclusion criteria, or requested subparts permit different complete answers;
- answer_granularity: multiple levels of specificity would be scored differently even though each answers the wording.

Do not mark ambiguity merely because the question is hard, obscure, multi-hop, paraphrased, or requires web research. Do not accept a second candidate that violates an explicit constraint. Normal aliases, equivalent rounding, and semantically equivalent phrasings are one answer, not ambiguity. If evidence is insufficient to prove two valid readings, return uncertain rather than guessing.

Real dataset example (knowledge ambiguity):
Question: "The ornate gold crown shown in the image, which is adorned with gemstones, pearls, and a cross, has a native name from a certain language. According to a census in a particular year, what is the approximate number of speakers of the West Slavic language that ranks third in its closeness to that language?"
Reference answer: "According to the 2011 census, ... Silesian ... had over 530,000 speakers."
Correct assessment: ambiguous. "Ranks third in closeness" does not specify a language-distance metric or candidate set, while "a census in a particular year" omits the year and statistical measure. The construction path's intended Silesian/2011 reading does not repair constraints absent from the question. A minimal repair must explicitly name the distance source or criterion and census year and measure, or directly ask for the intended language and statistic.

Return one JSON object only:
{
  "decision": "unique" | "ambiguous" | "uncertain",
  "ambiguity_types": ["visual_referent" | "visual_retrievability" | "undefined_knowledge_relation" | "unspecified_temporal_scope" | "unspecified_statistical_measure" | "entity_collision" | "scope_or_quantifier" | "answer_granularity" | "other"],
  "explicit_constraints": ["constraint stated in the question"],
  "candidate_interpretations": [
    {
      "candidate": "concrete referent or reading",
      "why_it_satisfies_all_constraints": "constraint-by-constraint explanation",
      "resulting_answer": "materially different answer"
    }
  ],
  "missing_disambiguator": "the smallest missing constraint, or empty string",
  "reason": "concise evidence-based explanation",
  "repair_suggestion": "exact minimal wording repair, or empty string"
}

For an ambiguous decision, provide at least two concrete candidate interpretations with different resulting answers and a non-empty missing_disambiguator. Otherwise return uncertain.
"""


ALLOWED_AMBIGUITY_TYPES = {
    "visual_referent",
    "visual_retrievability",
    "undefined_knowledge_relation",
    "unspecified_temporal_scope",
    "unspecified_statistical_measure",
    "entity_collision",
    "scope_or_quantifier",
    "answer_granularity",
    "other",
}


ALLOWED_CATEGORIES = {
    "wrong_gold",
    "incomplete_gold",
    "non_unique_referent",
    "missing_required_image",
    "unretrievable_visual_target",
    "corrupt_image",
    "wrong_image",
    "visual_ambiguity",
    "broken_relation",
    "temporal_error",
    "unsupported_strengthening",
    "unscorable_question",
    "conflicting_duplicate",
    "other",
}

# ``other`` is intentionally excluded: an unstructured complaint must never
# delete a training item automatically.
AUTO_DROP_CATEGORIES = ALLOWED_CATEGORIES - {"other"}
VISUAL_REFERENCE_RE = re.compile(
    r"\b(image|photo|photograph|picture|pictured|shown|depicted|illustration|visual)\b",
    re.IGNORECASE,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            yield value


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str) + "\n")
            count += 1
    return count


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")


def _parse_source(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"Source must have the form PREFIX=DIR, got {raw!r}")
    prefix, path_text = raw.split("=", 1)
    prefix = prefix.strip()
    source_dir = Path(path_text.strip()).expanduser().resolve()
    if not prefix or not prefix.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid source prefix: {prefix!r}")
    manifest_path = source_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing source manifest: {manifest_path}")
    return prefix, source_dir


def _normalise_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _record_images(record: dict[str, Any]) -> list[Any]:
    value = record.get("images")
    if value is None:
        value = record.get("image_url") or record.get("image_urls")
    if value is None:
        return []
    return value if isinstance(value, list) else [value]


def _image_reference(value: Any) -> str:
    if isinstance(value, dict):
        value = value.get("path") or value.get("url") or value.get("image_url")
    return str(value or "").strip()


def _inspect_images(record: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    inspections: list[dict[str, Any]] = []
    issues: list[str] = []
    for index, raw_value in enumerate(_record_images(record)):
        reference = _image_reference(raw_value)
        item: dict[str, Any] = {"index": index, "reference": reference}
        if not reference:
            item["status"] = "missing_reference"
            issues.append("missing_image_reference")
        elif reference.startswith(("http://", "https://", "data:image/")):
            item["status"] = "remote_or_inline_unchecked"
        else:
            path = Path(reference).expanduser()
            item["path"] = str(path)
            if not path.is_file():
                item["status"] = "missing_file"
                issues.append("missing_image_file")
            else:
                try:
                    with Image.open(path) as image:
                        image.verify()
                    with Image.open(path) as image:
                        item.update(
                            {
                                "status": "readable",
                                "format": image.format,
                                "width": image.width,
                                "height": image.height,
                                "mode": image.mode,
                                "bytes": path.stat().st_size,
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    item["status"] = "corrupt_or_unreadable"
                    item["error"] = f"{type(exc).__name__}: {exc}"[:500]
                    issues.append("corrupt_image")
        inspections.append(item)

    return inspections, sorted(set(issues))


def _deterministic_checks(
    record: dict[str, Any],
    *,
    duplicate_answer_count: int,
) -> dict[str, Any]:
    question = str(record.get("question") or "").strip()
    answer = record.get("answer")
    answer_text = str(answer or "").strip()
    image_inspections, image_issues = _inspect_images(record)
    issues = list(image_issues)
    if not question:
        issues.append("missing_question")
    if not answer_text:
        issues.append("missing_gold_answer")
    if duplicate_answer_count > 1:
        issues.append("conflicting_duplicate")

    fatal_categories: list[str] = []
    if not question or not answer_text:
        fatal_categories.append("unscorable_question")
    if "corrupt_image" in issues and image_inspections and all(
        item.get("status") == "corrupt_or_unreadable" for item in image_inspections
    ):
        fatal_categories.append("corrupt_image")
    if "missing_image_file" in issues and VISUAL_REFERENCE_RE.search(question) and image_inspections and all(
        item.get("status") in {"missing_file", "missing_reference"}
        for item in image_inspections
    ):
        fatal_categories.append("missing_required_image")

    return {
        "issues": sorted(set(issues)),
        "fatal_categories": sorted(set(fatal_categories)),
        "image_inspections": image_inspections,
        "conflicting_normalized_answer_count": duplicate_answer_count,
    }


def _local_image_data_url(
    path: Path,
    *,
    max_edge: int,
    max_original_bytes: int,
) -> str:
    if path.stat().st_size <= max_original_bytes:
        mime_type = mimetypes.guess_type(str(path))[0] or "image/png"
        if mime_type.startswith("image/"):
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            return f"data:{mime_type};base64,{encoded}"

    with Image.open(path) as raw_image:
        image = ImageOps.exif_transpose(raw_image)
        image.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        if "A" in image.getbands():
            image.save(output, format="PNG", optimize=True)
            mime_type = "image/png"
        else:
            image.convert("RGB").save(output, format="JPEG", quality=90, optimize=True)
            mime_type = "image/jpeg"
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _image_content_blocks(
    record: dict[str, Any],
    inspections: list[dict[str, Any]],
    *,
    max_images: int,
    max_edge: int,
    max_original_bytes: int,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for raw_value, inspection in zip(_record_images(record), inspections, strict=False):
        if len(blocks) >= max_images:
            break
        reference = _image_reference(raw_value)
        if inspection.get("status") == "readable":
            url = _local_image_data_url(
                Path(reference),
                max_edge=max_edge,
                max_original_bytes=max_original_bytes,
            )
        elif reference.startswith(("http://", "https://", "data:image/")):
            url = reference
        else:
            continue
        blocks.append({"type": "image_url", "image_url": {"url": url}})
    return blocks


def _parse_json_object(raw: str) -> dict[str, Any] | None:
    text = str(raw or "").strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    candidates = [text]
    object_match = re.search(r"\{.*\}", text, re.DOTALL)
    if object_match:
        candidates.append(object_match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _parse_judge_result(raw: str) -> dict[str, Any]:
    parsed = _parse_json_object(raw)
    if parsed is None:
        return {
            "parse_ok": False,
            "decision": "review",
            "objective_failure": False,
            "issue_categories": [],
            "reason": "judge_output_not_json",
            "raw": str(raw or "")[:4000],
        }

    decision = str(parsed.get("decision") or "").strip().lower()
    if decision not in {"keep_hard", "repair", "drop", "review"}:
        return {
            "parse_ok": False,
            "decision": "review",
            "objective_failure": False,
            "issue_categories": [],
            "reason": f"invalid_decision:{decision}",
            "raw": str(raw or "")[:4000],
        }

    categories = parsed.get("issue_categories") or []
    if isinstance(categories, str):
        categories = [categories]
    categories = [str(value).strip().lower() for value in categories]
    categories = [value for value in categories if value in ALLOWED_CATEGORIES]
    alternatives = parsed.get("alternative_answers") or []
    if not isinstance(alternatives, list):
        alternatives = []
    evidence = parsed.get("evidence") or []
    if isinstance(evidence, str):
        evidence = [evidence]
    if not isinstance(evidence, list):
        evidence = []

    return {
        "parse_ok": True,
        "decision": decision,
        "objective_failure": bool(parsed.get("objective_failure", False)),
        "issue_categories": categories[:8],
        "reason": str(parsed.get("reason") or "")[:2000],
        "gold_assessment": str(parsed.get("gold_assessment") or "uncertain")[:40],
        "unique_answerability": str(parsed.get("unique_answerability") or "uncertain")[:40],
        "alternative_answers": alternatives[:5],
        "evidence": [str(value)[:1000] for value in evidence[:8]],
        "repair_suggestion": str(parsed.get("repair_suggestion") or "")[:2000],
        "raw": str(raw or "")[:4000],
    }


def _parse_uniqueness_result(raw: str) -> dict[str, Any]:
    parsed = _parse_json_object(raw)
    if parsed is None:
        return {
            "parse_ok": False,
            "decision": "uncertain",
            "ambiguity_types": [],
            "reason": "uniqueness_judge_output_not_json",
            "raw": str(raw or "")[:4000],
        }

    decision = str(parsed.get("decision") or "").strip().lower()
    if decision not in {"unique", "ambiguous", "uncertain"}:
        return {
            "parse_ok": False,
            "decision": "uncertain",
            "ambiguity_types": [],
            "reason": f"invalid_uniqueness_decision:{decision}",
            "raw": str(raw or "")[:4000],
        }

    ambiguity_types = parsed.get("ambiguity_types") or []
    if isinstance(ambiguity_types, str):
        ambiguity_types = [ambiguity_types]
    ambiguity_types = [str(value).strip().lower() for value in ambiguity_types]
    ambiguity_types = [
        value for value in ambiguity_types if value in ALLOWED_AMBIGUITY_TYPES
    ]

    constraints = parsed.get("explicit_constraints") or []
    if isinstance(constraints, str):
        constraints = [constraints]
    if not isinstance(constraints, list):
        constraints = []

    candidates = parsed.get("candidate_interpretations") or []
    if not isinstance(candidates, list):
        candidates = []
    valid_candidates = []
    for candidate in candidates[:5]:
        if not isinstance(candidate, dict):
            continue
        normalized = {
            "candidate": str(candidate.get("candidate") or "")[:1000],
            "why_it_satisfies_all_constraints": str(
                candidate.get("why_it_satisfies_all_constraints") or ""
            )[:1500],
            "resulting_answer": str(candidate.get("resulting_answer") or "")[:1000],
        }
        if all(normalized.values()):
            valid_candidates.append(normalized)

    missing_disambiguator = str(parsed.get("missing_disambiguator") or "")[:1500]
    validation_error = ""
    if decision == "ambiguous":
        distinct_answers = {
            _normalise_text(candidate["resulting_answer"])
            for candidate in valid_candidates
            if _normalise_text(candidate["resulting_answer"])
        }
        if len(valid_candidates) < 2 or len(distinct_answers) < 2:
            validation_error = "ambiguous_requires_two_concrete_distinct_answers"
        elif not missing_disambiguator.strip():
            validation_error = "ambiguous_requires_missing_disambiguator"
        if validation_error:
            decision = "uncertain"

    return {
        "parse_ok": True,
        "decision": decision,
        "ambiguity_types": ambiguity_types[:8],
        "explicit_constraints": [str(value)[:1000] for value in constraints[:12]],
        "candidate_interpretations": valid_candidates,
        "missing_disambiguator": missing_disambiguator,
        "reason": str(parsed.get("reason") or "")[:2000],
        "repair_suggestion": str(parsed.get("repair_suggestion") or "")[:2000],
        "validation_error": validation_error,
        "raw": str(raw or "")[:4000],
    }


def _aggregate_decision(
    judge_results: list[dict[str, Any]],
    *,
    fatal_categories: list[str],
    min_drop_votes: int,
    uniqueness_result: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    if fatal_categories:
        return "drop", {
            "basis": "deterministic_fatal_check",
            "drop_votes": 0,
            "shared_drop_categories": fatal_categories,
        }

    valid = [result for result in judge_results if result.get("parse_ok")]
    drop_results: list[dict[str, Any]] = []
    category_votes: Counter[str] = Counter()
    for result in valid:
        categories = set(result.get("issue_categories") or []) & AUTO_DROP_CATEGORIES
        ambiguity = bool(categories & {"non_unique_referent", "visual_ambiguity"})
        alternatives = result.get("alternative_answers") or []
        if (
            result.get("decision") == "drop"
            and result.get("objective_failure")
            and categories
            and result.get("evidence")
            and (not ambiguity or len(alternatives) >= 2)
        ):
            drop_results.append(result)
            category_votes.update(categories)
    shared_categories = sorted(
        category for category, votes in category_votes.items() if votes >= min_drop_votes
    )
    if len(drop_results) >= min_drop_votes and shared_categories:
        return "drop", {
            "basis": "independent_drop_consensus",
            "drop_votes": len(drop_results),
            "shared_drop_categories": shared_categories,
        }

    decisions = Counter(str(result.get("decision")) for result in valid)
    uniqueness_ambiguous = bool(
        uniqueness_result
        and uniqueness_result.get("parse_ok")
        and uniqueness_result.get("decision") == "ambiguous"
    )
    if (
        valid
        and decisions.get("keep_hard", 0) == len(valid)
        and len(valid) == len(judge_results)
        and not uniqueness_ambiguous
    ):
        final = "keep_hard"
        basis = "all_judges_keep"
    elif decisions.get("repair", 0) > 0 and decisions.get("drop", 0) == 0:
        final = "repair"
        basis = "repair_requested_without_drop_consensus"
    elif uniqueness_ambiguous and decisions.get("drop", 0) == 0:
        final = "review"
        basis = "uniqueness_quality_disagreement"
    else:
        final = "review"
        basis = "insufficient_or_disagreeing_judges"
    return final, {
        "basis": basis,
        "drop_votes": len(drop_results),
        "shared_drop_categories": shared_categories,
        "judge_decisions": dict(decisions),
        "uniqueness_decision": (
            uniqueness_result.get("decision") if uniqueness_result else "not_run"
        ),
    }


def _build_audit_payload(
    record: dict[str, Any],
    source_manifest: dict[str, Any],
    deterministic: dict[str, Any],
) -> dict[str, Any]:
    previous_quality = source_manifest.get("quality_judge") or {}
    audit_input = previous_quality.get("audit_input") or {}
    return {
        "question_id": record.get("id"),
        "question": record.get("question"),
        "reference_answer": record.get("answer"),
        "difficulty_level": source_manifest.get("difficulty_level"),
        "difficulty_features": source_manifest.get("difficulty_features") or {},
        "previous_relation_audit": {
            "decision": previous_quality.get("decision"),
            "reject_categories": previous_quality.get("reject_categories") or [],
            "reasons": previous_quality.get("reasons") or [],
        },
        "construction_hop_chain": audit_input.get("hop_chain") or [],
        "path_summary": audit_input.get("path_summary") or {},
        "source_path_features": source_manifest.get("path_features") or {},
        "task_semantics": {
            "agent_may_search_web": True,
            "agent_may_search_images": True,
            "empty_initial_images_is_allowed": True,
        },
        "deterministic_checks": {
            "issues": deterministic.get("issues") or [],
            "image_inspections": deterministic.get("image_inspections") or [],
            "conflicting_normalized_answer_count": deterministic.get(
                "conflicting_normalized_answer_count", 1
            ),
        },
    }


def _build_uniqueness_prompt(
    record: dict[str, Any],
    source_manifest: dict[str, Any],
    deterministic: dict[str, Any],
) -> str:
    payload = _build_audit_payload(record, source_manifest, deterministic)
    return (
        "Audit whether this hard benchmark item uniquely determines its answer, "
        "using only constraints visible to the solver and evidence the solver can "
        "discover with web/image-search tools. Attached images, if any, are initial "
        "inputs; when none are attached, visual discovery is an intended part of the "
        "task and must not be treated as a missing-image defect.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def _build_prompt(
    record: dict[str, Any],
    source_manifest: dict[str, Any],
    deterministic: dict[str, Any],
    uniqueness_result: dict[str, Any],
) -> str:
    payload = _build_audit_payload(record, source_manifest, deterministic)
    payload["independent_uniqueness_audit"] = {
        key: value
        for key, value in uniqueness_result.items()
        if key not in {"raw", "model_alias"}
    }
    return (
        "Audit this hard benchmark item using the policy in the system message. "
        "Attached images, if any, are initial inputs. When none are attached, the "
        "agent is expected to discover visual evidence with web/image-search tools; "
        "do not report missing_required_image solely for an empty images field. "
        "Do not infer that the item is defective merely from its hard label. "
        "The independent uniqueness audit is a hypothesis to verify, not ground truth.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    )


def _judge_once(
    *,
    prompt: str,
    image_blocks: list[dict[str, Any]],
    model_alias: str,
    max_tokens: int,
    case_id: str,
    repeat: int,
) -> dict[str, Any]:
    from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(image_blocks)
    try:
        response = LLM_WORKER.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=AUDIT_SYSTEM_PROMPT),
                    ModelMessage(role="user", content=content),
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                metadata={
                    "trace_label": "rl_hard_question_quality_audit",
                    "case_id": case_id,
                    "judge_repeat": repeat,
                },
            )
        )
        result = _parse_judge_result(str(response.content or ""))
    except Exception as exc:  # noqa: BLE001
        result = {
            "parse_ok": False,
            "decision": "review",
            "objective_failure": False,
            "issue_categories": [],
            "reason": f"judge_request_failed:{type(exc).__name__}:{exc}"[:2000],
        }
    result["model_alias"] = model_alias
    result["repeat"] = repeat
    return result


def _uniqueness_judge_once(
    *,
    prompt: str,
    image_blocks: list[dict[str, Any]],
    model_alias: str,
    max_tokens: int,
    case_id: str,
) -> dict[str, Any]:
    from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    content.extend(image_blocks)
    try:
        response = LLM_WORKER.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=UNIQUENESS_SYSTEM_PROMPT),
                    ModelMessage(role="user", content=content),
                ],
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                metadata={
                    "trace_label": "rl_hard_question_uniqueness_audit",
                    "case_id": case_id,
                },
            )
        )
        result = _parse_uniqueness_result(str(response.content or ""))
    except Exception as exc:  # noqa: BLE001
        result = {
            "parse_ok": False,
            "decision": "uncertain",
            "ambiguity_types": [],
            "reason": f"uniqueness_judge_request_failed:{type(exc).__name__}:{exc}"[:2000],
        }
    result["model_alias"] = model_alias
    return result


def _audit_one(
    *,
    record: dict[str, Any],
    source_manifest: dict[str, Any],
    duplicate_answer_count: int,
    uniqueness_model_alias: str,
    model_aliases: list[str],
    judge_repeats: int,
    max_tokens: int,
    min_drop_votes: int,
    max_images: int,
    image_max_edge: int,
    max_original_image_bytes: int,
    audit_signature: str,
) -> dict[str, Any]:
    deterministic = _deterministic_checks(
        record,
        duplicate_answer_count=duplicate_answer_count,
    )
    judge_results: list[dict[str, Any]] = []
    uniqueness_result: dict[str, Any] = {
        "parse_ok": False,
        "decision": "uncertain",
        "ambiguity_types": [],
        "reason": "not_run_due_to_deterministic_fatal_check",
        "model_alias": uniqueness_model_alias,
    }
    if not deterministic["fatal_categories"]:
        try:
            image_blocks = _image_content_blocks(
                record,
                deterministic["image_inspections"],
                max_images=max_images,
                max_edge=image_max_edge,
                max_original_bytes=max_original_image_bytes,
            )
            uniqueness_result = _uniqueness_judge_once(
                prompt=_build_uniqueness_prompt(record, source_manifest, deterministic),
                image_blocks=image_blocks,
                model_alias=uniqueness_model_alias,
                max_tokens=max_tokens,
                case_id=str(record.get("id")),
            )
            prompt = _build_prompt(
                record,
                source_manifest,
                deterministic,
                uniqueness_result,
            )
            for model_alias in model_aliases:
                for repeat in range(judge_repeats):
                    judge_results.append(
                        _judge_once(
                            prompt=prompt,
                            image_blocks=image_blocks,
                            model_alias=model_alias,
                            max_tokens=max_tokens,
                            case_id=str(record.get("id")),
                            repeat=repeat,
                        )
                    )
        except Exception as exc:  # noqa: BLE001
            judge_results.append(
                {
                    "parse_ok": False,
                    "decision": "review",
                    "objective_failure": False,
                    "issue_categories": [],
                    "reason": f"audit_preparation_failed:{type(exc).__name__}:{exc}"[:2000],
                }
            )

    final_decision, aggregation = _aggregate_decision(
        judge_results,
        fatal_categories=deterministic["fatal_categories"],
        min_drop_votes=min_drop_votes,
        uniqueness_result=uniqueness_result,
    )
    return {
        "case_id": str(record.get("id")),
        "question": str(record.get("question") or ""),
        "reference_answer": record.get("answer"),
        "images": [_image_reference(value) for value in _record_images(record)],
        "difficulty_level": "hard",
        "final_decision": final_decision,
        "aggregation": aggregation,
        "deterministic_checks": deterministic,
        "uniqueness_result": uniqueness_result,
        "judge_results": judge_results,
        "audit_signature": audit_signature,
        "audited_at": _utc_now(),
    }


def _load_source_manifests(
    sources: list[tuple[str, Path]],
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    records: dict[str, dict[str, Any]] = {}
    source_counts: dict[str, int] = {}
    for prefix, source_dir in sources:
        selected_count = 0
        for item in _iter_jsonl(source_dir / "manifest.jsonl"):
            if not item.get("selected"):
                continue
            local_id = str(item.get("id") or item.get("question_id") or "").strip()
            if not local_id:
                raise ValueError(f"Selected source item without id in {source_dir}")
            full_id = f"{prefix}__{local_id}"
            if full_id in records:
                raise ValueError(f"Duplicate merged source id: {full_id}")
            records[full_id] = item
            selected_count += 1
        source_counts[prefix] = selected_count
    return records, source_counts


def _load_dataset(path: Path) -> list[dict[str, Any]]:
    records = list(_iter_jsonl(path))
    ids: set[str] = set()
    for index, record in enumerate(records):
        case_id = str(record.get("id") or record.get("question_id") or "").strip()
        if not case_id:
            raise ValueError(f"Dataset row {index} has no id/question_id")
        if case_id in ids:
            raise ValueError(f"Duplicate dataset id: {case_id}")
        ids.add(case_id)
        record["id"] = case_id
    return records


def _duplicate_answer_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    answers_by_question: dict[str, set[str]] = defaultdict(set)
    for record in records:
        question = _normalise_text(record.get("question"))
        answer = _normalise_text(record.get("answer"))
        if question:
            answers_by_question[question].add(answer)
    return {
        str(record["id"]): len(answers_by_question[_normalise_text(record.get("question"))])
        for record in records
    }


def _audit_record_reusable(
    record: dict[str, Any],
    *,
    expected_quality_judge_calls: int,
) -> bool:
    if record.get("error"):
        return False
    deterministic = record.get("deterministic_checks") or {}
    if deterministic.get("fatal_categories"):
        return True
    uniqueness = record.get("uniqueness_result") or {}
    if not uniqueness.get("parse_ok") or uniqueness.get("validation_error"):
        return False
    judge_results = record.get("judge_results") or []
    return (
        len(judge_results) == expected_quality_judge_calls
        and all(result.get("parse_ok") for result in judge_results)
    )


def _load_audit_cache(
    path: Path,
    audit_signature: str,
    *,
    expected_quality_judge_calls: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    stats = {
        "records": 0,
        "malformed_records": 0,
        "matching_records": 0,
        "matching_cases": 0,
        "reusable_cases": 0,
        "retryable_cases": 0,
    }
    if not path.is_file():
        return {}, stats
    cached: dict[str, dict[str, Any]] = {}
    matching_case_ids: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            stats["records"] += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                stats["malformed_records"] += 1
                logger.warning(
                    "[resume] ignoring malformed audit record path=%s line=%d error=%s",
                    path,
                    line_number,
                    exc,
                )
                continue
            if not isinstance(record, dict) or record.get("audit_signature") != audit_signature:
                continue
            case_id = str(record.get("case_id") or "").strip()
            if not case_id:
                continue
            stats["matching_records"] += 1
            matching_case_ids.add(case_id)
            if _audit_record_reusable(
                record,
                expected_quality_judge_calls=expected_quality_judge_calls,
            ):
                cached[case_id] = record
    stats["matching_cases"] = len(matching_case_ids)
    stats["reusable_cases"] = len(cached)
    stats["retryable_cases"] = len(matching_case_ids - set(cached))
    return cached, stats


def _write_id_file(path: Path, ids: Iterable[str]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for case_id in ids:
            handle.write(f"{case_id}\n")


def _matches_ambiguous_repair_review_filter(record: dict[str, Any]) -> bool:
    """Return whether an audited item matches the conservative ambiguity filter.

    Failed, malformed, or validation-downgraded uniqueness results are never
    removed by this rule. They remain retryable/reviewable rather than being
    mistaken for confirmed ambiguity.
    """

    if record.get("final_decision") not in {"repair", "review"}:
        return False
    uniqueness = record.get("uniqueness_result") or {}
    return bool(
        uniqueness.get("parse_ok")
        and not uniqueness.get("validation_error")
        and uniqueness.get("decision") == "ambiguous"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit gold correctness and answerability of hard RL questions without rerunning rollouts"
    )
    parser.add_argument("--dataset-jsonl", type=Path, required=True)
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="PREFIX=DIR",
        help="Filtered source directory containing manifest.jsonl; repeat per source",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--uniqueness-judge-alias",
        default="gpt54_internal_azure",
        help="Independent model alias used for the single uniqueness audit",
    )
    parser.add_argument(
        "--judge-model-alias",
        action="append",
        default=None,
        help=(
            "Judge alias; repeat to use different independent judges "
            "(defaults: gpt54_internal_azure and gemini37flash_internal_azure)"
        ),
    )
    parser.add_argument("--judge-repeats", type=int, default=1)
    parser.add_argument("--workers", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1800)
    parser.add_argument("--min-drop-votes", type=int, default=2)
    parser.add_argument("--max-images", type=int, default=2)
    parser.add_argument("--image-max-edge", type=int, default=1536)
    parser.add_argument("--max-original-image-bytes", type=int, default=4 * 1024 * 1024)
    parser.add_argument("--limit", type=int, default=None, help="Audit only the first N pending hard items")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--llm-usage-debug",
        action="store_true",
        help="Enable model_worker's per-request [llm-usage] output",
    )
    resume_group = parser.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        help="Resume from reusable records in hard_quality_audit.jsonl (default)",
    )
    resume_group.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore matching cached audit records",
    )
    parser.set_defaults(resume=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.llm_usage_debug:
        os.environ["LLM_WORKER_PRINT_USAGE"] = "1"

    dataset_path = args.dataset_jsonl.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(dataset_path)
    if args.judge_repeats <= 0 or args.workers <= 0:
        raise ValueError("--judge-repeats and --workers must be positive")
    if args.max_tokens <= 0 or args.max_images <= 0 or args.image_max_edge <= 0:
        raise ValueError("Token and image limits must be positive")
    uniqueness_model_alias = str(args.uniqueness_judge_alias or "").strip()
    if not uniqueness_model_alias:
        raise ValueError("--uniqueness-judge-alias must not be empty")
    model_aliases = args.judge_model_alias or [
        "gpt54_internal_azure",
        "gemini37flash_internal_azure",
    ]
    quality_judge_call_count = len(model_aliases) * args.judge_repeats
    total_judge_call_count = quality_judge_call_count + 1
    if args.min_drop_votes <= 0 or args.min_drop_votes > quality_judge_call_count:
        raise ValueError(
            f"--min-drop-votes must be in [1, {quality_judge_call_count}], "
            f"got {args.min_drop_votes}"
        )

    sources = [_parse_source(raw) for raw in args.source]
    dataset = _load_dataset(dataset_path)
    source_manifests, source_counts = _load_source_manifests(sources)
    dataset_by_id = {str(record["id"]): record for record in dataset}
    missing_source_ids = sorted(set(dataset_by_id) - set(source_manifests))
    if missing_source_ids:
        raise ValueError(
            f"{len(missing_source_ids)} dataset IDs have no selected source manifest; "
            f"examples={missing_source_ids[:5]}"
        )

    hard_ids = [
        case_id
        for case_id in dataset_by_id
        if source_manifests[case_id].get("difficulty_level") == "hard"
    ]
    duplicate_counts = _duplicate_answer_counts(dataset)
    signature_payload = {
        "prompt_sha256": _sha256(AUDIT_SYSTEM_PROMPT),
        "dataset": str(dataset_path),
        "dataset_size": dataset_path.stat().st_size,
        "sources": [(prefix, str(path)) for prefix, path in sources],
        "uniqueness_prompt_sha256": _sha256(UNIQUENESS_SYSTEM_PROMPT),
        "uniqueness_model_alias": uniqueness_model_alias,
        "model_aliases": model_aliases,
        "judge_repeats": args.judge_repeats,
        "max_tokens": args.max_tokens,
        "min_drop_votes": args.min_drop_votes,
        "max_images": args.max_images,
        "image_max_edge": args.image_max_edge,
        "max_original_image_bytes": args.max_original_image_bytes,
    }
    audit_signature = _sha256(signature_payload)
    audit_path = output_dir / "hard_quality_audit.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume:
        cached, resume_stats = _load_audit_cache(
            audit_path,
            audit_signature,
            expected_quality_judge_calls=quality_judge_call_count,
        )
    else:
        cached = {}
        resume_stats = {
            "records": 0,
            "malformed_records": 0,
            "matching_records": 0,
            "matching_cases": 0,
            "reusable_cases": 0,
            "retryable_cases": 0,
        }
    pending_ids = [case_id for case_id in hard_ids if case_id not in cached]
    if args.limit is not None:
        pending_ids = pending_ids[: max(0, args.limit)]

    logger.info(
        "Dataset=%d hard=%d cached=%d pending=%d quality_judge_calls_per_item=%d "
        "total_judge_calls_per_item=%d",
        len(dataset),
        len(hard_ids),
        len(cached),
        len(pending_ids),
        quality_judge_call_count,
        total_judge_call_count,
    )
    logger.info(
        "[resume] enabled=%d path=%s reusable=%d retryable_failed=%d "
        "matching_records=%d malformed_records=%d remaining=%d",
        int(args.resume),
        audit_path,
        resume_stats["reusable_cases"],
        resume_stats["retryable_cases"],
        resume_stats["matching_records"],
        resume_stats["malformed_records"],
        len(pending_ids),
    )
    if args.dry_run:
        print(json.dumps({
            "dataset_count": len(dataset),
            "source_selected_counts": source_counts,
            "hard_count": len(hard_ids),
            "cached_count": len(cached),
            "pending_count": len(pending_ids),
            "uniqueness_judge_calls_per_item": 1,
            "quality_judge_calls_per_item": quality_judge_call_count,
            "total_judge_calls_per_item": total_judge_call_count,
            "planned_judge_calls": len(pending_ids) * total_judge_call_count,
            "resume_enabled": args.resume,
            "resume_stats": resume_stats,
            "llm_usage_debug": args.llm_usage_debug,
            "audit_signature": audit_signature,
        }, ensure_ascii=False, indent=2))
        return 0

    append_lock = threading.Lock()
    mode = "a" if audit_path.exists() else "w"
    with audit_path.open(mode, encoding="utf-8") as audit_handle:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    _audit_one,
                    record=dataset_by_id[case_id],
                    source_manifest=source_manifests[case_id],
                    duplicate_answer_count=duplicate_counts[case_id],
                    uniqueness_model_alias=uniqueness_model_alias,
                    model_aliases=model_aliases,
                    judge_repeats=args.judge_repeats,
                    max_tokens=args.max_tokens,
                    min_drop_votes=args.min_drop_votes,
                    max_images=args.max_images,
                    image_max_edge=args.image_max_edge,
                    max_original_image_bytes=args.max_original_image_bytes,
                    audit_signature=audit_signature,
                ): case_id
                for case_id in pending_ids
            }
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Hard quality audit",
                unit="item",
            ):
                case_id = futures[future]
                try:
                    result = future.result()
                except Exception as exc:  # noqa: BLE001
                    result = {
                        "case_id": case_id,
                        "difficulty_level": "hard",
                        "final_decision": "review",
                        "aggregation": {"basis": "unhandled_audit_error"},
                        "deterministic_checks": {},
                        "uniqueness_result": {
                            "parse_ok": False,
                            "decision": "uncertain",
                            "reason": "unhandled_audit_error",
                            "model_alias": uniqueness_model_alias,
                        },
                        "judge_results": [],
                        "error": f"{type(exc).__name__}: {exc}"[:2000],
                        "audit_signature": audit_signature,
                        "audited_at": _utc_now(),
                    }
                cached[case_id] = result
                with append_lock:
                    audit_handle.write(
                        json.dumps(result, ensure_ascii=False, separators=(",", ":"), default=str)
                        + "\n"
                    )
                    audit_handle.flush()

    completed_ids = [case_id for case_id in hard_ids if case_id in cached]
    decision_counts = Counter(cached[case_id]["final_decision"] for case_id in completed_ids)
    uniqueness_decision_counts = Counter(
        str(cached[case_id].get("uniqueness_result", {}).get("decision") or "missing")
        for case_id in completed_ids
    )
    drop_category_counts: Counter[str] = Counter()
    for case_id in completed_ids:
        if cached[case_id]["final_decision"] == "drop":
            drop_category_counts.update(
                cached[case_id].get("aggregation", {}).get("shared_drop_categories") or []
            )
    for decision in ("keep_hard", "repair", "drop", "review"):
        ids = [case_id for case_id in completed_ids if cached[case_id]["final_decision"] == decision]
        _write_id_file(output_dir / f"hard_{decision}_ids.txt", ids)

    ambiguous_repair_review_ids = [
        case_id
        for case_id in completed_ids
        if _matches_ambiguous_repair_review_filter(cached[case_id])
    ]
    high_confidence_remove_ids = [
        case_id
        for case_id in completed_ids
        if cached[case_id]["final_decision"] == "drop"
        or _matches_ambiguous_repair_review_filter(cached[case_id])
    ]
    _write_id_file(
        output_dir / "hard_ambiguous_repair_review_ids.txt",
        ambiguous_repair_review_ids,
    )
    _write_id_file(
        output_dir / "hard_quality_filter_ids.txt",
        high_confidence_remove_ids,
    )

    complete = len(completed_ids) == len(hard_ids)
    report = {
        "status": "completed" if complete else "partial",
        "created_at": _utc_now(),
        "dataset_jsonl": str(dataset_path),
        "dataset_count": len(dataset),
        "source_selected_counts": source_counts,
        "hard_count": len(hard_ids),
        "hard_audited_count": len(completed_ids),
        "hard_pending_count": len(hard_ids) - len(completed_ids),
        "decision_counts": dict(decision_counts),
        "uniqueness_decision_counts": dict(uniqueness_decision_counts),
        "decision_rates": {
            decision: count / len(completed_ids)
            for decision, count in decision_counts.items()
        } if completed_ids else {},
        "drop_category_counts": dict(drop_category_counts),
        "uniqueness_judge_calls_per_item": 1,
        "quality_judge_calls_per_item": quality_judge_call_count,
        "total_judge_calls_per_item": total_judge_call_count,
        "resume": {
            "enabled": args.resume,
            "initial_stats": resume_stats,
        },
        "automatic_drop_policy": {
            "min_drop_votes": args.min_drop_votes,
            "requires_shared_objective_category": True,
            "deterministic_fatal_checks_can_drop": True,
        },
        "quality_filter_policy": {
            "remove_final_drop": True,
            "remove_validated_ambiguous_repair_or_review": True,
            "ambiguous_repair_review_count": len(ambiguous_repair_review_ids),
            "total_remove_count": len(high_confidence_remove_ids),
            "id_file": str(output_dir / "hard_quality_filter_ids.txt"),
        },
        "audit_signature": audit_signature,
        "signature_payload": signature_payload,
    }

    if complete:
        safe_records = [
            record
            for record in dataset
            if str(record["id"]) not in cached
            or cached[str(record["id"])]["final_decision"] != "drop"
        ]
        quality_filtered_records = [
            record
            for record in dataset
            if str(record["id"]) not in cached
            or (
                cached[str(record["id"])]["final_decision"] != "drop"
                and not _matches_ambiguous_repair_review_filter(cached[str(record["id"])])
            )
        ]
        strict_records = [
            record
            for record in dataset
            if str(record["id"]) not in cached
            or cached[str(record["id"])]["final_decision"] == "keep_hard"
        ]
        safe_count = _write_jsonl(output_dir / "rl_dataset_drop_filtered.jsonl", safe_records)
        quality_filtered_count = _write_jsonl(
            output_dir / "rl_dataset_quality_filtered.jsonl",
            quality_filtered_records,
        )
        strict_count = _write_jsonl(output_dir / "rl_dataset_strict.jsonl", strict_records)
        report["outputs"] = {
            "drop_filtered": {
                "path": str(output_dir / "rl_dataset_drop_filtered.jsonl"),
                "count": safe_count,
                "removed": len(dataset) - safe_count,
            },
            "quality_filtered": {
                "path": str(output_dir / "rl_dataset_quality_filtered.jsonl"),
                "count": quality_filtered_count,
                "removed": len(dataset) - quality_filtered_count,
            },
            "strict": {
                "path": str(output_dir / "rl_dataset_strict.jsonl"),
                "count": strict_count,
                "removed": len(dataset) - strict_count,
            },
        }
    _write_json(output_dir / "report.json", report)
    logger.info("Audit status=%s decisions=%s output=%s", report["status"], dict(decision_counts), output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
