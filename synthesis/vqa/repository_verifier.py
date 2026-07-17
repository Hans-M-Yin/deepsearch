"""Graph-backed repository verifier for generated VQA samples.

This module currently runs two verification branches for each question:

1. repository-grounded verifier
   - Builds a mixed repository of relevant and distractor documents/images from
     the persisted synthesis graph.
   - Asks the answer model to solve the question using only those repository
     materials.
   - Uses the judge model to check whether the predicted answer is correct.

2. question-only shortcut verifier
   - Gives the model only the question itself, plus the question-attached image
     when the VQA sample includes one.
   - Asks the model to answer from memory or shortcuts without external
     documents, search, or tools.
   - Uses the judge model to check whether that shortcut answer is correct.
   - If this branch still answers correctly, the sample is rejected with
     ``closed_book_shortcut``.

Important note
--------------
There is no standalone ``question_only`` CLI in this file.

The main command below already runs both branches above inside one verifier run:

    python -m synthesis.vqa.repository_verifier
      --vqa-dir /path/to/vqa_dir
      --graph-dir /path/to/graph_dir
      --answer-model-alias <answer_model>
      --judge-model-alias <judge_model>

Runtime outputs
---------------
Running this module writes two files into ``vqa_dir``:

- ``repository_verification_results.jsonl``
  One record per question. Each record includes:
  - ``repository``: assembled mixed repository items
  - ``solver_result``: raw result of the repository-grounded solve branch
  - ``question_only_solver_result``: raw result of the shortcut branch
  - ``checks``: per-check details, including ``answer_judgment`` and
    ``question_only_shortcut``
  - ``final_keep`` and ``reject_reasons``

- ``repository_verification_summary.json``
  Aggregate counts such as:
  - ``verified_total``
  - ``final_keep_total``
  - ``answer_correct_total``
  - ``question_only_shortcut_total``
  - invalid / out-of-scope citation totals
  - insufficient evidence total

The CLI also prints a summary report to stdout.

Verbose merged runner
---------------------
If you want one explicit command that runs both verifier branches and also
prints the model inputs and outputs, use:

    python -m synthesis.vqa.run_repository_verifier_with_io
      --vqa-dir /path/to/vqa_dir
      --graph-dir /path/to/graph_dir
      --answer-model-alias <answer_model>
      --judge-model-alias <judge_model>

This merged runner delegates to the debug verifier with
``--run-verification`` enabled. It prints:
- ``Repository Bundle``
- ``Answer Model Request``
- ``Question-Only Shortcut Request``
- ``Answer Model Raw Output``
- ``Question-Only Shortcut Raw Output``
- ``Judge Model Request`` / ``Judge Model Raw Output``
- ``Question-Only Judge Request`` / ``Question-Only Judge Raw Output``

Subset debugging
----------------
If you want to inspect only specific questions or samples, use:

    python -m synthesis.vqa.debug.debug_repository_verifier
      --vqa-dir /path/to/vqa_dir
      --graph-dir /path/to/graph_dir
      --question-id q_000001
      --run-verification
      --answer-model-alias <answer_model>
      --judge-model-alias <judge_model>
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import random
import re
import textwrap
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol

from synthesis.model_worker import ModelMessage, ModelRequest
from synthesis.store import JsonlGraphStore

from .graph_view import GraphView


PROMPT_REPOSITORY_SOLVER = """
You are solving a multi-hop question using only the provided repository materials.

You will receive:
- one question
- a mixed repository containing documents and images
- some repository items are relevant, some are distractors

Rules:
- Use only the provided repository materials. Do not use outside knowledge.
- Treat document labels like [DOC 1] and image labels like [IMG 2] as the citation handles.
- When you make a factual claim based on a repository item, cite it inline using labels like [DOC 1] or [IMG 2].
- Every major reasoning step must include at least one inline citation.
- If the repository does not contain enough evidence to finish the reasoning chain, write exactly one line beginning with:
  Insufficient evidence: <brief reason>
- If you solve it, write a concise but complete natural-language reasoning chain.
- End every solved response with exactly one final line:
  Final answer: <answer>
"""


PROMPT_REPOSITORY_ANSWER_JUDGE = """
You are judging whether a repository-grounded answer should count as correct for a VQA question.

You will receive:
- the question
- the gold answer
- the predicted answer

Judge semantic correctness, not exact string match.

Guidelines:
- Accept paraphrases and semantically equivalent answers.
- If the predicted answer is substantially incomplete, off-topic, or refers to a different entity/object, mark it incorrect.
- If the predicted answer is empty, or the model says evidence is insufficient, mark it incorrect.

Return JSON in exactly this format:
{
  "correct": true,
  "confidence": 0.0,
  "reason": "",
  "normalized_gold_answer": "",
  "normalized_predicted_answer": ""
}
"""


PROMPT_QUESTION_ONLY_SHORTCUT_SOLVER = """
You are a knowledge expert. Below, you will be given a relatively complex knowledge question. In some cases, the image attached to the question will also be provided.

Please answer the question based only on your own memory and the attached question image, if one is provided.

Requirements:
- You do not have access to external documents, web search, or tools.
- The question may be difficult, but there may also be shortcuts. Your only goal is to provide the correct answer, and you may avoid difficult intermediate reasoning if a shortcut is enough.
- If you are confident, prefer step-by-step reasoning before answering.
- If you answer, end with exactly one final line:
  Final answer: <answer>
- If you think the question is ambiguous, or if it contains an obvious factual error, write exactly one line beginning with:
  Cannot answer: <brief reason>
"""


def build_repository_solver_request(
    *,
    bundle: RepositoryBundle,
    answer_model_alias: str | None,
    answer_max_tokens: int,
    user_content: list[dict[str, Any]],
) -> ModelRequest:
    question_id = bundle.question_id or bundle.sample_id or bundle.path_id or "repository_question"
    return ModelRequest(
        model=answer_model_alias,
        messages=[
            ModelMessage(role="system", content=PROMPT_REPOSITORY_SOLVER),
            ModelMessage(role="user", content=user_content),
        ],
        max_tokens=answer_max_tokens,
        metadata={"trace_label": f"repository_solve:{question_id}"},
    )


def build_repository_answer_judge_request(
    *,
    question: str,
    gold_answer: str,
    predicted_answer: str,
    judge_model_alias: str | None,
    judge_max_tokens: int,
    question_id: str,
) -> ModelRequest:
    payload = {
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
    }
    return ModelRequest(
        model=judge_model_alias,
        messages=[
            ModelMessage(role="system", content=PROMPT_REPOSITORY_ANSWER_JUDGE),
            ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False, indent=2)),
        ],
        response_format={"type": "json_object"},
        max_tokens=judge_max_tokens,
        metadata={"trace_label": f"repository_answer_judge:{question_id}"},
    )


def build_question_only_shortcut_request(
    *,
    question: str,
    answer_model_alias: str | None,
    answer_max_tokens: int,
    question_id: str,
    image_url: str | None = None,
) -> ModelRequest:
    if image_url:
        user_content: str | list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Question:\n{question}\n\n"
                    "The next image is the image attached to the question. "
                    "Use only the question text and this attached image.\n\n"
                    "Answer naturally."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": _resolve_multimodal_image_url(image_url)},
            },
        ]
    else:
        user_content = f"Question:\n{question}\n\nAnswer naturally."
    return ModelRequest(
        model=answer_model_alias,
        messages=[
            ModelMessage(role="system", content=PROMPT_QUESTION_ONLY_SHORTCUT_SOLVER),
            ModelMessage(role="user", content=user_content),
        ],
        max_tokens=answer_max_tokens,
        metadata={"trace_label": f"question_only_shortcut:{question_id}"},
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0



def _stable_seed(*parts: object) -> int:
    payload = "||".join("" if part is None else str(part) for part in parts)
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _normalize_text_key(text: str | None) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip().lower()



def _question_text(question_record: dict[str, Any]) -> str:
    for field_name in ("question", "final_question", "polished_question", "draft_question"):
        value = str(question_record.get(field_name) or "").strip()
        if value:
            return value
    return ""


def _extract_question_input_image_url(
    *,
    question_record: dict[str, Any],
    sample_record: dict[str, Any] | None,
) -> str | None:
    candidates: list[Any] = [
        question_record.get("image_url"),
        question_record.get("input_image_url"),
    ]
    sample = sample_record or {}
    candidates.append(sample.get("input_image_url"))
    metadata = sample.get("metadata") or {}
    if isinstance(metadata, dict):
        candidates.append(metadata.get("input_image_url"))
    for candidate in candidates:
        value = str(candidate or "").strip()
        if value:
            return value
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{path}:{line_no} must contain one JSON object per line")
            records.append(payload)
    return records


def _append_jsonl(handle, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
    handle.write("\n")


def _infer_graph_dir(vqa_dir: str | Path) -> Path:
    current = Path(vqa_dir).expanduser().resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "nodes.jsonl").exists() and (candidate / "edges.jsonl").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not infer graph_dir from {current}. Pass --graph-dir explicitly."
    )


def _format_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def _format_wrapped(label: str, value: str, *, width: int, indent: int = 2) -> list[str]:
    prefix = " " * indent
    if not value:
        return [f"{prefix}{label}: -"]
    wrapped = textwrap.wrap(
        value,
        width=max(20, width - indent - len(label) - 2),
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return [f"{prefix}{label}: -"]
    lines = [f"{prefix}{label}: {wrapped[0]}"]
    continuation_prefix = " " * (indent + len(label) + 2)
    for segment in wrapped[1:]:
        lines.append(f"{continuation_prefix}{segment}")
    return lines


def _classify_trajectory(node_types: list[Any]) -> str:
    normalized = [str(item).strip().lower() for item in node_types if str(item).strip()]
    image_positions = [index for index, node_type in enumerate(normalized) if node_type == "image"]
    image_count = len(image_positions)
    if image_count == 0:
        return "text_only"
    if image_count >= 2:
        return "multi_image"
    image_index = image_positions[0]
    if image_index == 0:
        return "image_first"
    if image_index == len(normalized) - 1:
        return "image_end"
    return "unclassified"


def _normalize_citation_label(label: Any) -> str:
    text = re.sub(r"\s+", " ", str(label or "").strip())
    match = re.search(r"(?:\[)?\s*(doc|img)\s*[-_ ]?(\d+)\s*(?:\])?", text, flags=re.IGNORECASE)
    if not match:
        return text.upper()
    prefix = match.group(1).upper()
    number = int(match.group(2))
    return f"{prefix} {number}"


def _extract_citation_labels(text: str) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"\[\s*(doc|img)\s*[-_ ]?(\d+)\s*\]", str(text or ""), flags=re.IGNORECASE):
        label = f"{match.group(1).upper()} {int(match.group(2))}"
        if label not in seen:
            seen.add(label)
            labels.append(label)
    return labels


def _strip_citation_labels(text: str) -> str:
    return re.sub(r"\s*\[\s*(?:doc|img)\s*[-_ ]?\d+\s*\]", "", str(text or ""), flags=re.IGNORECASE).strip()


def _split_reasoning_paragraphs(text: str) -> list[str]:
    body = str(text or "").strip()
    body = re.sub(r"(?im)^\s*final\s+answer\s*:\s*.*$", "", body).strip()
    body = re.sub(r"(?im)^\s*insufficient\s+evidence\s*:\s*.*$", "", body).strip()
    body = re.sub(r"(?im)^\s*cannot\s+answer\s*:\s*.*$", "", body).strip()
    paragraphs = [part.strip() for part in re.split(r"\n\s*\n", body) if part.strip()]
    if len(paragraphs) <= 1:
        lines = [line.strip() for line in body.splitlines() if line.strip()]
        if len(lines) > 1:
            paragraphs = lines
    return paragraphs


def _parse_final_answer(text: str) -> str:
    matches = list(re.finditer(r"(?im)^\s*final\s+answer\s*:\s*(.+?)\s*$", str(text or "")))
    if not matches:
        return ""
    return _strip_citation_labels(matches[-1].group(1).strip())


def _parse_prefixed_reason(text: str, *, prefix: str) -> str:
    pattern = rf"(?im)^\s*{re.escape(prefix)}\s*:\s*(.+?)\s*$"
    match = re.search(pattern, str(text or ""))
    return match.group(1).strip() if match else ""


def _parse_freeform_solver_response(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    insufficient_reason = _parse_prefixed_reason(raw_text, prefix="Insufficient evidence")
    if insufficient_reason:
        return {
            "status": "insufficient_evidence",
            "answer": "",
            "reasoning_steps": [],
            "used_evidence": [],
            "insufficient_reason": insufficient_reason,
            "raw_text": raw_text,
        }

    reasoning_steps: list[dict[str, Any]] = []
    for index, paragraph in enumerate(_split_reasoning_paragraphs(raw_text), start=1):
        citations = _extract_citation_labels(paragraph)
        claim = _strip_citation_labels(paragraph)
        if claim or citations:
            reasoning_steps.append({"step": index, "claim": claim, "citations": citations})
    used_evidence = sorted({label for step in reasoning_steps for label in (step.get("citations") or [])})
    answer = _parse_final_answer(raw_text)
    return {
        "status": "solved" if answer else "insufficient_evidence",
        "answer": answer,
        "reasoning_steps": reasoning_steps,
        "used_evidence": used_evidence,
        "insufficient_reason": "" if answer else "missing_final_answer",
        "raw_text": raw_text,
    }


def _parse_freeform_question_only_response(text: str) -> dict[str, Any]:
    raw_text = str(text or "").strip()
    cannot_answer_reason = _parse_prefixed_reason(raw_text, prefix="Cannot answer")
    if cannot_answer_reason:
        return {
            "status": "cannot_answer",
            "answer": "",
            "shortcut_basis": "",
            "cannot_answer_reason": cannot_answer_reason,
            "confidence": 0.0,
            "raw_text": raw_text,
        }
    answer = _parse_final_answer(raw_text)
    shortcut_basis = "\n\n".join(_split_reasoning_paragraphs(raw_text))
    return {
        "status": "answered" if answer else "cannot_answer",
        "answer": answer,
        "shortcut_basis": shortcut_basis,
        "cannot_answer_reason": "" if answer else "missing_final_answer",
        "confidence": 0.0,
        "raw_text": raw_text,
    }


def _resolve_multimodal_image_url(image_url: str) -> str:
    normalized = str(image_url or "").strip()
    if not normalized:
        return normalized
    lowered = normalized.lower()
    if lowered.startswith(("http://", "https://", "data:")):
        return normalized
    if lowered.startswith("file://"):
        local_path = Path(normalized[7:])
    else:
        local_path = Path(normalized)
    if not local_path.exists() or not local_path.is_file():
        return normalized
    mime_type, _ = mimetypes.guess_type(local_path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


class JsonGeneratingModelClient(Protocol):
    def generate(self, request: ModelRequest) -> Any:
        """Run one request and return a model response."""

    def generate_json(self, request: ModelRequest) -> dict[str, Any]:
        """Run one request and return a JSON object."""


@dataclass(slots=True)
class RepositoryVerificationConfig:
    random_seed: int = 0
    max_relevant_docs_per_edge: int = 2
    max_sibling_doc_distractors_per_edge: int = 1
    max_random_doc_distractors: int = 2
    max_sibling_image_distractors_per_image: int = 1
    max_random_image_distractors: int = 1
    min_reasoning_steps: int = 1
    min_unique_citations: int = 2
    answer_max_tokens: int = 1200
    question_only_answer_max_tokens: int = 256
    judge_max_tokens: int = 512

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class RepositoryItem:
    label: str
    item_type: str
    text: str | None = None
    image_url: str | None = None
    is_relevant: bool = False
    selection_reason: str | None = None
    source_edge_id: str | None = None
    source_node_id: str | None = None
    evidence_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class RepositoryBundle:
    question_id: str
    sample_id: str
    path_id: str
    question: str
    gold_answer: str
    items: list[RepositoryItem] = field(default_factory=list)
    relevant_labels: list[str] = field(default_factory=list)
    distractor_labels: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))



@dataclass(slots=True)
class RepositoryAssembler:
    graph: GraphView
    config: RepositoryVerificationConfig = field(default_factory=RepositoryVerificationConfig)

    _doc_evidence_priority: dict[str, int] = field(
        init=False,
        default_factory=lambda: {
            "web_text": 0,
            "llm_output": 1,
            "search_result": 2,
            "visual_target": 3,
            "caption": 4,
            "ocr": 5,
            "image": 6,
        },
    )

    def build_bundle(
        self,
        *,
        question_record: dict[str, Any],
        sample_record: dict[str, Any],
    ) -> RepositoryBundle:
        question_id = str(question_record.get("question_id") or "")
        sample_id = str(question_record.get("sample_id") or sample_record.get("sample_id") or "")
        path = sample_record.get("path") or {}
        path_id = str(question_record.get("path_id") or path.get("path_id") or "")
        rng = random.Random(_stable_seed(question_id, sample_id, path_id, self.config.random_seed))

        relevant_edges = self._ordered_relevant_edges(sample_record)
        path_edge_ids = {str(edge.get("edge_id") or "") for edge in relevant_edges if edge.get("edge_id")}
        path_node_ids = [
            str(node_id)
            for node_id in (path.get("node_ids") or [])
            if str(node_id or "").strip()
        ]
        path_node_id_set = set(path_node_ids)

        relevant_docs = self._collect_relevant_docs(relevant_edges)
        writer_docs = self._collect_writer_stage_docs(sample_record)
        distractor_docs = self._collect_doc_distractors(
            relevant_edges=relevant_edges,
            excluded_edge_ids=path_edge_ids,
            excluded_doc_keys={item["doc_key"] for item in [*relevant_docs, *writer_docs]},
            rng=rng,
        )
        relevant_images = self._collect_relevant_images(path_node_ids)
        distractor_images = self._collect_image_distractors(
            relevant_images=relevant_images,
            excluded_node_ids=path_node_id_set,
            rng=rng,
        )

        raw_items = [
            *relevant_docs,
            *writer_docs,
            *relevant_images,
            *distractor_docs,
            *distractor_images,
        ]
        rng.shuffle(raw_items)

        items: list[RepositoryItem] = []
        relevant_labels: list[str] = []
        distractor_labels: list[str] = []
        doc_index = 0
        img_index = 0
        for raw_item in raw_items:
            if raw_item.get("item_type") == "doc":
                doc_index += 1
                label = f"DOC {doc_index}"
            else:
                img_index += 1
                label = f"IMG {img_index}"
            item = RepositoryItem(
                label=label,
                item_type=str(raw_item.get("item_type") or "doc"),
                text=raw_item.get("text"),
                image_url=raw_item.get("image_url"),
                is_relevant=bool(raw_item.get("is_relevant")),
                selection_reason=raw_item.get("selection_reason"),
                source_edge_id=raw_item.get("source_edge_id"),
                source_node_id=raw_item.get("source_node_id"),
                evidence_id=raw_item.get("evidence_id"),
                metadata=dict(raw_item.get("metadata") or {}),
            )
            items.append(item)
            if item.is_relevant:
                relevant_labels.append(item.label)
            else:
                distractor_labels.append(item.label)

        return RepositoryBundle(
            question_id=question_id,
            sample_id=sample_id,
            path_id=path_id,
            question=_question_text(question_record),
            gold_answer=str(question_record.get("answer") or "").strip(),
            items=items,
            relevant_labels=relevant_labels,
            distractor_labels=distractor_labels,
            metadata={
                "relevant_edge_ids": [edge.get("edge_id") for edge in relevant_edges if edge.get("edge_id")],
                "path_node_ids": list(path_node_ids),
                "item_count": len(items),
                "doc_count": sum(1 for item in items if item.item_type == "doc"),
                "image_count": sum(1 for item in items if item.item_type == "image"),
                "relevant_doc_count": sum(1 for item in items if item.item_type == "doc" and item.is_relevant),
                "relevant_image_count": sum(1 for item in items if item.item_type == "image" and item.is_relevant),
                "writer_stage_doc_count": sum(
                    1
                    for item in items
                    if item.item_type == "doc" and str(item.selection_reason or "").startswith("writer_")
                ),
            },
        )

    def build_solver_user_content(self, *, bundle: RepositoryBundle) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        intro = (
            f"Question:\n{bundle.question}\n\n"
            "Repository materials follow. Some are relevant and some are distractors. "
            "Use only these materials."
        )
        blocks.append({"type": "text", "text": intro})
        for item in bundle.items:
            if item.item_type == "doc":
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[{item.label}]\n{str(item.text or '').strip()}",
                    }
                )
                continue
            if not item.image_url:
                blocks.append(
                    {
                        "type": "text",
                        "text": f"[{item.label}]\n[image missing]",
                    }
                )
                continue
            caption = str((item.metadata or {}).get("caption") or "").strip()
            if caption:
                image_intro = (
                    f"[{item.label}]\n"
                    f"Caption: {caption}\n"
                    f"The next image is repository item {item.label}. Cite it as {item.label} if you use it."
                )
            else:
                image_intro = (
                    f"[{item.label}]\n"
                    f"The next image is repository item {item.label}. Cite it as {item.label} if you use it."
                )
            blocks.append(
                {
                    "type": "text",
                    "text": image_intro,
                }
            )
            blocks.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _resolve_multimodal_image_url(item.image_url)},
                }
            )
        blocks.append(
            {
                "type": "text",
                "text": (
                    "Answer naturally with inline citations such as [DOC 1] or [IMG 2]. "
                    "If evidence is insufficient, start one line with 'Insufficient evidence:'. "
                    "If you solve it, end with one line 'Final answer: <answer>'."
                ),
            }
        )
        return blocks

    def _ordered_relevant_edges(self, sample_record: dict[str, Any]) -> list[dict[str, Any]]:
        ordered: list[dict[str, Any]] = []
        seen_edge_ids: set[str] = set()
        for hop in (sample_record.get("hop_chain") or []):
            edge = self._resolve_edge_from_hop(hop)
            edge_id = str((edge or {}).get("edge_id") or "")
            if not edge_id or edge_id in seen_edge_ids:
                continue
            ordered.append(edge)
            seen_edge_ids.add(edge_id)
        for edge_id in ((sample_record.get("path") or {}).get("edge_ids") or []):
            edge = self.graph.get_edge(str(edge_id))
            edge_key = str((edge or {}).get("edge_id") or "")
            if not edge_key or edge_key in seen_edge_ids:
                continue
            ordered.append(edge)
            seen_edge_ids.add(edge_key)
        return ordered

    def _resolve_edge_from_hop(self, hop: Any) -> dict[str, Any] | None:
        if not isinstance(hop, dict):
            return None
        edge_id = str(hop.get("edge_id") or "").strip()
        if edge_id:
            edge = self.graph.get_edge(edge_id)
            if edge is not None:
                return edge
        src_node_id = str(hop.get("src_node_id") or "").strip()
        dst_node_id = str(hop.get("dst_node_id") or "").strip()
        if src_node_id and dst_node_id:
            return self.graph.get_edge_id_between(src_node_id, dst_node_id)
        return None

    def _collect_relevant_docs(self, relevant_edges: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen_doc_keys: set[str] = set()
        for edge in relevant_edges:
            for candidate in self._doc_candidates_from_edge(
                edge,
                max_items=self.config.max_relevant_docs_per_edge,
                is_relevant=True,
                selection_prefix="relevant",
            ):
                doc_key = str(candidate.get("doc_key") or "")
                if not doc_key or doc_key in seen_doc_keys:
                    continue
                seen_doc_keys.add(doc_key)
                results.append(candidate)
        return results

    def _collect_doc_distractors(
        self,
        *,
        relevant_edges: list[dict[str, Any]],
        excluded_edge_ids: set[str],
        excluded_doc_keys: set[str],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen_doc_keys = set(excluded_doc_keys)
        for edge in relevant_edges:
            src_node_id = str(edge.get("src_node_id") or "")
            if not src_node_id:
                continue
            siblings = [
                item
                for item in self.graph.neighbors(src_node_id)
                if str(item.get("edge_id") or "") not in excluded_edge_ids
            ]
            rng.shuffle(siblings)
            kept = 0
            for sibling in siblings:
                for candidate in self._doc_candidates_from_edge(
                    sibling,
                    max_items=1,
                    is_relevant=False,
                    selection_prefix="sibling_distractor",
                ):
                    doc_key = str(candidate.get("doc_key") or "")
                    if not doc_key or doc_key in seen_doc_keys:
                        continue
                    seen_doc_keys.add(doc_key)
                    results.append(candidate)
                    kept += 1
                    break
                if kept >= self.config.max_sibling_doc_distractors_per_edge:
                    break

        all_edges = self.graph.store.list_edges()
        rng.shuffle(all_edges)
        max_total = len(relevant_edges) * self.config.max_sibling_doc_distractors_per_edge + self.config.max_random_doc_distractors
        for edge in all_edges:
            if len(results) >= max_total:
                break
            edge_id = str(edge.get("edge_id") or "")
            if edge_id in excluded_edge_ids:
                continue
            for candidate in self._doc_candidates_from_edge(
                edge,
                max_items=1,
                is_relevant=False,
                selection_prefix="random_distractor",
            ):
                doc_key = str(candidate.get("doc_key") or "")
                if not doc_key or doc_key in seen_doc_keys:
                    continue
                seen_doc_keys.add(doc_key)
                results.append(candidate)
                break
        return results

    def _collect_writer_stage_docs(self, sample_record: dict[str, Any]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen_doc_keys: set[str] = set()

        def add_doc(*, text: Any, selection_reason: str, metadata: dict[str, Any]) -> None:
            normalized = str(text or "").strip()
            if not normalized:
                return
            doc_key = _normalize_text_key(f"{selection_reason}\n{normalized}")
            if not doc_key or doc_key in seen_doc_keys:
                return
            seen_doc_keys.add(doc_key)
            results.append(
                {
                    "item_type": "doc",
                    "text": normalized,
                    "doc_key": doc_key,
                    "is_relevant": True,
                    "selection_reason": selection_reason,
                    "source_edge_id": None,
                    "source_node_id": None,
                    "evidence_id": None,
                    "metadata": metadata,
                }
            )

        opening_package = sample_record.get("opening_package") or {}
        if isinstance(opening_package, dict):
            for fact in self._string_list(opening_package.get("source_supporting_facts")):
                add_doc(
                    text=fact,
                    selection_reason="writer_opening_source_supporting_fact",
                    metadata={
                        "writer_stage": "opening_package",
                        "source_clue": opening_package.get("source_clue"),
                        "packaged_first_hop": opening_package.get("packaged_first_hop"),
                    },
                )
            for field_name, reason in (
                ("packaged_first_hop", "writer_opening_packaged_first_hop"),
                ("why_relevant", "writer_opening_why_relevant"),
                ("first_hop_support", "writer_opening_first_hop_support"),
            ):
                add_doc(
                    text=opening_package.get(field_name),
                    selection_reason=reason,
                    metadata={
                        "writer_stage": "opening_package",
                        "source_clue": opening_package.get("source_clue"),
                    },
                )

        for stage_name in ("question_target_ask", "target_ask"):
            target_ask = sample_record.get(stage_name) or {}
            if not isinstance(target_ask, dict):
                continue
            for fact in self._string_list(target_ask.get("supporting_facts")):
                add_doc(
                    text=fact,
                    selection_reason=f"writer_{stage_name}_supporting_fact",
                    metadata={
                        "writer_stage": stage_name,
                        "ask_target": target_ask.get("ask_target"),
                        "answer": target_ask.get("answer"),
                    },
                )
            for field_name, reason_suffix in (
                ("reasoning", "reasoning"),
                ("support", "support"),
            ):
                add_doc(
                    text=target_ask.get(field_name),
                    selection_reason=f"writer_{stage_name}_{reason_suffix}",
                    metadata={
                        "writer_stage": stage_name,
                        "ask_target": target_ask.get("ask_target"),
                        "answer": target_ask.get("answer"),
                    },
                )

        question_terminal_bridge = sample_record.get("question_terminal_bridge") or {}
        if isinstance(question_terminal_bridge, dict):
            for field_name, reason in (
                ("raw_ask_target", "writer_question_terminal_raw_ask_target"),
                ("rewritten_ask_target", "writer_question_terminal_rewritten_ask_target"),
                ("target_image", "writer_question_terminal_target_image"),
            ):
                add_doc(
                    text=question_terminal_bridge.get(field_name),
                    selection_reason=reason,
                    metadata={
                        "writer_stage": "question_terminal_bridge",
                        "answer": question_terminal_bridge.get("answer"),
                    },
                )

        return results

    def _collect_relevant_images(self, path_node_ids: list[str]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen_node_ids: set[str] = set()
        for node_id in path_node_ids:
            node = self.graph.get_node(node_id) or {}
            if node.get("node_type") != "image":
                continue
            image_url = self._preferred_image_url(node)
            if not image_url or node_id in seen_node_ids:
                continue
            seen_node_ids.add(node_id)
            metadata = node.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            results.append(
                {
                    "item_type": "image",
                    "image_url": image_url,
                    "is_relevant": True,
                    "selection_reason": "relevant_path_image",
                    "source_node_id": node_id,
                    "metadata": {
                        "title": node.get("title"),
                        "caption": self._image_caption(node),
                        "source_text_node_id": metadata.get("source_text_node_id"),
                        "image_origin": metadata.get("image_origin"),
                    },
                }
            )
        return results

    def _collect_image_distractors(
        self,
        *,
        relevant_images: list[dict[str, Any]],
        excluded_node_ids: set[str],
        rng: random.Random,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen_node_ids = set(excluded_node_ids)
        image_nodes = [self.graph.get_node(node_id) or {} for node_id in self.graph.list_node_ids(node_type="image")]
        for relevant in relevant_images:
            source_node_id = str(relevant.get("source_node_id") or "")
            node = self.graph.get_node(source_node_id) or {}
            metadata = node.get("metadata") or {}
            if not isinstance(metadata, dict):
                metadata = {}
            sibling_candidates = [
                candidate
                for candidate in image_nodes
                if self._is_sibling_image_candidate(
                    candidate,
                    source_text_node_id=metadata.get("source_text_node_id"),
                    source_page_url=node.get("source_page_url"),
                    excluded_node_ids=seen_node_ids,
                )
            ]
            rng.shuffle(sibling_candidates)
            for candidate in sibling_candidates[: self.config.max_sibling_image_distractors_per_image]:
                image_url = self._preferred_image_url(candidate)
                candidate_node_id = str(candidate.get("node_id") or "")
                if not image_url or not candidate_node_id:
                    continue
                seen_node_ids.add(candidate_node_id)
                candidate_metadata = candidate.get("metadata") or {}
                if not isinstance(candidate_metadata, dict):
                    candidate_metadata = {}
                results.append(
                    {
                        "item_type": "image",
                        "image_url": image_url,
                        "is_relevant": False,
                        "selection_reason": "sibling_image_distractor",
                        "source_node_id": candidate_node_id,
                        "metadata": {
                            "title": candidate.get("title"),
                            "caption": self._image_caption(candidate),
                            "source_text_node_id": candidate_metadata.get("source_text_node_id"),
                        },
                    }
                )

        random_candidates = [
            node
            for node in image_nodes
            if str(node.get("node_id") or "") not in seen_node_ids
        ]
        rng.shuffle(random_candidates)
        for candidate in random_candidates[: self.config.max_random_image_distractors]:
            image_url = self._preferred_image_url(candidate)
            candidate_node_id = str(candidate.get("node_id") or "")
            if not image_url or not candidate_node_id:
                continue
            seen_node_ids.add(candidate_node_id)
            results.append(
                {
                    "item_type": "image",
                    "image_url": image_url,
                    "is_relevant": False,
                    "selection_reason": "random_image_distractor",
                    "source_node_id": candidate_node_id,
                    "metadata": {"title": candidate.get("title"), "caption": self._image_caption(candidate)},
                }
            )
        return results

    @staticmethod
    def _string_list(value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item or "").strip()]
        text = str(value or "").strip()
        return [text] if text else []

    @staticmethod
    def _image_caption(node: dict[str, Any]) -> str:
        candidates: list[Any] = [
            node.get("caption"),
            node.get("summary"),
        ]
        metadata = node.get("metadata") or {}
        if isinstance(metadata, dict):
            candidates.extend(
                [
                    metadata.get("caption"),
                    metadata.get("search_caption"),
                    metadata.get("image_caption"),
                    metadata.get("visual_target"),
                ]
            )
        for candidate in candidates:
            text = str(candidate or "").strip()
            if text:
                return text
        return ""

    def _doc_candidates_from_edge(
        self,
        edge: dict[str, Any],
        *,
        max_items: int,
        is_relevant: bool,
        selection_prefix: str,
    ) -> list[dict[str, Any]]:
        edge_id = str(edge.get("edge_id") or "")
        src_node_id = str(edge.get("src_node_id") or "")
        quote_candidates: list[dict[str, Any]] = []
        evidence_candidates: list[tuple[int, dict[str, Any]]] = []
        for evidence_ref in (edge.get("evidence_refs") or []):
            if not isinstance(evidence_ref, dict):
                continue
            quote = str(evidence_ref.get("quote") or "").strip()
            evidence_id = str(evidence_ref.get("evidence_id") or "").strip() or None
            if quote:
                quote_candidates.append(
                    {
                        "item_type": "doc",
                        "text": quote,
                        "doc_key": _normalize_text_key(quote),
                        "is_relevant": is_relevant,
                        "selection_reason": f"{selection_prefix}_edge_quote",
                        "source_edge_id": edge_id or None,
                        "source_node_id": src_node_id or None,
                        "evidence_id": evidence_id,
                        "metadata": {
                            "edge_type": edge.get("edge_type"),
                            "relation": edge.get("relation"),
                        },
                    }
                )
                continue
            if not evidence_id:
                continue
            evidence_record = self.graph.store.get_evidence(evidence_id) or {}
            content = str(evidence_record.get("content") or "").strip()
            if not content:
                continue
            evidence_type = str(evidence_record.get("evidence_type") or "")
            priority = self._doc_evidence_priority.get(evidence_type, 999)
            evidence_candidates.append(
                (
                    priority,
                    {
                        "item_type": "doc",
                        "text": content,
                        "doc_key": _normalize_text_key(content),
                        "is_relevant": is_relevant,
                        "selection_reason": f"{selection_prefix}_edge_evidence",
                        "source_edge_id": edge_id or None,
                        "source_node_id": src_node_id or None,
                        "evidence_id": evidence_id,
                        "metadata": {
                            "edge_type": edge.get("edge_type"),
                            "relation": edge.get("relation"),
                            "evidence_type": evidence_type,
                        },
                    },
                )
            )

        candidates = quote_candidates[:max_items]
        if candidates:
            return candidates

        evidence_candidates.sort(key=lambda item: (item[0], str((item[1].get("evidence_id") or ""))))
        return [candidate for _, candidate in evidence_candidates[:max_items]]

    def _is_sibling_image_candidate(
        self,
        candidate: dict[str, Any],
        *,
        source_text_node_id: Any,
        source_page_url: Any,
        excluded_node_ids: set[str],
    ) -> bool:
        candidate_node_id = str(candidate.get("node_id") or "")
        if not candidate_node_id or candidate_node_id in excluded_node_ids:
            return False
        if candidate.get("node_type") != "image":
            return False
        metadata = candidate.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        candidate_source_text = str(metadata.get("source_text_node_id") or "").strip()
        if source_text_node_id and candidate_source_text and candidate_source_text == str(source_text_node_id):
            return True
        candidate_source_page = str(candidate.get("source_page_url") or "").strip()
        if source_page_url and candidate_source_page and candidate_source_page == str(source_page_url):
            return True
        return False

    @staticmethod
    def _preferred_image_url(node: dict[str, Any]) -> str | None:
        for field_name in ("image_url", "oss_uri", "thumb_oss_uri"):
            image_url = str(node.get(field_name) or "").strip()
            if image_url:
                return image_url
        return None



@dataclass(slots=True)
class OfflineGraphRepositoryVerifier:
    assembler: RepositoryAssembler
    model_client: JsonGeneratingModelClient
    answer_model_alias: str
    judge_model_alias: str
    output_file_name: str = "repository_verification_results.jsonl"
    summary_file_name: str = "repository_verification_summary.json"

    def run(self, *, vqa_dir: str | Path) -> dict[str, Any]:
        vqa_path = Path(vqa_dir).expanduser().resolve()
        questions_path = vqa_path / "questions.jsonl"
        samples_path = vqa_path / "samples.jsonl"
        output_path = vqa_path / self.output_file_name
        summary_path = vqa_path / self.summary_file_name

        if not questions_path.exists():
            raise FileNotFoundError(f"questions.jsonl does not exist: {questions_path}")
        if not samples_path.exists():
            raise FileNotFoundError(f"samples.jsonl does not exist: {samples_path}")

        question_records = _load_jsonl(questions_path)
        sample_records = _load_jsonl(samples_path)
        samples_by_id = {
            str(record.get("sample_id")): record
            for record in sample_records
            if record.get("sample_id") is not None
        }
        existing_records = _load_jsonl(output_path) if output_path.exists() else []
        existing_by_question_id = {
            str(record.get("question_id")): record
            for record in existing_records
            if record.get("question_id") is not None
        }

        summary = {
            "vqa_dir": str(vqa_path),
            "questions_total": len(question_records),
            "verified_total": 0,
            "reused_total": 0,
            "newly_verified_total": 0,
            "reverified_total": 0,
            "final_keep_total": 0,
            "answer_correct_total": 0,
            "question_only_shortcut_total": 0,
            "invalid_citation_total": 0,
            "out_of_scope_citation_total": 0,
            "insufficient_evidence_total": 0,
            "output_path": str(output_path),
            "answer_model_alias": self.answer_model_alias,
            "judge_model_alias": self.judge_model_alias,
            "repository_config": self.assembler.config.to_dict(),
            "trajectory_type_counts": {
                "text_only": 0,
                "image_first": 0,
                "image_end": 0,
                "multi_image": 0,
                "unclassified": 0,
            },
            "created_at": _utc_now(),
            "updated_at": _utc_now(),
        }

        with output_path.open("w", encoding="utf-8") as handle:
            for index, question_record in enumerate(question_records, start=1):
                sample = samples_by_id.get(str(question_record.get("sample_id") or ""))
                fingerprint = self._question_fingerprint(question_record=question_record, sample_record=sample)
                question_id = str(question_record.get("question_id") or index)
                existing = existing_by_question_id.get(question_id)

                if self._can_reuse_existing_record(existing=existing, fingerprint=fingerprint):
                    verification_record = dict(existing)
                    verification_record["question_number"] = index
                    verification_record["reuse_status"] = "reused"
                    summary["reused_total"] += 1
                else:
                    verification_record = self.verify_question_record(
                        question_record=question_record,
                        sample_record=sample,
                        question_index=index,
                        question_fingerprint=fingerprint,
                    )
                    verification_record["reuse_status"] = "reverified" if existing is not None else "new"
                    if existing is not None:
                        summary["reverified_total"] += 1
                    else:
                        summary["newly_verified_total"] += 1
                _append_jsonl(handle, verification_record)
                summary["verified_total"] += 1
                if verification_record.get("final_keep"):
                    summary["final_keep_total"] += 1
                if (verification_record.get("checks") or {}).get("answer_judgment", {}).get("correct"):
                    summary["answer_correct_total"] += 1
                if not (verification_record.get("checks") or {}).get("citations_exist", {}).get("passed"):
                    summary["invalid_citation_total"] += 1
                if not (verification_record.get("checks") or {}).get("citations_within_relevant_scope", {}).get("passed"):
                    summary["out_of_scope_citation_total"] += 1
                if (verification_record.get("solver_result") or {}).get("status") == "insufficient_evidence":
                    summary["insufficient_evidence_total"] += 1
                if not ((verification_record.get("checks") or {}).get("question_only_shortcut") or {}).get("passed", True):
                    summary["question_only_shortcut_total"] += 1
                trajectory_type = str(verification_record.get("trajectory_type") or "unclassified")
                if trajectory_type not in summary["trajectory_type_counts"]:
                    trajectory_type = "unclassified"
                summary["trajectory_type_counts"][trajectory_type] += 1

        summary["updated_at"] = _utc_now()
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    def verify_question_record(
        self,
        *,
        question_record: dict[str, Any],
        sample_record: dict[str, Any] | None,
        question_index: int,
        question_fingerprint: str,
    ) -> dict[str, Any]:
        sample = sample_record or {}
        question = _question_text(question_record)
        gold_answer = str(question_record.get("answer") or "").strip()
        node_types = list(((sample.get("path") or {}).get("node_types") or []))
        trajectory_type = _classify_trajectory(node_types)
        bundle = self.assembler.build_bundle(question_record=question_record, sample_record=sample)
        solver_result = self._solve_question(bundle=bundle)
        reasoning_check = self._check_reasoning_complete(bundle=bundle, solver_result=solver_result)
        citations_exist = self._check_citations_exist(bundle=bundle, solver_result=solver_result)
        citations_scope = self._check_citations_within_scope(bundle=bundle, solver_result=solver_result)
        question_id = str(question_record.get("question_id") or question_index)
        question_input_image_url = _extract_question_input_image_url(
            question_record=question_record,
            sample_record=sample,
        )
        answer_judgment = self._judge_answer(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=str(solver_result.get("answer") or ""),
            question_id=question_id,
        )
        question_only_solver_result = self._solve_question_only_shortcut(
            question=question,
            question_id=question_id,
            image_url=question_input_image_url,
        )
        question_only_answer_judgment = self._judge_answer(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=str(question_only_solver_result.get("answer") or ""),
            question_id=f"{question_id}:question_only",
        )
        question_only_shortcut = {
            "passed": not bool(question_only_answer_judgment.get("correct")),
            "solver_status": question_only_solver_result.get("status") or "-",
            "predicted_answer": str(question_only_solver_result.get("answer") or ""),
            "shortcut_basis": str(question_only_solver_result.get("shortcut_basis") or ""),
            "cannot_answer_reason": str(question_only_solver_result.get("cannot_answer_reason") or ""),
            "confidence": _safe_float(question_only_solver_result.get("confidence")),
            "answer_judgment": question_only_answer_judgment,
        }

        checks = {
            "repository_non_empty": {
                "passed": bool(bundle.items),
                "item_count": len(bundle.items),
                "relevant_count": len(bundle.relevant_labels),
            },
            "reasoning_complete": reasoning_check,
            "citations_exist": citations_exist,
            "citations_within_relevant_scope": citations_scope,
            "answer_judgment": answer_judgment,
            "question_only_shortcut": question_only_shortcut,
        }

        reject_reasons: list[str] = []
        if not checks["repository_non_empty"]["passed"]:
            reject_reasons.append("repository_empty")
        if not reasoning_check.get("passed"):
            reject_reasons.append(str(reasoning_check.get("reason") or "reasoning_incomplete"))
        if not citations_exist.get("passed"):
            reject_reasons.append("invalid_citation_labels")
        if not citations_scope.get("passed"):
            reject_reasons.append("used_distractor_or_out_of_scope_evidence")
        if not answer_judgment.get("correct"):
            reject_reasons.append("answer_incorrect")
        if not question_only_shortcut.get("passed"):
            reject_reasons.append("closed_book_shortcut")

        return {
            "question_number": question_index,
            "question_id": question_record.get("question_id"),
            "sample_id": question_record.get("sample_id"),
            "path_id": question_record.get("path_id"),
            "status": question_record.get("status"),
            "question": question,
            "gold_answer": gold_answer,
            "question_input_image_url": question_input_image_url,
            "node_types": node_types,
            "trajectory_type": trajectory_type,
            "question_fingerprint": question_fingerprint,
            "verifier_config": self._verifier_config(),
            "repository": bundle.to_dict(),
            "solver_result": solver_result,
            "question_only_solver_result": question_only_solver_result,
            "checks": checks,
            "final_keep": not reject_reasons,
            "reject_reasons": reject_reasons,
            "verified_at": _utc_now(),
        }

    def _solve_question(self, *, bundle: RepositoryBundle) -> dict[str, Any]:
        request = build_repository_solver_request(
            bundle=bundle,
            answer_model_alias=self.answer_model_alias,
            answer_max_tokens=self.assembler.config.answer_max_tokens,
            user_content=self.assembler.build_solver_user_content(bundle=bundle),
        )
        try:
            response = self.model_client.generate(request)
        except Exception as exc:
            return {
                "status": "error",
                "answer": "",
                "reasoning_steps": [],
                "used_evidence": [],
                "insufficient_reason": "",
                "error": f"{exc.__class__.__name__}: {exc}",
            }

        parsed = _parse_freeform_solver_response(str(getattr(response, "content", response) or ""))

        raw_status = str(parsed.get("status") or parsed.get("final_status") or "").strip().lower()
        status = raw_status if raw_status in {"solved", "insufficient_evidence"} else ("solved" if parsed.get("answer") else "insufficient_evidence")
        reasoning_steps = []
        raw_steps = parsed.get("reasoning_steps") or parsed.get("steps") or []
        if isinstance(raw_steps, list):
            for index, item in enumerate(raw_steps, start=1):
                if not isinstance(item, dict):
                    continue
                citations = item.get("citations") or item.get("evidence") or []
                if not isinstance(citations, list):
                    citations = [citations]
                reasoning_steps.append(
                    {
                        "step": item.get("step") if item.get("step") is not None else index,
                        "claim": str(item.get("claim") or item.get("reasoning") or item.get("text") or "").strip(),
                        "citations": [
                            _normalize_citation_label(citation)
                            for citation in citations
                            if str(citation or "").strip()
                        ],
                    }
                )

        used_evidence = parsed.get("used_evidence") or []
        if not isinstance(used_evidence, list):
            used_evidence = [used_evidence]
        normalized_used = [
            _normalize_citation_label(item)
            for item in used_evidence
            if str(item or "").strip()
        ]
        if not normalized_used:
            normalized_used = sorted(
                {
                    citation
                    for step in reasoning_steps
                    for citation in (step.get("citations") or [])
                }
            )

        return {
            "status": status,
            "answer": str(parsed.get("answer") or parsed.get("final_answer") or "").strip(),
            "reasoning_steps": reasoning_steps,
            "used_evidence": normalized_used,
            "insufficient_reason": str(parsed.get("insufficient_reason") or "").strip(),
            "raw_text": str(parsed.get("raw_text") or ""),
            "raw": parsed,
        }

    def _solve_question_only_shortcut(
        self,
        *,
        question: str,
        question_id: str,
        image_url: str | None = None,
    ) -> dict[str, Any]:
        request = build_question_only_shortcut_request(
            question=question,
            answer_model_alias=self.answer_model_alias,
            answer_max_tokens=self.assembler.config.question_only_answer_max_tokens,
            question_id=question_id,
            image_url=image_url,
        )
        try:
            response = self.model_client.generate(request)
        except Exception as exc:
            return {
                "status": "error",
                "answer": "",
                "shortcut_basis": "",
                "confidence": 0.0,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        parsed = _parse_freeform_question_only_response(str(getattr(response, "content", response) or ""))
        raw_status = str(parsed.get("status") or "").strip().lower()
        raw_answer = str(parsed.get("answer") or parsed.get("final_answer") or "").strip()
        raw_shortcut_basis = str(parsed.get("shortcut_basis") or parsed.get("reason") or parsed.get("analysis") or "").strip()
        status = raw_status if raw_status in {"answered", "cannot_answer"} else ("answered" if raw_answer else "cannot_answer")
        cannot_answer_reason = ""
        answer = raw_answer
        shortcut_basis = raw_shortcut_basis
        if status == "cannot_answer":
            cannot_answer_reason = str(parsed.get("cannot_answer_reason") or "").strip() or raw_answer or raw_shortcut_basis
            answer = ""
        return {
            "status": status,
            "answer": answer,
            "shortcut_basis": shortcut_basis,
            "cannot_answer_reason": cannot_answer_reason,
            "confidence": _safe_float(parsed.get("confidence")),
            "raw_text": str(parsed.get("raw_text") or ""),
            "raw": parsed,
        }

    def _judge_answer(
        self,
        *,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        question_id: str,
    ) -> dict[str, Any]:
        request = build_repository_answer_judge_request(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer,
            judge_model_alias=self.judge_model_alias,
            judge_max_tokens=self.assembler.config.judge_max_tokens,
            question_id=question_id,
        )
        try:
            parsed = self.model_client.generate_json(request)
        except Exception as exc:
            return {
                "correct": False,
                "confidence": 0.0,
                "reason": "judge_model_error",
                "normalized_gold_answer": gold_answer,
                "normalized_predicted_answer": predicted_answer,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        return {
            "correct": bool(parsed.get("correct")),
            "confidence": _safe_float(parsed.get("confidence")),
            "reason": str(parsed.get("reason") or ""),
            "normalized_gold_answer": str(parsed.get("normalized_gold_answer") or gold_answer),
            "normalized_predicted_answer": str(parsed.get("normalized_predicted_answer") or predicted_answer),
            "raw": parsed,
        }

    def _check_reasoning_complete(
        self,
        *,
        bundle: RepositoryBundle,
        solver_result: dict[str, Any],
    ) -> dict[str, Any]:
        if solver_result.get("status") == "error":
            return {"passed": False, "reason": "solver_error", "detail": solver_result.get("error") or ""}
        if solver_result.get("status") == "insufficient_evidence":
            return {
                "passed": False,
                "reason": "insufficient_evidence_claimed",
                "detail": str(solver_result.get("insufficient_reason") or ""),
            }
        reasoning_steps = list(solver_result.get("reasoning_steps") or [])
        if len(reasoning_steps) < self.assembler.config.min_reasoning_steps:
            return {
                "passed": False,
                "reason": "too_few_reasoning_steps",
                "detail": f"reasoning_steps={len(reasoning_steps)}",
            }
        missing_claims = [step.get("step") for step in reasoning_steps if not str(step.get("claim") or "").strip()]
        missing_citations = [step.get("step") for step in reasoning_steps if not (step.get("citations") or [])]
        if missing_claims:
            return {
                "passed": False,
                "reason": "missing_reasoning_claim",
                "detail": f"steps={missing_claims}",
            }
        if missing_citations:
            return {
                "passed": False,
                "reason": "missing_reasoning_citations",
                "detail": f"steps={missing_citations}",
            }
        used_labels = sorted(
            {
                _normalize_citation_label(label)
                for step in reasoning_steps
                for label in (step.get("citations") or [])
                if str(label or "").strip()
            }
        )
        minimum_unique = min(self.assembler.config.min_unique_citations, max(1, len(bundle.relevant_labels)))
        if len(used_labels) < minimum_unique:
            return {
                "passed": False,
                "reason": "too_few_unique_citations",
                "detail": f"used={len(used_labels)} required={minimum_unique}",
            }
        return {
            "passed": True,
            "reason": "ok",
            "detail": f"reasoning_steps={len(reasoning_steps)} unique_citations={len(used_labels)}",
        }

    @staticmethod
    def _check_citations_exist(
        *,
        bundle: RepositoryBundle,
        solver_result: dict[str, Any],
    ) -> dict[str, Any]:
        available = {item.label for item in bundle.items}
        cited = {
            _normalize_citation_label(label)
            for label in (solver_result.get("used_evidence") or [])
            if str(label or "").strip()
        }
        invalid = sorted(label for label in cited if label not in available)
        return {
            "passed": not invalid,
            "cited_labels": sorted(cited),
            "invalid_labels": invalid,
        }

    @staticmethod
    def _check_citations_within_scope(
        *,
        bundle: RepositoryBundle,
        solver_result: dict[str, Any],
    ) -> dict[str, Any]:
        relevant = set(bundle.relevant_labels)
        cited = {
            _normalize_citation_label(label)
            for label in (solver_result.get("used_evidence") or [])
            if str(label or "").strip()
        }
        out_of_scope = sorted(label for label in cited if label not in relevant)
        return {
            "passed": not out_of_scope,
            "cited_labels": sorted(cited),
            "out_of_scope_labels": out_of_scope,
        }

    def _question_fingerprint(
        self,
        *,
        question_record: dict[str, Any],
        sample_record: dict[str, Any] | None,
    ) -> str:
        payload = {
            "question_id": question_record.get("question_id"),
            "sample_id": question_record.get("sample_id"),
            "path_id": question_record.get("path_id"),
            "status": question_record.get("status"),
            "question": _question_text(question_record),
            "answer": question_record.get("answer"),
            "hop_chain": list((sample_record or {}).get("hop_chain") or []),
            "repository_config": self.assembler.config.to_dict(),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _verifier_config(self) -> dict[str, Any]:
        return {
            "answer_model_alias": self.answer_model_alias,
            "judge_model_alias": self.judge_model_alias,
            "question_only_answer_model_alias": self.answer_model_alias,
            "repository_config": self.assembler.config.to_dict(),
        }

    def _can_reuse_existing_record(
        self,
        *,
        existing: dict[str, Any] | None,
        fingerprint: str,
    ) -> bool:
        if not existing:
            return False
        if str(existing.get("question_fingerprint") or "") != fingerprint:
            return False
        existing_config = existing.get("verifier_config") or {}
        return existing_config == self._verifier_config()



def format_repository_bundle(
    bundle: RepositoryBundle,
    *,
    width: int = 100,
    include_hidden: bool = True,
) -> str:
    lines = [
        "=" * 96,
        f"Repository Bundle | question_id={bundle.question_id or '-'} | sample_id={bundle.sample_id or '-'} | path_id={bundle.path_id or '-'}",
        "-" * 96,
    ]
    lines.extend(_format_wrapped("question", bundle.question, width=width, indent=2))
    lines.extend(_format_wrapped("gold_answer", bundle.gold_answer, width=width, indent=2))
    lines.append(
        f"  summary: items={len(bundle.items)} docs={sum(1 for item in bundle.items if item.item_type == 'doc')} "
        f"images={sum(1 for item in bundle.items if item.item_type == 'image')} "
        f"relevant={len(bundle.relevant_labels)} distractors={len(bundle.distractor_labels)}"
    )
    if include_hidden:
        lines.append(f"  relevant_labels: {', '.join(bundle.relevant_labels) if bundle.relevant_labels else '-'}")
        lines.append(f"  distractor_labels: {', '.join(bundle.distractor_labels) if bundle.distractor_labels else '-'}")
    lines.append("-" * 96)
    lines.append("Items")
    for item in bundle.items:
        tag = "relevant" if item.is_relevant else "distractor"
        lines.append(f"  [{item.label}] type={item.item_type} tag={tag} reason={item.selection_reason or '-'}")
        if include_hidden:
            if item.source_edge_id:
                lines.append(f"    source_edge_id: {item.source_edge_id}")
            if item.source_node_id:
                lines.append(f"    source_node_id: {item.source_node_id}")
        if item.item_type == "doc":
            lines.extend(_format_wrapped("text", str(item.text or ""), width=width, indent=4))
        else:
            lines.extend(_format_wrapped("image_url", str(item.image_url or ""), width=width, indent=4))
        if include_hidden and item.metadata:
            hidden_summary = json.dumps(item.metadata, ensure_ascii=False, sort_keys=True)
            lines.extend(_format_wrapped("metadata", hidden_summary, width=width, indent=4))
    lines.append("=" * 96)
    return "\n".join(lines)


def format_verification_record(record: dict[str, Any], *, width: int = 100) -> str:
    checks = dict(record.get("checks") or {})
    solver = dict(record.get("solver_result") or {})
    question_only_solver = dict(record.get("question_only_solver_result") or {})
    lines = [
        "=" * 96,
        f"Repository Verification | question_id={record.get('question_id') or '-'} | sample_id={record.get('sample_id') or '-'}",
        "-" * 96,
        f"  final_keep: {bool(record.get('final_keep'))}",
        f"  reject_reasons: {', '.join(record.get('reject_reasons') or []) or '-'}",
        f"  trajectory_type: {record.get('trajectory_type') or '-'}",
        f"  solver_status: {solver.get('status') or '-'}",
        f"  question_only_solver_status: {question_only_solver.get('status') or '-'}",
    ]
    lines.extend(_format_wrapped("predicted_answer", str(solver.get("answer") or ""), width=width, indent=2))
    lines.extend(_format_wrapped("question_only_answer", str(question_only_solver.get("answer") or ""), width=width, indent=2))
    lines.extend(_format_wrapped("question_only_reason", str(question_only_solver.get("cannot_answer_reason") or question_only_solver.get("shortcut_basis") or ""), width=width, indent=2))
    lines.extend(_format_wrapped("gold_answer", str(record.get("gold_answer") or ""), width=width, indent=2))
    lines.append("-" * 96)
    lines.append("Checks")
    for check_name in (
        "repository_non_empty",
        "reasoning_complete",
        "citations_exist",
        "citations_within_relevant_scope",
        "answer_judgment",
        "question_only_shortcut",
    ):
        payload = checks.get(check_name) or {}
        lines.append(f"  {check_name}: {json.dumps(payload, ensure_ascii=False, sort_keys=True)}")
    if solver.get("reasoning_steps"):
        lines.append("-" * 96)
        lines.append("Reasoning Steps")
        for step in solver.get("reasoning_steps") or []:
            lines.append(f"  step={step.get('step')} citations={','.join(step.get('citations') or []) or '-'}")
            lines.extend(_format_wrapped("claim", str(step.get("claim") or ""), width=width, indent=4))
    lines.append("=" * 96)
    return "\n".join(lines)


def print_summary_report(summary: dict[str, Any]) -> None:
    total = int(summary.get("verified_total") or 0)
    final_keep_total = int(summary.get("final_keep_total") or 0)
    answer_correct_total = int(summary.get("answer_correct_total") or 0)
    question_only_shortcut_total = int(summary.get("question_only_shortcut_total") or 0)
    invalid_citation_total = int(summary.get("invalid_citation_total") or 0)
    out_of_scope_citation_total = int(summary.get("out_of_scope_citation_total") or 0)
    insufficient_evidence_total = int(summary.get("insufficient_evidence_total") or 0)
    trajectory_counts = dict(summary.get("trajectory_type_counts") or {})

    print("repository_verification_report:")
    print(f"  total_questions: {total}")
    print(f"  final_keep: {final_keep_total}/{total} ({_format_ratio(final_keep_total, total)})")
    print(f"  answer_correct: {answer_correct_total}/{total} ({_format_ratio(answer_correct_total, total)})")
    print(f"  question_only_shortcut: {question_only_shortcut_total}/{total} ({_format_ratio(question_only_shortcut_total, total)})")
    print(f"  invalid_citation: {invalid_citation_total}/{total} ({_format_ratio(invalid_citation_total, total)})")
    print(f"  out_of_scope_citation: {out_of_scope_citation_total}/{total} ({_format_ratio(out_of_scope_citation_total, total)})")
    print(f"  insufficient_evidence: {insufficient_evidence_total}/{total} ({_format_ratio(insufficient_evidence_total, total)})")
    print("  trajectory_types:")
    for key in ("text_only", "image_first", "image_end", "multi_image", "unclassified"):
        count = int(trajectory_counts.get(key) or 0)
        print(f"    {key}: {count}/{total} ({_format_ratio(count, total)})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", required=True, help="Directory containing questions.jsonl and samples.jsonl.")
    parser.add_argument("--graph-dir", default=None, help="Graph directory containing nodes.jsonl and edges.jsonl. Inferred from vqa_dir when omitted.")
    parser.add_argument("--answer-model-alias", required=True, help="Model alias used to solve each repository-grounded question and each question-only shortcut attempt.")
    parser.add_argument("--judge-model-alias", required=True, help="Model alias used to judge the predicted answer.")
    parser.add_argument("--random-seed", type=int, default=0, help="Random seed used for distractor sampling.")
    parser.add_argument("--max-relevant-docs-per-edge", type=int, default=2)
    parser.add_argument("--max-sibling-doc-distractors-per-edge", type=int, default=1)
    parser.add_argument("--max-random-doc-distractors", type=int, default=2)
    parser.add_argument("--max-sibling-image-distractors-per-image", type=int, default=1)
    parser.add_argument("--max-random-image-distractors", type=int, default=1)
    parser.add_argument("--min-reasoning-steps", type=int, default=1)
    parser.add_argument("--min-unique-citations", type=int, default=2)
    parser.add_argument("--question-only-answer-max-tokens", type=int, default=256)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    graph_dir = Path(args.graph_dir).expanduser().resolve() if args.graph_dir else _infer_graph_dir(args.vqa_dir)
    store = JsonlGraphStore(graph_dir)
    graph = GraphView(store)
    config = RepositoryVerificationConfig(
        random_seed=args.random_seed,
        max_relevant_docs_per_edge=args.max_relevant_docs_per_edge,
        max_sibling_doc_distractors_per_edge=args.max_sibling_doc_distractors_per_edge,
        max_random_doc_distractors=args.max_random_doc_distractors,
        max_sibling_image_distractors_per_image=args.max_sibling_image_distractors_per_image,
        max_random_image_distractors=args.max_random_image_distractors,
        min_reasoning_steps=args.min_reasoning_steps,
        min_unique_citations=args.min_unique_citations,
        question_only_answer_max_tokens=args.question_only_answer_max_tokens,
    )

    from synthesis.model_worker import LLM_WORKER

    verifier = OfflineGraphRepositoryVerifier(
        assembler=RepositoryAssembler(graph=graph, config=config),
        model_client=LLM_WORKER,
        answer_model_alias=args.answer_model_alias,
        judge_model_alias=args.judge_model_alias,
    )
    summary = verifier.run(vqa_dir=args.vqa_dir)
    print_summary_report(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
