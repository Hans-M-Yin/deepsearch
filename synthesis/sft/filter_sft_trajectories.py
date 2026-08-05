#!/usr/bin/env python3
"""Re-judge and quality-filter generated SFT trajectories.

This is a pre-processing gate for the ShareGPT converter.  It applies a strict
serial pipeline; a later judge only receives records that passed the previous
stage:

1. Remove trajectories truncated by the ReAct/tool-calling max-turn limit.
2. Re-judge final-answer correctness with ``--simple-model-alias`` and compare
   it with the stored ``answer_judge.is_correct`` value.
3. Score logic coherence, answer non-exposure, and tool-use quality with
   ``--quality-model-alias``.  Remove records with any dimension below 6.5.
4. On the dimension-pass subset only, remove records whose average score is
   below 7.

The output directory contains ``accepted_trajectories.jsonl`` for the next
conversion step, ``filtered_trajectories.jsonl`` for auditability,
``judge_results.jsonl`` with one compact decision per source row, and
``filter_report.json`` with category-specific source IDs and indices.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelWorkerClient


_FIXED_REQUEST_ID = "3200636808"
_MAX_TURN_STATUS = {
    "max_turns_reached",
    "max_tool_calling_turns",
    "max_react_turns",
}
_CATEGORY_NAMES = (
    "wrong_answer_ids",
    "max_turn_reached_ids",
    "tool_call_low_score_ids",
    "answer_exposure_low_score_ids",
    "logic_coherence_low_score_ids",
    "average_score_low_ids",
)
_QUALITY_DIMENSION_PASSING_SCORE = 6.5
_QUALITY_AVERAGE_PASSING_SCORE = 7.0


PROMPT_SIMPLE_CORRECTNESS = """
You are a professional answer-evaluation expert. You will be given a complex multi-hop knowledge question, its final gold answer, and a candidate answer to be verified. Your task is to evaluate whether the candidate answer is semantically consistent with the gold answer, that is, whether it should be judged correct or incorrect.

Requirements:
1. If the candidate answer is semantically consistent with the gold answer, it should be judged correct. Harmless paraphrases, aliases, or formatting differences are allowed.
2. Judge correctness flexibly according to what the question actually asks. The gold answer may contain additional explanatory details, or it may be more concise than the candidate answer. Your judgment should be based on the question itself: as long as the candidate answer correctly addresses what the question asks, and the part it answers is semantically consistent with the gold answer, it should be judged correct. Do not consider extra factual details in the gold answer, and do not consider extra content in the candidate answer.
3. Return one JSON object and only one JSON object, with no Markdown:
{
"predict": "pass|reject",
"confidence": 0.0,
"reason": ""
}
""".strip()


PROMPT_QUALITY = """Next, I will give you a multi-hop knowledge deep-search question, the intermediate answers and final gold answer, as well as a constructed model reasoning trajectory. You need to score this reasoning trajectory from three dimensions—logic coherence, answer exposure, and tool use—to judge whether it is suitable for training a knowledge-search model. Each dimension should be scored from 1 to 10, with 6.5 as the passing line. If the trajectory performs well on a given dimension and is suitable as SFT data, it should receive a high score (8.5–10). Conversely, if it performs poorly on that dimension and is not suitable to be used directly as SFT training data, it should receive a low score (1–6). A score of 6.5 is considered passing. The three dimensions are as follows:
logic_coherence: Is the visible reasoning process coherent, step-by-step, and grounded in the question and observations? Points should be deducted for unexplained jumps, contradictions, unsupported conclusions, and reasoning that does not naturally follow from the previous tool result.
answer_exposure: This is a “non-leakage” score: 10 means the answer is not leaked and the reasoning path discovers it naturally; 1 means the assistant reveals the gold answer or private hop-chain facts before the evidence supports them, making the reasoning unnatural. The private reference context below is for evaluators only. It must never be mentioned as a source in the visible assistant transcript. If the final <answer> block is given only after sufficient evidence has been obtained, that should not be penalized.
tool_use: Does each step choose an appropriate tool, and are the arguments used properly? The agent should not blindly repeat the same tool/query; before claiming that a search-result page contains certain content, it should first inspect the selected result with read_url; for candidates returned by t2i/i2i, it should not claim that those images are visible before read_url(image_id) succeeds. Redundant, aimless, malformed, or poorly targeted calls should be penalized.

Basic rules:
1. t2t_search returns compact text-result metadata, not full page contents. t2i_search and i2i_search return candidate metadata; their images are not visible until read_url(image_id) succeeds. read_url(source_page_id) provides webpage evidence; read_url(image_id) provides an image. If a tool result has ok=false, it cannot count as successful evidence.
2. It is allowed for the response to directly use the search snippet as evidence, even without reading the full webpage in detail.
3. Tool-call errors may appear in the response; successfully handling errors and promptly changing reasoning direction is worth extra credit.
4. Return strictly JSON only:
{
"logic_coherence": 1~10,
"answer_exposure": 1~10,
"tool_use": 1~10,
"logic_reason": "",
"answer_exposure_reason": "",
"tool_use_reason": "",
"overall_reason": ""
}

""".strip()


@dataclass(slots=True)
class FilterConfig:
    quality_model_alias: str
    simple_model_alias: str
    quality_max_tokens: int = 1200
    simple_max_tokens: int = 800
    max_transcript_chars: int = 50000
    max_reference_chars: int = 12000
    temperature: float | None = None


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        value = json.loads(candidate)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, dict) else None


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    half = max(1, limit // 2)
    return f"{text[:half]}\n...[truncated]...\n{text[-half:]}"


def _message_content_for_judge(content: Any) -> str:
    """Render text without embedding data URLs or private initial prompts."""

    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return _json_text(content).strip() if content is not None else ""
    parts: list[str] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        part_type = str(part.get("type") or "").lower()
        if part_type in {"text", "input_text"}:
            text = str(part.get("text") or "").strip()
            if text:
                parts.append(text)
        elif part_type in {"image", "image_url", "image_path", "input_image"}:
            parts.append("<image>")
    return "\n".join(parts).strip()


def _transcript(record: dict[str, Any], *, max_chars: int) -> str:
    """Format only generated assistant/tool turns, omitting the answer-bearing user prompt."""

    lines: list[str] = []
    for message in record.get("raw_messages") or []:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "").lower()
        if role == "system":
            continue
        if role == "assistant":
            lines.append("[assistant]")
            content = _message_content_for_judge(message.get("content"))
            if content:
                lines.append(content)
            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                lines.append("tool_calls=" + _json_text(tool_calls))
        elif role == "tool":
            name = str(message.get("name") or "tool")
            lines.append(f"[tool:{name}]")
            lines.append(_message_content_for_judge(message.get("content")))
        elif role == "user":
            # The initial user prompt contains `Answer:` and private facts. Keep only
            # the existence/order of a later image attachment, never its text.
            content = message.get("content")
            has_image = bool(
                isinstance(content, list)
                and any(str(item.get("type") or "").lower() in {"image", "image_url", "image_path", "input_image"} for item in content if isinstance(item, dict))
            )
            if has_image:
                lines.extend(["[user_attachment]", "<image>"])
    return _truncate("\n".join(lines), max_chars)


def _private_reference_context(record: dict[str, Any], *, max_chars: int) -> str:
    context = {
        "gold_answer": record.get("gold_answer") or "",
        "hop_chain": record.get("hop_chain") or [],
        "hop_chain_coverage": record.get("hop_chain_coverage") or {},
    }
    return _truncate(json.dumps(context, ensure_ascii=False, indent=2), max_chars)


def _last_assistant_text(record: dict[str, Any]) -> str:
    for message in reversed(record.get("raw_messages") or []):
        if isinstance(message, dict) and str(message.get("role") or "").lower() == "assistant":
            return _message_content_for_judge(message.get("content"))
    return ""


def _record_id(record: dict[str, Any], source_index: int) -> str:
    for key in ("sample_id", "question_id", "path_id"):
        value = str(record.get(key) or "").strip()
        if value:
            return value
    return f"source_index_{source_index}"


def _stored_correctness(record: dict[str, Any]) -> bool | None:
    judge = record.get("answer_judge")
    if not isinstance(judge, dict):
        return None
    value = judge.get("is_correct")
    if isinstance(value, bool):
        return value
    verdict = str(judge.get("verdict") or "").strip().lower()
    if verdict in {"pass", "correct", "accepted"}:
        return True
    if verdict in {"reject", "incorrect", "wrong"}:
        return False
    return None


def _max_turn_reason(record: dict[str, Any]) -> str | None:
    summary = record.get("generation_summary") or {}
    if not isinstance(summary, dict):
        summary = {}
    candidates = [summary.get("generation_status"), summary.get("stop_reason")]
    candidates.extend(summary.get("failure_reasons") or [])
    for candidate in candidates:
        text = str(candidate or "").strip().lower()
        if text in _MAX_TURN_STATUS or "max_turn" in text or "max react turns" in text:
            return text
    return None


def _coerce_score(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"quality judge field {name!r} must be numeric")
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"quality judge field {name!r} is not numeric") from exc
    if not math.isfinite(score):
        raise ValueError(f"quality judge field {name!r} is not finite")
    if not 1.0 <= score <= 10.0:
        raise ValueError(f"quality judge field {name!r} is outside 1..10: {score}")
    return score


class SftTrajectoryFilter:
    def __init__(self, *, config: FilterConfig, model_client: ModelWorkerClient | None = None) -> None:
        self.config = config
        self.model_client = model_client or LLM_WORKER

    def _generate_json(
        self,
        *,
        model_alias: str,
        system_prompt: str,
        payload: dict[str, Any],
        max_tokens: int,
        trace_label: str,
    ) -> tuple[dict[str, Any] | None, str, str | None]:
        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=system_prompt),
                        ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False, indent=2)),
                    ],
                    temperature=self.config.temperature,
                    max_tokens=max_tokens,
                    response_format={"type": "json_object"},
                    metadata={
                        "trace_label": trace_label,
                        "session_id": _FIXED_REQUEST_ID,
                        "prompt_cache_key": _FIXED_REQUEST_ID,
                        "user_id": _FIXED_REQUEST_ID,
                        "x_tt_logid": _FIXED_REQUEST_ID,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001 - isolate one bad judge call
            return None, "", f"{exc.__class__.__name__}: {exc}"
        raw = str(getattr(response, "content", "") or "")
        parsed = _extract_json_object(raw)
        if parsed is None:
            return None, raw, "model_output_not_json"
        return parsed, raw, None

    def _simple_judge(self, record: dict[str, Any], record_id: str) -> dict[str, Any]:
        payload = {
            "question": record.get("question") or "",
            "gold_answer": record.get("gold_answer") or "",
            "model_extracted_answer": record.get("extracted_answer") or "",
            "final_assistant_message": _last_assistant_text(record),
        }
        parsed, raw, error = self._generate_json(
            model_alias=self.config.simple_model_alias,
            system_prompt=PROMPT_SIMPLE_CORRECTNESS,
            payload=payload,
            max_tokens=self.config.simple_max_tokens,
            trace_label=f"sft_filter_simple_correctness:{record_id}",
        )
        if parsed is None:
            return {"status": "error", "error": error, "raw_model_output": raw, "answer_correct": None}
        # PROMPT_SIMPLE_CORRECTNESS deliberately uses ``predict`` as its
        # contract. Do not fall back to the legacy ``verdict`` field: doing so
        # would make the parser accept a response that violates the prompt.
        predict = str(parsed.get("predict") or "").strip().lower()
        if predict not in {"pass", "reject"}:
            return {
                **parsed,
                "status": "error",
                "error": "missing_or_invalid_predict",
                "raw_model_output": raw,
                "answer_correct": None,
            }
        answer_correct = predict == "pass"
        return {
            **parsed,
            "status": "ok",
            "predict": predict,
            # Keep a normalized alias for downstream audit consumers.  It is
            # never read as a fallback when parsing a new model response.
            "verdict": predict,
            "answer_correct": answer_correct,
            "raw_model_output": raw,
            "model_alias": self.config.simple_model_alias,
        }

    def _quality_judge(self, record: dict[str, Any], record_id: str) -> dict[str, Any]:
        payload = {
            "question": record.get("question") or "",
            "gold_answer": record.get("gold_answer") or "",
            "model_extracted_answer": record.get("extracted_answer") or "",
            "private_reference_context": _private_reference_context(record, max_chars=self.config.max_reference_chars),
            "sft_response_transcript": _transcript(record, max_chars=self.config.max_transcript_chars),
            "generation_summary": record.get("generation_summary") or {},
        }
        parsed, raw, error = self._generate_json(
            model_alias=self.config.quality_model_alias,
            system_prompt=PROMPT_QUALITY,
            payload=payload,
            max_tokens=self.config.quality_max_tokens,
            trace_label=f"sft_filter_quality:{record_id}",
        )
        if parsed is None:
            return {"status": "error", "error": error, "raw_model_output": raw}
        try:
            scores = {
                "logic_coherence": _coerce_score(parsed.get("logic_coherence"), "logic_coherence"),
                "answer_exposure": _coerce_score(parsed.get("answer_exposure"), "answer_exposure"),
                "tool_use": _coerce_score(parsed.get("tool_use"), "tool_use"),
            }
        except ValueError as exc:
            return {
                **parsed,
                "status": "error",
                "error": str(exc),
                "raw_model_output": raw,
            }
        average = sum(scores.values()) / 3.0
        return {
            **parsed,
            **scores,
            "average_score": round(average, 4),
            "status": "ok",
            "raw_model_output": raw,
            "model_alias": self.config.quality_model_alias,
        }

    def evaluate_record(self, record: dict[str, Any], *, source_index: int, source_line: int) -> dict[str, Any]:
        record_id = _record_id(record, source_index)
        result: dict[str, Any] = {
            "record_id": record_id,
            "source_index": source_index,
            "source_position": source_index + 1,
            "source_line": source_line,
            "question_id": record.get("question_id"),
            "sample_id": record.get("sample_id"),
            "path_id": record.get("path_id"),
            "decision": "keep",
            "filter_reasons": [],
        }

        max_turn_reason = _max_turn_reason(record)
        if max_turn_reason:
            result["decision"] = "reject"
            result["filter_stage"] = "max_turn"
            result["filter_reasons"] = ["max_turn_reached"]
            result["max_turn_reason"] = max_turn_reason
            return result

        stored = _stored_correctness(record)
        simple = self._simple_judge(record, record_id)
        result["stored_answer_correct"] = stored
        result["simple_judge"] = simple
        new_correct = simple.get("answer_correct") if simple.get("status") == "ok" else None
        result["simple_answer_correct"] = new_correct
        if stored is not None and new_correct is not None and stored != new_correct:
            result["correctness_disagreement"] = True
        else:
            result["correctness_disagreement"] = False

        reasons: list[str] = []
        if simple.get("status") != "ok":
            reasons.append("simple_judge_error")
        elif new_correct is not True:
            reasons.append("wrong_answer" if new_correct is False else "correctness_uncertain")
        if reasons:
            result["decision"] = "reject"
            result["filter_stage"] = "answer_correctness"
            result["filter_reasons"] = reasons
            return result

        quality = self._quality_judge(record, record_id)
        result["quality_judge"] = quality
        if quality.get("status") != "ok":
            result["decision"] = "reject"
            result["filter_stage"] = "quality_judge"
            result["filter_reasons"] = ["quality_judge_error"]
            return result

        result["quality_scores"] = {
            key: quality[key]
            for key in ("logic_coherence", "answer_exposure", "tool_use", "average_score")
        }
        dimension_reasons: list[str] = []
        if quality["tool_use"] < _QUALITY_DIMENSION_PASSING_SCORE:
            dimension_reasons.append("tool_call_low_score")
        if quality["answer_exposure"] < _QUALITY_DIMENSION_PASSING_SCORE:
            dimension_reasons.append("answer_exposure_low_score")
        if quality["logic_coherence"] < _QUALITY_DIMENSION_PASSING_SCORE:
            dimension_reasons.append("logic_coherence_low_score")
        if dimension_reasons:
            result["decision"] = "reject"
            result["filter_stage"] = "quality_dimensions"
            result["filter_reasons"] = dimension_reasons
            return result

        if quality["average_score"] < _QUALITY_AVERAGE_PASSING_SCORE:
            result["decision"] = "reject"
            result["filter_stage"] = "quality_average"
            result["filter_reasons"] = ["average_score_low"]
        return result


def _load_jsonl(path: Path) -> Iterable[tuple[int, int, dict[str, Any]]]:
    source_index = 0
    with path.open("r", encoding="utf-8") as handle:
        for source_line, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict):
                raise ValueError(f"expected a JSON object on line {source_line}")
            yield source_index, source_line, value
            source_index += 1


def _write_jsonl(handle: Any, value: Any) -> None:
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def _category_report(results: list[dict[str, Any]]) -> dict[str, Any]:
    ids: dict[str, list[str]] = {name: [] for name in _CATEGORY_NAMES}
    source_indices: dict[str, list[int]] = {name: [] for name in _CATEGORY_NAMES}
    source_positions: dict[str, list[int]] = {name: [] for name in _CATEGORY_NAMES}
    details: dict[str, list[dict[str, Any]]] = {name: [] for name in _CATEGORY_NAMES}
    correctness_disagreements: list[dict[str, Any]] = []
    extra: dict[str, list[str]] = {
        "correctness_uncertain_ids": [],
        "simple_judge_error_ids": [],
        "quality_judge_error_ids": [],
    }
    for result in results:
        record_id = str(result["record_id"])
        if result.get("correctness_disagreement"):
            correctness_disagreements.append(
                {
                    "record_id": record_id,
                    "source_index": result["source_index"],
                    "source_position": result["source_position"],
                    "source_line": result["source_line"],
                    "stored_answer_correct": result.get("stored_answer_correct"),
                    "simple_answer_correct": result.get("simple_answer_correct"),
                }
            )
        for reason in result.get("filter_reasons") or []:
            mapping = {
                "wrong_answer": "wrong_answer_ids",
                "max_turn_reached": "max_turn_reached_ids",
                "tool_call_low_score": "tool_call_low_score_ids",
                "answer_exposure_low_score": "answer_exposure_low_score_ids",
                "logic_coherence_low_score": "logic_coherence_low_score_ids",
                "average_score_low": "average_score_low_ids",
            }
            category = mapping.get(reason)
            if category:
                ids[category].append(record_id)
                source_indices[category].append(int(result["source_index"]))
                source_positions[category].append(int(result["source_position"]))
                details[category].append(
                    {
                        "record_id": record_id,
                        "source_index": result["source_index"],
                        "source_position": result["source_position"],
                        "source_line": result["source_line"],
                        "quality_scores": result.get("quality_scores"),
                    }
                )
            elif reason == "correctness_uncertain":
                extra["correctness_uncertain_ids"].append(record_id)
            elif reason == "simple_judge_error":
                extra["simple_judge_error_ids"].append(record_id)
            elif reason == "quality_judge_error":
                extra["quality_judge_error_ids"].append(record_id)
    stage_counts = Counter(str(result.get("filter_stage") or "accepted") for result in results)
    after_max_turn = sum(1 for result in results if result.get("filter_stage") != "max_turn")
    after_correctness = sum(
        1
        for result in results
        if result.get("filter_stage") not in {"max_turn", "answer_correctness"}
    )
    after_dimensions = sum(
        1
        for result in results
        if result.get("filter_stage") not in {"max_turn", "answer_correctness", "quality_judge", "quality_dimensions"}
    )
    return {
        "pipeline_order": ["max_turn", "answer_correctness", "quality_dimensions", "quality_average"],
        "categories_are_nonexclusive": True,
        "stage_counts": dict(sorted(stage_counts.items())),
        "stage_survivors": {
            "after_max_turn": after_max_turn,
            "after_answer_correctness": after_correctness,
            "after_quality_dimensions": after_dimensions,
            "after_quality_average": stage_counts.get("accepted", 0),
        },
        "filtered_ids": ids,
        "filtered_source_indices": source_indices,
        "filtered_source_positions": source_positions,
        "category_details": details,
        "correctness_disagreements": correctness_disagreements,
        "additional_review_ids": extra,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Raw debug_vqa_batch trajectory JSONL.")
    parser.add_argument("--output-dir", required=True, help="Directory for accepted/rejected JSONL and report JSON.")
    parser.add_argument("--quality-model-alias", required=True, help="LLM_WORKER alias for the three-dimensional quality judge.")
    parser.add_argument("--simple-model-alias", required=True, help="LLM_WORKER alias for the answer-correctness rejudge.")
    parser.add_argument("--quality-max-tokens", type=int, default=1200)
    parser.add_argument("--simple-max-tokens", type=int, default=800)
    parser.add_argument("--max-transcript-chars", type=int, default=50000)
    parser.add_argument("--max-reference-chars", type=int, default=12000)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip this many non-empty input records.")
    parser.add_argument("--limit", type=int, default=0, help="Process at most this many records; <=0 means all.")
    return parser


def filter_jsonl(
    input_jsonl: str | Path,
    output_dir: str | Path,
    *,
    quality_model_alias: str,
    simple_model_alias: str,
    quality_max_tokens: int = 1200,
    simple_max_tokens: int = 800,
    max_transcript_chars: int = 50000,
    max_reference_chars: int = 12000,
    temperature: float | None = None,
    offset: int = 0,
    limit: int = 0,
    model_client: ModelWorkerClient | None = None,
) -> dict[str, Any]:
    input_path = Path(input_jsonl).expanduser().resolve()
    final_dir = Path(output_dir).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if final_dir.exists():
        raise FileExistsError(f"output directory already exists; choose a new path: {final_dir}")
    if offset < 0 or limit < 0:
        raise ValueError("offset and limit must be non-negative")
    final_dir.parent.mkdir(parents=True, exist_ok=True)
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{final_dir.name}.tmp-", dir=str(final_dir.parent)))
    try:
        config = FilterConfig(
            quality_model_alias=quality_model_alias,
            simple_model_alias=simple_model_alias,
            quality_max_tokens=quality_max_tokens,
            simple_max_tokens=simple_max_tokens,
            max_transcript_chars=max_transcript_chars,
            max_reference_chars=max_reference_chars,
            temperature=temperature,
        )
        judge = SftTrajectoryFilter(config=config, model_client=model_client)
        accepted_path = stage_dir / "accepted_trajectories.jsonl"
        filtered_path = stage_dir / "filtered_trajectories.jsonl"
        results_path = stage_dir / "judge_results.jsonl"
        results: list[dict[str, Any]] = []
        total_available = 0
        selected = 0
        with accepted_path.open("w", encoding="utf-8") as accepted_handle, filtered_path.open("w", encoding="utf-8") as filtered_handle, results_path.open("w", encoding="utf-8") as results_handle:
            for source_index, source_line, record in _load_jsonl(input_path):
                total_available += 1
                if source_index < offset or (limit > 0 and selected >= limit):
                    continue
                selected += 1
                result = judge.evaluate_record(record, source_index=source_index, source_line=source_line)
                results.append(result)
                enriched = dict(record)
                enriched["sft_trajectory_filter"] = result
                _write_jsonl(results_handle, result)
                _write_jsonl(accepted_handle if result["decision"] == "keep" else filtered_handle, enriched)
        category = _category_report(results)
        decision_counts = Counter(str(item.get("decision") or "unknown") for item in results)
        report = {
            "input_jsonl": str(input_path),
            "output_dir": str(final_dir),
            "quality_model_alias": quality_model_alias,
            "simple_model_alias": simple_model_alias,
            "offset": offset,
            "limit": limit,
            "total_available_records": total_available,
            "processed_records": selected,
            "decision_counts": dict(sorted(decision_counts.items())),
            **category,
        }
        (stage_dir / "filter_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stage_dir.rename(final_dir)
        return report
    except Exception:
        shutil.rmtree(stage_dir, ignore_errors=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    try:
        report = filter_jsonl(
            args.input_jsonl,
            args.output_dir,
            quality_model_alias=args.quality_model_alias,
            simple_model_alias=args.simple_model_alias,
            quality_max_tokens=args.quality_max_tokens,
            simple_max_tokens=args.simple_max_tokens,
            max_transcript_chars=args.max_transcript_chars,
            max_reference_chars=args.max_reference_chars,
            temperature=args.temperature,
            offset=args.offset,
            limit=args.limit,
        )
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(str(exc)) from exc
    print(json.dumps({"decision_counts": report["decision_counts"], "output_dir": report["output_dir"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
