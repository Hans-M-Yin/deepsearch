"""Verification hooks and offline verification runner for generated VQA samples."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelRouterWorkerClient

from .schemas import QuestionDraft, VerificationCheck, VerificationResult


PROMPT_VERIFY_GOLD_ANSWER_SANITY = """
You are verifying whether a gold answer is obviously incompatible with a generated multi-hop VQA question.

Your job is not to solve the full question from scratch. Your job is only to detect whether the provided gold answer
looks clearly wrong, off-topic, or inconsistent with the question and the provided hop chain.

Please be conservative:
- If the gold answer seems plausible or possibly correct, mark it as passing.
- Only fail when the answer is clearly unrelated, clearly of the wrong type, or clearly inconsistent with the question.
- Use the hop chain only as supporting context for what the question is about.

Return JSON in exactly this format:
{
  "passed": true,
  "confidence": 0.0,
  "issues": "",
  "reason": ""
}
"""


PROMPT_VERIFY_MODEL_ANSWER_JUDGE = """
You are judging whether a model-predicted answer should count as correct for a VQA question.

You will receive:
- the question
- the gold answer
- a model-predicted answer

Judge semantic correctness, not exact string match.

Guidelines:
- Accept paraphrases and semantically equivalent answers.
- If the predicted answer is substantially incomplete, off-topic, or refers to a different entity/object, mark it incorrect.
- Be strict about the core referent, but tolerant about wording differences.
- If the predicted answer is empty or says it cannot answer, mark it incorrect.

Return JSON in exactly this format:
{
  "correct": true,
  "confidence": 0.0,
  "reason": "",
  "normalized_gold_answer": "",
  "normalized_predicted_answer": ""
}
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class SampleVerifier:
    """First-pass verifier with structural placeholder checks."""

    def verify(self, *, question: QuestionDraft) -> VerificationResult:
        checks = [
            VerificationCheck(
                name="question_non_empty",
                passed=bool(question.question.strip()),
                detail="Question text must be non-empty.",
            ),
            VerificationCheck(
                name="answer_non_empty",
                passed=bool(question.answer.strip()),
                detail="Answer text must be non-empty.",
            ),
        ]
        final_keep = all(check.passed for check in checks)
        return VerificationResult(
            checks=checks,
            final_keep=final_keep,
            reject_reason=None if final_keep else "basic_validation_failed",
        )


@dataclass(slots=True)
class OfflineVqaVerifier:
    """Offline verifier that reads generated VQA outputs from a folder."""

    model_client: ModelRouterWorkerClient
    answer_model_alias: str
    judge_model_alias: str
    answer_max_tokens: int = 512
    judge_max_tokens: int = 1024
    output_file_name: str = "verification_results.jsonl"
    summary_file_name: str = "verification_summary.json"

    def run(self, *, vqa_dir: str | Path) -> dict[str, Any]:
        vqa_path = Path(vqa_dir)
        questions_path = vqa_path / "questions.jsonl"
        samples_path = vqa_path / "samples.jsonl"
        output_path = vqa_path / self.output_file_name
        summary_path = vqa_path / self.summary_file_name

        if not questions_path.exists():
            raise FileNotFoundError(f"questions.jsonl does not exist: {questions_path}")
        if not samples_path.exists():
            raise FileNotFoundError(f"samples.jsonl does not exist: {samples_path}")

        question_records = self._load_jsonl(questions_path)
        sample_records = self._load_jsonl(samples_path)
        samples_by_id = {
            str(record.get("sample_id")): record
            for record in sample_records
            if record.get("sample_id") is not None
        }
        existing_records = self._load_jsonl(output_path) if output_path.exists() else []
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
            "gold_sanity_passed": 0,
            "model_answer_correct": 0,
            "output_path": str(output_path),
            "judge_model_alias": self.judge_model_alias,
            "answer_model_alias": self.answer_model_alias,
            "answer_max_tokens": self.answer_max_tokens,
            "judge_max_tokens": self.judge_max_tokens,
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
                fingerprint = self._question_fingerprint(
                    question_record=question_record,
                    sample_record=sample,
                )
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
                self._append_jsonl(handle, verification_record)
                summary["verified_total"] += 1
                trajectory_type = str(verification_record.get("trajectory_type") or "unclassified")
                if trajectory_type not in summary["trajectory_type_counts"]:
                    trajectory_type = "unclassified"
                summary["trajectory_type_counts"][trajectory_type] += 1
                if (verification_record.get("checks") or {}).get("gold_answer_sanity", {}).get("passed"):
                    summary["gold_sanity_passed"] += 1
                if (verification_record.get("checks") or {}).get("model_answer_judgment", {}).get("correct"):
                    summary["model_answer_correct"] += 1

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
        question = str(question_record.get("question") or "").strip()
        gold_answer = str(question_record.get("answer") or "").strip()
        hop_chain = list((sample_record or {}).get("hop_chain") or [])
        node_types = list(((sample_record or {}).get("path") or {}).get("node_types") or [])
        trajectory_type = self._classify_trajectory(node_types)

        gold_sanity = self._check_gold_answer_sanity(
            question=question,
            gold_answer=gold_answer,
            hop_chain=hop_chain,
            question_id=str(question_record.get("question_id") or question_index),
        )
        predicted_answer = self._answer_question(
            question=question,
            question_id=str(question_record.get("question_id") or question_index),
        )
        model_answer_judgment = self._judge_model_answer(
            question=question,
            gold_answer=gold_answer,
            predicted_answer=predicted_answer.get("answer") or "",
            question_id=str(question_record.get("question_id") or question_index),
        )

        reject_reasons: list[str] = []
        if not gold_sanity.get("passed", False):
            reject_reasons.append("gold_answer_sanity_failed")
        if not model_answer_judgment.get("correct", False):
            reject_reasons.append("model_answer_incorrect")

        return {
            "question_number": question_index,
            "question_id": question_record.get("question_id"),
            "sample_id": question_record.get("sample_id"),
            "path_id": question_record.get("path_id"),
            "status": question_record.get("status"),
            "question": question,
            "gold_answer": gold_answer,
            "node_types": node_types,
            "trajectory_type": trajectory_type,
            "hop_chain": hop_chain,
            "question_fingerprint": question_fingerprint,
            "verifier_config": self._verifier_config(),
            "checks": {
                "gold_answer_sanity": gold_sanity,
                "model_answer": predicted_answer,
                "model_answer_judgment": model_answer_judgment,
            },
            "final_keep": not reject_reasons,
            "reject_reasons": reject_reasons,
            "verified_at": _utc_now(),
        }

    def _check_gold_answer_sanity(
        self,
        *,
        question: str,
        gold_answer: str,
        hop_chain: list[dict[str, Any]],
        question_id: str,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "gold_answer": gold_answer,
            "hop_chain": hop_chain,
        }
        try:
            parsed = self.model_client.generate_json(
                ModelRequest(
                    model=self.judge_model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_VERIFY_GOLD_ANSWER_SANITY),
                        ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False, indent=2)),
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=self.judge_max_tokens,
                    metadata={"trace_label": f"verify_gold_sanity:{question_id}"},
                )
            )
        except Exception as exc:
            return {
                "passed": False,
                "confidence": 0.0,
                "issues": f"Verifier call failed: {exc}",
                "reason": "judge_model_error",
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        return {
            "passed": bool(parsed.get("passed")),
            "confidence": _safe_float(parsed.get("confidence")),
            "issues": str(parsed.get("issues") or ""),
            "reason": str(parsed.get("reason") or ""),
            "raw": parsed,
        }

    def _answer_question(
        self,
        *,
        question: str,
        question_id: str,
    ) -> dict[str, Any]:
        prompt = (
            "Answer the following VQA question as directly and concisely as possible. "
            "If you are uncertain, still provide your best short answer."
        )
        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=self.answer_model_alias,
                    messages=[
                        ModelMessage(role="system", content=prompt),
                        ModelMessage(role="user", content=question),
                    ],
                    max_tokens=self.answer_max_tokens,
                    metadata={"trace_label": f"verify_answer:{question_id}"},
                )
            )
            answer = (response.content or "").strip()
            return {
                "model_alias": self.answer_model_alias,
                "answer": answer,
                "raw_model": response.model,
                "usage": response.usage,
                "metadata": response.metadata,
            }
        except Exception as exc:
            return {
                "model_alias": self.answer_model_alias,
                "answer": "",
                "error": f"{exc.__class__.__name__}: {exc}",
            }

    def _judge_model_answer(
        self,
        *,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        question_id: str,
    ) -> dict[str, Any]:
        payload = {
            "question": question,
            "gold_answer": gold_answer,
            "predicted_answer": predicted_answer,
        }
        try:
            parsed = self.model_client.generate_json(
                ModelRequest(
                    model=self.judge_model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_VERIFY_MODEL_ANSWER_JUDGE),
                        ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False, indent=2)),
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=self.judge_max_tokens,
                    metadata={"trace_label": f"verify_judge_answer:{question_id}"},
                )
            )
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

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw_line in enumerate(handle, start=1):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_no} is not valid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_no} must contain one JSON object per line")
                records.append(record)
        return records

    @staticmethod
    def _append_jsonl(handle, record: dict[str, Any]) -> None:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        handle.write("\n")

    @staticmethod
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
            "question": question_record.get("question"),
            "answer": question_record.get("answer"),
            "hop_chain": list((sample_record or {}).get("hop_chain") or []),
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _verifier_config(self) -> dict[str, Any]:
        return {
            "answer_model_alias": self.answer_model_alias,
            "judge_model_alias": self.judge_model_alias,
            "answer_max_tokens": self.answer_max_tokens,
            "judge_max_tokens": self.judge_max_tokens,
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


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _format_ratio(count: int, total: int) -> str:
    if total <= 0:
        return "0.00%"
    return f"{(count / total) * 100:.2f}%"


def print_summary_report(summary: dict[str, Any]) -> None:
    total = int(summary.get("verified_total") or 0)
    gold_sanity_passed = int(summary.get("gold_sanity_passed") or 0)
    model_answer_correct = int(summary.get("model_answer_correct") or 0)
    answer_mismatch_count = max(0, total - gold_sanity_passed)
    trajectory_counts = dict(summary.get("trajectory_type_counts") or {})

    print("verification_report:")
    print(f"  total_questions: {total}")
    print(
        "  gold_answer_mismatch: "
        f"{answer_mismatch_count}/{total} ({_format_ratio(answer_mismatch_count, total)})"
    )
    print(
        "  model_answer_correct: "
        f"{model_answer_correct}/{total} ({_format_ratio(model_answer_correct, total)})"
    )
    print("  trajectory_types:")
    for key in ("text_only", "image_first", "image_end", "multi_image", "unclassified"):
        count = int(trajectory_counts.get(key) or 0)
        print(f"    {key}: {count}/{total} ({_format_ratio(count, total)})")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vqa-dir", required=True, help="Directory containing questions.jsonl and samples.jsonl.")
    parser.add_argument(
        "--answer-model-alias",
        required=True,
        help="Model alias in synthesis/models.json used to answer each question.",
    )
    parser.add_argument(
        "--judge-model-alias",
        required=True,
        help="Model alias in synthesis/models.json used for both sanity check and answer judgment.",
    )
    parser.add_argument(
        "--output-file",
        default="verification_results.jsonl",
        help="Output JSONL file name written inside --vqa-dir.",
    )
    parser.add_argument(
        "--summary-file",
        default="verification_summary.json",
        help="Output summary JSON file name written inside --vqa-dir.",
    )
    parser.add_argument("--answer-max-tokens", type=int, default=512)
    parser.add_argument("--judge-max-tokens", type=int, default=1024)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    verifier = OfflineVqaVerifier(
        model_client=LLM_WORKER,
        answer_model_alias=args.answer_model_alias,
        judge_model_alias=args.judge_model_alias,
        answer_max_tokens=args.answer_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        output_file_name=args.output_file,
        summary_file_name=args.summary_file,
    )
    summary = verifier.run(vqa_dir=args.vqa_dir)
    print_summary_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
