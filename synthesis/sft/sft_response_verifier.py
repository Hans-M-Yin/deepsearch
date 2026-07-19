"""Verifier for generated SFT tool-use responses.

This module verifies *generated trajectories*, not standalone question quality.  It
runs a serial set of subtasks and stops early when a severe problem is found:

1. Generation status check (non-LLM): reject incomplete/max-turn samples.
2. Answer correctness judge (LLM): compare final answer with the gold answer.
3. Tool-call legality check: placeholder for future specialized checks.
4. Oracle/evidence judge (LLM): sentence-level audit for unsupported claims,
   private guidance leakage, overconfident evidence use, and optional local repair.
5. Image-search error check: placeholder for future multimodal image checks.

Typical usage:

    python -m synthesis.sft.sft_response_verifier \
      --input-jsonl raw_trajectories.jsonl \
      --output-jsonl verified_trajectories.jsonl \
      --rejected-jsonl rejected_trajectories.jsonl
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import traceback
from typing import Any

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelWorkerClient
from synthesis.sft.pipeline import format_messages


_DEFAULT_VERIFY_MODEL = "multimodal_process"
_SFT_FIXED_REQUEST_ID = "3200636808"


PROMPT_ANSWER_CORRECTNESS = """
You are a strict but fair answer-equivalence judge for generated SFT trajectories.

Task:
Decide whether the model's final answer is semantically equivalent to the gold answer for the given question.
Judge only answer correctness. Do not judge reasoning quality, tool use, evidence support, or whether the question is good.

Guidelines:
- Accept paraphrases, aliases, equivalent names, harmless formatting differences, and answers that include the correct answer with minor extra wording.
- Reject answers that refer to a different entity, are incomplete in a way that changes meaning, omit required units/qualifiers, or say the answer cannot be determined.
- If the gold answer is a short entity/name/number and the model answer contains it clearly as the final answer, mark correct unless the surrounding text contradicts it.
- If equivalence is genuinely unclear, use verdict "manual_review" rather than forcing pass/reject.

Few-shot examples:

Example A:
Question: Which city was the author born in?
Gold answer: Oak Park, Illinois
Model final answer: The answer is Oak Park.
Output:
{"verdict":"pass","answer_correct":true,"equivalence":"semantic","confidence":0.92,"reason":"Oak Park is the key city in the gold answer; omitting the state does not change the answer for this question.","reject_reasons":[]}

Example B:
Question: How many goals did the player score in the 2016-17 season?
Gold answer: 11
Model final answer: 18 goals
Output:
{"verdict":"reject","answer_correct":false,"equivalence":"incorrect","confidence":0.99,"reason":"The numeric answer differs from the gold answer.","reject_reasons":["wrong_final_answer"]}

Example C:
Question: Which organization nominated the person?
Gold answer: Democratic Party
Model final answer: the Democrats
Output:
{"verdict":"pass","answer_correct":true,"equivalence":"semantic","confidence":0.95,"reason":"The Democrats is an equivalent informal name for the Democratic Party in this context.","reject_reasons":[]}

Return strict JSON exactly following this schema:
{
  "verdict": "pass|reject|manual_review",
  "answer_correct": true,
  "equivalence": "exact|semantic|partial|incorrect|uncertain",
  "confidence": 0.0,
  "reason": "",
  "reject_reasons": []
}
""".strip()


PROMPT_ORACLE_EVIDENCE_REPAIR = """
You are the core quality auditor for generated SFT tool-use trajectories.

You are NOT judging whether the original question is elegant or whether the reference chain is the only possible solution. You are judging whether the generated SFT response is trustworthy training data.

You will receive:
- the question and gold answer,
- the generated final answer,
- the generated SFT response transcript containing assistant messages and tool observations,
- private construction-time reference facts / hop chain information.

Important: the private reference facts are given to you only as verifier context. In the actual SFT generation prompt, similar private guidance may have been available for verification only. A good SFT response must not reveal that such guidance exists, must not cite it as evidence, and must not use it in visible reasoning without tool support.

Your task:
Inspect the assistant response carefully. Focus on four failure modes:

1. Oracle/private-guidance leakage
   Reject if the assistant explicitly mentions or relies on private guidance, reference facts, gold answer, verification guidance, provided facts, hidden hints, or construction-time facts as evidence.

2. Unsupported or hallucinated evidence
   Reject if a critical factual claim is not supported by the question or prior tool observations. Be especially strict when the response identifies a person/object/image, states that a webpage says something, or uses a search result as proof.

3. Overconfident weak evidence
   Reject or repair if the assistant states a conclusion as certain when the tool output only weakly suggests it. A small wording repair may be enough if the overall evidence is otherwise sufficient.

4. Wrong-reason-correct-answer
   Reject if the final answer is correct but the trajectory reaches it through a clearly wrong or unsupported key step, such as a wrong entity identification, a wrong image, a misread source, or a sudden jump to a private reference fact.

Tool semantics you must enforce:
- t2t_search returns search result metadata/snippets only. It does not prove full webpage contents.
- t2i_search returns image search result metadata only. It does not show the images to the assistant.
- i2i_search returns visually similar image metadata only. It does not prove identity by itself and does not let the assistant inspect returned images.
- read_url may read a webpage or download an image. Only after a successful read_url image result may the assistant make visual observations about that downloaded image.
- If a tool output has ok=false, the assistant must not use it as successful evidence.

Repair policy:
- If issues are local and can be fixed only by editing assistant text (softening, deleting unsupported claims, or adding a brief evidence-grounded clarification), return verdict "repair" and provide edited assistant messages.
- Do not invent new tool outputs, new URLs, new evidence, or new tool calls.
- Do not change tool messages.
- Do not change the final answer unless the existing final answer was already supported and the change is purely formatting.
- If a critical step needs new evidence or a new tool call, reject instead of repairing.

Few-shot examples:

Example A: image metadata overclaim
Transcript excerpt:
Assistant: I searched images and the first result shows the actor holding the trophy, so I can identify him as Alex Smith.
Tool t2i_search: {"ok": true, "results": [{"title": "Alex Smith trophy photo", "image_url": "..."}]}
Judgment:
{"verdict":"reject","severity":"major","unsupported_claims":[{"assistant_step":1,"claim":"the first result shows the actor holding the trophy","support_status":"unsupported","severity":"major","suggested_fix":"reject"}],"reason":"t2i_search only returned metadata; no image was read, so the assistant cannot claim visual content."}

Example B: private guidance leakage
Transcript excerpt:
Assistant: The provided reference facts say that the intermediate entity is the Apollo program, so I will search for its launch failures.
Judgment:
{"verdict":"reject","severity":"critical","oracle_leakage":{"present":true,"evidence":["provided reference facts say"]},"reason":"The response explicitly reveals and relies on private reference facts."}

Example C: repairable overconfidence
Transcript excerpt:
Assistant: The search results prove that the person is Maria Chen.
Tool i2i_search: {"ok": true, "matches": [{"title": "Maria Chen at the ceremony", "link": "..."}]}
Judgment:
{"verdict":"repair","severity":"minor","unsupported_claims":[{"assistant_step":1,"claim":"prove that the person is Maria Chen","support_status":"weak","severity":"minor","suggested_fix":"soften"}],"repair":{"needed":true,"repaired_assistant_messages":[{"message_index":1,"content":"The reverse-image results include a result titled \"Maria Chen at the ceremony\", which suggests the person may be Maria Chen. I should verify this with an additional source before treating it as certain."}]}}

Example D: good evidence use
Transcript excerpt:
Assistant: The search result snippet suggests this may be the relevant source, so I will read it.
Tool t2t_search: {"ok": true, "results": [...]}
Assistant: The page states that the player scored 11 goals in 2016-17, so the answer is 11.
Tool read_url: {"ok": true, "content": "... scored 11 goals in the 2016-17 season ..."}
Judgment:
{"verdict":"pass","severity":"none","reason":"The key answer is supported by read_url content and there is no private guidance leakage."}

Return strict JSON exactly following this schema:
{
  "verdict": "pass|repair|reject|manual_review",
  "severity": "none|minor|major|critical",
  "oracle_leakage": {
    "present": false,
    "evidence": []
  },
  "unsupported_claims": [
    {
      "assistant_step": 0,
      "claim": "",
      "support_status": "unsupported|weak|contradicted",
      "severity": "minor|major|critical",
      "suggested_fix": "delete|soften|add_supporting_explanation|reject"
    }
  ],
  "wrong_reason_correct_answer": {
    "present": false,
    "evidence": [],
    "reason": ""
  },
  "repair": {
    "needed": false,
    "repair_scope": "assistant_text_only",
    "repaired_assistant_messages": [
      {
        "message_index": 0,
        "content": ""
      }
    ],
    "repair_notes": ""
  },
  "should_reject": false,
  "reject_reasons": [],
  "warnings": [],
  "reason": ""
}
""".strip()


@dataclass(slots=True)
class SftResponseVerifierConfig:
    default_model_alias: str = _DEFAULT_VERIFY_MODEL
    answer_model_alias: str | None = None
    oracle_model_alias: str | None = None
    answer_max_tokens: int = 1024
    oracle_max_tokens: int = 4096
    temperature: float | None = None
    max_transcript_chars: int = 45000
    max_reference_chars: int = 12000


class SftResponseVerifier:
    """Serial verifier for one generated SFT trajectory record."""

    def __init__(
        self,
        *,
        model_client: ModelWorkerClient | None = None,
        config: SftResponseVerifierConfig | None = None,
    ) -> None:
        self.model_client = model_client or LLM_WORKER
        self.config = config or SftResponseVerifierConfig()

    def verify_record(self, record: dict[str, Any]) -> dict[str, Any]:
        verification: dict[str, Any] = {
            "decision": "keep",
            "stopped_after_subtask": None,
            "reject_reasons": [],
            "warnings": [],
            "subtask_order": [
                "generation_status",
                "answer_correctness",
                "tool_call_legality_placeholder",
                "oracle_leakage_evidence_repair",
                "image_search_error_placeholder",
            ],
            "subtasks": {},
            "repaired": False,
        }

        generation = self._generation_status_check(record)
        verification["subtasks"]["generation_status"] = generation
        if generation.get("should_reject"):
            self._reject(verification, "generation_status", generation.get("reject_reasons") or ["generation_incomplete"])
            return verification

        answer = self._answer_correctness_judge(record)
        verification["subtasks"]["answer_correctness"] = answer
        if answer.get("should_reject"):
            self._reject(verification, "answer_correctness", answer.get("reject_reasons") or ["wrong_final_answer"])
            return verification
        if answer.get("verdict") == "manual_review":
            verification["warnings"].append("answer_correctness_manual_review")

        tool_placeholder = self._placeholder_subtask("tool_call_legality_placeholder")
        verification["subtasks"]["tool_call_legality_placeholder"] = tool_placeholder

        oracle = self._oracle_evidence_repair_judge(record)
        verification["subtasks"]["oracle_leakage_evidence_repair"] = oracle
        if oracle.get("should_reject") or oracle.get("verdict") == "reject":
            self._reject(verification, "oracle_leakage_evidence_repair", oracle.get("reject_reasons") or ["oracle_or_evidence_failure"])
            return verification
        if oracle.get("verdict") == "repair":
            repaired_messages = self._apply_assistant_repairs(record.get("raw_messages") or [], oracle)
            if repaired_messages is None:
                self._reject(verification, "oracle_leakage_evidence_repair", ["repair_application_failed"])
                return verification
            verification["decision"] = "repair"
            verification["repaired"] = True
            verification["repaired_raw_messages"] = repaired_messages
        elif oracle.get("verdict") == "manual_review":
            verification["decision"] = "manual_review"
            verification["warnings"].append("oracle_evidence_manual_review")

        image_placeholder = self._placeholder_subtask("image_search_error_placeholder")
        verification["subtasks"]["image_search_error_placeholder"] = image_placeholder
        return verification

    @staticmethod
    def _reject(verification: dict[str, Any], stopped_after: str, reasons: list[Any]) -> None:
        verification["decision"] = "reject"
        verification["stopped_after_subtask"] = stopped_after
        verification["reject_reasons"] = [str(item) for item in reasons if str(item or "").strip()]

    @staticmethod
    def _placeholder_subtask(name: str) -> dict[str, Any]:
        return {
            "subtask": name,
            "verdict": "skipped",
            "status": "skipped",
            "should_reject": False,
            "reason": "placeholder_not_implemented",
        }

    def _generation_status_check(self, record: dict[str, Any]) -> dict[str, Any]:
        summary = record.get("generation_summary") or {}
        raw_messages = record.get("raw_messages") or []
        extracted_answer = str(record.get("extracted_answer") or "").strip()
        reject_reasons: list[str] = []
        status = str(summary.get("generation_status") or "unknown")
        generation_complete = bool(summary.get("generation_complete"))
        if status in {"max_turns_reached", "parse_error_finalized"}:
            reject_reasons.append(status)
        if summary and not generation_complete:
            reject_reasons.append("generation_incomplete")
        for reason in summary.get("failure_reasons") or []:
            reason_text = str(reason or "").strip()
            if reason_text:
                reject_reasons.append(reason_text)
        if not raw_messages:
            reject_reasons.append("empty_raw_messages")
        if not extracted_answer:
            reject_reasons.append("empty_extracted_answer")
        reject_reasons = sorted(set(reject_reasons))
        return {
            "subtask": "generation_status",
            "verdict": "reject" if reject_reasons else "pass",
            "status": status,
            "generation_complete": generation_complete,
            "should_reject": bool(reject_reasons),
            "reject_reasons": reject_reasons,
            "summary": summary,
        }

    def _answer_correctness_judge(self, record: dict[str, Any]) -> dict[str, Any]:
        question_id = self._question_id(record)
        payload = {
            "question": record.get("question") or "",
            "gold_answer": record.get("gold_answer") or "",
            "model_extracted_answer": record.get("extracted_answer") or "",
            "final_assistant_message": self._last_assistant_text(record.get("raw_messages") or []),
        }
        parsed, raw_text = self._generate_json(
            model_alias=self.config.answer_model_alias or self.config.default_model_alias,
            system_prompt=PROMPT_ANSWER_CORRECTNESS,
            user_payload=payload,
            max_tokens=self.config.answer_max_tokens,
            trace_label=f"sft_answer_correctness:{question_id}",
        )
        verdict = str(parsed.get("verdict") or "manual_review").strip().lower()
        answer_correct = bool(parsed.get("answer_correct"))
        should_reject = verdict == "reject" or not answer_correct
        reject_reasons = list(parsed.get("reject_reasons") or [])
        if should_reject and not reject_reasons:
            reject_reasons = ["wrong_final_answer"]
        parsed.update(
            {
                "subtask": "answer_correctness",
                "model_alias": self.config.answer_model_alias or self.config.default_model_alias,
                "raw_model_output": raw_text,
                "should_reject": should_reject,
                "reject_reasons": [str(item) for item in reject_reasons],
            }
        )
        return parsed

    def _oracle_evidence_repair_judge(self, record: dict[str, Any]) -> dict[str, Any]:
        question_id = self._question_id(record)
        response_transcript = self._format_sft_response_transcript(record.get("raw_messages") or [])
        payload = {
            "question": record.get("question") or "",
            "gold_answer": record.get("gold_answer") or "",
            "model_extracted_answer": record.get("extracted_answer") or "",
            "sft_response_transcript": self._truncate(response_transcript, self.config.max_transcript_chars),
            "private_reference_context": self._truncate(
                json.dumps(self._reference_context(record), ensure_ascii=False, indent=2),
                self.config.max_reference_chars,
            ),
            "generation_summary": record.get("generation_summary") or {},
        }
        parsed, raw_text = self._generate_json(
            model_alias=self.config.oracle_model_alias or self.config.default_model_alias,
            system_prompt=PROMPT_ORACLE_EVIDENCE_REPAIR,
            user_payload=payload,
            max_tokens=self.config.oracle_max_tokens,
            trace_label=f"sft_oracle_evidence:{question_id}",
        )
        verdict = str(parsed.get("verdict") or "manual_review").strip().lower()
        severity = str(parsed.get("severity") or "none").strip().lower()
        oracle_leakage = parsed.get("oracle_leakage") or {}
        wrong_reason = parsed.get("wrong_reason_correct_answer") or {}
        unsupported_claims = parsed.get("unsupported_claims") or []
        reject_reasons = list(parsed.get("reject_reasons") or [])
        has_critical_unsupported = any(
            str(item.get("severity") or "").lower() in {"major", "critical"}
            and str(item.get("suggested_fix") or "").lower() == "reject"
            for item in unsupported_claims
            if isinstance(item, dict)
        )
        if bool((oracle_leakage or {}).get("present")):
            reject_reasons.append("oracle_leakage")
        if bool((wrong_reason or {}).get("present")):
            reject_reasons.append("wrong_reason_correct_answer")
        if has_critical_unsupported:
            reject_reasons.append("critical_unsupported_claim")
        if severity == "critical" and verdict != "repair":
            reject_reasons.append("critical_oracle_evidence_issue")
        should_reject = bool(parsed.get("should_reject")) or verdict == "reject" or bool(reject_reasons)
        parsed.update(
            {
                "subtask": "oracle_leakage_evidence_repair",
                "model_alias": self.config.oracle_model_alias or self.config.default_model_alias,
                "raw_model_output": raw_text,
                "should_reject": should_reject,
                "reject_reasons": sorted(set(str(item) for item in reject_reasons if str(item or "").strip())),
            }
        )
        return parsed

    def _generate_json(
        self,
        *,
        model_alias: str,
        system_prompt: str,
        user_payload: Any,
        max_tokens: int,
        trace_label: str,
    ) -> tuple[dict[str, Any], str]:
        response = self.model_client.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=system_prompt),
                    ModelMessage(
                        role="user",
                        content=json.dumps(user_payload, ensure_ascii=False, indent=2),
                    ),
                ],
                temperature=self.config.temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                metadata=self._worker_metadata(trace_label),
            )
        )
        raw_text = response.content or ""
        parsed = _extract_json_object(raw_text)
        if parsed is None:
            return {
                "verdict": "manual_review",
                "should_reject": False,
                "parse_error": "model_output_not_json",
                "reason": "Failed to parse judge output as JSON.",
            }, raw_text
        return parsed, raw_text

    @staticmethod
    def _worker_metadata(trace_label: str) -> dict[str, Any]:
        return {
            "trace_label": trace_label,
            "session_id": _SFT_FIXED_REQUEST_ID,
            "prompt_cache_key": _SFT_FIXED_REQUEST_ID,
            "user_id": _SFT_FIXED_REQUEST_ID,
            "x_tt_logid": _SFT_FIXED_REQUEST_ID,
        }

    @staticmethod
    def _question_id(record: dict[str, Any]) -> str:
        return str(record.get("question_id") or record.get("sample_id") or record.get("path_id") or "sft_response")

    @staticmethod
    def _last_assistant_text(messages: list[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "assistant":
                return _message_content_to_text(message.get("content")).strip()
        return ""

    @staticmethod
    def _format_sft_response_transcript(messages: list[dict[str, Any]]) -> str:
        lines: list[str] = []
        for index, message in enumerate(messages, start=1):
            role = str(message.get("role") or "")
            if role == "system":
                continue
            if role == "user":
                # The original user prompt may contain gold answers/private facts.
                # Verifier inputs provide the question/reference context separately.
                continue
            content = _message_content_to_text(message.get("content")).strip()
            if role == "tool":
                tool_name = str(message.get("name") or "tool")
                lines.append(f"[Message {index}] TOOL {tool_name}\n{content}")
            elif role == "assistant":
                lines.append(f"[Message {index}] ASSISTANT\n{content}")
            else:
                lines.append(f"[Message {index}] {role.upper()}\n{content}")
        return "\n\n".join(lines).strip()

    @staticmethod
    def _reference_context(record: dict[str, Any]) -> dict[str, Any]:
        source_metadata = record.get("source_metadata") or {}
        sample_record = source_metadata.get("sample_record") or {}
        question_record = source_metadata.get("question_record") or {}
        entry_hop = (
            sample_record.get("entry_hop")
            or sample_record.get("opening_package")
            or ((sample_record.get("metadata") or {}).get("entry_hop") if isinstance(sample_record.get("metadata"), dict) else None)
            or {}
        )
        question_hop_chain = sample_record.get("question_hop_chain") or []
        return {
            "raw_hop_chain": record.get("hop_chain") or [],
            "question_hop_chain": question_hop_chain,
            "opening_or_entry_hop_for_first_source": entry_hop,
            "target_ask": sample_record.get("target_ask") or {},
            "question_target_ask": sample_record.get("question_target_ask") or {},
            "question_terminal_bridge": sample_record.get("question_terminal_bridge") or {},
            "question_record_fields": {
                "question_id": question_record.get("question_id"),
                "sample_id": question_record.get("sample_id"),
                "path_id": question_record.get("path_id"),
            },
        }

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        if limit <= 0 or len(text) <= limit:
            return text
        head = text[: max(0, limit // 2)].rstrip()
        tail = text[-max(0, limit // 2) :].lstrip()
        return f"{head}\n\n...[truncated {len(text) - len(head) - len(tail)} chars]...\n\n{tail}"

    @staticmethod
    def _apply_assistant_repairs(messages: list[dict[str, Any]], oracle_result: dict[str, Any]) -> list[dict[str, Any]] | None:
        repair = oracle_result.get("repair") or {}
        patches = repair.get("repaired_assistant_messages") or []
        if not isinstance(patches, list) or not patches:
            return None
        updated = deepcopy(messages)
        for patch in patches:
            if not isinstance(patch, dict):
                return None
            try:
                message_index = int(patch.get("message_index"))
            except (TypeError, ValueError):
                return None
            content = patch.get("content")
            if not isinstance(content, str) or not content.strip():
                return None
            zero_index = message_index - 1
            if zero_index < 0 or zero_index >= len(updated):
                return None
            if updated[zero_index].get("role") != "assistant":
                return None
            updated[zero_index] = dict(updated[zero_index])
            updated[zero_index]["content"] = content
        return updated


def _message_content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, indent=2)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    fence_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", candidate, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        candidate = fence_match.group(1).strip()
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start >= 0 and end > start:
        try:
            parsed = json.loads(candidate[start : end + 1])
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if not isinstance(parsed, dict):
                raise ValueError(f"Expected JSON object on line {line_number}: {path}")
            records.append(parsed)
    return records


def _write_jsonl(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


def _verify_one(index: int, record: dict[str, Any], verifier: SftResponseVerifier) -> dict[str, Any]:
    verified = dict(record)
    verified["sft_response_verification"] = verifier.verify_record(record)
    verified["verification_index"] = index
    return verified


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-jsonl", required=True, help="Raw trajectories JSONL from synthesis.sft.debug_vqa_batch.")
    parser.add_argument("--output-jsonl", required=True, help="Records whose verification decision is keep or repair.")
    parser.add_argument("--rejected-jsonl", required=True, help="Records rejected or marked manual_review.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum records to verify. <=0 means all.")
    parser.add_argument("--offset", type=int, default=0, help="Start offset in input records.")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--model-alias", default=os.environ.get("SFT_RESPONSE_VERIFY_MODEL") or _DEFAULT_VERIFY_MODEL)
    parser.add_argument("--answer-model-alias", default=os.environ.get("SFT_RESPONSE_VERIFY_ANSWER_MODEL"))
    parser.add_argument("--oracle-model-alias", default=os.environ.get("SFT_RESPONSE_VERIFY_ORACLE_MODEL"))
    parser.add_argument("--answer-max-tokens", type=int, default=int(os.environ.get("SFT_RESPONSE_VERIFY_ANSWER_MAX_TOKENS", "1024")))
    parser.add_argument("--oracle-max-tokens", type=int, default=int(os.environ.get("SFT_RESPONSE_VERIFY_ORACLE_MAX_TOKENS", "4096")))
    parser.add_argument("--max-transcript-chars", type=int, default=int(os.environ.get("SFT_RESPONSE_VERIFY_MAX_TRANSCRIPT_CHARS", "45000")))
    parser.add_argument("--max-reference-chars", type=int, default=int(os.environ.get("SFT_RESPONSE_VERIFY_MAX_REFERENCE_CHARS", "12000")))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")

    records = _load_jsonl(Path(args.input_jsonl))
    selected = records[args.offset :]
    if args.limit and args.limit > 0:
        selected = selected[: args.limit]

    config = SftResponseVerifierConfig(
        default_model_alias=args.model_alias,
        answer_model_alias=args.answer_model_alias or args.model_alias,
        oracle_model_alias=args.oracle_model_alias or args.model_alias,
        answer_max_tokens=args.answer_max_tokens,
        oracle_max_tokens=args.oracle_max_tokens,
        max_transcript_chars=args.max_transcript_chars,
        max_reference_chars=args.max_reference_chars,
    )
    verifier = SftResponseVerifier(config=config)

    output_path = Path(args.output_jsonl)
    rejected_path = Path(args.rejected_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    summary: Counter[str] = Counter()
    reject_reasons: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as output_handle, rejected_path.open("w", encoding="utf-8") as rejected_handle:
        if args.workers == 1:
            iterator = ((_verify_one(index, record, verifier), None) for index, record in enumerate(selected, start=args.offset + 1))
            for verified, _ in iterator:
                verification = verified.get("sft_response_verification") or {}
                decision = str(verification.get("decision") or "manual_review")
                summary[decision] += 1
                for reason in verification.get("reject_reasons") or []:
                    reject_reasons[str(reason)] += 1
                _write_jsonl(output_handle if decision in {"keep", "repair"} else rejected_handle, verified)
        else:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(_verify_one, index, record, verifier): index
                    for index, record in enumerate(selected, start=args.offset + 1)
                }
                for future in as_completed(futures):
                    try:
                        verified = future.result()
                    except Exception as exc:  # noqa: BLE001
                        index = futures[future]
                        summary["verifier_exception"] += 1
                        error_record = {
                            "verification_index": index,
                            "sft_response_verification": {
                                "decision": "reject",
                                "reject_reasons": ["verifier_exception"],
                                "error_type": exc.__class__.__name__,
                                "error": str(exc),
                                "traceback": "".join(traceback.format_exception(exc)),
                            },
                        }
                        _write_jsonl(rejected_handle, error_record)
                        continue
                    verification = verified.get("sft_response_verification") or {}
                    decision = str(verification.get("decision") or "manual_review")
                    summary[decision] += 1
                    for reason in verification.get("reject_reasons") or []:
                        reject_reasons[str(reason)] += 1
                    _write_jsonl(output_handle if decision in {"keep", "repair"} else rejected_handle, verified)

    print("SFT Response Verification Summary")
    print(f"total: {len(selected)}")
    for key, count in summary.most_common():
        print(f"{key}: {count}")
    if reject_reasons:
        print("reject_reasons:")
        for key, count in reject_reasons.most_common():
            print(f"  {key}: {count}")
    print(f"output_jsonl: {output_path}")
    print(f"rejected_jsonl: {rejected_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
