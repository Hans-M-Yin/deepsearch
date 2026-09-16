"""Statement-level partial-credit reward for DeepResearch trajectories.

The RL dataset stores a compact, question-facing chain of merged textual
statements in ``task.extra_info.partial_credit.statements``.  This scorer reads
that self-contained metadata, gives a text-only audit prompt to an LLM judge,
and returns a small auxiliary trajectory reward in ``reward`` mode.

Images are deliberately omitted from the judge trace.  A marker records that
the rollout policy saw an image, but the marker is never accepted as evidence.
This keeps the first version restricted to text-verifiable process credit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
import os
import re
import time
from typing import Any, Mapping, Sequence
import weakref


_WHITESPACE_RE = re.compile(r"\s+")
_IMAGE_PLACEHOLDER_RE = re.compile(r"<image>", flags=re.IGNORECASE)
_LIMITERS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()


_JUDGE_SYSTEM_PROMPT = """You audit whether an agent established useful
intermediate states while solving a research question.

You receive an original question, a text-only execution trace, and gold intermediate statements from question construction. For every gold statement, decide whether the agent established its material conclusion from tool output and subsequently used that conclusion in its investigation.

Rules:
1. A tool observation includes both search-result titles/snippets and read_url page text. Either is valid evidence when it explicitly supports the statement. The assistant's search query alone is not evidence. If the evidence is omitted from the trace, you may skip verification of that evidence; this applies only to the verification of evidence from omitted images.

2. Each gold statement includes graph anchors source_object and target_object. For a non-entry hop (a non-empty source_object), a positive verdict requires that the cited textual tool observations explicitly ground both anchors and the relation from source_object to target_object. For an entry hop (no source_object), the cited textual tool observations must ground target_object and its relevant relation to the original question. Exact string equality is not required: an explicit, unambiguous alias, abbreviation, or parenthetical disambiguation is valid. Do not require the agent to repeat the full gold statement verbatim, nor to repeat source_object in its later action; Rule 3 governs how target_object must be adopted. Do not grant credit merely because an anchor occurs elsewhere in the trace or because the assistant guessed it in a query. When the relevant evidence was obtained only from an image that is omitted from this trace, you may skip verification of that visual evidence under Rule 6, provided the visible reasoning/action identifies and adopts the resulting transition and no displayed textual evidence contradicts it.

3. The later ``adoption_turn`` must visibly use the established transition: its reasoning, query, or tool goal must use ``target_object`` (or an unambiguous alias) to continue the investigation. Do not infer adoption only from a plausible final outcome.

4. You need not require that the agent list every detail in the statement to judge it true. When both source_object and target_object match, and the relation between them obtained by the agent through search is, on the whole, semantically consistent with the statement, it may be accepted even if some details inherent to the statement are not mentioned, as long as there is no contradiction. Conversely, when the agent's search and reasoning contain an explicit statement that contradicts the statement, it should be rejected. That is, follow this logic: first confirm whether both source_object and target_object explicitly appear and are distinguished from other easily-confused objects; then judge whether the relation between the objects in the search results and the reasoning process conforms to the statement, and if there is no obvious contradictory wording, it may be judged as accepted.

5. Never substitute or merge merely "similar" entities across sources. A source about one person, organization, venue, event, date, or location cannot serve as evidence for another one unless a cited tool observation explicitly states their identity/alias or the relation connecting them. For example: a source about Josef Mengele is never evidence for Johann Kremer; a source saying that a trial had forty defendants does not establish that a named individual was among them unless that individual is explicitly named.

6. To compress the trace length, all images have been omitted. If the agent's reasoning derived the information for a given statement from an image, then you may skip verification of that evidence in this case, because the evidence has already been omitted from the context.

7. Output exactly one verdict for every supplied statement ID. Give a positive verdict only when the trace itself supports the conclusion; do not reward a bare guess or unsupported model knowledge.

8. A request may contain only a trajectory slice. In that case, reason solely over the displayed turns: do not use facts from omitted earlier or later turns. A positive verdict needs both evidence and a later adoption turn within the displayed slice.

Return only valid JSON matching this schema:
{
  "statements": [
    {
      "statement_id": "hop_1",
      "verdict": "established_and_used" | "not_established",
      "evidence_turns": [3] | [3, 7] | [],
      "adoption_turn": 4 | null,
      "reason": "optional concise explanation"
    }
  ]
}

Formatting is a scoring requirement, not optional prose:
- Emit every key shown above for every statement, including negatives.
- For ``established_and_used``, ``evidence_turns`` must contain one or two
  integer turn numbers copied from ``--- Turn N ---`` headings. Each listed
  turn must contain textual tool output. ``adoption_turn`` must be one later
  integer turn number, at most four turns after the last evidence turn.
- For ``not_established``, emit ``"evidence_turns": []`` and
  ``"adoption_turn": null``. Do not invent turn numbers.
- Do not use ``supporting_turn`` or prose such as "the preceding turn".
  If you cannot identify valid numeric turn references, choose
  ``not_established``.

Example positive entry:
{"statement_id":"hop_1","verdict":"established_and_used",
 "evidence_turns":[4],"adoption_turn":5,
 "reason":"Turn 4's search snippet establishes the entity, and turn 5 searches using it."}
"""

_JUDGE_REASON_INSTRUCTION = """

Diagnostic requirement: ``reason`` is mandatory for every statement. It must
be one concise sentence (at most 35 words) explaining why the trace did or did
not establish and use the statement. Do not copy long page content into it.
"""


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _normalize_mode(value: Any) -> str:
    mode = str(value or "off").strip().lower()
    if mode in {"on", "true", "1", "apply"}:
        mode = "reward"
    elif mode in {"observe", "metrics", "metrics_only"}:
        mode = "shadow"
    return mode if mode in {"off", "shadow", "reward"} else "off"


def _normalize_text(value: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(value or "").strip())


@dataclass(frozen=True)
class PartialCreditConfig:
    """Environment-configurable policy for statement-level partial credit.

    ``shadow`` executes the judge and exports metrics while leaving the reward
    untouched.  Only ``reward`` applies the bounded positive bonus.
    """

    mode: str = "off"
    judge_model: str = ""
    judge_timeout_s: float = 90.0
    judge_concurrency: int = 4
    judge_max_tokens: int = 1400
    min_confidence: float = 0.85
    statement_reward: float = 0.05
    trajectory_reward_cap: float = 0.15
    max_trace_chars: int = 80_000
    # A non-positive value retains the legacy single-request full-trace path.
    # Chunking keeps long, multi-hop traces within the judge's reliable local
    # attention range while retaining absolute turn IDs for credit assignment.
    trace_chunk_steps: int = 15
    max_statements: int = 8
    min_quote_chars: int = 12
    # Kept off in training: explanations are useful for one-off audits but
    # are non-scalar diagnostic text and add judge output tokens.
    include_judge_reasons: bool = False

    @classmethod
    def from_env(cls) -> "PartialCreditConfig":
        return cls(
            mode=_normalize_mode(os.environ.get("PARTIAL_CREDIT_MODE", "off")),
            judge_model=str(
                os.environ.get("PARTIAL_CREDIT_JUDGE_MODEL")
                or os.environ.get("JUDGE_MODEL")
                or ""
            ).strip(),
            judge_timeout_s=_env_float(
                "PARTIAL_CREDIT_JUDGE_TIMEOUT_S", 90.0, minimum=1.0
            ),
            judge_concurrency=_env_int(
                "PARTIAL_CREDIT_JUDGE_CONCURRENCY", 4, minimum=1
            ),
            judge_max_tokens=_env_int(
                "PARTIAL_CREDIT_JUDGE_MAX_TOKENS", 1400, minimum=128
            ),
            min_confidence=min(
                1.0,
                _env_float("PARTIAL_CREDIT_MIN_CONFIDENCE", 0.85, minimum=0.0),
            ),
            statement_reward=_env_float(
                "PARTIAL_CREDIT_STATEMENT_REWARD", 0.05, minimum=0.0
            ),
            trajectory_reward_cap=_env_float(
                "PARTIAL_CREDIT_TRAJECTORY_CAP", 0.15, minimum=0.0
            ),
            max_trace_chars=_env_int(
                "PARTIAL_CREDIT_MAX_TRACE_CHARS", 256_000, minimum=1_000
            ),
            trace_chunk_steps=_env_int(
                "PARTIAL_CREDIT_TRACE_CHUNK_STEPS", 15, minimum=0
            ),
            max_statements=_env_int(
                "PARTIAL_CREDIT_MAX_STATEMENTS", 8, minimum=1
            ),
            min_quote_chars=_env_int(
                "PARTIAL_CREDIT_MIN_QUOTE_CHARS", 12, minimum=1
            ),
        )


def _empty_result(config: PartialCreditConfig, *, reason: str) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "scored": False,
        "reason": reason,
        "source_statement_count": 0,
        "eligible_statement_count": 0,
        "entry_statement_skipped_count": 0,
        "terminal_overlap_skipped_count": 0,
        "duplicate_statement_skipped_count": 0,
        "trace_turn_count": 0,
        "trace_chars": 0,
        "trace_chunk_steps": config.trace_chunk_steps,
        "trace_chunk_count": 0,
        "trace_chunk_scored_count": 0,
        "trace_chunk_failed_count": 0,
        "chunk_valid_verdict_count": 0,
        "chunk_judge_positive_occurrence_count": 0,
        "chunk_judge_positive_statement_count": 0,
        "chunk_judge_duplicate_positive_statement_count": 0,
        "chunk_judge_duplicate_positive_occurrence_count": 0,
        "chunk_judge_duplicate_positive_statement_rate": 0.0,
        "chunk_valid_duplicate_statement_count": 0,
        "chunk_valid_duplicate_occurrence_count": 0,
        "chunk_valid_duplicate_statement_rate": 0.0,
        "judge_request_s": 0.0,
        "judge_queue_wait_s": 0.0,
        "judge_valid_verdict_count": 0,
        "judge_rejected_verdict_count": 0,
        "verified_statement_count": 0,
        "raw_bonus": 0.0,
        "bonus": 0.0,
        "events": [],
    }


def _task_extra_info(task: Mapping[str, Any]) -> Mapping[str, Any]:
    nested = task.get("extra_info")
    return nested if isinstance(nested, Mapping) else {}


def resolve_partial_credit_statements(task: Mapping[str, Any]) -> tuple[list[dict[str, str]], dict[str, int], str]:
    """Return distinct text-verifiable statements embedded in task metadata.

    The VQA-to-RL conversion retains every non-empty, distinct merged hop,
    including an input-image entry hop when it can later be established from
    a textual tool observation.  No rollout process reads ``samples.jsonl``:
    this keeps reward computation self-contained and avoids HDFS/FUSE reads
    in the rollout hot path.
    """

    diagnostics = {
        "source_statement_count": 0,
        "entry_statement_skipped_count": 0,
        "terminal_overlap_skipped_count": 0,
        "duplicate_statement_skipped_count": 0,
    }
    info = _task_extra_info(task)
    partial_credit = info.get("partial_credit")
    # Be tolerant of callers that pass the registry's wrapped record rather
    # than the already-unwrapped runtime task.
    if not isinstance(partial_credit, Mapping):
        nested_info = info.get("extra_info")
        if isinstance(nested_info, Mapping):
            partial_credit = nested_info.get("partial_credit")
    if not isinstance(partial_credit, Mapping):
        partial_credit = task.get("partial_credit")
    if not isinstance(partial_credit, Mapping):
        return [], diagnostics, "missing_partial_credit_metadata"

    raw_statements = partial_credit.get("statements")
    # Pandas/pyarrow returns a nested parquet list column as ``np.ndarray``
    # in some DatasetRegistry paths.  Avoid importing NumPy in rollout code;
    # the standard ``tolist`` protocol is sufficient for that representation.
    tolist = getattr(raw_statements, "tolist", None)
    if callable(tolist):
        raw_statements = tolist()
    if not isinstance(raw_statements, list):
        return [], diagnostics, "missing_partial_credit_statements"

    statements: list[dict[str, str]] = []
    seen_statements: set[str] = set()
    for position, item in enumerate(raw_statements):
        if not isinstance(item, Mapping):
            continue
        statement = _normalize_text(item.get("statement"))
        if not statement:
            continue
        diagnostics["source_statement_count"] += 1
        dedupe_key = statement.casefold()
        if dedupe_key in seen_statements:
            diagnostics["duplicate_statement_skipped_count"] += 1
            continue
        seen_statements.add(dedupe_key)
        statement_id = str(item.get("statement_id") or "").strip()
        if not statement_id:
            statement_id = f"statement_{position}"
        source_object = _normalize_text(item.get("source_object"))
        # ``-`` is the source marker for an entry hop in raw compose payloads.
        # Treat it as no predecessor should an old converter have preserved it.
        if source_object == "-":
            source_object = ""
        target_object = _normalize_text(item.get("target_object"))
        statements.append(
            {
                "statement_id": statement_id,
                "statement": statement,
                "source_object": source_object,
                "target_object": target_object,
            }
        )
    if not statements:
        return [], diagnostics, "no_partial_credit_statements"
    return statements, diagnostics, "resolved_task_extra_info"


def _strip_tool_response_wrapper(text: Any) -> str:
    value = str(text or "")
    if "<tool_response>" in value and "</tool_response>" in value:
        value = value.split("<tool_response>", 1)[1].split("</tool_response>", 1)[0]
    return value.strip()


def _replace_image_placeholders(text: Any) -> str:
    value = str(text or "")
    return _IMAGE_PLACEHOLDER_RE.sub(
        "[VISUAL_OBSERVATION_OMITTED: the rollout policy received an image, but the judge did not. This marker is not textual evidence.]",
        value,
    )


def _step_action(step: Any) -> tuple[str, str]:
    action = getattr(step, "action", None)
    action_value = getattr(action, "action", {}) if action is not None else {}
    if not isinstance(action_value, Mapping):
        return "unknown", ""
    return str(action_value.get("type") or "unknown"), str(action_value.get("tool_call") or "")


def build_text_only_judge_trace(
    steps: Sequence[Any],
    *,
    question: str,
    turn_offset: int = 0,
) -> tuple[str, dict[int, dict[str, Any]], int]:
    """Build a text trace while withholding answer/image bytes.

    ``turn_offset`` preserves global one-based turn IDs when a long trajectory
    is judged in slices. It is zero for the legacy full-trace path.
    """

    question_text = _replace_image_placeholders(question)
    lines = ["[Original Question]", question_text or "(empty question)", "", "[Trajectory]"]
    turn_records: dict[int, dict[str, Any]] = {}

    for local_index, step in enumerate(steps, start=1):
        index = turn_offset + local_index
        action_type, tool_call = _step_action(step)
        observation = _replace_image_placeholders(
            _strip_tool_response_wrapper(getattr(step, "observation", ""))
        )
        response = _replace_image_placeholders(getattr(step, "model_response", ""))
        is_final = action_type == "final_answer"
        turn_records[index] = {
            "action_type": action_type,
            "is_tool_turn": action_type == "tool_call",
            "is_final": is_final,
            "observation": observation,
        }
        lines.append(f"--- Turn {index} ---")
        if is_final:
            # The answer is withheld to prevent hindsight leakage into an
            # earlier statement verdict.
            lines.append("Assistant terminated with a final answer [ANSWER_OMITTED].")
            continue
        if action_type == "tool_call":
            lines.append("Assistant action:")
            lines.append(response or tool_call or "(missing tool action text)")
            lines.append("Tool observation:")
            lines.append(observation or "(empty observation)")
        else:
            lines.append("Assistant non-tool action:")
            lines.append(response or "(empty action)")
    return "\n".join(lines), turn_records, len(steps)


def _build_judge_prompt(
    *,
    question: str,
    statements: Sequence[Mapping[str, str]],
    trace: str,
    slice_start_turn: int | None = None,
    slice_end_turn: int | None = None,
) -> str:
    slice_notice = ""
    if slice_start_turn is not None and slice_end_turn is not None:
        slice_notice = (
            "[Trajectory Slice]\n"
            f"Only absolute turns {slice_start_turn} through {slice_end_turn} are "
            "shown in this request. Do not rely on omitted turns. For a positive "
            "verdict, every evidence_turn and the later adoption_turn must be in "
            "this slice.\n\n"
        )
    return (
        "[Original Question]\n"
        f"{_replace_image_placeholders(question) or '(empty question)'}\n\n"
        "[Gold Intermediate Statements]\n"
        f"{json.dumps(list(statements), ensure_ascii=False)}\n\n"
        f"{slice_notice}"
        "[Text-only Trajectory]\n"
        f"{trace}\n"
    )


def _extract_response_text(data: Any) -> str:
    if not isinstance(data, Mapping):
        return ""
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, Mapping):
        return ""
    message = first.get("message")
    return str(message.get("content") or "") if isinstance(message, Mapping) else ""


def _parse_json_object(text: Any) -> dict[str, Any] | None:
    value = str(text or "").strip()
    if value.startswith("```"):
        value = re.sub(r"^```(?:json)?\s*|\s*```$", "", value, flags=re.IGNORECASE)
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    decoder = json.JSONDecoder()
    for index, char in enumerate(value):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(value[index:])
        except json.JSONDecodeError:
            continue
        return parsed if isinstance(parsed, dict) else None
    return None


def validate_judge_verdicts(
    payload: Mapping[str, Any] | None,
    *,
    statements: Sequence[Mapping[str, str]],
    turn_records: Mapping[int, Mapping[str, Any]],
    config: PartialCreditConfig,
) -> tuple[list[dict[str, Any]], int]:
    """Validate compact, tool-grounded partial-credit verdicts.

    The judge decides semantic support and whether the conclusion was used.
    This deterministic layer only prevents malformed turn references, image-only
    evidence, and duplicate statement credit. Legacy verdicts are accepted to
    keep old diagnostic fixtures readable.
    """

    statements_by_id = {
        str(item.get("statement_id") or ""): item for item in statements
    }
    expected_ids = set(statements_by_id)
    raw_items = payload.get("statements") if isinstance(payload, Mapping) else None
    if not isinstance(raw_items, list):
        return [], 0

    accepted: list[dict[str, Any]] = []
    rejected = 0
    seen_ids: set[str] = set()
    for item in raw_items:
        if not isinstance(item, Mapping):
            rejected += 1
            continue
        statement_id = str(item.get("statement_id") or "")
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict not in {"established_and_used", "verified_and_adopted"}:
            continue
        if not statement_id or statement_id not in expected_ids or statement_id in seen_ids:
            rejected += 1
            continue
        evidence_turns_raw = item.get("evidence_turns")
        if isinstance(evidence_turns_raw, list):
            if not 1 <= len(evidence_turns_raw) <= 2:
                rejected += 1
                continue
            try:
                evidence_turns = sorted({int(value) for value in evidence_turns_raw})
            except (TypeError, ValueError):
                rejected += 1
                continue
            if len(evidence_turns) != len(evidence_turns_raw):
                rejected += 1
                continue
        else:
            # Backward compatibility for saved diagnostic fixtures and
            # in-flight outputs generated with the earlier schema.
            try:
                evidence_turns = [int(item.get("supporting_turn"))]
            except (TypeError, ValueError):
                rejected += 1
                continue
        try:
            adoption_turn = int(item.get("adoption_turn"))
        except (TypeError, ValueError):
            rejected += 1
            continue
        adoption_record = turn_records.get(adoption_turn)
        if (
            adoption_record is None
            or adoption_turn <= evidence_turns[-1]
            # Preserve local credit assignment without requiring the exact
            # next turn. The adoption may be a tool action or final answer.
            or adoption_turn > evidence_turns[-1] + 4
            or any(
                (record := turn_records.get(turn)) is None
                or not bool(record.get("is_tool_turn"))
                or not _normalize_text(record.get("observation"))
                or "VISUAL_OBSERVATION_OMITTED" in str(record.get("observation"))
                for turn in evidence_turns
            )
        ):
            rejected += 1
            continue
        seen_ids.add(statement_id)
        accepted.append(
            {
                "statement_id": statement_id,
                "evidence_turns": evidence_turns,
                # Retained for existing metrics/tests that display a single
                # source turn; it is the latest evidence turn.
                "supporting_turn": evidence_turns[-1],
                "adoption_turn": adoption_turn,
            }
        )
    return accepted, rejected


def extract_judge_statement_assessments(
    payload: Mapping[str, Any] | None,
    *,
    statements: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    """Return one compact, diagnostic judge assessment per expected statement.

    This preserves only the judge's verdict, turn references, and short
    rationale—never the trace or page observation. It is enabled by the
    standalone pilot only, so production rollout metadata stays scalar-only.
    """

    expected_ids = [str(item.get("statement_id") or "") for item in statements]
    expected_id_set = {statement_id for statement_id in expected_ids if statement_id}
    raw_items = payload.get("statements") if isinstance(payload, Mapping) else None
    parsed: dict[str, dict[str, Any]] = {}
    if isinstance(raw_items, list):
        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            statement_id = str(item.get("statement_id") or "")
            if statement_id not in expected_id_set or statement_id in parsed:
                continue
            verdict = str(item.get("verdict") or "").strip().lower()
            if verdict not in {
                "established_and_used",
                "not_established",
                # Keep reports emitted by the old diagnostic schema readable.
                "verified_and_adopted",
                "not_verified",
            }:
                verdict = "invalid_verdict"
            reason = _normalize_text(item.get("reason"))[:500]
            evidence_turns = item.get("evidence_turns")
            if not isinstance(evidence_turns, list):
                # Make legacy output visible in the pilot without treating it
                # as valid. The deterministic validator remains authoritative.
                supporting_turn = item.get("supporting_turn")
                evidence_turns = (
                    [supporting_turn] if supporting_turn is not None else None
                )
            elif len(evidence_turns) > 4:
                evidence_turns = evidence_turns[:4] + ["…"]
            parsed[statement_id] = {
                "statement_id": statement_id,
                "verdict": verdict,
                "evidence_turns": evidence_turns,
                "adoption_turn": item.get("adoption_turn"),
                "reason": reason or "Judge returned no diagnostic reason.",
            }

    return [
        parsed.get(
            statement_id,
            {
                "statement_id": statement_id,
                "verdict": "missing_assessment",
                "evidence_turns": None,
                "adoption_turn": None,
                "reason": "Judge omitted this statement from its response.",
            },
        )
        for statement_id in expected_ids
        if statement_id
    ]


def _get_limiter(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    existing = _LIMITERS.get(loop)
    if existing is None or existing[0] != limit:
        limiter = asyncio.Semaphore(limit)
        _LIMITERS[loop] = (limit, limiter)
        return limiter
    return existing[1]


async def _request_judge(
    prompt: str, config: PartialCreditConfig
) -> tuple[str, float, float]:
    """Call the existing optional OpenAI-compatible judge gateway."""

    from vision_deepresearch_async_workflow.utils.api_gateway_client import (
        api_gateway_chat,
        is_api_gateway_configured,
    )

    if not is_api_gateway_configured():
        raise RuntimeError("judge_api_not_configured")
    queued_at = time.perf_counter()
    async with _get_limiter(config.judge_concurrency):
        started_at = time.perf_counter()
        data = await asyncio.to_thread(
            api_gateway_chat,
            messages=[
                {
                    "role": "system",
                    "content": (
                        _JUDGE_SYSTEM_PROMPT
                        + (_JUDGE_REASON_INSTRUCTION if config.include_judge_reasons else "")
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            model_marker=config.judge_model or None,
            max_tokens=config.judge_max_tokens,
            timeout=int(config.judge_timeout_s),
            temperature=0.0,
        )
        elapsed_s = time.perf_counter() - started_at
    return _extract_response_text(data), elapsed_s, started_at - queued_at


_POSITIVE_VERDICTS = frozenset({"established_and_used", "verified_and_adopted"})


def _judge_positive_statement_ids(
    payload: Mapping[str, Any] | None,
    *,
    statements: Sequence[Mapping[str, str]],
) -> set[str]:
    """Return the judge's semantic positives before deterministic validation."""

    raw_items = payload.get("statements") if isinstance(payload, Mapping) else None
    if not isinstance(raw_items, list):
        return set()
    expected_ids = {str(item.get("statement_id") or "") for item in statements}
    return {
        statement_id
        for item in raw_items
        if isinstance(item, Mapping)
        and (statement_id := str(item.get("statement_id") or "")) in expected_ids
        and str(item.get("verdict") or "").strip().lower() in _POSITIVE_VERDICTS
    }


def _annotate_chunk_assessments(
    assessments: list[dict[str, Any]], events: Sequence[Mapping[str, Any]]
) -> None:
    """Attach deterministic validation status to pilot-only judge diagnostics."""

    accepted_ids = {str(event["statement_id"]) for event in events}
    for assessment in assessments:
        if assessment.get("verdict") in _POSITIVE_VERDICTS:
            assessment["validation_status"] = (
                "accepted"
                if assessment.get("statement_id") in accepted_ids
                else "rejected"
            )
        else:
            assessment["validation_status"] = "not_positive"


async def _score_trace_chunk(
    *,
    question: str,
    statements: Sequence[Mapping[str, Any]],
    steps: Sequence[Any],
    turn_offset: int,
    chunk_index: int,
    config: PartialCreditConfig,
) -> dict[str, Any]:
    """Judge one contiguous slice using global turn numbers.

    The returned object intentionally contains no prompt or page content. It
    can be retained by the standalone pilot to audit local attribution.
    """

    turn_start = turn_offset + 1
    turn_end = turn_offset + len(steps)
    chunk: dict[str, Any] = {
        "chunk_index": chunk_index,
        "turn_start": turn_start,
        "turn_end": turn_end,
        "scored": False,
        "reason": "unscored",
        "trace_chars": 0,
        "judge_request_s": 0.0,
        "judge_queue_wait_s": 0.0,
        "judge_valid_verdict_count": 0,
        "judge_rejected_verdict_count": 0,
        "judge_positive_statement_ids": [],
        "events": [],
    }
    trace, turn_records, _ = build_text_only_judge_trace(
        steps, question=question, turn_offset=turn_offset
    )
    chunk["trace_chars"] = len(trace)
    if not turn_records:
        chunk["reason"] = "no_trajectory_turns"
        return chunk
    if len(trace) > config.max_trace_chars:
        chunk["reason"] = "trace_too_long"
        return chunk

    prompt = _build_judge_prompt(
        question=question,
        statements=statements,
        trace=trace,
        slice_start_turn=turn_start,
        slice_end_turn=turn_end,
    )
    try:
        judge_text, request_s, queue_wait_s = await _request_judge(prompt, config)
    except Exception as exc:  # noqa: BLE001 - auxiliary reward must fail open.
        chunk.update(
            {
                "reason": "judge_request_failed",
                "judge_error_type": type(exc).__name__,
                "judge_error": str(exc),
            }
        )
        return chunk

    chunk["judge_request_s"] = request_s
    chunk["judge_queue_wait_s"] = queue_wait_s
    payload = _parse_json_object(judge_text)
    if payload is None:
        chunk["reason"] = "judge_invalid_json"
        return chunk

    events, rejected = validate_judge_verdicts(
        payload,
        statements=statements,
        turn_records=turn_records,
        config=config,
    )
    chunk.update(
        {
            "scored": True,
            "reason": "scored",
            "judge_valid_verdict_count": len(events),
            "judge_rejected_verdict_count": rejected,
            "judge_positive_statement_ids": sorted(
                _judge_positive_statement_ids(
                    payload, statements=statements
                )
            ),
            "events": events,
        }
    )
    if config.include_judge_reasons:
        assessments = extract_judge_statement_assessments(
            payload,
            statements=statements,
        )
        _annotate_chunk_assessments(assessments, events)
        chunk["judge_statement_assessments"] = assessments
    return chunk


def _chunk_slices(
    steps: Sequence[Any], chunk_steps: int
) -> list[tuple[int, Sequence[Any]]]:
    """Return ``(zero_based_offset, slice)`` pairs for judge requests."""

    if chunk_steps <= 0:
        return [(0, steps)]
    return [
        (offset, steps[offset : offset + chunk_steps])
        for offset in range(0, len(steps), chunk_steps)
    ]


def _duplicate_stats(
    chunks_by_statement: Mapping[str, Sequence[int]],
) -> tuple[int, int, float]:
    """Return duplicated statement count, excess occurrence count, and rate."""

    if not chunks_by_statement:
        return 0, 0, 0.0
    duplicate_counts = [
        len(set(chunk_indices))
        for chunk_indices in chunks_by_statement.values()
        if len(set(chunk_indices)) > 1
    ]
    duplicate_statement_count = len(duplicate_counts)
    duplicate_occurrence_count = sum(count - 1 for count in duplicate_counts)
    return (
        duplicate_statement_count,
        duplicate_occurrence_count,
        duplicate_statement_count / len(chunks_by_statement),
    )


async def score_statement_partial_credit(
    *,
    task: Mapping[str, Any],
    steps: Sequence[Any],
    config: PartialCreditConfig | None = None,
) -> dict[str, Any]:
    """Judge one completed trajectory and return a fail-open reward record."""

    config = config or PartialCreditConfig.from_env()
    result = _empty_result(config, reason="mode_off")
    if config.mode == "off":
        return result

    statements, diagnostics, statement_reason = resolve_partial_credit_statements(task)
    result.update(diagnostics)
    if not statements:
        result["reason"] = statement_reason
        return result
    if len(statements) > config.max_statements:
        statements = statements[: config.max_statements]
        result["statement_limit_hit"] = True
    result["eligible_statement_count"] = len(statements)

    question = str(task.get("question") or task.get("query") or "")
    trace, turn_records, turn_count = build_text_only_judge_trace(steps, question=question)
    result["trace_turn_count"] = turn_count
    result["trace_chars"] = len(trace)
    if not turn_records:
        result["reason"] = "no_trajectory_turns"
        return result

    slices = _chunk_slices(steps, config.trace_chunk_steps)
    chunk_results = await asyncio.gather(
        *[
            _score_trace_chunk(
                question=question,
                statements=statements,
                steps=chunk_steps,
                turn_offset=offset,
                chunk_index=index,
                config=config,
            )
            for index, (offset, chunk_steps) in enumerate(slices)
        ]
    )
    result["trace_chunk_count"] = len(chunk_results)
    result["trace_chunk_scored_count"] = sum(
        bool(chunk.get("scored")) for chunk in chunk_results
    )
    result["trace_chunk_failed_count"] = len(chunk_results) - int(
        result["trace_chunk_scored_count"]
    )
    result["judge_request_s"] = sum(
        float(chunk.get("judge_request_s", 0.0)) for chunk in chunk_results
    )
    result["judge_queue_wait_s"] = sum(
        float(chunk.get("judge_queue_wait_s", 0.0)) for chunk in chunk_results
    )
    if not result["trace_chunk_scored_count"]:
        reasons = {str(chunk.get("reason") or "") for chunk in chunk_results}
        result["reason"] = reasons.pop() if len(reasons) == 1 else "all_chunks_unscored"
        for chunk in chunk_results:
            if chunk.get("judge_error_type"):
                result["judge_error_type"] = chunk["judge_error_type"]
                result["judge_error"] = chunk.get("judge_error", "")
                break
        if config.include_judge_reasons:
            result["chunk_results"] = chunk_results
        return result

    positive_chunks_by_statement: dict[str, list[int]] = {}
    valid_events_by_statement: dict[str, list[dict[str, Any]]] = {}
    flattened_assessments: list[dict[str, Any]] = []
    rejected = 0
    valid_event_occurrences = 0
    for chunk in chunk_results:
        chunk_index = int(chunk["chunk_index"])
        rejected += int(chunk.get("judge_rejected_verdict_count", 0))
        for statement_id in chunk.get("judge_positive_statement_ids", []):
            positive_chunks_by_statement.setdefault(str(statement_id), []).append(
                chunk_index
            )
        for event in chunk.get("events", []):
            if not isinstance(event, Mapping):
                continue
            valid_event_occurrences += 1
            annotated_event = dict(event)
            annotated_event.update(
                {
                    "chunk_index": chunk_index,
                    "chunk_turn_start": int(chunk["turn_start"]),
                    "chunk_turn_end": int(chunk["turn_end"]),
                }
            )
            valid_events_by_statement.setdefault(
                str(annotated_event["statement_id"]), []
            ).append(annotated_event)
        if config.include_judge_reasons:
            for assessment in chunk.get("judge_statement_assessments", []):
                if not isinstance(assessment, Mapping):
                    continue
                annotated_assessment = dict(assessment)
                annotated_assessment.update(
                    {
                        "chunk_index": chunk_index,
                        "chunk_turn_start": int(chunk["turn_start"]),
                        "chunk_turn_end": int(chunk["turn_end"]),
                    }
                )
                flattened_assessments.append(annotated_assessment)

    events: list[dict[str, Any]] = []
    selected_event_keys: set[tuple[str, int, int]] = set()
    for statement in statements:
        statement_id = str(statement["statement_id"])
        occurrences = valid_events_by_statement.get(statement_id, [])
        if not occurrences:
            continue
        selected = min(
            occurrences,
            key=lambda event: (
                int(event["chunk_index"]),
                int(event["adoption_turn"]),
            ),
        )
        events.append(selected)
        selected_event_keys.add(
            (
                statement_id,
                int(selected["chunk_index"]),
                int(selected["adoption_turn"]),
            )
        )

    positive_duplicate_stats = _duplicate_stats(positive_chunks_by_statement)
    valid_chunk_map = {
        statement_id: [int(event["chunk_index"]) for event in occurrences]
        for statement_id, occurrences in valid_events_by_statement.items()
    }
    valid_duplicate_stats = _duplicate_stats(valid_chunk_map)
    raw_bonus = min(
        len(events) * config.statement_reward,
        config.trajectory_reward_cap,
    )
    result.update(
        {
            "scored": True,
            "reason": "scored",
            "judge_valid_verdict_count": len(events),
            "judge_rejected_verdict_count": rejected,
            "verified_statement_count": len(events),
            "chunk_valid_verdict_count": valid_event_occurrences,
            "chunk_judge_positive_occurrence_count": sum(
                len(chunk_indices)
                for chunk_indices in positive_chunks_by_statement.values()
            ),
            "chunk_judge_positive_statement_count": len(
                positive_chunks_by_statement
            ),
            "chunk_judge_duplicate_positive_statement_count": positive_duplicate_stats[0],
            "chunk_judge_duplicate_positive_occurrence_count": positive_duplicate_stats[1],
            "chunk_judge_duplicate_positive_statement_rate": positive_duplicate_stats[2],
            "chunk_valid_duplicate_statement_count": valid_duplicate_stats[0],
            "chunk_valid_duplicate_occurrence_count": valid_duplicate_stats[1],
            "chunk_valid_duplicate_statement_rate": valid_duplicate_stats[2],
            "raw_bonus": raw_bonus,
            "bonus": raw_bonus if config.mode == "reward" else 0.0,
            # The caller consumes this private list immediately and exports
            # scalar metrics only. It intentionally contains no page content.
            "events": events,
        }
    )
    if config.include_judge_reasons:
        for assessment in flattened_assessments:
            key = (
                str(assessment.get("statement_id") or ""),
                int(assessment["chunk_index"]),
                int(assessment.get("adoption_turn") or -1),
            )
            if assessment.get("validation_status") == "accepted":
                assessment["aggregation_status"] = (
                    "selected_for_reward"
                    if key in selected_event_keys
                    else "duplicate_not_rewarded"
                )
            else:
                assessment["aggregation_status"] = assessment.get(
                    "validation_status", "not_positive"
                )
        result["judge_statement_assessments"] = flattened_assessments
        result["chunk_results"] = chunk_results
    return result


def partial_credit_metrics(record: Any) -> dict[str, float]:
    """Export scalar diagnostics only; never put judge prompts/pages in W&B."""

    result = record if isinstance(record, Mapping) else {}

    def value(name: str) -> float:
        try:
            return max(0.0, float(result.get(name, 0.0)))
        except (TypeError, ValueError):
            return 0.0

    mode = _normalize_mode(result.get("mode"))
    reason = str(result.get("reason") or "")
    return {
        "partial_credit/enabled": float(mode in {"shadow", "reward"}),
        "partial_credit/observation_only": float(mode == "shadow"),
        "partial_credit/reward_enabled": float(mode == "reward"),
        "partial_credit/scored": float(bool(result.get("scored"))),
        "partial_credit/missing_metadata": float(
            reason == "missing_partial_credit_metadata"
        ),
        "partial_credit/missing_statements": float(
            reason == "missing_partial_credit_statements"
        ),
        "partial_credit/no_eligible_statements": float(
            reason == "no_partial_credit_statements"
        ),
        "partial_credit/trace_too_long": float(reason == "trace_too_long"),
        "partial_credit/judge_request_failed": float(reason == "judge_request_failed"),
        "partial_credit/judge_invalid_json": float(reason == "judge_invalid_json"),
        "partial_credit/source_statement_count": value("source_statement_count"),
        "partial_credit/eligible_statement_count": value("eligible_statement_count"),
        "partial_credit/entry_statement_skipped_count": value(
            "entry_statement_skipped_count"
        ),
        "partial_credit/terminal_overlap_skipped_count": value(
            "terminal_overlap_skipped_count"
        ),
        "partial_credit/duplicate_statement_skipped_count": value(
            "duplicate_statement_skipped_count"
        ),
        "partial_credit/trace_turn_count": value("trace_turn_count"),
        "partial_credit/trace_chars": value("trace_chars"),
        "partial_credit/trace_chunk_steps": value("trace_chunk_steps"),
        "partial_credit/trace_chunk_count": value("trace_chunk_count"),
        "partial_credit/trace_chunk_scored_count": value(
            "trace_chunk_scored_count"
        ),
        "partial_credit/trace_chunk_failed_count": value(
            "trace_chunk_failed_count"
        ),
        "partial_credit/chunk_valid_verdict_count": value(
            "chunk_valid_verdict_count"
        ),
        "partial_credit/chunk_judge_positive_occurrence_count": value(
            "chunk_judge_positive_occurrence_count"
        ),
        "partial_credit/chunk_judge_positive_statement_count": value(
            "chunk_judge_positive_statement_count"
        ),
        "partial_credit/chunk_judge_duplicate_positive_statement_count": value(
            "chunk_judge_duplicate_positive_statement_count"
        ),
        "partial_credit/chunk_judge_duplicate_positive_occurrence_count": value(
            "chunk_judge_duplicate_positive_occurrence_count"
        ),
        "partial_credit/chunk_judge_duplicate_positive_statement_rate": value(
            "chunk_judge_duplicate_positive_statement_rate"
        ),
        "partial_credit/chunk_valid_duplicate_statement_count": value(
            "chunk_valid_duplicate_statement_count"
        ),
        "partial_credit/chunk_valid_duplicate_occurrence_count": value(
            "chunk_valid_duplicate_occurrence_count"
        ),
        "partial_credit/chunk_valid_duplicate_statement_rate": value(
            "chunk_valid_duplicate_statement_rate"
        ),
        "partial_credit/judge_request_s": value("judge_request_s"),
        "partial_credit/judge_queue_wait_s": value("judge_queue_wait_s"),
        "partial_credit/judge_valid_verdict_count": value(
            "judge_valid_verdict_count"
        ),
        "partial_credit/judge_rejected_verdict_count": value(
            "judge_rejected_verdict_count"
        ),
        "partial_credit/verified_statement_count": value(
            "verified_statement_count"
        ),
        "partial_credit/raw_bonus": value("raw_bonus"),
        "partial_credit/applied_bonus": value("bonus"),
    }


__all__ = [
    "PartialCreditConfig",
    "build_text_only_judge_trace",
    "partial_credit_metrics",
    "resolve_partial_credit_statements",
    "score_statement_partial_credit",
    "validate_judge_verdicts",
]
