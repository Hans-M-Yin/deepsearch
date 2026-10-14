"""Statement-level partial-credit reward for DeepResearch trajectories.

The RL dataset stores a compact, question-facing chain of merged textual
statements in ``task.extra_info.partial_credit.statements``.  This scorer reads
that self-contained metadata, gives a text-only audit prompt to an LLM judge,
and returns a small auxiliary trajectory reward in ``reward`` mode.

Images are deliberately omitted from the judge trace. A marker records that
the rollout policy saw an image; under the prompt's omitted-image rule, the
judge may credit a visible, subsequently adopted transition that it attributes
to that unavailable visual evidence.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import html
import json
import os
import re
import time
from typing import Any, Mapping, Sequence
import unicodedata
import weakref


_WHITESPACE_RE = re.compile(r"\s+")
_IMAGE_PLACEHOLDER_RE = re.compile(r"<image>", flags=re.IGNORECASE)
_LIMITERS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()
_JUDGE_EXECUTORS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, tuple[int, ThreadPoolExecutor]
] = weakref.WeakKeyDictionary()


_JUDGE_SYSTEM_PROMPT = """You audit whether an agent established useful
intermediate states while solving a research question.

You receive the question, the agent's trace, and gold intermediate statements. Judge each statement separately: did a tool result support the source-to-target conclusion, and did the agent then use that conclusion in a later tool action? The gold statements tell you what to check; they are NOT evidence that the agent found it. Do not fill gaps using your own knowledge or the gold answer.

For each statement_id, first read THAT statement's source, target, and relation. Compare them with the relation actually supported by the cited tool result. Keep this comparison in the reason. Do not replace the supplied statement with the agent's current subproblem, another statement, or the question's final goal. Rule 2 permits a direct identification of the SAME target in the SAME question context, not a different object or event. Judge each supplied ID exactly once; do not shift findings to the preceding or following ID.

Rules:
1. Search-result titles/snippets and read_url page text both count as evidence. A snippet is enough if it supports the required relation; opening the page is not mandatory. A search query, the assistant's unsupported claim, or two names appearing separately is not evidence of a relation. Find the actual supporting tool passage before giving a positive verdict. Rule 6 explains the limited exception for omitted images.

2. Check the core target and the relation that makes it relevant to the question. Optional required_relation and identity_constraints specify what must be proved; background_details need not be proved. Without those fields, use the supplied statement and question to distinguish the identifying relation from incidental context. Do not require the agent to follow the construction chain in order. A direct, evidenced identification in the SAME pictured scene/event can replace a detour through an organization or other construction-chain source. Explain that direct connection explicitly. Target-name occurrence alone or a fact about the target on an unrelated path is insufficient. Never waive a person/event/venue/date distinction needed to identify the object. Explicit contradictions remain disqualifying. Aliases, including a former name, can be used only when the tool evidence connects them.

3. In ``--- Turn N ---``, the assistant acts BEFORE receiving that turn's tool result. For tool evidence, inspect the displayed observations and prior memory in chronological order. Find the earliest point at which valid evidence sufficiently supports THIS statement, including any required image-to-object connection. Select one to five ``credit_turns`` that supplied that evidence, not the turns that later discuss or use it. A sufficient search snippet and sufficient read_url text have equal evidential standing: if a snippet already proves the relation, credit that search, not a later page repeating it. Do not move credit later merely because a page is longer, more detailed, or closer to adoption. If a later result is needed, explain the specific missing fact or unresolved ambiguity it adds. Exclude repeated confirmations, failed searches, and entity mentions without the required relation. When one observation is sufficient, choose the earliest sufficient one. Use multiple turns only when they supply different necessary pieces; removing any selected turn must leave a required piece unsupported by the remaining selected results. Among equally sufficient sets at the earliest point, choose the fewest turns, then the earliest turn numbers. For example, an image match and a separate page identifying that matched image may both be needed; a page naming an object alone does not establish its match to the input image. Five is a maximum, not a target. ``credit_turn`` is the latest selected turn. ``adoption_turn`` is the first later non-final tool action using the supported finding; this use check does not move credit to a later confirmation. Mentioning or testing an unresolved candidate is not adoption. A follow-up URL/resource ID can count without repeating the target name. Never infer earlier evidence from a later result. The retrospective original-image exception in Rule 6 remains separate, and corrected mistakes follow Rule 4.

Attribution example: turn 3's snippet explicitly states company X is listed on exchange Y; turn 4 reads a page repeating that fact; turn 6 searches using exchange Y. Select credit_turns=[3], credit_turn=3, adoption_turn=6, not [4] or [3,4]. If turn 3 only names X and turn 4 first establishes its listing on Y, select [4]. The same rule applies to evidence_only while waiting for a later adoption.

Adoption can be stated in the reasoning BEFORE a tool call, not only in its query or arguments. Count the first later non-final tool turn when the agent explicitly accepts the supported conclusion, treats that subproblem as resolved, and proceeds to the next question subproblem on that basis. The next query need not repeat the resolved object's name. For example, after tool evidence identifies a pictured chapel as being in Clumber Park, the agent says the location is now settled and proceeds to the question's painting search: this can count as adoption even if the painting query omits Clumber Park. Cite the explicit acceptance and explain the transition in the reason. Merely mentioning a candidate, repeating facts, taking an unrelated action, or searching a clue independently supplied by the question without connecting it to the accepted conclusion is NOT enough. Do not call a candidate unresolved when the displayed reasoning explicitly resolves it using valid evidence; do not invent acceptance when the agent still expresses doubt. Final-answer-only use remains excluded.

4. Require support for the main relation, not every descriptive detail. Missing an incidental detail is acceptable; missing the relation is not. "Nothing contradicts the gold statement" is not sufficient evidence. A different person, venue, event, or a date that changes which event is meant is not a minor detail. An explicit statement that contradicts the required relation cannot support a positive verdict. If the agent later corrects a mistake, judge the corrected evidence and its later use, not the earlier wrong action. A later failure on a different step does not erase an earlier supported and used finding.

5. Never substitute or merge merely "similar" entities across sources. A source about one person, organization, venue, event, date, or location cannot serve as evidence for another one unless a cited tool observation explicitly states their identity/alias or the relation connecting them. For example: a source about Josef Mengele is never evidence for Johann Kremer; a source saying that a trial had forty defendants does not establish that a named individual was among them unless that individual is explicitly named.

Examples of a wrong relation: evidence about Elizabeth of Poland and Louis does not establish that Saint Casimir's mother was Elisabeth of Austria. Evidence that GFS inspired HDFS does not establish that Hadoop inspired HMDFS. Shared topics, similar names, and a plausible next search do not make those relations interchangeable.

6. Images are replaced with omission markers. For tool-returned images, require that the image was available, the agent states the finding, and a later tool action uses it. Label pixels unverified; an image marker alone proves nothing. For an entry statement (empty source_object) identified from the original image, use evidence_source="input_image" ONLY if a later textual tool observation independently corroborates the image identification. A generic company page or a search result repeating the queried brand does NOT verify that brand in the original image. Set credit_turns to the ONE first tool action that used the visual identification, and adoption_turn equal to that action; evidence_turns and evidence_quotes point to the corroboration, which may arrive later. Set visual_claim_quote to an exact quote of the agent's visual identification in that credited action. This retrospective credit is an explicit exception, not invented prior tool evidence. If pixels were omitted and no independent corroboration is available, verdict="unverifiable" and no reward, not an assertion that the identification was wrong.

7. Give a positive verdict only if BOTH evidence and use pass the checks above (Rule 6 defines the retrospective input-image exception). For example, a snippet names Micah, but the agent dismisses Micah and searches for Maccabees: negative. The reasoning mentions Cronus but the next action pursues a different, unsupported identification: mentioning Cronus alone is not adoption. Conversely, a snippet links the correct source to the target and the next action investigates that target: positive, even if the final answer is wrong. Judge each statement once; repeated mentions do not earn extra credit.

8. A request may contain a slice plus [Prior Evidence Memory]. That memory contains exact tool excerpts and action text from earlier turns, NOT ground truth; recheck their meaning and relation before using them. Evidence/credit may come from displayed memory; adoption must be a non-final tool action in the CURRENT slice. There is no maximum evidence-to-adoption gap. Do not infer use solely from timing or a shared name. When evidence is supported but no adoption is visible yet, emit verdict="evidence_only" with evidence_turns, credit_turns, credit_turn and exact evidence_quotes; adoption_turn=null. This earns NO reward and only makes evidence available for a later slice. Do not fill omitted gaps with your own knowledge.

9. Do not add requirements from the next step. If the statement asks for a building's style, evidence identifying that style followed by a search using it can pass, even if the search fails to find the reformer mentioned in the question. Put that finding under the style statement, not the architect statement. Likewise, confirming an object's general description is not automatically confirmation that it is the object in the input image. If the agent still says that image-to-object match is unresolved, do not describe a search to test the candidate as an already confirmed identification.

10. Make negative reasons accurate too. Before saying an object was never mentioned, check the displayed reasoning and actions, including aliases. Mentioning can still lack evidence or adoption; state that limitation. For original-image claims without corroboration, say "pixels omitted; identification unverified". For missing slice evidence, say "not shown in this slice or memory", not "never found in the trajectory".

Example of useful direct progress: the question asks about a vessel in a particular painting/event. A tool identifies the foreground vessel as Intrepid and explicitly gives its former name Mastico; the next action investigates Mastico. This can establish the vessel-identification statement without separately proving a background departmental affiliation. It cannot pass if the observation is about a different painting, vessel, or event. Do not require a later action to repeat the preferred name when a supported alias is used.

Return only valid JSON matching this schema:
{
  "statements": [
    {
      "statement_id": "hop_1",
      "verdict": "established_and_used" | "evidence_only" | "unverifiable" | "not_established",
      "evidence_source": "tool" | "input_image",
      "evidence_quotes": [{"turn": 3, "quote": "exact supporting tool excerpt"}],
      "visual_claim_quote": "exact assistant visual claim for input_image, otherwise empty",
      "evidence_turns": [3] | [3, 7] | [],
      "credit_turn": 3 | null,
      "credit_turns": [3] | [2, 3] | [],
      "adoption_turn": 4 | null,
      "reason": "brief evidence and use explanation, or the specific missing check"
    }
  ]
}

Formatting is a scoring requirement, not optional prose:
- Emit every key shown above for every statement, including negatives.
- ``reason`` is mandatory, at most 80 words. Use the compact form
  "Gold: [this statement's relation]. Evidence: [relation actually supported,
  with the distinct needed contribution for EACH selected credit turn; if
  credit is later than an earlier relevant result, state what that result
  was missing, rather than merely calling the later result confirmation; include a short quote
  when available]. Use: [later action,
  or why it is missing]." For a mismatch, name the different objects or
  relation explicitly. If no supporting result exists, say so; do not invent
  a quote. For omitted images, label the visual claim as unverified pixels.
- For ``established_and_used``, ``evidence_turns`` must contain one to five
  integer turn numbers copied from ``--- Turn N ---`` headings or memory records. Each listed
  turn must contain a tool observation (including an omitted-image marker
  when Rule 6 applies). ``credit_turn`` must be exactly one
  member of ``evidence_turns``: select the action whose observation directly
  establishes the statement, not an action that merely uses it afterwards.
  ``adoption_turn`` must be the first later *non-final tool turn* visibly using
  the established conclusion; no fixed distance limit applies.
- ``credit_turns`` must contain one to five distinct integer turns from
  ``evidence_turns``. Choose only actions needed for the first establishment
  of this statement; do not list the whole search history or pad the list.
  Never output an ellipsis or a range. Every evidence turn must be no later
  than ``credit_turn`` and strictly earlier than ``adoption_turn``.
  ``credit_turn`` must equal the latest ``credit_turns`` entry.
  The adoption turn's own result
  cannot justify its earlier action. A final-answer turn is not adoption.
- These temporal/subset rules apply to evidence_source="tool". The separately
  corroborated input_image exception follows Rule 6 instead.
- evidence_quotes must give one exact supporting observation excerpt (12 to
  2000 characters preferred; use a short relevant passage, not a whole page; no ellipses or paraphrases)
  for EVERY evidence turn. Do not quote a query, assistant claim, or gold text
  as a tool observation. For tool-returned omitted images only, use the exact
  omission marker and explain the unverified visual claim in the reason.
- For evidence_only, keep the grounded evidence/credit fields and set adoption
  to null. Never label an unsupported candidate evidence_only.
- For ``not_established`` or ``unverifiable``, emit ``"evidence_turns": []``,
  ``"credit_turn": null``, ``"credit_turns": []``, and ``"adoption_turn": null``. Do not invent
  turn numbers.
- Do not use ``supporting_turn`` or prose such as "the preceding turn".
  If you cannot identify valid numeric turn references, choose
  ``not_established`` (or ``unverifiable`` for omitted input-image pixels).

Example positive entry:
{"statement_id":"hop_1","verdict":"established_and_used",
 "evidence_source":"tool","evidence_quotes":[{"turn":4,"quote":"Alpha was designed by Beta"}],"visual_claim_quote":"",
 "evidence_turns":[4],"credit_turn":4,"credit_turns":[4],"adoption_turn":5,
 "reason":"Gold: Alpha's architect is Beta. Evidence: turn 4 says 'Alpha was designed by Beta'. Use: turn 5 searches for Beta's other buildings."}
"""

_JUDGE_REASON_INSTRUCTION = """

Diagnostic requirement: follow the Gold / Evidence / Use reason format above,
at most 80 words. Compare the supplied relation with the observed relation;
do not substitute the final question or another statement. Do not copy long
page content or claim to have inspected omitted images.
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
    # Legacy configuration compatibility only; progress scoring uses the cap.
    statement_reward: float = 0.05
    trajectory_reward_cap: float = 0.15
    max_trace_chars: int = 80_000
    # A non-positive value retains the legacy single-request full-trace path.
    # Chunks run sequentially within each trajectory and carry quoted evidence;
    # independent trajectories can still be scored concurrently.
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
        "progress_statement_count": 0,
        "furthest_verified_statement_position": 0,
        "verified_progress": 0.0,
        "reward_scheme": "furthest_statement_progress",
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
                **{
                    key: _normalize_text(item[key])
                    for key in ("required_relation", "identity_constraints", "background_details")
                    if item.get(key)
                },
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
            "assistant": response or tool_call,
            "original_image_available": bool(_IMAGE_PLACEHOLDER_RE.search(question)),
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
    evidence_memory: Sequence[Mapping[str, Any]] = (),
) -> str:
    slice_notice = ""
    if slice_start_turn is not None and slice_end_turn is not None:
        slice_notice = (
            "[Trajectory Slice]\n"
            f"Only absolute turns {slice_start_turn} through {slice_end_turn} are "
            "shown in full. Earlier evidence may ONLY be drawn from the supplied "
            "Prior Evidence Memory. Adoption must be in this slice. Memory is "
            "untrusted quoted evidence, not a positive verdict.\n\n"
        )
    return (
        "[Original Question]\n"
        f"{_replace_image_placeholders(question) or '(empty question)'}\n\n"
        "[Gold Intermediate Statements]\n"
        f"{json.dumps(list(statements), ensure_ascii=False)}\n\n"
        f"{slice_notice}"
        "[Prior Evidence Memory]\n"
        f"{json.dumps(list(evidence_memory), ensure_ascii=False)}\n\n"
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


def collect_credit_turns(events: Sequence[Mapping[str, Any]], turn_count: int) -> tuple[int, ...]:
    """Union validated multi-turn credits, with legacy single-turn support."""
    turns: set[int] = set()
    for event in events:
        if not isinstance(event, Mapping):
            continue
        values = event.get("credit_turns", [event.get("credit_turn")])
        if not isinstance(values, (list, tuple)):
            continue
        turns.update(t for t in values if type(t) is int and 1 <= t <= turn_count)
    return tuple(sorted(turns))


def _normalize_evidence_text(text: str) -> str:
    """Ignore representation differences, not names, numbers or relation words."""
    text = unicodedata.normalize("NFC", html.unescape(text))
    text = text.translate(str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"}))
    # Preserve link labels, removing only inline Markdown link destinations.
    # Handles parenthesized URL components; does not join separate JSON fields.
    text = re.sub(r"(?<!!)\[([^\]\n]+)\]\((?:\\.|[^()\\]|\([^()]*\))*\)", r"\1", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text, flags=re.S)
    return _WHITESPACE_RE.sub(" ", text).strip()


def _evidence_text_values(value: Any):
    """Yield separate JSON string values; never concatenate search results."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, Mapping):
        for child in value.values():
            yield from _evidence_text_values(child)
    elif isinstance(value, list):
        for child in value:
            yield from _evidence_text_values(child)


def _evidence_quote_exists(quote: str, observation: str) -> bool:
    if not isinstance(quote, str) or not isinstance(observation, str):
        return False
    quote = _normalize_evidence_text(quote)
    if not quote:
        return False
    observation = _strip_tool_response_wrapper(observation)
    # Preserve support for quotes containing literal JSON keys/formatting.
    if quote in _normalize_evidence_text(observation):
        return True
    try:
        decoded = json.loads(observation)
    except (json.JSONDecodeError, RecursionError):
        return False
    return any(quote in _normalize_evidence_text(text)
               for text in _evidence_text_values(decoded))


def _grounded_quotes(item: Mapping[str, Any], records: Mapping[int, Mapping[str, Any]],
                     minimum: int) -> list[dict[str, Any]] | None:
    """Check provenance, not semantic truth. Every cited result needs a real quote."""
    quotes = item.get("evidence_quotes")
    turns = item.get("evidence_turns")
    if not isinstance(quotes, list) or not isinstance(turns, list) or not 1 <= len(turns) <= 5:
        return None
    if any(type(t) is not int for t in turns) or len(set(turns)) != len(turns):
        return None
    if len(quotes) != len(turns):
        return None
    checked = []
    for q in quotes:
        if not isinstance(q, Mapping) or type(q.get("turn")) is not int:
            return None
        turn = q["turn"]
        quote = _normalize_text(q.get("quote"))
        record = records.get(turn, {})
        owners = record.get("memory_statement_ids")
        if (turn not in turns or not record.get("is_tool_turn")
                or (owners is not None and item.get("statement_id") not in owners)
                # Long verbatim excerpts are inefficient, not fabricated. Keep a
                # bounded safety limit and verify the ENTIRE quote, never a prefix.
                or not minimum <= len(quote) <= 16000
                or not _evidence_quote_exists(quote, record.get("observation", ""))):
            return None
        checked.append({"turn": turn, "quote": quote})
    return checked if {q["turn"] for q in checked} == set(turns) else None


def validate_judge_verdicts(
    payload: Mapping[str, Any] | None,
    *,
    statements: Sequence[Mapping[str, str]],
    turn_records: Mapping[int, Mapping[str, Any]],
    config: PartialCreditConfig,
    rejection_reasons: dict[str, list[str]] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Validate compact, tool-grounded partial-credit verdicts.

    The judge decides semantic support and whether the conclusion was used.
    This deterministic layer only prevents malformed turn references, ambiguous
    credit locations, and duplicate statement credit. The judge owns semantic
    grounding, including the permitted omitted-image case, and alias resolution.
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
    def reject(statement_id: str, reason: str) -> None:
        nonlocal rejected
        rejected += 1
        if rejection_reasons is not None:
            rejection_reasons.setdefault(statement_id, []).append(reason)

    for item in raw_items:
        if not isinstance(item, Mapping):
            reject("", "invalid_statement_record")
            continue
        statement_id = str(item.get("statement_id") or "")
        verdict = str(item.get("verdict") or "").strip().lower()
        if verdict not in {"established_and_used", "verified_and_adopted"}:
            continue
        if not statement_id or statement_id not in expected_ids or statement_id in seen_ids:
            reject(statement_id, "unknown_or_duplicate_statement_id")
            continue
        if item.get("_adoption_outside_current_slice"):
            reject(statement_id, "adoption_or_corroboration_outside_current_slice")
            continue
        source = item.get("evidence_source", "tool")
        if not isinstance(source, str) or source not in {"tool", "input_image"}:
            reject(statement_id, "invalid_evidence_source")
            continue
        if "evidence_quotes" in item and _grounded_quotes(item, turn_records, config.min_quote_chars) is None:
            reject(statement_id, "ungrounded_evidence_quote")
            continue
        if source == "input_image":
            credit = item.get("credit_turn")
            evidence = item.get("evidence_turns")
            record = turn_records.get(credit, {}) if type(credit) is int else {}
            claim = _normalize_text(item.get("visual_claim_quote"))
            quotes = _grounded_quotes(item, turn_records, config.min_quote_chars)
            if (statements_by_id[statement_id].get("source_object")
                    or not record.get("original_image_available")
                    or not record.get("is_tool_turn")
                    or item.get("credit_turns") != [credit]
                    or type(item.get("adoption_turn")) is not int
                    or item.get("adoption_turn") != credit
                    or len(claim) < config.min_quote_chars
                    or claim not in _normalize_text(record.get("assistant"))
                    or not quotes or any(t < credit for t in evidence)
                    or any("VISUAL_OBSERVATION_OMITTED" in q["quote"] for q in quotes)):
                reject(statement_id, "input_image_missing_corroboration_or_action")
                continue
            seen_ids.add(statement_id)
            accepted.append({"statement_id": statement_id, "evidence_source": source,
                             "evidence_turns": evidence, "evidence_quotes": quotes,
                             "credit_turn": credit, "credit_turns": [credit],
                             "adoption_turn": credit, "visual_claim_quote": claim})
            continue
        evidence_turns_raw = item.get("evidence_turns")
        if isinstance(evidence_turns_raw, list):
            if not 1 <= len(evidence_turns_raw) <= 5:
                reject(statement_id, "evidence_turn_count_out_of_range")
                continue
            if "credit_turns" in item and any(type(t) is not int or t < 1 for t in evidence_turns_raw):
                reject(statement_id, "invalid_evidence_turn_number")
                continue
            try:
                evidence_turns = sorted({int(value) for value in evidence_turns_raw})
            except (TypeError, ValueError):
                reject(statement_id, "invalid_evidence_turn_number")
                continue
            if len(evidence_turns) != len(evidence_turns_raw):
                reject(statement_id, "duplicate_evidence_turns")
                continue
        else:
            reject(statement_id, "evidence_turns_not_list")
            continue
        try:
            credit_turn = int(item.get("credit_turn"))
        except (TypeError, ValueError):
            # New outputs must name the causal action explicitly. Falling
            # back to max(evidence_turns) recreates the turn-shift bug this
            # schema is meant to prevent.
            reject(statement_id, "invalid_credit_turn")
            continue
        if credit_turn not in evidence_turns:
            reject(statement_id, "credit_turn_not_in_evidence")
            continue
        # Older stored judge responses retain their original single-turn semantics.
        credit_turns_raw = item.get("credit_turns", [credit_turn])
        if (not isinstance(credit_turns_raw, list)
                or not 1 <= len(credit_turns_raw) <= 5
                or any(type(t) is not int or t < 1 for t in credit_turns_raw)):
            reject(statement_id, "invalid_credit_turns")
            continue
        credit_turns = sorted(set(credit_turns_raw))
        if len(credit_turns) != len(credit_turns_raw):
            reject(statement_id, "duplicate_credit_turns")
            continue
        if not set(credit_turns).issubset(evidence_turns) or max(credit_turns) != credit_turn:
            reject(statement_id, "credit_turns_evidence_mismatch")
            continue
        if "credit_turns" in item and max(evidence_turns) > credit_turn:
            reject(statement_id, "evidence_after_credit")
            continue
        try:
            adoption_turn = int(item.get("adoption_turn"))
        except (TypeError, ValueError):
            reject(statement_id, "invalid_adoption_turn")
            continue
        adoption_record = turn_records.get(adoption_turn)
        credit_record = turn_records.get(credit_turn)
        if (
            adoption_record is None
            or not bool(adoption_record.get("is_tool_turn"))
            or bool(adoption_record.get("is_final"))
            or adoption_turn <= credit_turn
            or credit_record is None
            or not bool(credit_record.get("is_tool_turn"))
            or not _normalize_text(credit_record.get("observation"))
            or any(
                (record := turn_records.get(turn)) is None
                or not bool(record.get("is_tool_turn"))
                or not _normalize_text(record.get("observation"))
                for turn in evidence_turns
            )
        ):
            failures = []
            if adoption_record is None:
                failures.append("adoption_turn_not_in_slice")
            elif bool(adoption_record.get("is_final")):
                failures.append("adoption_turn_is_final")
            elif not bool(adoption_record.get("is_tool_turn")):
                failures.append("adoption_turn_not_tool")
            if adoption_turn <= credit_turn:
                failures.append("adoption_not_after_credit")
            if credit_record is None or not bool(credit_record.get("is_tool_turn")) or not _normalize_text(credit_record.get("observation")):
                failures.append("credit_turn_missing_tool_observation")
            if any(not turn_records.get(t) or not bool(turn_records[t].get("is_tool_turn")) or not _normalize_text(turn_records[t].get("observation")) for t in evidence_turns):
                failures.append("evidence_turn_missing_tool_observation")
            reject(statement_id, ";".join(failures))
            continue
        seen_ids.add(statement_id)
        accepted.append(
            {
                "statement_id": statement_id,
                "evidence_turns": evidence_turns,
                # Singular field retains the completion anchor for existing
                # chunk aggregation. Multi-turn events explicitly name every
                # action eligible for shaping; evidence alone is not credit.
                "credit_turn": credit_turn,
                "adoption_turn": adoption_turn,
                **({"credit_turns": credit_turns} if "credit_turns" in item else {}),
                **({"evidence_source": source, "evidence_quotes": item["evidence_quotes"]}
                   if "evidence_quotes" in item else {}),
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
                "evidence_only",
                "unverifiable",
                # Keep reports emitted by the old diagnostic schema readable.
                "verified_and_adopted",
                "not_verified",
            }:
                verdict = "invalid_verdict"
            reason = str(item.get("reason") or "")
            evidence_turns = item.get("evidence_turns")
            parsed[statement_id] = {
                "statement_id": statement_id,
                "verdict": verdict,
                "evidence_turns": evidence_turns if isinstance(evidence_turns, list) else None,
                "credit_turn": item.get("credit_turn"),
                "credit_turns": item.get("credit_turns", [item.get("credit_turn")] if item.get("credit_turn") is not None else []),
                "adoption_turn": item.get("adoption_turn"),
                "reason": reason or "Judge returned no diagnostic reason.",
                "evidence_source": item.get("evidence_source", "tool"),
                "evidence_quotes": item.get("evidence_quotes", []),
                "visual_claim_quote": item.get("visual_claim_quote", ""),
            }

    return [
        parsed.get(
            statement_id,
            {
                "statement_id": statement_id,
                "verdict": "missing_assessment",
                "evidence_turns": None,
                "credit_turn": None,
                "credit_turns": [],
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


async def _run_judge_request(fn, request, concurrency: int):
    """Use a dedicated pool, not asyncio's small shared default executor.

    Threads are created lazily. The semaphore bounds submitted work to the
    same capacity, so unrelated tools/accuracy requests cannot exhaust it.
    """
    loop = asyncio.get_running_loop()
    existing = _JUDGE_EXECUTORS.get(loop)
    if existing is None or existing[0] != concurrency:
        if existing is not None:
            existing[1].shutdown(wait=False)
        executor = ThreadPoolExecutor(
            max_workers=concurrency, thread_name_prefix="partial-judge"
        )
        weakref.finalize(loop, executor.shutdown, wait=False)
        _JUDGE_EXECUTORS[loop] = (concurrency, executor)
    else:
        executor = existing[1]
    return await loop.run_in_executor(executor, fn, request)


class PartialCreditJudgeConfigError(ValueError):
    """Invalid local alias configuration must not silently become zero reward."""


def validate_partial_credit_judge_model(config: PartialCreditConfig) -> None:
    from synthesis.model_worker import LLM_WORKER

    if not config.judge_model or LLM_WORKER.get_model(config.judge_model) is None:
        raise PartialCreditJudgeConfigError(
            f"Partial-credit judge alias is not registered: {config.judge_model!r}. "
            "Check PARTIAL_CREDIT_JUDGE_MODEL and SYNTHESIS_MODEL_CONFIG."
        )


async def _request_judge(
    prompt: str, config: PartialCreditConfig
) -> tuple[str, float, float]:
    """Route a registered alias through the shared worker.

    Timeout and sampling budgets follow the worker's alias config;
    partial judging permits only one worker retry (two attempts total).
    the legacy judge_timeout_s is not a worker request timeout override.
    """
    from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest

    validate_partial_credit_judge_model(config)
    queued_at = time.perf_counter()
    async with _get_limiter(config.judge_concurrency):
        started_at = time.perf_counter()
        response = await _run_judge_request(
            LLM_WORKER.generate,
            ModelRequest(
                model=config.judge_model,
                messages=[
                    ModelMessage(role="system", content=(
                        _JUDGE_SYSTEM_PROMPT
                        + (_JUDGE_REASON_INSTRUCTION if config.include_judge_reasons else "")
                    )),
                    ModelMessage(role="user", content=prompt),
                ],
                metadata={"trace_label": "partial_credit_judge", "llm_retry_count": 1},
            ),
            config.judge_concurrency,
        )
        elapsed_s = time.perf_counter() - started_at
    return str(response.content or ""), elapsed_s, started_at - queued_at


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


def _collect_evidence_memory(payload, statements, turn_records, config):
    """Carry a bounded quoted packet per statement; never turn extraction into reward."""
    expected = {str(s["statement_id"]) for s in statements}
    memory = []
    items = payload.get("statements", []) if isinstance(payload, Mapping) else []
    if not isinstance(items, list):
        return memory
    for item in items:
        if not isinstance(item, Mapping) or item.get("statement_id") not in expected:
            continue
        if item.get("verdict") not in {"evidence_only", "established_and_used"} or item.get("evidence_source", "tool") != "tool":
            continue
        quotes = _grounded_quotes(item, turn_records, config.min_quote_chars)
        credits = item.get("credit_turns")
        anchor = item.get("credit_turn")
        if (not quotes or not isinstance(credits, list) or not 1 <= len(credits) <= 5
                or any(type(t) is not int for t in credits) or len(set(credits)) != len(credits)
                or type(anchor) is not int or anchor != max(credits)
                or not set(credits).issubset(item["evidence_turns"])
                or max(item["evidence_turns"]) > anchor):
            continue
        memory.append({
            "statement_id": item["statement_id"], "credit_turn": anchor,
            "credit_turns": credits, "evidence_quotes": quotes,
            "turns": [{"turn": q["turn"], "observation": q["quote"],
                       "assistant": turn_records[q["turn"]].get("assistant", ""),
                       "original_image_available": turn_records[q["turn"]].get("original_image_available", False)}
                      for q in quotes],
        })
    return memory


async def _score_trace_chunk(
    *,
    question: str,
    statements: Sequence[Mapping[str, Any]],
    steps: Sequence[Any],
    turn_offset: int,
    chunk_index: int,
    config: PartialCreditConfig,
    evidence_memory: Sequence[Mapping[str, Any]] = (),
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
    if not turn_records:
        chunk["reason"] = "no_trajectory_turns"
        return chunk
    # Only quoted observations actually sent to this request are valid references.
    for packet in evidence_memory:
        for old in packet.get("turns", []):
            turn = old["turn"]
            if turn >= turn_start:
                continue
            existing = turn_records.get(turn)
            if existing is None:
                turn_records[turn] = {**old, "is_tool_turn": True, "is_final": False,
                                      "memory_statement_ids": [packet["statement_id"]]}
            else:
                existing["observation"] += "\n" + old.get("observation", "")
                existing["memory_statement_ids"].append(packet["statement_id"])

    prompt = _build_judge_prompt(
        question=question,
        statements=statements,
        trace=trace,
        slice_start_turn=turn_start,
        slice_end_turn=turn_end,
        evidence_memory=evidence_memory,
    )
    # Include carried context in the limit; never silently truncate a quotation.
    chunk["trace_chars"] = len(prompt)
    chunk["prior_evidence_packet_count"] = len(evidence_memory)
    if len(prompt) > config.max_trace_chars:
        chunk["reason"] = "trace_too_long"
        return chunk
    try:
        judge_text, request_s, queue_wait_s = await _request_judge(prompt, config)
    except PartialCreditJudgeConfigError:
        raise
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

    chunk["_evidence_memory"] = _collect_evidence_memory(payload, statements, turn_records, config)
    chunk["evidence_memory_packet_count"] = len(chunk["_evidence_memory"])
    # Ordinary adoption must be in this slice. The input-image exception may
    # retroactively credit an earlier action, but needs NEW corroboration here.
    if isinstance(payload.get("statements"), list):
        for item in payload["statements"]:
            if not isinstance(item, dict) or item.get("verdict") not in _POSITIVE_VERDICTS:
                continue
            if item.get("evidence_source") == "input_image":
                evidence = item.get("evidence_turns")
                valid_here = isinstance(evidence, list) and any(
                    type(t) is int and turn_start <= t <= turn_end for t in evidence)
            else:
                a = item.get("adoption_turn")
                valid_here = type(a) is int and turn_start <= a <= turn_end
            if not valid_here:
                item["_adoption_outside_current_slice"] = True

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
        rejection_reasons: dict[str, list[str]] = {}
        validate_judge_verdicts(
            payload, statements=statements, turn_records=turn_records,
            config=config, rejection_reasons=rejection_reasons,
        )
        for assessment in assessments:
            assessment["validation_rejection_reasons"] = rejection_reasons.get(
                assessment["statement_id"], []
            ) if assessment.get("validation_status") == "rejected" else []
            if assessment.get("verdict") == "evidence_only":
                carried = any(p["statement_id"] == assessment["statement_id"] for p in chunk["_evidence_memory"])
                assessment["validation_status"] = "pending_evidence" if carried else "rejected_pending_evidence"
                assessment["validation_rejection_reasons"] = [] if carried else ["invalid_evidence_memory_packet"]
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


def statement_progress(events, statement_order):
    """Return furthest validated position and progress, not a prefix/count score.

    Use the complete resolved metadata order, before judge limits are applied.
    Unknown IDs cannot contribute; duplicate hits never accumulate reward.
    """
    positions = {str(s["statement_id"]): i for i, s in enumerate(statement_order, 1)}
    furthest = max(
        (positions.get(str(event.get("statement_id", "")), 0) for event in events),
        default=0,
    )
    return furthest, furthest / len(statement_order) if statement_order else 0.0


async def score_statement_partial_credit(
    *,
    task: Mapping[str, Any],
    steps: Sequence[Any],
    config: PartialCreditConfig | None = None,
    is_correct: bool = False,
) -> dict[str, Any]:
    """Audit progress; apply bonus only in reward mode to incorrect outcomes.

    Correct outcomes retain raw progress and events for diagnostics, but
    never receive an applied bonus. Training callers must pass correctness.
    """

    config = config or PartialCreditConfig.from_env()
    result = _empty_result(config, reason="mode_off")
    if config.mode == "off":
        return result

    statements, diagnostics, statement_reason = resolve_partial_credit_statements(task)
    result.update(diagnostics)
    if not statements:
        result["reason"] = statement_reason
        return result
    statement_order = list(statements)
    result["progress_statement_count"] = len(statement_order)
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
    chunk_results = []
    memory_by_statement: dict[str, dict[str, Any]] = {}
    entry_action_memory = []
    if _IMAGE_PLACEHOLDER_RE.search(question):
        first_tool = next(((t, r) for t, r in turn_records.items() if r["is_tool_turn"]), None)
        if first_tool:
            t, record = first_tool
            for statement in statements:
                if not statement.get("source_object"):
                    entry_action_memory.append({
                        "statement_id": statement["statement_id"],
                        "note": "Original-image first action only; NOT corroborating evidence.",
                        "turns": [{"turn": t, "assistant": record["assistant"],
                                   "observation": "", "original_image_available": True}],
                    })
    # Sequential within a trajectory so pending evidence can be used later.
    # Different trajectories still share the existing asynchronous limiter.
    for index, (offset, chunk_steps) in enumerate(slices):
        chunk = await _score_trace_chunk(
                question=question,
                statements=statements,
                steps=chunk_steps,
                turn_offset=offset,
                chunk_index=index,
                config=config,
                evidence_memory=(entry_action_memory if offset else []) + list(memory_by_statement.values()),
            )
        for packet in chunk.pop("_evidence_memory", []):
            memory_by_statement[packet["statement_id"]] = packet
        chunk_results.append(chunk)
        if chunk.get("reason") == "judge_request_failed":
            # The worker has exhausted this request's retry budget. Do not
            # keep incomplete progress or spend requests on later chunks.
            break
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
    request_failure = next(
        (chunk for chunk in chunk_results if chunk.get("reason") == "judge_request_failed"),
        None,
    )
    if request_failure is not None:
        result.update({
            "reason": "judge_request_failed",
            "scored": False,
            "raw_bonus": 0.0,
            "bonus": 0.0,
            "events": [],
            "judge_error_type": request_failure.get("judge_error_type"),
            "judge_error": request_failure.get("judge_error", ""),
        })
        if config.include_judge_reasons:
            result["chunk_results"] = chunk_results
        return result
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
                int(event["credit_turn"]),
            ),
        )
        events.append(selected)
        selected_event_keys.add(
            (
                statement_id,
                int(selected["chunk_index"]),
                int(selected["credit_turn"]),
            )
        )

    positive_duplicate_stats = _duplicate_stats(positive_chunks_by_statement)
    valid_chunk_map = {
        statement_id: [int(event["chunk_index"]) for event in occurrences]
        for statement_id, occurrences in valid_events_by_statement.items()
    }
    valid_duplicate_stats = _duplicate_stats(valid_chunk_map)
    furthest, progress = statement_progress(events, statement_order)
    raw_bonus = config.trajectory_reward_cap * progress
    result.update(
        {
            "scored": True,
            "reason": "scored",
            "judge_valid_verdict_count": len(events),
            "judge_rejected_verdict_count": rejected,
            "verified_statement_count": len(events),
            "furthest_verified_statement_position": furthest,
            "verified_progress": progress,
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
            "bonus": raw_bonus if config.mode == "reward" and not is_correct else 0.0,
            "reward_suppressed_correct": bool(is_correct),
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
                int(assessment.get("credit_turn") or -1),
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
        "partial_credit/progress_statement_count": value("progress_statement_count"),
        "partial_credit/furthest_verified_statement_position": value("furthest_verified_statement_position"),
        "partial_credit/verified_progress": value("verified_progress"),
        "partial_credit/applied_bonus": value("bonus"),
        "partial_credit/reward_suppressed_correct": value("reward_suppressed_correct"),
    }


__all__ = [
    "PartialCreditConfig",
    "build_text_only_judge_trace",
    "partial_credit_metrics",
    "resolve_partial_credit_statements",
    "score_statement_partial_credit",
    "validate_judge_verdicts",
]
