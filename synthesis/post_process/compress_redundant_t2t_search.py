#!/usr/bin/env python3
"""Detect and compress redundant text/image searches in ShareGPT trajectories.

The script deliberately makes a narrow edit:

* only ``t2t_search`` and ``t2i_search`` action blocks are eligible;
* a block includes the assistant action and its immediately following
  observation;
* a block is protected when a resource id from its observation is later used
  by ``read_url`` and no earlier retained search already supplied that id;
* tool calls, tool observations, and final answers are never rewritten;
* one detector LLM sees the complete numbered trajectory and returns only
  contiguous redundant text/image-search turn groups plus reasons;
* after removing a redundant block (or a consecutive run of them), an LLM
  rewrites only the thinking section of the next retained assistant turn.

For long-running jobs, the CLI also provides a tqdm progress bar and a
per-output resumable checkpoint. Completed records are appended to
``<state-dir>/completed_records.jsonl`` and flushed/fsynced immediately. The
final ``--output`` is materialized in the original JSON/JSONL format after the
selected range is complete; the checkpoint is the recovery source if the job
is interrupted before that final materialization.

No per-call temperature or completion-token limit is supplied.  Those values
are controlled exclusively by the registered model alias in ``model_worker``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from synthesis.model_worker import (
    LLM_WORKER,
    ModelMessage,
    ModelRequest,
    ResponsesModelRequest,
)

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    tqdm = None


TARGET_SAMPLE = "q_004035_sample_path_126bf6699d62f0dc_path_126bf6699d62f0dc_57b19ed7f8"
ELIGIBLE_SEARCH_ACTIONS = frozenset({"t2t_search", "t2i_search"})

DETECTOR_SYSTEM_PROMPT = r"""You inspect a complete numbered agent trajectory and identify redundant text/image-search spans.

The purpose of deletion is to remove dead search work from training data. This
teaches the model to stop repeating an unproductive text/image-search direction and
to move to a genuinely different search strategy, while preserving every
useful observation and every later action.

What counts as a redundant span:

1. It contains one or more consecutive numbered turns, and every turn in the
   span is an observed `t2t_search` or `t2i_search` action. The turns in a
   returned group must be consecutive numbers in the displayed trajectory.
2. The span searches the same or essentially the same sub-goal as an earlier
   search. Similar wording alone is not enough: inspect the query, the returned
   titles/snippets, and the reasoning after the observation.
3. Across the span, the searches add no actionable entity, usable fact,
   meaningful contradiction, or real change of search direction. A newly named
   candidate alone does not make a search useful. If a candidate is merely
   mentioned in the result list and then dismissed in a brief generic sentence
   (for example, "this is unrelated"), without being read, cited later, used
   by a later tool call, or contributing a unique fact/contradiction that
   changes the subsequent reasoning, the search is redundant. Repeated generic
   results, repeated irrelevant noise, and query wording tweaks that do not
   advance the investigation are redundant.
   For `t2i_search`, an image result that was never inspected with `read_url`
   is not visual evidence. Its title or snippet may still be useful if it
   concretely changes the later reasoning or next tool call; otherwise a
   repeated image search followed only by generic dismissal is redundant.
4. Do not delete a useful first search merely because a later search is
   similar. Do not delete the final pivot that changes direction or produces a
   new lead. Keep a search if its observation supplies a clue that later
   reasoning or a later `read_url` actually needs. Judge the whole search
   block: do not delete it if another result from the same observation is
   independently used later, even when one candidate in that observation is
   dismissed.
5. A result resource used later by `read_url` must not be orphaned. If a later
   read depends on a resource returned only by a proposed span, do not propose
   that span. If the same resource was already returned by an earlier retained
   text/image search, the repeated provider may be deleted.

What is never eligible:

- `i2i_search`, `read_url`, or any other non-`t2t_search`/`t2i_search` turn;
- an observation by itself;
- an answer turn or a turn whose removal changes the final answer;
- a search whose evidence materially affects later reasoning, a later tool
  call, or the final answer, even if its query shares many words with an
  earlier query. A candidate that is only named and then discarded without
  such downstream use is not protected merely because it is new.

Be conservative about useful evidence, but do delete clearly dead repeated
search spans. Each group should be the smallest contiguous span that is truly
redundant. Groups must not overlap. If there is no safe redundant span, return
an empty list.

Output ONLY valid JSON, with no markdown or extra text, in exactly this form:
{"redundant_groups":[{"turns":[12,13],"reason":"Both turns repeat the same search sub-goal and add no new clue; the next retained turn pivots away."}]}

`turns` must contain integer turn numbers exactly as displayed, in ascending
order, and each group must be contiguous. Use only turn numbers whose label
explicitly says `t2t_search` or `t2i_search` and is eligible for deletion. The reason must
briefly explain the repeated sub-goal and why the observations contain no new
useful information. Report every redundant span found anywhere in the complete
trajectory; do not return only a subset of the redundant spans."""

QUALITY_GATE_SYSTEM_PROMPT = r"""You are a strict local quality checker for a trajectory edit.

The trajectory editor has removed one or more redundant text/image-search blocks
and rewired the reasoning at the marked merge boundary. Inspect ONLY the marked
boundary sections. Do not re-solve the question, judge the final answer, or
re-evaluate the whole trajectory.

Reject only when the edit creates a real problem that would be harmful in SFT:

- the revised reasoning does not follow from the retained observation before it;
- an important premise needed by the next action was lost;
- the revised reasoning invents evidence or makes an unsupported bridge;
- the next action's purpose is no longer coherent with the revised reasoning;
- the edit creates a contradiction, broken reference, or malformed reasoning flow.

Do NOT reject merely because the revised reasoning is concise, stylistically
different, or does not repeat every detail. The question and the retained tool
call are context only; they are not targets for a new correctness judgment.

Output ONLY valid JSON in exactly this form:
{"decision":"accept","reason":"...","issues":[]}

Use `decision` = `reject` only for a concrete logical discontinuity or unsupported
bridge at one of the marked merge boundaries. If several boundaries are shown,
inspect all of them before deciding."""

EDITOR_SYSTEM_PROMPT = r"""You are a careful reasoning editor repairing a trajectory after one or more
redundant text/image-search blocks have been removed.

Return ONLY a single `<thinking>...</thinking>` section. Do not return a tool
call, an observation, an answer, JSON, or commentary about this editing task.

Rules:
1. Preserve the next retained assistant turn's tool call or final answer
   exactly; the caller will restore it programmatically.
2. Do not invent facts, candidates, evidence, or search results. Use only the
   question and the retained observations shown in the context.
3. Connect the last retained observation before the deleted search run to the
   next retained action. Explicitly acknowledge what the retained observation
   established, why continuing the same text-search path was not useful when
   that is supported by the evidence, and why the next action is a sensible
   way to make progress. Do not claim that a search failed if the observation
   does not show that.
4. Keep the original direction of the next action. Do not change its query,
   arguments, resource id, or final answer.
5. Keep the language natural and concise enough to avoid padding. The purpose
   is to remove dead search steps while preserving logical continuity.
"""

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.I | re.S)
_THINKING_RE = re.compile(r"<thinking>\s*(.*?)\s*</thinking>", re.I | re.S)
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", re.I | re.S)
_RESOURCE_ID_RE = re.compile(r"\b(?:page|image|img)_[A-Za-z0-9_-]+\b")
_DATA_IMAGE_RE = re.compile(r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", re.I)
_FSYNC_WARNING_EMITTED = False
_CHECKPOINT_TERMINAL_STATUSES = frozenset({"ok", "quality_rejected"})


@dataclass(frozen=True)
class ActionBlock:
    block_id: str
    turn_number: int
    assistant_index: int
    observation_index: int | None
    end_index: int
    action_name: str
    action: dict[str, Any]
    assistant_value: str
    observation_value: str


def _metadata(trace_label: str) -> dict[str, Any]:
    # Keep the same tracing/session convention as the other synthesis tools.
    return {
        "trace_label": trace_label,
        "session_id": "3200636808",
        "prompt_cache_key": "3200636808",
        "user_id": "3200636808",
        "x_tt_logid": "3200636808",
    }


def _record_id(record: dict[str, Any], index: int) -> str:
    for key in ("id", "sample_id", "question_id", "path_id"):
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return f"record_{index}"


def _question(record: dict[str, Any]) -> str:
    for message in record.get("conversations", []):
        if str(message.get("from", "")).lower() == "human":
            return str(message.get("value", ""))
    return ""


def _parse_tool_calls(value: str) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for match in _TOOL_CALL_RE.finditer(value):
        raw = match.group(1).strip()
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            calls.append(parsed)
    return calls


def _tool_name(call: dict[str, Any]) -> str:
    return str(call.get("name") or call.get("function", {}).get("name") or "").strip()


def _call_arguments(call: dict[str, Any]) -> dict[str, Any]:
    arguments = call.get("arguments")
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
    return arguments if isinstance(arguments, dict) else {}


def _block_query(block: ActionBlock) -> str:
    return str(_call_arguments(block.action).get("query") or "")


def _safe_text(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    # Trajectories normally contain <image> placeholders rather than base64,
    # but never send an accidental image payload to the detector/editor.
    return _DATA_IMAGE_RE.sub("[IMAGE_DATA_OMITTED]", text)


def _compact_text(value: Any, max_chars: int = 9000) -> str:
    text = _safe_text(value)
    if len(text) <= max_chars:
        return text
    head = max_chars * 2 // 3
    tail = max_chars - head
    return text[:head] + f"\n...[{len(text) - max_chars} chars omitted]...\n" + text[-tail:]


def _merge_adjacent_assistant_values(prefix: str, suffix: str) -> str:
    """Merge two consecutive ShareGPT assistant values into one message."""

    prefix_text = str(prefix or "").strip()
    suffix_text = str(suffix or "").strip()
    suffix_match = re.search(r"<thinking\b[^>]*>.*?</thinking>", suffix_text, re.DOTALL | re.IGNORECASE)
    if not suffix_match:
        return "\n".join(part for part in (prefix_text, suffix_text) if part)

    prefix_match = re.search(r"<thinking\b[^>]*>.*?</thinking>", prefix_text, re.DOTALL | re.IGNORECASE)
    prefix_thinking = prefix_match.group(0) if prefix_match else f"<thinking>\n{prefix_text}\n</thinking>"
    prefix_body = re.sub(
        r"^<thinking\b[^>]*>|</thinking>$",
        "",
        prefix_thinking.strip(),
        flags=re.IGNORECASE,
    ).strip()
    suffix_thinking = suffix_match.group(0)
    suffix_body = re.sub(
        r"^<thinking\b[^>]*>|</thinking>$",
        "",
        suffix_thinking.strip(),
        flags=re.IGNORECASE,
    ).strip()
    merged_thinking = "<thinking>\n" + "\n\n".join(
        part for part in (prefix_body, suffix_body) if part
    ) + "\n</thinking>"
    prefix_rest = (
        prefix_text[: prefix_match.start()] + prefix_text[prefix_match.end() :]
        if prefix_match
        else ""
    ).strip()
    suffix_rest = (suffix_text[: suffix_match.start()] + suffix_text[suffix_match.end() :]).strip()
    return "\n".join(part for part in (merged_thinking, prefix_rest, suffix_rest) if part)


def _normalize_adjacent_assistant_messages(record: dict[str, Any]) -> tuple[dict[str, Any], int]:
    """Repair legacy ``gpt -> gpt`` pairs before detection/editing."""

    updated = copy.deepcopy(record)
    conversations = updated.get("conversations")
    if not isinstance(conversations, list):
        return updated, 0
    normalized: list[dict[str, Any]] = []
    merged_pairs = 0
    for message in conversations:
        if (
            normalized
            and isinstance(message, dict)
            and str(normalized[-1].get("from", "")).lower() == "gpt"
            and str(message.get("from", "")).lower() == "gpt"
        ):
            normalized[-1]["value"] = _merge_adjacent_assistant_values(
                str(normalized[-1].get("value", "")),
                str(message.get("value", "")),
            )
            merged_pairs += 1
            continue
        normalized.append(message)
    updated["conversations"] = normalized
    return updated, merged_pairs


def _parse_blocks(conversations: list[dict[str, Any]]) -> list[ActionBlock]:
    blocks: list[ActionBlock] = []
    turn_number = -1
    for index, message in enumerate(conversations):
        if str(message.get("from", "")).lower() != "gpt":
            continue
        turn_number += 1
        assistant_value = _safe_text(message.get("value", ""))
        calls = _parse_tool_calls(assistant_value)
        if len(calls) != 1:
            continue
        observation_index: int | None = None
        observation_value = ""
        if index + 1 < len(conversations):
            next_message = conversations[index + 1]
            if str(next_message.get("from", "")).lower() == "observation":
                observation_index = index + 1
                observation_value = _safe_text(next_message.get("value", ""))
        blocks.append(
            ActionBlock(
                block_id=f"b{len(blocks):04d}",
                turn_number=turn_number,
                assistant_index=index,
                observation_index=observation_index,
                end_index=observation_index if observation_index is not None else index,
                action_name=_tool_name(calls[0]),
                action=calls[0],
                assistant_value=assistant_value,
                observation_value=observation_value,
            )
        )
    return blocks


def _block_text(block: ActionBlock) -> str:
    payload = {
        "block_id": block.block_id,
        "turn_number": block.turn_number,
        "conversation_indices": [block.assistant_index, block.observation_index],
        "action_name": block.action_name,
        "assistant_action": _compact_text(block.assistant_value),
        "observation": _compact_text(block.observation_value),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _trajectory_for_detector(record: dict[str, Any], blocks: list[ActionBlock]) -> str:
    """Render the complete trajectory with stable assistant-turn numbers.

    The detector needs to see all turns, including non-text tools, so that it
    cannot accidentally select across a read/image-search boundary.  Only
    base64 image payloads are omitted; ordinary text and tool observations are
    retained in full.
    """
    timeline: list[str] = []
    conversations = record.get("conversations") or []
    by_assistant = {block.assistant_index: block for block in blocks}
    consumed_observations = {
        block.observation_index for block in blocks if block.observation_index is not None
    }
    turn_number = -1
    for index, message in enumerate(conversations):
        role = str(message.get("from", ""))
        value = _safe_text(message.get("value", ""))
        role_lower = role.lower()
        if role_lower == "human":
            timeline.append(f"[Question]\n{value}")
            continue
        if role_lower == "gpt":
            turn_number += 1
            block = by_assistant.get(index)
            if block is not None:
                eligibility = (
                    "ELIGIBLE_SEARCH"
                    if block.action_name in ELIGIBLE_SEARCH_ACTIONS and block.observation_index is not None
                    else "NOT_ELIGIBLE"
                )
                section = [
                    f"[Turn {turn_number:03d} | action_block_id={block.block_id} | "
                    f"tool={block.action_name} | {eligibility}]",
                    "Assistant action:",
                    block.assistant_value,
                ]
                if block.observation_index is not None:
                    section.extend(["Tool observation:", block.observation_value])
                timeline.append("\n".join(section))
            else:
                timeline.append(f"[Turn {turn_number:03d} | assistant]\n{value}")
            continue
        if role_lower == "observation" and index in consumed_observations:
            continue
        timeline.append(f"[{role} at conversation index {index}]\n{value}")
    return "\n\n".join(timeline)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
    candidates.extend(fenced)
    start = text.find("{")
    if start >= 0:
        candidates.append(text[start:])
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _call_model(alias: str, system_prompt: str, user_prompt: str, trace_label: str) -> str:
    # Some registered GPT aliases are backed by the Responses API rather than
    # chat completions. Detect that from the worker configuration so that the
    # request uses ``input``/``instructions`` instead of ``messages``.
    config = getattr(LLM_WORKER, "_configs", {}).get(alias, {})
    endpoint = str(config.get("azure_endpoint") or config.get("base_url") or "")
    api_mode = str(config.get("api_mode") or "").lower()
    use_responses = api_mode == "responses" or "/responses" in endpoint.lower()
    try:
        if use_responses:
            request: Any = ResponsesModelRequest(
                model=alias,
                instructions=system_prompt,
                input=[
                    {
                        "role": "user",
                        "content": [{"type": "input_text", "text": user_prompt}],
                    }
                ],
                metadata=_metadata(trace_label),
            )
            response = LLM_WORKER.responses_generate(request)
        else:
            request = ModelRequest(
                model=alias,
                messages=[
                    ModelMessage(role="system", content=system_prompt),
                    ModelMessage(role="user", content=user_prompt),
                ],
                metadata=_metadata(trace_label),
            )
            response = LLM_WORKER.generate(request)
    except Exception as exc:
        # model_worker intentionally keeps its retry log compact. Emit the
        # exception chain here as well, because APIConnectionError's useful
        # socket/DNS/TLS cause is often stored in __cause__ rather than str(exc).
        chain: list[str] = []
        current: BaseException | None = exc
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(
                f"{type(current).__module__}.{type(current).__name__}: {str(current)!r}"
            )
            current = current.__cause__ or current.__context__
        print(
            "[redundancy-model-error]"
            f" alias={alias} trace_label={trace_label}"
            f" exception_chain={json.dumps(chain, ensure_ascii=False)}",
            file=sys.stderr,
            flush=True,
        )
        raise
    content = response.content if response is not None else ""
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("model returned an empty response")
    return content


def _later_read_resource_ids(conversations: list[dict[str, Any]], after_index: int) -> set[str]:
    ids: set[str] = set()
    for message in conversations[after_index + 1 :]:
        if str(message.get("from", "")).lower() != "gpt":
            continue
        for call in _parse_tool_calls(str(message.get("value", ""))):
            if _tool_name(call) != "read_url":
                continue
            arguments = call.get("arguments")
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            if isinstance(arguments, dict):
                resource_id = arguments.get("resource_id")
                if resource_id is not None:
                    ids.add(str(resource_id))
    return ids


def _mechanically_protected(
    block: ActionBlock,
    conversations: list[dict[str, Any]],
    blocks: list[ActionBlock] | None = None,
    remove_ids: set[str] | None = None,
) -> tuple[bool, list[str]]:
    result_ids = set(_RESOURCE_ID_RE.findall(block.observation_value))
    read_ids = _later_read_resource_ids(conversations, block.end_index)
    protected: list[str] = []
    remove_ids = remove_ids or set()
    for resource_id in sorted(result_ids & read_ids):
        # A later read does not make this particular search indispensable when
        # the same resource was already returned by an earlier retained search.
        earlier_provider_exists = False
        if blocks is not None:
            earlier_provider_exists = any(
                candidate.block_id not in remove_ids
                and candidate.action_name in ELIGIBLE_SEARCH_ACTIONS
                and candidate.observation_index is not None
                and candidate.end_index < block.end_index
                and resource_id in _RESOURCE_ID_RE.findall(candidate.observation_value)
                for candidate in blocks
            )
        if not earlier_provider_exists:
            protected.append(resource_id)
    return bool(protected), protected


def _candidate_blocks(blocks: list[ActionBlock]) -> list[ActionBlock]:
    return [
        block
        for block in blocks
        if block.action_name in ELIGIBLE_SEARCH_ACTIONS and block.observation_index is not None
    ]


def _empty_detection(blocks: list[ActionBlock]) -> dict[str, Any]:
    return {
        "raw_response": "",
        "parsed": {"redundant_groups": []},
        "blocks": blocks,
        "candidate_block_ids": [block.block_id for block in _candidate_blocks(blocks)],
        "pre_rejected_block_ids": {},
        "accepted_block_ids": [],
        "accepted_groups": [],
        "rejected_groups": {},
        "reason_by_block_id": {},
    }


def _as_turn_numbers(value: Any) -> list[int] | None:
    if not isinstance(value, list) or not value:
        return None
    result: list[int] = []
    for item in value:
        if isinstance(item, bool):
            return None
        try:
            number = int(item)
        except (TypeError, ValueError):
            return None
        result.append(number)
    return result


def _parse_redundant_groups(
    parsed: dict[str, Any],
    blocks: list[ActionBlock],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    """Parse and validate the detector's contiguous turn groups.

    The new format is ``redundant_groups[].turns``.  A small legacy fallback
    accepts the old ``decisions[].block_id`` format so old audit samples remain
    inspectable, but new runs are required to use turn-number groups.
    """
    by_turn = {block.turn_number: block for block in _candidate_blocks(blocks)}
    by_id = {block.block_id: block for block in _candidate_blocks(blocks)}
    raw_groups = parsed.get("redundant_groups")
    if not isinstance(raw_groups, list):
        raw_groups = []
    if not raw_groups:
        # Backward-compatible parsing only; the current prompt never asks for
        # this format.
        for decision in parsed.get("decisions", []) if isinstance(parsed.get("decisions"), list) else []:
            if not isinstance(decision, dict):
                continue
            if str(decision.get("decision", "")).lower() not in {"delete", "remove", "redundant"}:
                continue
            block = by_id.get(str(decision.get("block_id", "")))
            if block is not None:
                raw_groups.append(
                    {
                        "turns": [block.turn_number],
                        "reason": decision.get("reason", ""),
                    }
                )
        legacy_ids = parsed.get("redundant_block_ids")
        if isinstance(legacy_ids, list):
            for block_id in legacy_ids:
                block = by_id.get(str(block_id))
                if block is not None:
                    raw_groups.append({"turns": [block.turn_number], "reason": "legacy detector output"})

    accepted: list[dict[str, Any]] = []
    rejected: dict[str, str] = {}
    seen_turns: set[int] = set()
    for group_number, item in enumerate(raw_groups):
        key = f"group_{group_number:03d}"
        if not isinstance(item, dict):
            rejected[key] = "group is not an object"
            continue
        turns = _as_turn_numbers(item.get("turns", item.get("turn_numbers")))
        reason = str(item.get("reason", "")).strip()
        if turns is None:
            rejected[key] = "turns must be a non-empty integer list"
            continue
        if turns != sorted(set(turns)):
            rejected[key] = "turns must be unique and in ascending order"
            continue
        if turns != list(range(turns[0], turns[-1] + 1)):
            rejected[key] = "turns are not consecutive"
            continue
        if not reason:
            rejected[key] = "missing deletion reason"
            continue
        missing = [turn for turn in turns if turn not in by_turn]
        if missing:
            rejected[key] = f"turns are not observed eligible text/image-search turns: {missing}"
            continue
        overlap = sorted(set(turns) & seen_turns)
        if overlap:
            rejected[key] = f"group overlaps previously accepted group at turns {overlap}"
            continue
        seen_turns.update(turns)
        selected_blocks = [by_turn[turn] for turn in turns]
        accepted.append(
            {
                "group_id": key,
                "turns": turns,
                "block_ids": [block.block_id for block in selected_blocks],
                "reason": reason,
            }
        )
    return accepted, rejected


def _detect(record: dict[str, Any], record_index: int, alias: str) -> dict[str, Any]:
    conversations = record.get("conversations") or []
    blocks = _parse_blocks(conversations)
    eligible_blocks = _candidate_blocks(blocks)
    if not eligible_blocks:
        return _empty_detection(blocks)
    detector_prompt = (
        f"Record id: {_record_id(record, record_index)}\n"
        f"Original question:\n{_question(record)}\n\n"
        "Below is the complete trajectory. The number in `[Turn NNN]` is the "
        "assistant-turn number that must be used in your JSON output.\n\n"
        "<complete_numbered_trajectory>\n"
        f"{_trajectory_for_detector(record, blocks)}\n"
        "</complete_numbered_trajectory>"
    )
    raw = _call_model(alias, DETECTOR_SYSTEM_PROMPT, detector_prompt, "redundant_t2t_detector")
    parsed = _extract_json_object(raw)
    if parsed is None:
        raise ValueError("detector response was not a JSON object")
    proposed_groups, parse_rejected = _parse_redundant_groups(parsed, blocks)
    by_id = {block.block_id: block for block in eligible_blocks}
    all_proposed_ids = {
        block_id for group in proposed_groups for block_id in group["block_ids"]
    }
    accepted_groups: list[dict[str, Any]] = []
    rejected_groups = dict(parse_rejected)
    for group in proposed_groups:
        protected_resources: set[str] = set()
        for block_id in group["block_ids"]:
            protected, resource_ids = _mechanically_protected(
                by_id[block_id],
                conversations,
                blocks,
                remove_ids=all_proposed_ids,
            )
            if protected:
                protected_resources.update(resource_ids)
        if protected_resources:
            rejected_groups[group["group_id"]] = (
                "removing this group would orphan later read_url resources: "
                + ",".join(sorted(protected_resources))
            )
            continue
        accepted_groups.append(group)
    accepted = [block_id for group in accepted_groups for block_id in group["block_ids"]]
    reason_by_id = {
        block_id: group["reason"]
        for group in accepted_groups
        for block_id in group["block_ids"]
    }
    return {
        "raw_response": raw,
        "parsed": parsed,
        "blocks": blocks,
        "candidate_block_ids": [block.block_id for block in eligible_blocks],
        "pre_rejected_block_ids": {},
        "accepted_block_ids": accepted,
        "accepted_groups": accepted_groups,
        "rejected_groups": rejected_groups,
        "reason_by_block_id": reason_by_id,
    }


def _group_block_ids(blocks: list[ActionBlock], accepted_ids: Iterable[str]) -> list[list[str]]:
    accepted = set(accepted_ids)
    groups: list[list[str]] = []
    current: list[str] = []
    for block in blocks:
        if block.block_id in accepted:
            current.append(block.block_id)
        elif current:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _editor_context(
    record: dict[str, Any],
    blocks: list[ActionBlock],
    group: list[str],
) -> tuple[str, ActionBlock | None, list[ActionBlock]]:
    by_id = {block.block_id: block for block in blocks}
    indices = [blocks.index(by_id[block_id]) for block_id in group]
    first_position, last_position = min(indices), max(indices)
    target: ActionBlock | None = None
    for block in blocks[last_position + 1 :]:
        if block.block_id not in group:
            target = block
            break
    # Include a little more context before the deletion, but keep the edit
    # local. The detector already saw the full trajectory.
    before = blocks[max(0, first_position - 4) : first_position]
    removed = [by_id[block_id] for block_id in group]
    after = [target] if target is not None else []
    pieces = [
        f"Record id: {_record_id(record, -1)}",
        f"Original question:\n{_question(record)}",
        "Retained context immediately before the redundant run:",
        "\n\n".join(_block_text(block) for block in before) or "[none]",
        "Redundant text/image-search block(s) to delete:",
        "\n\n".join(_block_text(block) for block in removed),
        "Next retained assistant action to keep exactly:",
        _block_text(target) if target is not None else "[none]",
    ]
    return "\n\n".join(pieces), target, removed


def _thinking_only(raw: str) -> str:
    match = _THINKING_RE.search(raw)
    if match:
        thinking = match.group(1).strip()
    else:
        thinking = raw.strip()
    thinking = _ANSWER_RE.sub("", thinking).strip()
    if "<tool_call>" in thinking.lower() or "</tool_call>" in thinking.lower():
        raise ValueError("editor returned a tool call instead of thinking only")
    if not thinking:
        raise ValueError("editor returned empty thinking")
    return f"<thinking>\n{thinking}\n</thinking>"


def _replace_thinking(original: str, rewritten_thinking: str) -> str:
    match = _THINKING_RE.search(original)
    if match:
        return original[: match.start()] + rewritten_thinking + original[match.end() :]
    return rewritten_thinking + ("\n" + original if original.strip() else "")


def _rewrite_group(
    record: dict[str, Any],
    blocks: list[ActionBlock],
    group: list[str],
    alias: str,
) -> dict[str, Any]:
    context, target, removed = _editor_context(record, blocks, group)
    if target is None:
        return {
            "group": group,
            "status": "skipped",
            "reason": "no_later_assistant_turn_to_reconnect",
            "removed": [block.block_id for block in removed],
        }
    prompt = (
        context
        + "\n\nYour task: delete the redundant block(s) listed above and rewrite only "
        "the thinking section of the next retained assistant action so the "
        "reasoning flows directly from the last retained evidence to that "
        "unchanged action. Output only <thinking>...</thinking>."
    )
    raw = _call_model(alias, EDITOR_SYSTEM_PROMPT, prompt, "redundant_t2t_editor")
    rewritten = _thinking_only(raw)
    return {
        "group": group,
        "status": "ok",
        "target_block_id": target.block_id,
        "target_assistant_index": target.assistant_index,
        "removed": [block.block_id for block in removed],
        "raw_response": raw,
        "rewritten_thinking": rewritten,
    }


def _apply_edits(
    record: dict[str, Any],
    blocks: list[ActionBlock],
    edit_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    updated = copy.deepcopy(record)
    conversations = updated.get("conversations")
    if not isinstance(conversations, list):
        return updated, []
    by_id = {block.block_id: block for block in blocks}
    successful = [item for item in edit_results if item.get("status") == "ok"]
    # Apply from right to left so conversation indices remain valid.
    for item in sorted(successful, key=lambda value: int(value["target_assistant_index"]), reverse=True):
        target_index = int(item["target_assistant_index"])
        if target_index >= len(conversations):
            raise RuntimeError(f"target conversation index disappeared: {target_index}")
        target_message = conversations[target_index]
        if str(target_message.get("from", "")).lower() != "gpt":
            raise RuntimeError(f"target is no longer a gpt message: {target_index}")
        target_message["value"] = _replace_thinking(
            str(target_message.get("value", "")), str(item["rewritten_thinking"])
        )
    remove_blocks: list[ActionBlock] = []
    for item in successful:
        for block_id in item.get("removed", []):
            if block_id in by_id:
                remove_blocks.append(by_id[block_id])
    remove_indices = {
        index
        for block in remove_blocks
        for index in (block.assistant_index, block.observation_index)
        if index is not None
    }
    if remove_indices:
        updated["conversations"] = [
            message for index, message in enumerate(conversations) if index not in remove_indices
        ]
    return updated, [block.block_id for block in remove_blocks]


def _quality_gate_context(
    record: dict[str, Any],
    blocks: list[ActionBlock],
    edit_results: list[dict[str, Any]],
) -> str:
    """Build only the local before/after splice context for the final gate."""
    by_id = {block.block_id: block for block in blocks}
    sections: list[str] = []
    merge_number = 0
    for edit in edit_results:
        if edit.get("status") != "ok":
            continue
        group = [str(block_id) for block_id in edit.get("group", [])]
        group_blocks = [by_id[block_id] for block_id in group if block_id in by_id]
        if not group_blocks:
            continue
        target_id = str(edit.get("target_block_id") or "")
        target = by_id.get(target_id)
        if target is None:
            continue
        positions = [blocks.index(block) for block in group_blocks]
        first_position = min(positions)
        prior = blocks[first_position - 1] if first_position > 0 else None
        revised_target = _replace_thinking(
            target.assistant_value,
            str(edit.get("rewritten_thinking") or ""),
        )
        merge_number += 1
        sections.append(
            "\n".join(
                [
                    f"[MERGE BOUNDARY {merge_number}]",
                    "Inspect this exact splice and nothing else.",
                    "Retained context immediately before the deleted span:",
                    _block_text(prior) if prior is not None else "[none; the deleted span was at the start]",
                    "Original next retained assistant turn:",
                    _compact_text(target.assistant_value),
                    "Revised next retained assistant turn:",
                    _compact_text(revised_target),
                    "The next retained tool call and its arguments are unchanged.",
                ]
            )
        )
    return "\n\n".join(
        [
            f"Original question:\n{_question(record)}",
            *sections,
        ]
    )


def _quality_gate(
    record: dict[str, Any],
    blocks: list[ActionBlock],
    edit_results: list[dict[str, Any]],
    alias: str,
) -> dict[str, Any]:
    prompt = (
        "Question:\n"
        f"{_question(record)}\n\n"
        "The following contains one or more local merge boundaries created by deleting "
        "redundant searches. Check only whether the revised reasoning is logically "
        "connected and safe for SFT. Do not solve the question.\n\n"
        "<local_merge_boundaries>\n"
        f"{_quality_gate_context(record, blocks, edit_results)}\n"
        "</local_merge_boundaries>"
    )
    started = time.perf_counter()
    raw = _call_model(alias, QUALITY_GATE_SYSTEM_PROMPT, prompt, "redundant_quality_gate")
    parsed = _extract_json_object(raw)
    if parsed is None:
        raise ValueError("quality gate response was not a JSON object")
    decision = str(parsed.get("decision") or "").strip().lower()
    if decision not in {"accept", "reject"}:
        raise ValueError("quality gate decision must be accept or reject")
    reason = str(parsed.get("reason") or "").strip()
    issues = parsed.get("issues")
    if not isinstance(issues, list):
        issues = []
    return {
        "decision": decision,
        "reason": reason,
        "issues": issues,
        "raw_response": raw,
        "parsed": parsed,
        "elapsed_s": time.perf_counter() - started,
    }


def process_record(
    record: dict[str, Any],
    record_index: int,
    detector_alias: str,
    quality_alias: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
    started = time.perf_counter()
    original = copy.deepcopy(record)
    try:
        working_record, normalized_adjacent_assistants = _normalize_adjacent_assistant_messages(record)
        detection = _detect(working_record, record_index, detector_alias)
        groups = _group_block_ids(detection["blocks"], detection["accepted_block_ids"])
        edit_results: list[dict[str, Any]] = []
        for group in groups:
            edit_results.append(
                _rewrite_group(working_record, detection["blocks"], group, detector_alias)
            )
        updated, removed = _apply_edits(working_record, detection["blocks"], edit_results)
        audit = {
            "id": _record_id(record, record_index),
            "input_index": record_index,
            "status": "ok",
            "detector_raw_response": detection["raw_response"],
            "detector_parsed": detection["parsed"],
            "candidate_block_ids": detection["candidate_block_ids"],
            "pre_rejected_block_ids": detection.get("pre_rejected_block_ids", {}),
            "detector_accepted_block_ids": detection["accepted_block_ids"],
            "detector_accepted_groups": detection.get("accepted_groups", []),
            "detector_rejected_groups": detection.get("rejected_groups", {}),
            "detector_reason_by_block_id": detection["reason_by_block_id"],
            "edit_results": edit_results,
            "removed_block_ids": removed,
            "normalized_adjacent_assistant_pairs": normalized_adjacent_assistants,
            "changed": updated != original,
            "elapsed_s": time.perf_counter() - started,
        }
        if removed:
            quality_gate = _quality_gate(
                working_record,
                detection["blocks"],
                edit_results,
                quality_alias,
            )
            audit["quality_gate"] = quality_gate
            if quality_gate["decision"] == "reject":
                audit["status"] = "quality_rejected"
                audit["candidate_changed"] = updated != original
                audit["changed"] = False
                return original, audit, updated
        return updated, audit, None
    except Exception as exc:
        return (
            original,
            {
                "id": _record_id(record, record_index),
                "input_index": record_index,
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": time.perf_counter() - started,
            },
            None,
        )


def _load_records(path: Path) -> tuple[list[dict[str, Any]], bool]:
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        data = json.loads(text)
        if not isinstance(data, list):
            raise ValueError("JSON input must be a list")
        return data, False
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"JSONL line {line_number} is not an object")
        records.append(value)
    return records, True


def _flush_and_sync(handle: Any) -> None:
    """Publish buffered bytes, including on HDFS-backed FUSE mounts.

    ``flush`` makes the bytes visible to the Python process and ``fsync`` asks
    the filesystem/FUSE layer to publish them durably.  Some HDFS mounts do
    not implement fsync; in that case we keep the already-flushed data and
    emit one warning instead of failing an otherwise completed sample.
    """
    handle.flush()
    try:
        os.fsync(handle.fileno())
    except (AttributeError, OSError) as exc:
        global _FSYNC_WARNING_EMITTED
        if not _FSYNC_WARNING_EMITTED:
            _FSYNC_WARNING_EMITTED = True
            print(
                f"[redundancy-output] fsync unavailable; relying on flush/close: {exc}",
                file=sys.stderr,
                flush=True,
            )


def _write_jsonl_records(handle: Any, records: list[dict[str, Any]]) -> None:
    """Append JSONL records and publish them immediately.

    This follows the write pattern used by ``synthesis/sft/debug_vqa_batch``:
    append, flush, and fsync after each checkpoint batch.  The caller keeps the
    handle open so HDFS does not need to reopen the file for every record.
    """
    if not records:
        return
    for record in records:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    _flush_and_sync(handle)


def _write_records(path: Path, records: list[dict[str, Any]], jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if jsonl:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        else:
            json.dump(records, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        _flush_and_sync(handle)


def _default_state_dir(output: Path) -> Path:
    return output.parent / f".{output.name}.state"


def _resume_key(record: dict[str, Any], record_index: int) -> tuple[int, str]:
    """Stable key for one input position in a resumable run."""
    return record_index, _record_id(record, record_index)


def _load_checkpoint_records(
    path: Path,
) -> dict[tuple[int, str], dict[str, Any]]:
    """Load the latest checkpoint entry for each input record.

    A truncated final JSONL line is ignored.  The last entry wins, which lets
    a later successful retry supersede an earlier error entry.
    """
    latest: dict[tuple[int, str], dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                print(
                    f"[redundancy-resume] ignoring malformed checkpoint line "
                    f"{path}:{line_number}: {exc}",
                    file=sys.stderr,
                    flush=True,
                )
                continue
            if not isinstance(payload, dict):
                continue
            try:
                record_index = int(payload["input_index"])
            except (KeyError, TypeError, ValueError):
                continue
            record_id = str(payload.get("id") or "").strip()
            if not record_id:
                continue
            latest[(record_index, record_id)] = payload
    return latest


def _checkpoint_entry(
    record_index: int,
    record: dict[str, Any],
    audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "input_index": record_index,
        "id": _record_id(record, record_index),
        "status": audit.get("status"),
        "record": record,
        "audit": audit,
    }


def _parse_sample_ids(values: list[str] | None) -> set[str] | None:
    if not values:
        return None
    result: set[str] = set()
    for value in values:
        result.update(item.strip() for item in value.split(",") if item.strip())
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-jsonl", type=Path, required=True)
    parser.add_argument("--model-alias", required=True)
    parser.add_argument(
        "--quality-model-alias",
        default=None,
        help="Alias for the local post-edit quality gate; defaults to --model-alias.",
    )
    parser.add_argument(
        "--quality-rejected-jsonl",
        type=Path,
        default=None,
        help="Archive for samples rejected by the post-edit quality gate.",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sample-id", action="append", help="repeatable or comma-separated")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the per-output checkpoint JSONL and skip records whose latest "
            "checkpoint status is terminal (ok or quality_rejected). The offset/limit "
            "range may be expanded on resume."
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=None,
        help="Checkpoint directory; defaults to a hidden directory next to --output.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        help="Flush completed records to the checkpoint after this many records (default: 1).",
    )
    parser.add_argument(
        "--selected-only-output",
        action="store_true",
        help="write only selected records; useful for a targeted test",
    )
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.offset < 0:
        parser.error("--offset must be >= 0")
    if args.checkpoint_every < 1:
        parser.error("--checkpoint-every must be >= 1")
    quality_model_alias = args.quality_model_alias or args.model_alias
    quality_rejected_path = args.quality_rejected_jsonl or args.output.with_name(
        f"{args.output.stem}.quality_rejected.jsonl"
    )

    records, jsonl = _load_records(args.input)
    start = args.offset
    stop = len(records) if args.limit is None else min(len(records), start + max(args.limit, 0))
    requested_ids = _parse_sample_ids(args.sample_id)
    selected_indices = [
        index
        for index in range(start, stop)
        if requested_ids is None or _record_id(records[index], index) in requested_ids
    ]
    if args.sample_id and not selected_indices:
        print("No records matched --sample-id", file=sys.stderr)
        return 2

    state_dir = args.state_dir or _default_state_dir(args.output)
    checkpoint_path = state_dir / "completed_records.jsonl"
    if (
        not args.resume
        and not args.dry_run
        and checkpoint_path.exists()
        and checkpoint_path.stat().st_size > 0
    ):
        parser.error(
            f"Checkpoint already exists: {checkpoint_path}; use --resume, "
            "--state-dir with a new directory, or remove the old checkpoint intentionally."
        )

    checkpoint_entries = _load_checkpoint_records(checkpoint_path) if args.resume else {}
    resumable_indices: set[int] = set()
    rejected_checkpoint_indices: set[int] = set()
    successful_checkpoint_records: dict[int, dict[str, Any]] = {}
    for (record_index, record_id), payload in checkpoint_entries.items():
        if payload.get("status") not in _CHECKPOINT_TERMINAL_STATUSES:
            continue
        if record_index < 0 or record_index >= len(records):
            continue
        if _resume_key(records[record_index], record_index) != (record_index, record_id):
            continue
        if payload.get("status") == "ok":
            successful_checkpoint_records[record_index] = payload
        elif payload.get("status") == "quality_rejected":
            rejected_checkpoint_indices.add(record_index)
        if record_index in selected_indices:
            resumable_indices.add(record_index)

    pending_indices = [index for index in selected_indices if index not in resumable_indices]
    state_handle: Any | None = None
    audit_handle: Any | None = None
    quality_rejected_handle: Any | None = None
    if not args.dry_run:
        state_dir.mkdir(parents=True, exist_ok=True)
        state_handle = checkpoint_path.open("a", encoding="utf-8")
        quality_rejected_path.parent.mkdir(parents=True, exist_ok=True)
        quality_rejected_handle = quality_rejected_path.open("a", encoding="utf-8")
    args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
    audit_handle = args.audit_jsonl.open("a", encoding="utf-8")

    print(
        json.dumps(
            {
                "input": str(args.input),
                "selected": len(selected_indices),
                "pending": len(pending_indices),
                "resumed": len(resumable_indices),
                "offset": args.offset,
                "limit": args.limit,
                "workers": args.workers,
                "checkpoint_every": args.checkpoint_every,
                "state_dir": str(state_dir),
                "checkpoint": str(checkpoint_path),
                "quality_model_alias": quality_model_alias,
                "quality_rejected_jsonl": str(quality_rejected_path),
                "dry_run": args.dry_run,
                "model_alias": args.model_alias,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    processed: dict[
        int,
        tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None],
    ] = {}
    pending_checkpoint_entries: list[dict[str, Any]] = []
    pending_audits: list[dict[str, Any]] = []
    resumed_audits = [
        payload.get("audit") if isinstance(payload.get("audit"), dict) else {}
        for (record_index, _), payload in checkpoint_entries.items()
        if record_index in resumable_indices
        and payload.get("status") in _CHECKPOINT_TERMINAL_STATUSES
    ]
    ok_count = sum(audit.get("status") == "ok" for audit in resumed_audits)
    quality_rejected_count = len(
        [
            index
            for index in rejected_checkpoint_indices
            if index in selected_indices
        ]
    )
    error_count = 0
    changed_count = sum(
        bool(audit.get("changed"))
        for audit in resumed_audits
        if audit.get("status") == "ok"
    )
    removed_count = sum(
        len(audit.get("removed_block_ids", []))
        for audit in resumed_audits
        if audit.get("status") == "ok"
    )
    merge_group_count = sum(
        len(audit.get("edit_results", []))
        for audit in resumed_audits
        if audit.get("status") == "ok"
    )
    merge_success_count = sum(
        sum(
            item.get("status") == "ok"
            for item in audit.get("edit_results", [])
            if isinstance(item, dict)
        )
        for audit in resumed_audits
        if audit.get("status") == "ok"
    )
    quality_check_count = sum("quality_gate" in audit for audit in resumed_audits)
    quality_accept_count = sum(
        (audit.get("quality_gate") or {}).get("decision") == "accept"
        for audit in resumed_audits
    )

    progress = (
        tqdm(
            total=len(selected_indices),
            initial=len(resumable_indices),
            desc="Compressing trajectories",
            unit="trajectory",
            dynamic_ncols=True,
        )
        if tqdm is not None
        else None
    )

    def update_progress() -> None:
        if progress is not None:
            progress.set_postfix(
                resumed=len(resumable_indices),
                ok=ok_count,
                errors=error_count,
                quality_rejected=quality_rejected_count,
                merge_groups=merge_group_count,
                merge_ok=merge_success_count,
                quality_checks=quality_check_count,
                quality_accept=quality_accept_count,
                changed=changed_count,
                removed=removed_count,
            )

    def publish_pending() -> None:
        if pending_checkpoint_entries and state_handle is not None:
            _write_jsonl_records(state_handle, pending_checkpoint_entries)
            pending_checkpoint_entries.clear()
        if pending_audits and audit_handle is not None:
            _write_jsonl_records(audit_handle, pending_audits)
            pending_audits.clear()

    update_progress()
    try:
        if pending_indices:
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(
                        process_record,
                        records[index],
                        index,
                        args.model_alias,
                        quality_model_alias,
                    ): index
                    for index in pending_indices
                }
                for future in as_completed(futures):
                    index = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # defensive: process_record normally catches this
                        result = (
                            copy.deepcopy(records[index]),
                            {
                                "id": _record_id(records[index], index),
                                "input_index": index,
                                "status": "error",
                                "error_type": type(exc).__name__,
                                "error": str(exc),
                            },
                            None,
                        )
                    processed[index] = result
                    updated, audit, quality_candidate = result
                    if not args.dry_run:
                        pending_checkpoint_entries.append(
                            _checkpoint_entry(index, updated, audit)
                        )
                        if audit.get("status") == "quality_rejected" and quality_rejected_handle is not None:
                            _write_jsonl_records(
                                quality_rejected_handle,
                                [
                                    {
                                        "input_index": index,
                                        "id": _record_id(records[index], index),
                                        "original_record": records[index],
                                        "candidate_record": quality_candidate,
                                        "audit": audit,
                                    }
                                ],
                            )
                    pending_audits.append(audit)
                    if len(pending_audits) >= args.checkpoint_every:
                        publish_pending()

                    if audit.get("status") == "ok":
                        ok_count += 1
                    elif audit.get("status") == "quality_rejected":
                        quality_rejected_count += 1
                    else:
                        error_count += 1
                    edit_results = audit.get("edit_results", [])
                    if isinstance(edit_results, list):
                        merge_group_count += len(edit_results)
                        merge_success_count += sum(
                            item.get("status") == "ok"
                            for item in edit_results
                            if isinstance(item, dict)
                        )
                    quality_gate = audit.get("quality_gate")
                    if isinstance(quality_gate, dict):
                        quality_check_count += 1
                        if quality_gate.get("decision") == "accept":
                            quality_accept_count += 1
                    changed_count += int(bool(audit.get("changed")))
                    removed_count += len(audit.get("removed_block_ids", []))
                    if progress is not None:
                        progress.update(1)
                    else:
                        print(
                            f"Processed {len(resumable_indices) + len(processed)}/"
                            f"{len(selected_indices)}: {_record_id(records[index], index)}",
                            flush=True,
                        )
                    update_progress()
    finally:
        # With the default checkpoint-every=1, every completed record has
        # already been published.  This also makes larger batches safe on a
        # normal interrupt and mirrors debug_vqa_batch's final flush.
        publish_pending()
        if audit_handle is not None:
            audit_handle.close()
        if state_handle is not None:
            state_handle.close()
        if quality_rejected_handle is not None:
            quality_rejected_handle.close()
        if progress is not None:
            progress.close()

    output_records = copy.deepcopy(records)
    for index, payload in successful_checkpoint_records.items():
        if 0 <= index < len(output_records):
            output_records[index] = copy.deepcopy(payload["record"])
    for index, (updated, audit, _) in processed.items():
        if audit.get("status") == "quality_rejected":
            continue
        if audit.get("status") == "ok":
            output_records[index] = updated

    excluded_indices = set(rejected_checkpoint_indices)
    excluded_indices.update(
        index
        for index, (_, audit, _) in processed.items()
        if audit.get("status") == "quality_rejected"
    )

    if not args.dry_run:
        if args.selected_only_output:
            _write_records(
                args.output,
                [
                    output_records[index]
                    for index in selected_indices
                    if index not in excluded_indices
                ],
                jsonl,
            )
        else:
            _write_records(
                args.output,
                [
                    record
                    for index, record in enumerate(output_records)
                    if index not in excluded_indices
                ],
                jsonl,
            )

    summary = {
        "selected": len(selected_indices),
        "pending_processed": len(processed),
        "resumed": len(resumable_indices),
        "ok": ok_count,
        "errors": error_count,
        "quality_rejected": quality_rejected_count,
        "merge_groups": merge_group_count,
        "merge_success": merge_success_count,
        "quality_checks": quality_check_count,
        "quality_accept": quality_accept_count,
        "changed": changed_count,
        "removed_blocks": removed_count,
        "dry_run": args.dry_run,
        "output": None if args.dry_run else str(args.output),
        "audit_jsonl": str(args.audit_jsonl),
        "checkpoint": None if args.dry_run else str(checkpoint_path),
        "quality_rejected_jsonl": None if args.dry_run else str(quality_rejected_path),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0 if summary["errors"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
