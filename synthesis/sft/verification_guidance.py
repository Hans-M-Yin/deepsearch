"""Lightweight verification-guidance helpers for manual ReAct trajectory generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from synthesis.model_worker import LLM_WORKER
from synthesis.model_worker import ModelMessage
from synthesis.model_worker import ModelRequest

PROMPT_CHECK_CURRENT_HOP = """
You are checking whether the agent has already solved the current hop in a multi-hop search trajectory.

You will receive:
1. The original question.
2. The current hop information.
3. The visible trajectory so far.

Return strict JSON:
{
  "found": false,
  "reason": ""
}

Rules:
- Mark found=true only if the current hop target has already been correctly identified with sufficiently strong evidence in the visible trajectory.
- If the target is only guessed, weakly suggested, or unsupported, return found=false.
- Do not output markdown or any text outside the JSON object.
"""

PROMPT_CHECK_INITIAL_SOURCE = """
You are checking whether the agent has already identified the initial source entity of a multi-hop search trajectory.

You will receive:
1. The original question.
2. The initial source information for the first search stage.
3. The visible trajectory so far.

Return strict JSON:
{
  "found": false,
  "reason": ""
}

Rules:
- Mark found=true only if the initial source entity has already been correctly identified with sufficiently strong evidence in the visible trajectory.
- If the source is only guessed, weakly suggested, or unsupported, return found=false.
- Do not require the agent to have solved the first hop target yet. At this stage, only check whether the starting source entity has been correctly identified.
- Do not output markdown or any text outside the JSON object.
"""


@dataclass(slots=True)
class HopJudgeConfig:
    model_alias: str | None = None
    max_tokens: int = 1024


@dataclass(slots=True)
class ManualReActHopState:
    question: str
    gold_answer: str
    hop_chain: list[dict[str, Any]]
    current_hop_index: int = 0
    judge: HopJudgeConfig = field(default_factory=HopJudgeConfig)

    def first_hop(self) -> dict[str, Any] | None:
        if not self.hop_chain:
            return None
        hop = self.hop_chain[0]
        return hop if isinstance(hop, dict) else None

    def has_text_start(self) -> bool:
        hop = self.first_hop()
        if not isinstance(hop, dict):
            return False
        src_node_id = str(hop.get("src_node_id") or "").strip().lower()
        return src_node_id.startswith("text_")

    def in_initial_source_stage(self) -> bool:
        return self.current_hop_index < 0 and self.has_text_start()

    def current_hop(self) -> dict[str, Any] | None:
        if self.in_initial_source_stage():
            return None
        self.advance_past_image_hops()
        if self.current_hop_index < 0 or self.current_hop_index >= len(self.hop_chain):
            return None
        hop = self.hop_chain[self.current_hop_index]
        return hop if isinstance(hop, dict) else None

    def has_remaining_hops(self) -> bool:
        return self.current_hop() is not None

    def advance(self) -> None:
        if self.current_hop_index < len(self.hop_chain):
            self.current_hop_index += 1
        self.advance_past_image_hops()

    def advance_past_image_hops(self) -> None:
        if self.current_hop_index < 0:
            return
        while self.current_hop_index < len(self.hop_chain):
            hop = self.hop_chain[self.current_hop_index]
            if not _is_image_target_hop(hop):
                break
            self.current_hop_index += 1

    def verification_guidance(self) -> str:
        if self.in_initial_source_stage():
            return ""
        hop = self.current_hop()
        if hop is None:
            answer = self.gold_answer.strip()
            if not answer:
                return ""
            return (
                f"""(CURRENT VERIFICATION GUIDANCE:
The correct answer to this question is: {answer}
Note that this answer is provided for you to verify whether you have found the correct answer. If the correct answer cannot be derived from your previous analysis and tool results, or if you have instead arrived at an incorrect answer, then you need to incorporate a logically natural self-correction into your solution process—for example, explain why the current answer may be wrong, and then continue planning tool usage and searching for clues that can lead to the correct answer. Be careful not to reveal the correct answer provided here at any point in your solution process; you must continue writing the response as if you were a responder who does not know the correct answer.)"""
            )
        statement = str(hop.get("statement") or "").strip()
        source = str(hop.get("source") or "").strip()
        target = str(hop.get("target") or "").strip()
        return (
            f"""(CURRENT VERIFICATION GUIDANCE:
The current search stage is: "{statement}", where {source} is the known original object, and you are inferring {target} based on the clues.
You need to use the above information to verify whether the solution process you are writing has achieved its objective. It is possible that the current search results are still insufficient to determine the target of the current search; in that case, you should continue planning the next search step so that the clues in the search results can eventually support deriving '{target}'. It is also possible that the target inferred from the current search results is not the correct '{target}'; in that case, you need to include a self-correction step in your solution process, explain why the current inferred target may be incorrect, then rethink the tool calls and continue searching until the clues are sufficient to derive '{target}'.
Note that the '{statement}' and '{target}' provided here are only for verifying the correctness of the process you are writing. You must not reveal this provided information at any point in your solution process; instead, you must continue writing the response as if you were a responder who still does not know the target.)"""
        )


def build_hop_state(
    *,
    question: str,
    gold_answer: str,
    hop_chain: list[dict[str, Any]],
    judge_model_alias: str | None,
    judge_max_tokens: int,
) -> ManualReActHopState:
    initial_hop_index = 0
    first_hop = hop_chain[0] if hop_chain else {}
    if isinstance(first_hop, dict):
        src_node_id = str(first_hop.get("src_node_id") or "").strip().lower()
        if src_node_id.startswith("text_"):
            initial_hop_index = -1
    state = ManualReActHopState(
        question=str(question or "").strip(),
        gold_answer=str(gold_answer or "").strip(),
        hop_chain=list(hop_chain or []),
        current_hop_index=initial_hop_index,
        judge=HopJudgeConfig(
            model_alias=(judge_model_alias or "").strip() or None,
            max_tokens=max(1, int(judge_max_tokens)),
        ),
    )
    state.advance_past_image_hops()
    return state


def maybe_advance_hop(
    *,
    state: ManualReActHopState,
    trajectory_messages: list[dict[str, Any]],
) -> None:
    model_alias = state.judge.model_alias
    if not model_alias:
        return
    try:
        LLM_WORKER.get_model(model_alias)
    except Exception:
        return

    if state.in_initial_source_stage():
        first_hop = state.first_hop()
        if first_hop is None:
            return
        prompt = PROMPT_CHECK_INITIAL_SOURCE
        payload = {
            "question": state.question,
            "initial_source_stage": {
                "source": str(first_hop.get("source") or ""),
                "source_node_id": str(first_hop.get("src_node_id") or ""),
                "first_hop_statement": str(first_hop.get("statement") or ""),
                "first_hop_relation": str(first_hop.get("relation") or ""),
                "first_hop_target": str(first_hop.get("target") or ""),
            },
            "trajectory": _trajectory_text(trajectory_messages),
        }
    else:
        hop = state.current_hop()
        if hop is None:
            return
        prompt = PROMPT_CHECK_CURRENT_HOP
        payload = {
            "question": state.question,
            "current_hop": {
                "hop_index": int(hop.get("hop_index") or state.current_hop_index),
                "source": str(hop.get("source") or ""),
                "relation": str(hop.get("relation") or ""),
                "target": str(hop.get("target") or ""),
                "statement": str(hop.get("statement") or ""),
            },
            "trajectory": _trajectory_text(trajectory_messages),
        }
    try:
        parsed = LLM_WORKER.generate_json(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=prompt),
                    ModelMessage(role="user", content=json.dumps(payload, ensure_ascii=False, indent=2)),
                ],
                response_format={"type": "json_object"},
                max_tokens=state.judge.max_tokens,
                metadata={"trace_label": f"manual_react_check_hop:{state.current_hop_index}"},
            )
        )
    except Exception:
        return

    if bool(parsed.get("found")):
        state.advance()


def _trajectory_text(messages: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for message in messages:
        role = str(message.get("role") or "")
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
        else:
            text = json.dumps(content, ensure_ascii=False, indent=2).strip()
        if role == "tool":
            role = f"tool[{str(message.get('name') or '').strip()}]"
        if text:
            lines.append(f"{role}: {text}")
    return "\n\n".join(lines).strip()


def _is_image_target_hop(hop: dict[str, Any] | None) -> bool:
    if not isinstance(hop, dict):
        return False
    dst_node_id = str(hop.get("dst_node_id") or "").strip().lower()
    return dst_node_id.startswith("image_")
