"""Lightweight verification-guidance helpers for manual ReAct trajectory generation."""

from __future__ import annotations

import base64
import json
import mimetypes
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any

from synthesis.model_worker import LLM_WORKER
from synthesis.model_worker import ModelMessage
from synthesis.model_worker import ModelRequest

PROMPT_CHECK_CURRENT_HOP = """
Next, you will be given a multi-turn interaction from an agent. In this interaction, the agent is inferring what the target entity is based on a set of clues. You will also be given the correct answer for that target entity, and your job is to determine whether the agent has correctly inferred the target.
Rules:
If the agent has not inferred the correct answer — for example, if it has not yet reached a conclusion, is still searching for clues, or has finished searching but inferred an incorrect target — then we consider that the agent has not found the target, and you should judge it as false.
If the agent has inferred the target based on the clues and the evidence is sufficient, then you should judge it as true.
If the target is an image, you need to determine comprehensively whether the model has found the target based on the agent’s image search query, the search results, and whether the agent actually opened and viewed the image (that is, whether the image appears in the context).
Please return strict JSON:
{
"found": true/false,
"reason": "Give the reason for your judgment"
}
"""

PROMPT_CHECK_INITIAL_SOURCE = """
Next, you will be given a multi-turn interaction from an agent. In this interaction, the agent is inferring what the target entity is based on a set of clues. You will also be given the correct answer for that target entity, and your job is to determine whether the agent has correctly inferred the target.
Rules:
If the agent has not inferred the correct answer — for example, if it has not yet reached a conclusion, is still searching for clues, or has finished searching but inferred an incorrect target — then we consider that the agent has not found the target, and you should judge it as false.
If the agent has inferred the target based on the clues and the evidence is sufficient, then you should judge it as true.
If the target is an image, you need to determine comprehensively whether the model has found the target based on the agent’s image search query, the search results, and whether the agent actually opened and viewed the image (that is, whether the image appears in the context).
Please return strict JSON:
{
"found": true/false,
"reason": "Give the reason for your judgment"
}
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
        if self.current_hop_index < 0 or self.current_hop_index >= len(self.hop_chain):
            return None
        hop = self.hop_chain[self.current_hop_index]
        return hop if isinstance(hop, dict) else None

    def has_remaining_hops(self) -> bool:
        return self.current_hop() is not None

    def advance(self) -> None:
        if self.current_hop_index < len(self.hop_chain):
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
    return state


def maybe_advance_hop(
    *,
    state: ManualReActHopState,
    trajectory_messages: list[dict[str, Any]],
    image_registry: dict[str, Any] | None = None,
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
        user_prompt = _build_initial_source_check_prompt(
            question=state.question,
            trajectory_messages=trajectory_messages,
            source=str(first_hop.get("source") or ""),
            image_registry=image_registry,
        )
    else:
        hop = state.current_hop()
        if hop is None:
            return
        prompt = PROMPT_CHECK_CURRENT_HOP
        user_prompt = _build_current_hop_check_prompt(
            question=state.question,
            trajectory_messages=trajectory_messages,
            statement=str(hop.get("statement") or ""),
            target=str(hop.get("target") or ""),
            image_registry=image_registry,
        )
    try:
        parsed = LLM_WORKER.generate_json(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=prompt),
                    ModelMessage(role="user", content=user_prompt),
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


def _trajectory_content_parts(
    messages: list[dict[str, Any]],
    *,
    image_registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    content_parts: list[dict[str, Any]] = []
    for message in messages:
        role = str(message.get("role") or "")
        if role not in {"assistant", "tool"}:
            continue
        if role == "assistant":
            content_parts.append({"type": "text", "text": "Assistant:\n"})
            content_parts.extend(_judge_content_parts(message.get("content"), image_registry=image_registry))
        else:
            tool_name = str(message.get("name") or "tool").strip()
            content_parts.append({"type": "text", "text": f"Tool Result ({tool_name}):\n"})
            content_parts.extend(
                _judge_tool_content_parts(
                    message.get("content"),
                    image_registry=image_registry,
                )
            )
        content_parts.append({"type": "text", "text": "\n\n"})
    while content_parts and content_parts[-1].get("type") == "text" and not str(content_parts[-1].get("text") or "").strip():
        content_parts.pop()
    return content_parts


def _judge_tool_content_parts(content: Any, *, image_registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    parts = _judge_content_parts(content, image_registry=image_registry)
    if isinstance(content, str):
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            for key in ("image_id", "cropped_image_id"):
                image_id = parsed.get(key)
                image_part = _build_image_part_from_source(image_id, image_registry=image_registry)
                if image_part is not None:
                    parts.append({"type": "text", "text": "\n"})
                    parts.append(image_part)
            for key in ("image_url", "cropped_image_url"):
                image_source = parsed.get(key)
                image_part = _build_image_part_from_source(image_source, image_registry=image_registry)
                if image_part is not None:
                    parts.append({"type": "text", "text": "\n"})
                    parts.append(image_part)
    return parts


def _judge_content_parts(content: Any, *, image_registry: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if content in (None, ""):
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content.strip()}] if content.strip() else []
    if isinstance(content, list):
        parts: list[dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                text = str(item).strip()
                if text:
                    parts.append({"type": "text", "text": text})
                continue
            part_type = str(item.get("type") or "")
            if part_type in {"text", "input_text"}:
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append({"type": "text", "text": text})
                continue
            if part_type in {"image_url", "image", "input_image", "image_path", "image_ref"}:
                source = (
                    item.get("image")
                    or item.get("path")
                    or item.get("url")
                    or item.get("image_url")
                    or item.get("ref")
                )
                image_part = _build_image_part_from_source(source, image_registry=image_registry)
                if image_part is not None:
                    parts.append(image_part)
                continue
            parts.append({"type": "text", "text": json.dumps(item, ensure_ascii=False, indent=2)})
        return parts
    if isinstance(content, (dict, tuple)):
        return [{"type": "text", "text": json.dumps(content, ensure_ascii=False, indent=2).strip()}]
    text = str(content).strip()
    return [{"type": "text", "text": text}] if text else []


def _build_initial_source_check_prompt(
    *,
    question: str,
    trajectory_messages: list[dict[str, Any]],
    source: str,
    image_registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    content_parts: list[dict[str, Any]] = []
    if str(question).strip():
        content_parts.append({"type": "text", "text": f"Question:\n{question.strip()}\n\n"})
    content_parts.extend(_trajectory_content_parts(trajectory_messages, image_registry=image_registry))
    content_parts.append(
        {
            "type": "text",
            "text": "----- Target to be verified -----\n" f"Check whether the agent inferred '{source}'.",
        }
    )
    return content_parts


def _build_current_hop_check_prompt(
    *,
    question: str,
    trajectory_messages: list[dict[str, Any]],
    statement: str,
    target: str,
    image_registry: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    content_parts: list[dict[str, Any]] = []
    if str(question).strip():
        content_parts.append({"type": "text", "text": f"Question:\n{question.strip()}\n\n"})
    content_parts.extend(_trajectory_content_parts(trajectory_messages, image_registry=image_registry))
    content_parts.append(
        {
            "type": "text",
            "text": (
                "----- Target to be verified -----\n"
                f"information statement: {statement}\n"
                f"check whether the agent inferred the target: {target}"
            ),
        }
    )
    return content_parts


def _build_image_part_from_source(
    source: Any,
    *,
    image_registry: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    resolved = _resolve_image_payload(source, image_registry=image_registry)
    image_url = _image_source_to_model_url(resolved)
    if not image_url:
        return None
    return {"type": "image_url", "image_url": image_url}


def _resolve_image_payload(source: Any, *, image_registry: dict[str, Any] | None = None) -> Any:
    if isinstance(source, str) and image_registry and source in image_registry:
        return image_registry[source]
    if isinstance(source, dict) and "url" in source:
        return source.get("url")
    if isinstance(source, dict) and "image_url" in source:
        image_url = source.get("image_url")
        if isinstance(image_url, dict):
            return image_url.get("url")
        return image_url
    return source


def _image_source_to_model_url(source: Any) -> str | None:
    if not source:
        return None
    if isinstance(source, str):
        candidate = source.strip()
        if not candidate:
            return None
        if candidate.startswith(("http://", "https://", "data:image")):
            return candidate
        path = Path(candidate)
        if path.exists() and path.is_file():
            mime_type, _ = mimetypes.guess_type(str(path))
            mime_type = mime_type or "image/png"
            payload = base64.b64encode(path.read_bytes()).decode("utf-8")
            return f"data:{mime_type};base64,{payload}"
    return None
