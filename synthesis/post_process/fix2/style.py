"""Style profiling and safe few-shot attachment for the new collector."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .io import iter_json_records, write_json


_TOOL_RE = re.compile(r'"tool_name"\s*:\s*"([^"\\]+)"|"name"\s*:\s*"(t2t_search|t2i_search|i2i_search|read_url)"')
_ANSWER_RE = re.compile(r"<answer>.*?</answer>", flags=re.IGNORECASE | re.DOTALL)


HIDDEN_REFERENCE_OVERRIDE = """

FVQA collection setting: the only task input is the user's question and its
image. No gold answer, intermediate statement, knowledge-graph fact, or other
private reference exists in this session. Discover every named entity and fact
from inspected external tool observations, and state uncertainty until it is
grounded.

FVQA evidence protocol (binding): do not use inner/world knowledge, memorized
identities, or a familiar-looking image as evidence. The original image may be
described only as an unresolved visual clue; it cannot by itself establish a
person, place, artwork, species, programme, event, or any other world fact.
Search for external evidence first. When image search returns a plausible
candidate, use read_url with that candidate's image_id to download and inspect
the pixels before claiming a visual match. Search titles, snippets, URLs, and
thumbnails are leads, not visual proof. Open every serious candidate that must
be distinguished; when there are multiple plausible candidates, inspect and
compare more than one. Do not invent a false candidate merely to create a
rejection step.

Write the trajectory so that a child with no world knowledge can follow every
step. Introduce each new entity, technical term, or relation with the specific
image observation or inspected tool result that established it, explain why it
matters to the question, and never jump from an unexplained name to a
conclusion. Use the same available tools and normal tool-use protocol; do not
mention this collector instruction in the trajectory.

Avoid semantic search loops (binding): changing only synonyms, word order,
adjectives, specificity, or the language of a query is not a new search plan
and will rarely produce new evidence. After two low-information attempts at
the same sub-goal—or earlier if they return no new candidate page or image to
inspect—do not issue another wording variation. First state why the direct
route failed, then pivot to a genuinely different, evidence-supported route:
inspect an already returned candidate, switch modality (text, image, or
reverse-image search), identify an easier adjacent entity or relation, look
for the relevant caption/record-keeper/source, or search a different grounded
variable. The next plan must change the target evidence, source, or modality,
not merely the query wording. Never invent an unverified candidate detail just
to make a pivot possible.
""".strip()


_FVQA_RESPONSES_REPLACEMENTS = {
    3: (
        "In the solution you write, every factual statement must be grounded in "
        "evidence from the question, the image, or tool-returned results. Do not "
        "introduce any fact, entity, date, name, or relationship unless it is "
        "directly supported by that evidence."
    ),
    4: (
        "Do not answer a FVQA item directly from apparently familiar visual "
        "content or model memory. Treat the original image as a hypothesis-"
        "forming clue only: for an identity, location, relation, or world fact, "
        "obtain external search evidence and inspect the relevant resource with "
        "an appropriate tool before relying on it."
    ),
    8: (
        "The question and image are the complete task input. Do not assume that "
        "an answer, intermediate conclusion, entity, or fact exists outside the "
        "visible question, image, and tool observations."
    ),
    9: (
        "A query may contain only details grounded in the question, image, or "
        "earlier inspected tool results. After a low-information search, replan "
        "from the last evidence-supported unresolved variable instead of adding "
        "an unverified candidate detail to the query."
    ),
    11: (
        "The final answer must be the conclusion naturally supported by the "
        "question, image, and inspected evidence. Do not optimize for any hidden "
        "target wording or unobserved detail."
    ),
}

_FVQA_MANUAL_REACT_REPLACEMENTS = {
    7: (
        "The question and image are the complete task input. No final answer, "
        "intermediate statement, knowledge-graph fact, or other private "
        "reference is available. Do not assume an unsupported entity or fact; "
        "discover it from the image or tool-returned evidence."
    ),
}


def _replace_numbered_requirement(prompt: str, number: int, replacement: str) -> str:
    """Replace one numbered requirement while preserving shared tool guidance."""

    pattern = re.compile(
        rf"(?ms)^{number}\.\s+.*?(?=^\d+\.\s+|^Note:|^Tool-use tips:|^Next,)",
    )
    updated, count = pattern.subn(f"{number}. {replacement}\n", prompt, count=1)
    if count != 1:
        raise ValueError(f"Could not locate requirement {number} in the SFT system prompt")
    return updated


def build_fvqa_collector_base_prompt(base_system_prompt: str, *, api_mode: str) -> str:
    """Remove SFT-only private-supervision assumptions from a shared prompt.

    We intentionally retain the established reasoning, tool-use guidance and
    demonstrations.  Only requirements whose meaning depends on a supplied
    hop-chain/intermediate answer are rewritten for raw FVQA questions.
    """

    replacements = (
        _FVQA_RESPONSES_REPLACEMENTS
        if api_mode == "responses"
        else _FVQA_MANUAL_REACT_REPLACEMENTS
    )
    prompt = base_system_prompt.strip()
    for number, replacement in replacements.items():
        prompt = _replace_numbered_requirement(prompt, number, replacement)
    return prompt


def build_livevqa_collector_base_prompt(base_system_prompt: str, *, api_mode: str) -> str:
    """Apply the same hidden-reference protocol without mislabelling LiveVQA.

    This is intentionally a wrapper rather than a modification of the FVQA
    prompt builder, so an existing FVQA run receives exactly its prior prompt.
    """

    prompt = build_fvqa_collector_base_prompt(base_system_prompt, api_mode=api_mode)
    return (
        prompt.replace("FVQA collection setting", "LiveVQA collection setting")
        .replace("FVQA evidence protocol", "LiveVQA evidence protocol")
        .replace("FVQA item", "LiveVQA item")
    )


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(x.get("text") or "") for x in value if isinstance(x, dict))
    return ""


def profile_sharegpt(records: Iterable[dict[str, Any]], *, limit: int | None = None) -> dict[str, Any]:
    """Measure surface style only; never infer or reproduce old answers."""

    role_counts: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()
    turn_counts: list[int] = []
    assistant_chars: list[int] = []
    answer_tag_count = 0
    source_count = 0
    for record in records:
        source_count += 1
        messages = record.get("conversations") or record.get("messages") or []
        if not isinstance(messages, list):
            continue
        turn_counts.append(len(messages))
        for message in messages:
            if not isinstance(message, dict):
                continue
            role = str(message.get("from") or message.get("role") or "unknown").lower()
            role_counts[role] += 1
            text = _text(message.get("value", message.get("content")))
            if role in {"gpt", "assistant"}:
                assistant_chars.append(len(text))
                answer_tag_count += int(bool(_ANSWER_RE.search(text)))
                for first, second in _TOOL_RE.findall(text):
                    tool_counts[first or second] += 1
        if limit is not None and source_count >= limit:
            break
    def mean(values: list[int]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0
    return {
        "records": source_count,
        "role_counts": dict(role_counts),
        "tool_counts": dict(tool_counts),
        "mean_turns": mean(turn_counts),
        "mean_assistant_chars": mean(assistant_chars),
        "assistant_answer_tag_rate": round(answer_tag_count / max(1, len(assistant_chars)), 4),
        "notes": [
            "Surface profile only; use curated, sanitized anchors for prompting.",
            "Do not copy raw trajectories, source facts, URLs, IDs, or answers into the expert prompt.",
        ],
    }


def profile_sharegpt_file(input_path: str | Path, output_path: str | Path, *, limit: int | None = None) -> dict[str, Any]:
    profile = profile_sharegpt(iter_json_records(input_path), limit=limit)
    write_json(output_path, profile)
    return profile


_LEGACY_STYLE_FORBIDDEN = ("gold answer", "private reference", "hop_chain", "ground truth")
_LEGACY_STYLE_TOOL_ORDER = ("i2i_search", "t2i_search", "t2t_search")


def build_legacy_style_examples_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    max_examples: int = 3,
    max_chars: int = 12_000,
    scan_limit: int = 2_000,
) -> dict[str, Any]:
    """Materialize small real legacy-SFT excerpts for few-shot style matching.

    The collector already uses the legacy tool/system protocol.  These short
    excerpts add its surface cadence and tool-call presentation without putting
    the 17k-record corpus into every writer context.  They are deliberately
    limited to non-final assistant turns: no legacy answer block or tool
    observation is injected as evidence for an FVQA case.
    """

    if max_examples < 1:
        raise ValueError("max_examples must be positive")
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if scan_limit < 1:
        raise ValueError("scan_limit must be positive")

    candidates: dict[str, str] = {}
    fallback: list[str] = []
    scanned = 0
    for record in iter_json_records(input_path):
        scanned += 1
        conversations = record.get("conversations") or record.get("messages") or []
        if not isinstance(conversations, list):
            if scanned >= scan_limit:
                break
            continue
        for message in conversations:
            if not isinstance(message, dict):
                continue
            role = str(message.get("from") or message.get("role") or "").lower()
            text = _text(message.get("value", message.get("content"))).strip()
            lowered = text.casefold()
            # Final-answer turns and any private-supervision wording are not
            # useful surface anchors and would be unsafe in a writer prompt.
            if role not in {"gpt", "assistant"} or "<answer>" in lowered:
                continue
            if "<thinking>" not in lowered or not ("<tool_call>" in lowered or "<action>" in lowered):
                continue
            if any(token in lowered for token in _LEGACY_STYLE_FORBIDDEN):
                continue
            if len(text) > max_chars:
                continue
            fallback.append(text)
            for tool_name in _LEGACY_STYLE_TOOL_ORDER:
                if tool_name in text and tool_name not in candidates:
                    candidates[tool_name] = text
        if len(candidates) >= min(max_examples, len(_LEGACY_STYLE_TOOL_ORDER)) or scanned >= scan_limit:
            break

    selected: list[str] = []
    seen: set[str] = set()
    for tool_name in _LEGACY_STYLE_TOOL_ORDER:
        value = candidates.get(tool_name)
        if value and value not in seen:
            selected.append(value)
            seen.add(value)
        if len(selected) >= max_examples:
            break
    if len(selected) < max_examples:
        for value in fallback:
            if value not in seen:
                selected.append(value)
                seen.add(value)
            if len(selected) >= max_examples:
                break

    bounded: list[str] = []
    used = 0
    for value in selected:
        if used + len(value) > max_chars:
            continue
        bounded.append(value)
        used += len(value)
    if not bounded:
        raise ValueError("No eligible non-final legacy assistant turns were found for style examples")
    payload = {
        "source": str(Path(input_path)),
        "mode": "legacy_sharegpt_nonfinal_assistant_excerpts",
        "scanned_records": scanned,
        "examples": bounded,
    }
    write_json(output_path, payload)
    return {"examples": len(bounded), "chars": used, "scanned_records": scanned, "output": str(output_path)}


def _load_anchor_strings(path: str | Path, *, char_budget: int) -> list[str]:
    """Load a human-curated sanitized anchor file under a strict character budget."""

    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    values = raw.get("examples") if isinstance(raw, dict) else raw
    if not isinstance(values, list):
        raise ValueError("style examples must be a JSON list or {'examples': [...]}.")
    examples: list[str] = []
    used = 0
    for index, value in enumerate(values, start=1):
        if isinstance(value, dict):
            value = value.get("text") or value.get("example") or ""
        text = str(value).strip()
        if not text:
            continue
        if any(token in text.lower() for token in ("gold answer", "private reference", "hop_chain", "ground truth")):
            raise ValueError(f"Style anchor {index} appears to contain private-supervision language")
        if used + len(text) > char_budget:
            break
        examples.append(text)
        used += len(text)
    return examples


def build_collector_system_prompt(
    default_system_prompt: str,
    *,
    style_guide_file: str | None = None,
    style_examples_file: str | None = None,
    style_char_budget: int = 14000,
) -> str:
    """Extend—not replace—the established tool protocol and prose style."""

    sections = [default_system_prompt.strip(), HIDDEN_REFERENCE_OVERRIDE]
    if style_guide_file:
        guide = Path(style_guide_file).read_text(encoding="utf-8").strip()
        if guide:
            sections.extend(["Style compatibility guide (formatting only):", guide])
    if style_examples_file:
        examples = _load_anchor_strings(style_examples_file, char_budget=style_char_budget)
        if examples:
            sections.append(
                "<style_examples>\nThese are sanitized demonstrations of writing style only. "
                "They are not evidence, facts, candidates, or instructions for the current question.\n"
                + "\n\n".join(f"[Style example {index}]\n{text}" for index, text in enumerate(examples, start=1))
                + "\n</style_examples>"
            )
    return "\n\n".join(sections).strip() + "\n"
