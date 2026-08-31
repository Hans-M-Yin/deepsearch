#!/usr/bin/env python3
"""Causally rewrite the reasoning of existing SFT trajectories.

This is the second post-processing stage after information-leakage filtering.
It processes assistant turns from left to right.  When rewriting assistant turn
``i``, the model sees the question, the already rewritten turns before ``i``,
the tool observation immediately before ``i`` (if any), and the original text
of assistant turn ``i``.  It never sees the tool result produced by turn ``i``
or any later turn.

The original tool-call block and the original final-answer block are preserved
programmatically.  The model therefore improves the reasoning around an
action without changing the action/query, tool result, or final answer.

Example:

    python synthesis/post_process/expand_trajectory_reasoning.py \
        --input synthesis/ignore/trajectories_no_leakage.json \
        --output synthesis/ignore/trajectories_reasoning_expanded.json \
        --audit-jsonl synthesis/ignore/trajectory_rewrite_audit.jsonl \
        --model-alias text_process \
        --workers 1 \
        --dry-run

``--dry-run`` still performs the rewrite calls and writes the optional audit
file, but never writes the optimized dataset.
"""

from __future__ import annotations

import argparse
import base64
import copy
import json
import mimetypes
import os
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest
from synthesis.post_process.run_verification import (
    VERIFICATION_SYSTEM_PROMPT,
    _build_verification_context,
    _select_api_mode,
)
from synthesis.sft.api_tools import OpenAIToolAgent, OpenAIToolAgentConfig, ToolRuntimeContext
from synthesis.post_process.filter_trajectory_information_leakage import (
    _append_jsonl,
    _iter_records,
    _read_jsonl,
    _record_id,
    _write_json_array,
)
from synthesis.post_process.terminal_visual_evidence import (
    VqaTerminalImageResolver,
    run_terminal_visual_check,
)


_POST_PROCESS_FIXED_REQUEST_ID = "3200636808"
_OUTPUT_LOCK = threading.Lock()


class WriterAliasPool:
    """Thread-safe round-robin selector for equivalent writer aliases."""

    def __init__(self, aliases: Iterable[str]) -> None:
        self.aliases = tuple(str(alias).strip() for alias in aliases if str(alias).strip())
        if not self.aliases:
            raise ValueError("WriterAliasPool requires at least one model alias.")
        self._next_index = 0
        self._lock = threading.Lock()

    def next_alias(self) -> str:
        with self._lock:
            alias = self.aliases[self._next_index]
            self._next_index = (self._next_index + 1) % len(self.aliases)
        return alias


def _next_writer_alias(model_alias: str | WriterAliasPool) -> str:
    return model_alias.next_alias() if isinstance(model_alias, WriterAliasPool) else model_alias


def _model_worker_metadata(trace_label: str) -> dict[str, str]:
    """Attach the routing/cache metadata used by the SFT API tools."""

    return {
        "trace_label": trace_label,
        "session_id": _POST_PROCESS_FIXED_REQUEST_ID,
        "prompt_cache_key": _POST_PROCESS_FIXED_REQUEST_ID,
        "user_id": _POST_PROCESS_FIXED_REQUEST_ID,
        "x_tt_logid": _POST_PROCESS_FIXED_REQUEST_ID,
    }


OPTIMIZER_SYSTEM_PROMPT = """You are a reasoning-polishing editor. You will be given: the original question, the reasoning and tool results from turns 1 through i-1, and the latest turn of the model's answer that you are to revise. While keeping everything logically consistent with what comes before and after, polish this latest turn so it is more complete, more rigorous, and reads more like a person genuinely searching and thinking as they go.

# Iron Rules (violating any one makes the output invalid)
1. Do NOT change this turn's tool call (name, arguments, and id all stay exactly as they are), do NOT change the final conclusion, and do NOT change the direction of the next step's reasoning.
2. Do NOT fabricate anything the tools did not return. Anything "written in but not actually visible in the image / results" counts as fabrication. You are polishing, not rewriting.
3. Common sense may be used only to "identify and rule out," never to "supply an answer." That is: within candidates or results that are already established, you may use public common sense (e.g. a person's well-known distinguishing features, a place's basic attributes) to judge who is who and to eliminate distractors that fit the wording but are misleading, spelling out the choice against the question's specific conditions — this is necessary for filtering out misleading options and is allowed. But you may NEVER use common sense, or knowledge you happen to remember, to directly arrive at the answer or to fill in a key intermediate conclusion that should have come from search.
4. Weak evidence must be honestly called weak. If this turn genuinely amounts to "verification didn't succeed, so I'm proceeding on common sense for now," say so plainly; never polish it into "confirmed / verified." Laundering weak evidence into strong evidence is poisoning the data.
5. Polishing reasoning does not mean producing repetitive or redundant output; it means supplying the missing logic, bridging the places where the original answer jumps ahead or leaves steps implicit, and strengthening the reasoning while preserving a reasonable level of fluency.

# Things To Do
1. Polish the clue analysis. This latest turn is often too terse — it typically grabs one piece of information from the previous result and calls the next tool without much explanation. Flesh it out: state which of the previous returns were useful and which weren't; whether there were any distractors or ambiguous items that fit the wording but are actually misleading; and why the chosen item wins over those distractors — tie the reasoning to the question's concrete conditions (date, material, place, identity, left-right position, etc.), not just "this one seemed more relevant." Dismiss the obviously irrelevant ones in a single clause; don't pad.
2. Polish the logical reasoning. Draw on the earlier turns and on the question's requirements, goals, and clues; make the cross-checking and analysis explicit so the logic is tighter. Only fill in gaps or leaps that the original answer actually has — don't add filler just to look fuller.
3. Keep the language natural and conversational. Write like a person searching and deciding on their own, not like a report or a set of clauses; no piles of bullets. Preserve every [ph_...] placeholder exactly, character for character. Output only the polished version of this turn.
4. If the current turn is the final turn (that is, it contains <answer></answer>), then please insert a <thinking></thinking> section before <answer>. In that thinking section, continue examining the previous turn’s tool result. If the result is image content, then you should expand the reasoning by connecting the image content to the final answer—for example, by providing some explanation or elaboration of the image based on the answer content (but without changing the answer)—and then confirm the final answer and the reason for it. After that, still within the thinking section, summarize the entire reasoning process, including a review of the question requirements, what was being asked, the clues, and the reasoning chain. Finally, close the thinking section and reuse the original answer inside <answer></answer>. Make sure the full reasoning process is coherent, complete, and transitions naturally into the answer.
5. Handle possible distractors in this order. First, inspect the question, the complete preceding history, the immediately preceding tool observation, ordinary public common sense, and straightforward logical comparison. If any candidate can already be ruled out by a concrete contradiction — for example, a wrong date, place, role, relationship, object type, material, event, or other stated condition — explain that contradiction in ordinary polishing and do not issue a verification request. If it is merely generic or loosely related noise that would not change the current reasoning direction, dismiss it briefly in ordinary polishing and do not issue a request. Only issue a verification request when all of the following are true: (a) the immediately preceding observation is a multi-result search and contains a specific unselected, externally searchable candidate; (b) that candidate genuinely fits the current referent or sub-goal well enough to be a live competing interpretation; (c) the current original assistant turn does not choose or already rule out that candidate; (d) no direct contradiction can be established from the currently available question, history, search results, common sense, or simple logic; and (e) if the candidate were actually the referent, the next reasoning direction, tool target, or final answer would materially change. Do not invent a distractor from memory, and do not use a candidate that appeared only outside the immediately preceding search observation. If all conditions hold, do not polish this turn yet. Instead, output exactly one structured verification request in the following form, with no other text: <verification_request>{"referent":"...","object_to_verify":"...","distractor_result_index":0,"verification_goal":"...","verification_scope":"external_candidate"}</verification_request>. The 'referent' must be the exact phrase or an unambiguous description in the original question. The 'object_to_verify' must be a concise candidate name taken directly from the stated search result — do not add a longer descriptive clause, inferred attributes, or a replacement answer. The 'distractor_result_index' is the zero-based index of that candidate in the immediately preceding search observation. The 'verification_goal' must identify the unresolved question condition that external research should test about this one candidate; it must not name or assume a correct alternative. This request is for named entities, places, artworks, organizations, historical records, or other independently searchable objects. It must not be an image region, hand, color, animal, count, pose, left/right position, spatial assignment, or any other fact that can only be resolved by directly looking at the provided image. Do not issue a verification request for pure visual recognition or image inspection; handle that uncertainty in ordinary polishing and state the evidence limitation honestly. The verification goal must seek an external, non-visual property such as date, place, role, relationship, participation, organization, material, provenance, or object type. It must never ask to compare an image, portrait, photograph, appearance, pose, color, hand, left/right position, or any other visual feature. Do not try to predict whether verification will succeed within five tool calls; the tool budget is the verifier’s execution constraint, not a trigger condition. Use a request only for a real, unresolved ambiguity that matters to the next reasoning step; otherwise follow the polishing instructions above.
6. For a final turn containing <answer></answer>, inspect the immediately preceding observation. If it provides an image, issue a terminal-image check only when either: (1) the original trajectory explicitly says this current image is too small, low-resolution, blurry, unclear, or otherwise makes the final visual answer difficult or uncertain; or (2) you personally cannot derive the visual content asserted by the final <answer></answer> from this image. Do not polish this turn yet. Instead output exactly one of the following, with no other text: `<visual_evidence_check_request>{"scope":"terminal_image_quality","reason":"trajectory_quality_complaint"}</visual_evidence_check_request>` for (1), or `<visual_evidence_check_request>{"scope":"terminal_image_quality","reason":"writer_cannot_read"}</visual_evidence_check_request>` for (2). Never use this for an old image, a webpage/text failure, a URL containing words such as thumbnail, a normal uncertainty, or an image that is clear enough to support the requested feature.

# Examples

## Example 1 — Spell out the distractor + look back at the question
BEFORE (thin — only mentions the winner):
"The reverse-image search returned several records. One describes an 'East facing pediment on Northern Library' with a frieze of the burning of Khandava forest, which matches the image. I will inspect that source page."

AFTER:
"The reverse-image search pulled back a whole batch, and the one that fits best is the record describing an 'East facing pediment on Northern Library' with a frieze of the burning of the Khandava forest — the library-like building and that narrative relief in the image both line up with it, so I'll go with it for now; what this step is really settling is 'which temple is in the picture.' There's also a result titled 'Keleti Mebon' pointing to a different Angkor temple, which looks vaguely similar, but I won't take it at face value: titles like this are often noise, the same image gets embedded on unrelated pages all the time, and it doesn't match the burning-forest pediment the image actually shows, so I'm not picking it. The rest are generic Angkor travel pages with no caption specific to this pediment, so they're no help. So I'll open the source page of the one I chose and verify it rather than just trusting the title. As for what the question ultimately wants — the original name and its meaning — I haven't reached that yet; that's for the next hop."

Why this is right: it says why it picked that one and why it rejected the tempting "Keleti Mebon" distractor (tying it to the concrete condition — the burning-forest pediment), waves off the rest in one line, restates which condition is being solved right now, and doesn't touch the tool call or the conclusion — and it reads like a person thinking, not stiff.

## Example 2 — Be honest when verification fails (don't write a failed check up as a passed one)
BEFORE:
"The above tool results returned no useful content. Based on the right-hand figure's distinctive feature, we tentatively take him to be Hitler; the next step is to find his trial in the early 1920s."

AFTER:
"Both cropped reverse-image searches came up short: the first was all uniform-and-equipment pages, and the second, tightened onto the head, still gave nothing pointing to a specific person. So the cropping route can't independently pin down who's on which side here, and I can't treat that noise as confirmation. What's still solid is the earlier step — the Getty caption said this photo is Mussolini and Hitler; all that's unresolved is which of them is on which side. The man on the right has a very obvious toothbrush moustache, which is Hitler's single most recognizable feature; Mussolini and Hitler look quite different anyway, and the right-hand man's appearance is closer to Hitler's. So I'll make the reasonable inference that the one on the right is Hitler. I'm aware this is just a face-recognition guess for now, not a verified identification."

Why this is right: it completes the observation (says why both crops failed and why the noise is discarded), reconnects to the Getty fact established earlier, restates the small open question, and leaves a "can be walked back" hook; it uses the toothbrush moustache and the difference in appearance as public common sense to identify and rule out (which is allowed), but it does NOT use common sense to hand over any answer, and it does NOT fabricate details the image doesn't clearly show (e.g. no inventing specific uniforms or medals); and it doesn't inflate a failed verification into a confirmed one. Weak evidence is honestly called weak.
"""


INTEGRATION_SYSTEM_PROMPT = OPTIMIZER_SYSTEM_PROMPT + """

# Integration instructions
The verification request and process have already been run and are included in
the user message. Do not issue another verification request. The original
trajectory's main direction and original tool call remain unchanged. The verification
evidence only explains why the unselected candidate is not the referent; it
does not establish or name any replacement candidate as correct. Rewrite the
current main turn around that evidence and return exactly one JSON object with
these three fields:
{"before_verification":"...","verification_turns":["..."],"after_verification":"..."}

The `before_verification` field must explain why the current evidence leaves a
real ambiguity and why the verification is needed. The `after_verification`
field must explain what the verification search established, why the
distractor is ruled out, and how the main reasoning continues toward the
original next step. `verification_turns` must contain one rewritten reasoning
string for each verifier assistant turn that contains a tool call, in the same
order as those turns appear in the supplied verification trajectory. These
strings may explain the observation and the choice of the next search, but must
not contain `<tool_call>` blocks. The program will restore every original tool
call byte-for-byte. Do not rewrite, summarize, or omit any verification tool
result, and do not change the verifier's final YES/NO judgment. The program
will preserve those messages and all tool results exactly.

The original main tool call remains unchanged. The program will only insert
this verification trajectory when the verifier returns a complete,
evidence-backed NO. If verification returns YES, times out, or lacks usable
tool evidence, the program will skip this rewrite and preserve the original
turn unchanged.

Your entire response must be the valid JSON object described above. Output no
prose, Markdown fence, explanation, or extra keys before or after that object.
""".strip()


VISUAL_VALIDATED_FINAL_SYSTEM_PROMPT = OPTIMIZER_SYSTEM_PROMPT + """

# Terminal visual-evidence check
The immediately preceding downloaded image was independently checked with a
visual-only question and supports the stored final answer. Polish this final
turn normally, but state only visual details that are actually needed for the
answer. Remove any earlier complaint that this final image is too small,
blurred, low-resolution, unreadable, or insufficient; if a caveat remains
useful, replace it with a short, neutral qualification rather than repeating a
quality protest that the completed visual check has resolved. Do not mention
this quality-control check, hidden metadata, a graph, or any other external
processing.
""".strip()


VISUAL_REPAIRED_FINAL_SYSTEM_PROMPT = OPTIMIZER_SYSTEM_PROMPT + """

# Terminal visual-evidence repair
The immediately preceding image placeholder has been replaced by the original
terminal image node used when this question was constructed. A visual-only
check on that replacement image supports the stored final answer. Polish this
final turn normally and ground every visual claim in the immediately preceding
image only. When an old visual description is vague, speculative, or does not
match what the current image shows, delete or rewrite that description to
follow the current image; never preserve it merely to make the old prose sound
consistent. Do not invent an extra visual detail to justify the stored answer.
The stored final answer itself is preserved programmatically, so a genuine
core contradiction would have been filtered by the preceding visual check.
Do not mention replacement, quality control, a graph, metadata, or any hidden
processing.
""".strip()


@dataclass(slots=True)
class TurnDecision:
    action: str
    rewritten: str | None = None
    verification_request: dict[str, Any] | None = None
    visual_evidence_request: bool = False
    visual_evidence_reason: str | None = None
    editor_note: str = ""

_TAG_RE = re.compile(r"<(?P<tag>thinking|tool_call|answer)\b[^>]*>.*?</(?P=tag)>", re.DOTALL | re.IGNORECASE)
_VISUAL_ONLY_CANDIDATE_PATTERNS = (
    r"\b(?:left|right|far[- ]left|far[- ]right)\b",
    r"\bhand(?:s)?\b",
    r"\b(?:color|colour|green|red|blue|yellow|white|black)\b",
    r"\b(?:swan|goose|sea snake|serpent|eagle|bird)\b",
    r"\b(?:wreath|cape|glove|pose|posture)\b",
    r"\b(?:count|number of|how many)\b",
)
_VISUAL_VERIFICATION_GOAL_PATTERNS = (
    r"\b(?:visual(?:ly)?|visible|appearance|portrait|photograph|photo|picture|image)\b",
    r"\b(?:tail[- ]?fin|hand(?:s)?|colour|color|pose|posture|left|right|wreath)\b",
    r"\b(?:compare|inspect|identify|recognize|recognise|look at|shown)\b.*\b(?:image|photo|portrait|appearance|visual)\b",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Input ShareGPT JSON array or JSONL.")
    parser.add_argument("--output", type=Path, help="Optimized output dataset; required unless --dry-run.")
    parser.add_argument("--audit-jsonl", type=Path, help="Optional JSONL audit of every rewritten trajectory.")
    parser.add_argument(
        "--model-alias",
        default=os.environ.get("POST_PROCESS_MODEL_ALIAS") or os.environ.get("TEXT_PROCESS_MODEL") or "",
        help="Primary registered Model Worker alias used for writer and integration calls.",
    )
    parser.add_argument(
        "--model-alias-2",
        default="",
        help=(
            "Optional equivalent second writer alias. When set, writer and integration "
            "calls round-robin between --model-alias and this alias."
        ),
    )
    parser.add_argument(
        "--verification-model-alias",
        default=os.environ.get("VERIFICATION_MODEL_ALIAS") or "",
        help="Optional dedicated alias used only when the writer emits a verification request.",
    )
    parser.add_argument(
        "--visual-vqa-model-alias",
        default=os.environ.get("VISUAL_VQA_MODEL_ALIAS") or "",
        help="Dedicated Model Worker alias used to answer the local visual evidence question.",
    )
    parser.add_argument("--vqa-dir", type=Path, help="VQA run directory containing samples.jsonl/questions.jsonl.")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        help="Graph directory containing nodes.jsonl; required only when a triggered record lacks a terminal image URL in VQA metadata.",
    )
    parser.add_argument(
        "--visual-evidence-workdir",
        type=Path,
        default=ROOT / "synthesis/.ignore/terminal_visual_evidence",
        help="Cache directory for terminal image-node downloads used by visual evidence repair.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Optional writer/integration output budget. Omit to use the registered model alias default.",
    )
    parser.add_argument(
        "--history-window-turns",
        type=int,
        default=None,
        help=(
            "Only show the writer the most recent N completed assistant turns and "
            "their following observations. The original question and its initial "
            "image(s) are still supplied. By default the full causal history is used."
        ),
    )
    parser.add_argument("--verification-max-tokens", type=int, default=8000)
    parser.add_argument(
        "--verification-max-turns",
        type=int,
        default=5,
        help="Maximum verification tool-use rounds; the final YES/NO response is allowed after this limit.",
    )
    parser.add_argument(
        "--verification-workdir",
        type=Path,
        default=ROOT / "synthesis/.ignore/verification_runs",
        help=(
            "Directory for verification-run artifacts. The default preserves the "
            "existing standalone-script behavior."
        ),
    )
    parser.add_argument(
        "--debug-output",
        choices=("full", "verification_only", "silence", "slience"),
        default="full",
        help="Debug trace mode: full, verification_only, or silence (slience is accepted as an alias).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print each rewritten turn immediately; by default only the completed trajectory is printed.",
    )
    parser.add_argument(
        "--no-action",
        action="store_true",
        help="Only debug-print the selected original trajectories; do not call any model or modify them.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Print the complete original trajectory followed by the complete rewritten trajectory.",
    )
    parser.add_argument(
        "--verification-debug-turns",
        type=int,
        default=3,
        help="Number of original assistant turns to include after an integrated verification.",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel trajectories; turns within one trajectory stay serial.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--sample-id",
        "--sample_id",
        dest="sample_ids",
        action="append",
        default=[],
        help="Only process these record IDs. Repeat the option or provide comma-separated IDs.",
    )
    offset_group = parser.add_mutually_exclusive_group()
    offset_group.add_argument("--start", type=int, default=None, help="Legacy name for --offset.")
    offset_group.add_argument("--offset", type=int, default=None, help="Zero-based input offset.")
    parser.add_argument("--dry-run", action="store_true", help="Run rewriting but do not write the optimized dataset.")
    parser.add_argument("--resume", action="store_true", help="Resume completed rewritten trajectories from checkpoint state.")
    parser.add_argument(
        "--force-overwrite-state",
        action="store_true",
        help=(
            "Discard this run's rewrite checkpoint state and start it again. "
            "Only the derived state files are reset; the input dataset is never modified."
        ),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        help="Directory for resumable JSONL checkpoint state (default: derived from --audit-jsonl or --output).",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=20,
        help="Publish rewritten trajectory checkpoints after this many completed records (default: 20).",
    )
    return parser.parse_args()


def _get_conversation_key(record: dict[str, Any]) -> str:
    for key in ("conversations", "messages"):
        if isinstance(record.get(key), list):
            return key
    raise ValueError("Trajectory record has no conversations/messages list")


def _role_and_content(message: Any) -> tuple[str, str]:
    if not isinstance(message, dict):
        return "unknown", str(message)
    role = str(message.get("from") or message.get("role") or message.get("speaker") or "unknown")
    content = message.get("value")
    if content is None:
        content = message.get("content")
    if content is None:
        content = message.get("response_text")
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False)
    return role, str(content if content is not None else "")


def _raw_role_and_content(message: Any) -> tuple[str, Any]:
    """Return a trajectory message without flattening multimodal content."""

    if not isinstance(message, dict):
        return "unknown", str(message)
    role = str(message.get("from") or message.get("role") or message.get("speaker") or "unknown")
    content = message.get("value")
    if content is None:
        content = message.get("content")
    if content is None:
        content = message.get("response_text")
    return role, content if content is not None else ""


def _is_assistant(role: str) -> bool:
    return role.lower() in {"gpt", "assistant", "model"}


def _record_question(record: dict[str, Any]) -> str:
    """Return the explicit question, falling back to the first user turn."""

    question = str(record.get("question") or record.get("query") or "").strip()
    if question:
        return question
    try:
        messages = record[_get_conversation_key(record)]
    except (KeyError, ValueError):
        return ""
    for message in messages:
        role, content = _role_and_content(message)
        if role.lower() in {"human", "user"}:
            return content.strip()
    return ""


def _render_history(messages: list[Any]) -> str:
    chunks: list[str] = []
    for i, message in enumerate(messages):
        role, content = _role_and_content(message)
        chunks.append(f"[TURN {i}][{role}]\n{content}")
    return "\n\n".join(chunks)


def _chat_role(role: str) -> str:
    normalized = role.lower()
    if normalized in {"gpt", "assistant", "model"}:
        return "assistant"
    # ShareGPT trajectories store tool results as `observation`.  They do not
    # carry a tool_call_id, so they cannot safely be sent as an OpenAI `tool`
    # message.  Keep them as a separate user message instead of flattening the
    # entire history into one text blob.
    return "user"


def _resolve_image_path(image_path: str) -> Path | None:
    candidate = Path(image_path)
    if candidate.is_absolute():
        candidates = [candidate]
    else:
        # ShareGPT records store paths relative to the dataset directory, for
        # example ``images/dataset00_sharegpt_dataset_8k/...jpg``.  The
        # post-processor runs from the repository root, so checking only
        # ``ROOT / image_path`` silently turns every such image into an
        # unavailable placeholder.  Try the repository roots first for
        # backward compatibility, then each local sharegpt dataset root.
        bases = [ROOT, ROOT.parent]
        data_root = ROOT / "data"
        if data_root.is_dir():
            bases.append(data_root)
            bases.extend(
                sorted(
                    path
                    for path in data_root.iterdir()
                    if path.is_dir() and path.name.startswith("sharegpt_dataset")
                )
            )
        candidates = [base / candidate for base in bases]
    for resolved in candidates:
        if resolved.is_file():
            return resolved
    return None


def _image_content_part(image_path: str) -> dict[str, Any] | None:
    """Create an OpenAI-compatible image content part for a local trajectory image."""

    if image_path.startswith(("http://", "https://")):
        return {"type": "image_url", "image_url": {"url": image_path}}
    resolved = _resolve_image_path(image_path)
    if resolved is None:
        return None
    mime_type = _image_mime_type(resolved)
    if mime_type is None:
        return None
    encoded = base64.b64encode(resolved.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}


def _image_mime_type(path: Path) -> str | None:
    """Return an image MIME type accepted by OpenAI-compatible endpoints.

    Some environments do not register newer image suffixes in ``mimetypes``.
    Never emit ``application/octet-stream`` as an image data URL: vLLM rejects
    that MIME type with a 400 before the model sees the request.
    """

    guessed = mimetypes.guess_type(str(path))[0]
    if guessed and guessed.startswith("image/"):
        return guessed
    suffix_mimes = {
        ".avif": "image/avif",
        ".bmp": "image/bmp",
        ".gif": "image/gif",
        ".heic": "image/heic",
        ".heif": "image/heif",
        ".jpeg": "image/jpeg",
        ".jpg": "image/jpeg",
        ".png": "image/png",
        ".tif": "image/tiff",
        ".tiff": "image/tiff",
        ".webp": "image/webp",
    }
    return suffix_mimes.get(path.suffix.casefold())


def _observation_content(
    content: Any,
    image_path: str | None,
) -> Any:
    """Wrap an observation and attach its corresponding read_url image, if available."""

    image_part = _image_content_part(image_path) if image_path else None
    if isinstance(content, list):
        parts: list[Any] = [{"type": "text", "text": "<tool_observation>"}, *content]
        if image_part is not None:
            parts.append(image_part)
        parts.append({"type": "text", "text": "</tool_observation>"})
        return parts
    text = f"<tool_observation>\n{content}\n"
    if image_path and image_part is None:
        text += f"[Referenced image unavailable locally: {image_path}]\n"
    text += "</tool_observation>"
    if image_part is not None:
        return [{"type": "text", "text": text}, image_part]
    return text


def _image_placeholder_count_in_messages(messages: Iterable[Any]) -> int:
    """Count trajectory image placeholders while preserving their global order."""

    return sum(_role_and_content(message)[1].count("<image>") for message in messages)


def _initial_question_image_paths(record: dict[str, Any]) -> list[str]:
    """Return images attached before the first assistant response.

    Windowed history deliberately omits old messages, but it must not omit the
    supplied question image.  Image paths are globally aligned with ``<image>``
    placeholders, so only the placeholders before the first assistant turn are
    the initial question assets.
    """

    try:
        messages = record[_get_conversation_key(record)]
    except (KeyError, ValueError):
        return []
    image_count = 0
    for message in messages:
        role, content = _role_and_content(message)
        if _is_assistant(role):
            break
        image_count += content.count("<image>")
    return [str(path) for path in (record.get("images") or [])[:image_count]]


def _window_history_by_assistant_turns(
    history: list[Any],
    max_assistant_turns: int | None,
) -> tuple[list[Any], int, bool]:
    """Keep the last completed assistant/tool rounds and report omitted images.

    A "turn" here means one prior assistant response plus the observation(s)
    that follow it.  The current assistant response is never in ``history``.
    Returning an image offset keeps image placeholders aligned after old turns
    are removed.
    """

    if max_assistant_turns is None:
        return history, 0, False
    assistant_indices = [
        index for index, message in enumerate(history)
        if _is_assistant(_role_and_content(message)[0])
    ]
    if len(assistant_indices) <= max_assistant_turns:
        return history, 0, False
    start = assistant_indices[-max_assistant_turns]
    omitted = history[:start]
    return history[start:], _image_placeholder_count_in_messages(omitted), True


def _user_content_with_image(content: Any, image_path: str | None) -> Any:
    """Attach the next trajectory image to a user message when available."""

    image_part = _image_content_part(image_path) if image_path else None
    if image_part is None:
        return content
    if isinstance(content, list):
        return [*content, image_part]
    return [{"type": "text", "text": str(content)}, image_part]


def _build_optimizer_messages(
    record: dict[str, Any],
    record_index: int,
    assistant_index: int,
    history: list[Any],
    original_response: str,
    system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
    integration_context: Any | None = None,
    history_image_offset: int = 0,
    earlier_history_omitted: bool = False,
) -> list[ModelMessage]:
    question = _record_question(record)
    messages = [ModelMessage(role="system", content=system_prompt)]
    image_paths = [str(path) for path in (record.get("images") or [])]
    image_index = history_image_offset
    for message in history:
        role, content = _raw_role_and_content(message)
        image_path = None
        if "<image>" in str(content) and image_index < len(image_paths):
            image_path = image_paths[image_index]
            image_index += 1
        if role.lower() == "observation":
            content = _observation_content(content, image_path)
        else:
            content = _user_content_with_image(content, image_path)
        messages.append(ModelMessage(role=_chat_role(role), content=content))

    current_prompt_prefix = (
        "Earlier trajectory history has been omitted for this editing request. "
        "Use only the original question, its supplied image(s), and the recent "
        "history below; do not assume any omitted fact.\n\n"
        if earlier_history_omitted
        else ""
    )
    current_prompt = (
        current_prompt_prefix
        +
        f"Record id: {_record_id(record, record_index)}\n"
        f"Current assistant message index: {assistant_index}\n"
        f"Original user question: {question}\n\n"
        "Revise only the following current assistant turn. The preceding "
        "messages are the causal history available for this request; do not "
        "use any future tool result.\n\n"
        + (
            "The following verification trajectory was produced in a separate "
            "context. Use its tool results as evidence and integrate it before "
            "the original main tool call:\n<verification_trajectory>\n"
            f"{json.dumps(integration_context, ensure_ascii=False, indent=2)}\n"
            "</verification_trajectory>\n\n"
            if integration_context is not None
            else "Output either the polished turn or the structured verification request required by the system prompt.\n\n"
        )
        + "<current_original_assistant_response>\n"
        + f"{original_response}\n"
        + "</current_original_assistant_response>"
    )
    current_content: Any = current_prompt
    if earlier_history_omitted:
        initial_images = _initial_question_image_paths(record)
        initial_parts = [
            part for path in initial_images
            if (part := _image_content_part(path)) is not None
        ]
        if initial_parts:
            current_content = [{"type": "text", "text": current_prompt}, *initial_parts]
    messages.append(
        ModelMessage(
            role="user",
            content=current_content,
        )
    )
    return messages


def _parse_verification_request(text: str) -> dict[str, Any] | None:
    match = re.search(
        r"<verification_request>\s*(\{.*?\})\s*</verification_request>",
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    string_fields = ("referent", "object_to_verify", "verification_goal", "verification_scope")
    if any(not isinstance(payload.get(key), str) or not payload[key].strip() for key in string_fields):
        return None
    distractor_index = payload.get("distractor_result_index")
    if not isinstance(distractor_index, int) or isinstance(distractor_index, bool):
        return None
    if payload["verification_scope"].strip() != "external_candidate":
        return None
    object_to_verify = payload["object_to_verify"].strip()
    if object_to_verify.lower() in {
        "another object",
        "a different object",
        "another candidate",
        "a different candidate",
        "a different number",
        "another item",
    }:
        return None
    if any(re.search(pattern, object_to_verify, flags=re.IGNORECASE) for pattern in _VISUAL_ONLY_CANDIDATE_PATTERNS):
        return None
    verification_goal = payload["verification_goal"].strip()
    if any(re.search(pattern, verification_goal, flags=re.IGNORECASE) for pattern in _VISUAL_VERIFICATION_GOAL_PATTERNS):
        return None
    return {
        "referent": payload["referent"].strip(),
        "object_to_verify": object_to_verify,
        "distractor_result_index": distractor_index,
        "verification_goal": verification_goal,
        "verification_scope": "external_candidate",
    }


def _parse_visual_evidence_check_request(text: str) -> str | None:
    """Return the writer-declared reason for a terminal image check."""

    match = re.fullmatch(
        r"\s*<visual_evidence_check_request>\s*(\{.*?\})\s*</visual_evidence_check_request>\s*",
        str(text or ""),
        re.IGNORECASE | re.DOTALL,
    )
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or payload.get("scope") != "terminal_image_quality":
        return None
    reason = str(payload.get("reason") or "").strip()
    if reason not in {"trajectory_quality_complaint", "writer_cannot_read"}:
        return None
    return reason


def _history_ends_with_image_observation(history: list[Any]) -> bool:
    if not history:
        return False
    role, content = _role_and_content(history[-1])
    return role.lower() in {"observation", "tool"} and "<image>" in content


def _immediate_multi_result_search_observation(history: list[Any]) -> list[dict[str, Any]] | None:
    """Return the last search result list, only when it has competing records."""

    if not history:
        return None
    role, content = _role_and_content(history[-1])
    if role.lower() not in {"observation", "tool"}:
        return None
    try:
        payload = json.loads(str(content))
    except json.JSONDecodeError:
        return None
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) < 2 or not all(isinstance(item, dict) for item in results):
        return None
    return results


def _normalized_provenance_text(value: Any) -> str:
    """Normalize a compact search record for conservative candidate matching."""

    text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).casefold()
    return re.sub(r"\s+", " ", re.sub(r"[^\w]+", " ", text)).strip()


def _result_mentions_candidate(result: dict[str, Any], candidate: str) -> bool:
    normalized_candidate = _normalized_provenance_text(candidate)
    normalized_result = _normalized_provenance_text(result)
    if not normalized_candidate:
        return False
    if normalized_candidate in normalized_result:
        return True

    # The writer is asked to copy a concise result title, but models sometimes
    # append a descriptive clause.  Accept a clear title prefix or a long
    # contiguous name anchor while still requiring the candidate to be rooted
    # in this exact compact result.
    candidate_tokens = normalized_candidate.split()
    title_tokens = _normalized_provenance_text(result.get("title") or "").split()
    if len(candidate_tokens) >= 3 and len(title_tokens) >= 3:
        shared_prefix = min(len(candidate_tokens), len(title_tokens), 6)
        if shared_prefix >= 3 and candidate_tokens[:shared_prefix] == title_tokens[:shared_prefix]:
            return True
    for width in range(min(len(candidate_tokens), 6), 3, -1):
        for start in range(0, len(candidate_tokens) - width + 1):
            if " ".join(candidate_tokens[start : start + width]) in normalized_result:
                return True
    return False


def _validate_verification_request_against_history(
    request: dict[str, Any],
    history: list[Any],
) -> str | None:
    """Return a rejection reason unless the distractor is grounded in one search hit."""

    results = _immediate_multi_result_search_observation(history)
    if results is None:
        return "not_immediate_multi_result_search"
    distractor_index = request.get("distractor_result_index")
    if not isinstance(distractor_index, int):
        return "missing_distractor_result_index"
    if not 0 <= distractor_index < len(results):
        return "distractor_result_index_out_of_range"
    if not _result_mentions_candidate(results[distractor_index], str(request.get("object_to_verify") or "")):
        return "distractor_candidate_not_in_declared_result"
    return None


def _verification_request_key(request: dict[str, Any]) -> tuple[str, str]:
    """Stable per-record key used to prevent repeated verification of B."""

    return (
        _normalized_provenance_text(request.get("referent") or ""),
        _normalized_provenance_text(request.get("object_to_verify") or ""),
    )


def _extract_block(text: str, tag: str) -> str | None:
    match = re.search(rf"<{tag}\b[^>]*>.*?</{tag}>", text, re.DOTALL | re.IGNORECASE)
    return match.group(0) if match else None


def _extract_thinking(text: str) -> str:
    block = _extract_block(text, "thinking")
    if block:
        return block
    return f"<thinking>\n{text.strip()}\n</thinking>"


def _safe_preserve_blocks(original: str, proposed: str) -> str:
    """Keep tool-call and answer blocks byte-for-byte from the source turn."""

    original_tool = _extract_block(original, "tool_call")
    original_answer = _extract_block(original, "answer")
    proposed_thinking = _extract_thinking(proposed)
    pieces = [proposed_thinking]
    if original_tool:
        pieces.append(original_tool)
    if original_answer:
        pieces.append(original_answer)
    return "\n".join(pieces)


def _print_turn_rewrite(
    record_id: str,
    message_index: int,
    original: str,
    rewritten: str,
) -> None:
    """Print one rewrite in the same readable trace style as api_tools.py."""

    with _OUTPUT_LOCK:
        print(f"\n=== Trajectory {record_id} | Turn {message_index} Before ===", flush=True)
        print(original, flush=True)
        print(f"\n=== Trajectory {record_id} | Turn {message_index} After ===", flush=True)
        print(rewritten, flush=True)


def _debug_message_text(message: Any, *, omit_observation: bool = True) -> tuple[str, str]:
    role, content = _role_and_content(message)
    if omit_observation and role.lower() in {"observation", "tool"}:
        return role, "<tool result omitted from debug output>"
    return role, content


def _format_trajectory_debug(
    record: dict[str, Any],
    record_id: str,
    ranges: list[tuple[int, int]] | None = None,
    marker_ranges: list[tuple[int, int]] | None = None,
) -> str:
    """Render a plain-text trajectory trace without tool-result bodies."""

    key = _get_conversation_key(record)
    messages = record[key]
    selected: set[int] | None = None
    if ranges is not None:
        selected = set()
        for start, end in ranges:
            selected.update(range(max(0, start), min(len(messages), end)))
    marker_starts: dict[int, list[int]] = {}
    marker_ends: dict[int, list[int]] = {}
    for marker_index, (start, end) in enumerate(marker_ranges or [], start=1):
        marker_starts.setdefault(max(0, start), []).append(marker_index)
        marker_ends.setdefault(min(len(messages), end), []).append(marker_index)
    lines = [f"=== Trajectory {record_id} ==="]
    for index, message in enumerate(messages):
        for marker_index in marker_starts.get(index, []):
            lines.append(f"\n===== VERIFICATION {marker_index} START =====")
        if selected is not None and index not in selected:
            continue
        role, content = _debug_message_text(message)
        lines.append(f"\n[Turn {index:03d}][{role}]")
        lines.append(content)
        for marker_index in marker_ends.get(index + 1, []):
            lines.append(f"\n===== VERIFICATION {marker_index} END =====")
    for marker_index in marker_ends.get(len(messages), []):
        if not any(f"VERIFICATION {marker_index} END" in line for line in lines):
            lines.append(f"\n===== VERIFICATION {marker_index} END =====")
    return "\n".join(lines)


def _extend_debug_window(
    messages: list[Any],
    start: int,
    end: int,
    additional_assistant_turns: int,
) -> tuple[int, int]:
    """Extend a verification window over subsequent original assistant turns."""

    assistant_seen = 0
    cursor = min(end, len(messages))
    while cursor < len(messages):
        role, _ = _role_and_content(messages[cursor])
        if _is_assistant(role):
            assistant_seen += 1
            if assistant_seen > additional_assistant_turns:
                break
        cursor += 1
    return start, cursor


def _print_completed_trajectory(
    record: dict[str, Any],
    record_id: str,
    debug_output: str,
    verification_ranges: list[tuple[int, int]],
    verification_debug_turns: int,
) -> None:
    """Print the selected plain-text trace after all turns are rewritten."""

    mode = "silence" if debug_output == "slience" else debug_output
    if mode == "silence":
        return
    ranges: list[tuple[int, int]] | None = None
    if mode == "verification_only":
        if not verification_ranges:
            return
        messages = record[_get_conversation_key(record)]
        ranges = [
            _extend_debug_window(messages, start, end, verification_debug_turns)
            for start, end in verification_ranges
        ]

    with _OUTPUT_LOCK:
        print(
            _format_trajectory_debug(
                record,
                record_id,
                ranges,
                marker_ranges=verification_ranges,
            ),
            flush=True,
        )


def _print_compare_trajectories(
    original: dict[str, Any],
    rewritten: dict[str, Any],
    record_id: str,
    debug_output: str,
) -> None:
    """Print original and rewritten full traces together for side-by-side review."""

    mode = "silence" if debug_output == "slience" else debug_output
    if mode == "silence":
        return
    with _OUTPUT_LOCK:
        print(f"\n=== Trajectory {record_id} | BEFORE (original) ===", flush=True)
        print(_format_trajectory_debug(original, record_id), flush=True)
        print(f"\n=== Trajectory {record_id} | AFTER (rewritten) ===", flush=True)
        print(_format_trajectory_debug(rewritten, record_id), flush=True)


def _replace_content(message: dict[str, Any], rewritten: str) -> dict[str, Any]:
    result = copy.deepcopy(message)
    if "value" in result:
        result["value"] = rewritten
    elif "content" in result:
        result["content"] = rewritten
    elif "response_text" in result:
        result["response_text"] = rewritten
    else:
        result["value"] = rewritten
    return result


def _rewrite_turn(
    record: dict[str, Any],
    record_index: int,
    assistant_index: int,
    history: list[Any],
    original_response: str,
    model_alias: str | WriterAliasPool,
    max_tokens: int | None,
    system_prompt: str = OPTIMIZER_SYSTEM_PROMPT,
    history_image_offset: int = 0,
    earlier_history_omitted: bool = False,
) -> TurnDecision:
    selected_model_alias = _next_writer_alias(model_alias)
    request = ModelRequest(
        model=selected_model_alias,
        messages=_build_optimizer_messages(
            record,
            record_index,
            assistant_index,
            history,
            original_response,
            system_prompt=system_prompt,
            history_image_offset=history_image_offset,
            earlier_history_omitted=earlier_history_omitted,
        ),
        metadata=_model_worker_metadata("post_process_trajectory_reasoning"),
    )
    response = LLM_WORKER.generate(request)
    proposed = response.content if response else ""
    if not isinstance(proposed, str) or not proposed.strip():
        raise ValueError("optimizer returned an empty rewritten response")
    terminal_image_final = (
        "<answer" in original_response.lower()
        and _history_ends_with_image_observation(history)
    )
    visual_reason = _parse_visual_evidence_check_request(proposed)
    if visual_reason is not None:
        if not terminal_image_final:
            return TurnDecision(
                action="rewrite",
                rewritten=original_response,
                editor_note="visual_evidence_request_rejected_not_eligible",
            )
        return TurnDecision(
            action="visual_evidence_check",
            visual_evidence_request=True,
            visual_evidence_reason=visual_reason,
            editor_note=f"writer_requested_terminal_visual_evidence_check_{visual_reason}",
        )
    verification_request = _parse_verification_request(proposed)
    if verification_request is None and re.search(r"<verification_request\b", proposed, re.IGNORECASE):
        # An invalid, visual-only, or schema-incomplete request must never leak
        # into the training trajectory as pseudo-reasoning.
        return TurnDecision(
            action="rewrite",
            rewritten=original_response,
            editor_note="verification_request_rejected_invalid_or_visual",
        )
    if verification_request is not None:
        rejection_reason = _validate_verification_request_against_history(verification_request, history)
        if rejection_reason is not None:
            return TurnDecision(
                action="rewrite",
                rewritten=original_response,
                verification_request=verification_request,
                editor_note=f"verification_request_rejected_{rejection_reason}",
            )
        return TurnDecision(
            action="verification_request",
            verification_request=verification_request,
            editor_note="writer_requested_verification",
        )
    return TurnDecision(
        action="rewrite",
        rewritten=_safe_preserve_blocks(original_response, proposed),
        editor_note="plain_text_optimizer_response",
    )


def _run_verification(
    *,
    record: dict[str, Any],
    record_index: int,
    assistant_index: int,
    history: list[Any],
    original_response: str,
    request: dict[str, Any],
    model_alias: str,
    max_tokens: int,
    max_turns: int,
    workdir: Path | str | None = None,
) -> Any:
    question = _record_question(record)
    verification_prompt = _build_verification_context(
        question=question,
        referent=request["referent"],
        distractor_candidate=request["object_to_verify"],
        verification_goal=request["verification_goal"],
        preceding_context=_render_history(history),
        current_main_turn=original_response,
    )
    case_id = f"{_record_id(record, record_index)}_turn_{assistant_index}_verification"
    context = ToolRuntimeContext(
        working_dir=str(workdir or (ROOT / "synthesis/.ignore/verification_runs")),
        case_id=case_id,
    )
    image_parts: list[dict[str, Any]] = []
    for image_index, image_path in enumerate(record.get("images") or [], start=1):
        image_source = str(image_path)
        context.register_image(image_source)
        image_part = _image_content_part(image_source)
        if image_part is not None:
            image_parts.append(
                {
                    "type": "text",
                    "text": f"[Trajectory image {image_index}; inspect this image if it is relevant.]",
                }
            )
            image_parts.append(image_part)
    verifier_user_content: Any = [{"type": "text", "text": verification_prompt}, *image_parts]
    api_mode = _select_api_mode(model_alias, "auto")
    agent = OpenAIToolAgent(
        OpenAIToolAgentConfig(
            model=model_alias,
            api_mode=api_mode,
            # The agent counts model responses, while this CLI option means
            # tool uses.  Reserve one normal final response plus one fallback
            # response if the model asks for a forbidden extra tool.
            max_turns=max_turns + 2,
            max_tool_calls=max_turns,
            timeout_s=600.0,
            system_prompt=VERIFICATION_SYSTEM_PROMPT,
            print_rounds=False,
        )
    )
    return agent.run(
        messages=[
            {"role": "system", "content": VERIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": verifier_user_content},
        ],
        context=context,
    )


def _normalized_verification_final(text: Any) -> str:
    value = str(text or "").strip()
    answer = _extract_block(value, "answer")
    if answer:
        value = re.sub(r"</?answer\b[^>]*>", "", answer, flags=re.IGNORECASE).strip()
    return value.strip().upper().rstrip(".")


def _is_usable_verification_evidence(tool_result: Any) -> bool:
    """Accept only readable page text, not an attempted or failed tool call."""

    if str(getattr(tool_result, "name", "") or "") != "read_url":
        return False
    output = getattr(tool_result, "output", None)
    if not isinstance(output, dict) or output.get("ok") is False:
        return False
    content = str(output.get("content") or "").strip()
    if not content:
        return False
    lower = content.casefold()
    failure_prefixes = (
        "unable to read the requested page",
        "insufficient evidence:",
        "requested page could not",
    )
    return not any(lower.startswith(prefix) for prefix in failure_prefixes)


def _verification_evidence_count(result: Any) -> int:
    return sum(
        1
        for tool_result in (getattr(result, "tool_results", []) or [])
        if _is_usable_verification_evidence(tool_result)
    )


def _verification_succeeded(result: Any) -> bool:
    final_text = _normalized_verification_final(getattr(result, "final_text", ""))
    metadata = getattr(result, "metadata", {}) or {}
    return (
        bool(metadata.get("generation_complete"))
        and final_text == "NO"
        and _verification_evidence_count(result) > 0
    )


def _normalize_manual_action(text: str) -> str:
    """Convert api_tools' manual-ReAct action block to the SFT tool_call form."""

    match = re.search(r"<action>\s*(\{.*?\})\s*</action>", text, re.DOTALL | re.IGNORECASE)
    if not match:
        return text
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return text
    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return text
    if tool_name == "finish":
        # ``finish`` is a control action used only by the manual-ReAct runner;
        # it is not one of the trajectory tools.  Keeping it as a tool call
        # would train the model to emit a fake ``finish`` invocation mid-trace.
        conclusion = str(arguments.get("answer") or "").strip()
        prefix = text[: match.start()].rstrip()
        suffix = text[match.end() :].lstrip()
        return "\n".join(part for part in (prefix, conclusion, suffix) if part)
    tool_call = {"name": tool_name, "arguments": arguments}
    replacement = "<tool_call>\n" + json.dumps(tool_call, ensure_ascii=False) + "\n</tool_call>"
    return text[: match.start()] + replacement + text[match.end() :]


def _assistant_message_to_trajectory_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, list):
        content = json.dumps(content, ensure_ascii=False)
    text = _normalize_manual_action(str(content or ""))
    tool_calls = message.get("tool_calls") or message.get("function_calls") or []
    if tool_calls and not _extract_block(text, "tool_call"):
        tool_call = tool_calls[0] if isinstance(tool_calls[0], dict) else {}
        function = tool_call.get("function") or {}
        name = function.get("name") or tool_call.get("name")
        arguments = function.get("arguments")
        if arguments is None:
            arguments = tool_call.get("arguments") or {}
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw_arguments": arguments}
        if isinstance(name, str) and isinstance(arguments, dict):
            payload = {"name": name, "arguments": arguments}
            text = text.rstrip() + "\n<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"
    return text


def _is_verification_verdict_message(text: str) -> bool:
    """Identify the verifier's terminal YES/NO message before trajectory insertion."""

    value = str(text or "").strip()
    thinking = _extract_block(value, "thinking")
    if thinking:
        value = re.sub(r"<thinking\b[^>]*>.*?</thinking>", "", value, flags=re.DOTALL | re.IGNORECASE)
    answer = _extract_block(value, "answer")
    if answer:
        value = re.sub(r"</?answer\b[^>]*>", "", answer, flags=re.IGNORECASE)
    value = re.sub(r"<[^>]+>", "", value).strip().upper().rstrip(".")
    return value in {"YES", "NO"}


def _verification_messages_to_trajectory(result: Any) -> tuple[list[dict[str, str]], list[str]]:
    """Convert the verifier's runtime messages to ShareGPT messages."""

    messages: list[dict[str, str]] = []
    image_paths: list[str] = []
    tool_results = list(getattr(result, "tool_results", []) or [])
    tool_index = 0
    skipped_initial_user = False
    for message in getattr(result, "messages", []) or []:
        role = str(message.get("role") or "") if isinstance(message, dict) else ""
        if role == "system":
            continue
        content = message.get("content", "") if isinstance(message, dict) else ""
        if isinstance(content, list):
            content = json.dumps(content, ensure_ascii=False)
        content_text = str(content or "")
        if role == "user":
            # The initial verification task is already represented by the main
            # trajectory. Later user messages are image attachments generated by
            # read_url and are represented on the observation below.
            if not skipped_initial_user:
                skipped_initial_user = True
            continue
        if role == "assistant":
            trajectory_text = _assistant_message_to_trajectory_text(message)
            # The verifier's terminal judgment is control metadata, not a
            # reasoning turn.  In particular, do not splice a bare ``NO`` or
            # ``YES`` into the main trajectory; the integration writer already
            # receives the judgment through the verification audit/result.
            if _is_verification_verdict_message(trajectory_text):
                continue
            messages.append({"from": "gpt", "value": trajectory_text})
            continue
        if role == "tool":
            if tool_index < len(tool_results):
                tool_result = tool_results[tool_index]
                tool_index += 1
                new_images = getattr(tool_result, "new_images", {}) or {}
                new_image_paths = [str(path) for path in new_images.values()]
                for path in new_image_paths:
                    image_paths.append(str(path))
                if new_image_paths:
                    # Keep the number and order of placeholders exactly aligned
                    # with image_paths.  A later splice inserts these paths at
                    # the corresponding point in the main trajectory's image
                    # list, so a subsequent editor turn sees the right image.
                    content_text = content_text.rstrip() + "\n" + "\n".join(
                        "<image>" for _ in new_image_paths
                    )
            messages.append({"from": "observation", "value": content_text})
    return messages, image_paths


def _image_placeholder_count(messages: list[Any]) -> int:
    """Count ShareGPT image placeholders in a message prefix.

    The ``images`` array is positional: each ``<image>`` in conversation
    order consumes one entry.  Verification messages are inserted in the
    middle of a trajectory, so their assets must be inserted at this exact
    offset instead of appended to the end of the array.
    """

    return sum(_role_and_content(message)[1].count("<image>") for message in messages)


def _extract_json_object(text: str) -> dict[str, Any] | None:
    candidate = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", candidate, re.DOTALL | re.IGNORECASE)
    if fenced:
        candidate = fenced.group(1).strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            parsed = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            return None
    return parsed if isinstance(parsed, dict) else None


def _reasoning_only(text: str) -> str:
    value = str(text or "").strip()
    thinking = _extract_block(value, "thinking")
    if thinking:
        value = thinking
    value = re.sub(r"<tool_call>.*?</tool_call>", "", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"<answer>.*?</answer>", "", value, flags=re.DOTALL | re.IGNORECASE)
    value = re.sub(r"<verification_request>.*?</verification_request>", "", value, flags=re.DOTALL | re.IGNORECASE)
    if value.lower().startswith("<thinking>") and value.lower().endswith("</thinking>"):
        return value
    return f"<thinking>\n{value.strip()}\n</thinking>"


def _merge_assistant_reasoning(prefix: str, suffix: str) -> str:
    """Join two assistant fragments into one valid ShareGPT assistant message.

    Verification integration used to emit the original rewritten reasoning as
    one ``gpt`` message and then emit the verifier's first ``gpt`` message as a
    second message.  That is not a rendering issue: it creates two adjacent
    assistant messages in the actual conversation.  The two reasoning blocks
    belong to the same assistant action, so keep one thinking block and append
    the suffix's tool call/other preserved blocks unchanged.
    """

    prefix_text = str(prefix or "").strip()
    suffix_text = str(suffix or "").strip()
    suffix_match = re.search(r"<thinking\b[^>]*>.*?</thinking>", suffix_text, re.DOTALL | re.IGNORECASE)
    if not suffix_match:
        return "\n".join(part for part in (prefix_text, suffix_text) if part)

    prefix_thinking = _extract_thinking(prefix_text)
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
    thinking_parts = [part for part in (prefix_body, suffix_body) if part]
    merged_thinking = "<thinking>\n" + "\n\n".join(thinking_parts) + "\n</thinking>"

    suffix_rest = (suffix_text[: suffix_match.start()] + suffix_text[suffix_match.end() :]).strip()
    return "\n".join(part for part in (merged_thinking, suffix_rest) if part)


def _integration_turn(
    *,
    record: dict[str, Any],
    record_index: int,
    assistant_index: int,
    history: list[Any],
    original_response: str,
    verification_request: dict[str, Any],
    verification_messages: list[dict[str, Any]],
    model_alias: str | WriterAliasPool,
    max_tokens: int | None,
    history_image_offset: int = 0,
    earlier_history_omitted: bool = False,
) -> tuple[str, list[dict[str, str]], str]:
    selected_model_alias = _next_writer_alias(model_alias)
    request = ModelRequest(
        model=selected_model_alias,
        messages=_build_optimizer_messages(
            record,
            record_index,
            assistant_index,
            history,
            original_response,
            system_prompt=INTEGRATION_SYSTEM_PROMPT,
            integration_context={
                "verification_request": verification_request,
                "verification_messages": verification_messages,
            },
            history_image_offset=history_image_offset,
            earlier_history_omitted=earlier_history_omitted,
        ),
        metadata=_model_worker_metadata("post_process_trajectory_integration"),
    )
    response = LLM_WORKER.generate(request)
    proposed = response.content if response else ""
    payload = _extract_json_object(proposed)
    if (
        not payload
        or not isinstance(payload.get("before_verification"), str)
        or not isinstance(payload.get("after_verification"), str)
    ):
        raise ValueError("integration writer did not return the required integration JSON fields")
    rewritten_turns = payload.get("verification_turns")
    if not isinstance(rewritten_turns, list):
        raise ValueError("integration writer did not return verification_turns")
    rewritable_originals = [
        item
        for item in verification_messages
        if item.get("from") == "gpt" and _extract_block(str(item.get("value", "")), "tool_call")
    ]
    if len(rewritten_turns) != len(rewritable_originals):
        raise ValueError(
            "integration writer returned an unexpected number of verification turns: "
            f"expected={len(rewritable_originals)} actual={len(rewritten_turns)}"
        )
    rewritten_messages: list[dict[str, str]] = []
    rewrite_index = 0
    for item in verification_messages:
        copied = {"from": str(item.get("from") or "observation"), "value": str(item.get("value", ""))}
        if copied["from"] == "gpt" and _extract_block(copied["value"], "tool_call"):
            candidate = rewritten_turns[rewrite_index]
            if not isinstance(candidate, str):
                raise ValueError("verification_turns must contain strings")
            copied["value"] = _safe_preserve_blocks(copied["value"], candidate)
            rewrite_index += 1
        rewritten_messages.append(copied)
    return (
        _reasoning_only(payload["before_verification"]),
        rewritten_messages,
        _reasoning_only(payload["after_verification"]),
    )


def _splice_verified_turn(
    *,
    messages: list[Any],
    message_index: int,
    original_message: dict[str, Any],
    before_verification: str,
    after_verification: str,
    verification_messages: list[dict[str, str]],
) -> int:
    original = _role_and_content(original_message)[1]
    original_tool = _extract_block(original, "tool_call")
    original_answer = _extract_block(original, "answer")
    after = after_verification
    if original_tool:
        after += "\n" + original_tool
    if original_answer:
        after += "\n" + original_answer
    before_message = _replace_content(original_message, before_verification)
    after_message = _replace_content(original_message, after)
    inserted = [
        {"from": "gpt" if item.get("from") == "gpt" else "observation", "value": item.get("value", "")}
        for item in verification_messages
    ]

    # The verifier normally starts with an assistant action containing its
    # first tool call.  Merge the main trajectory's pre-verification thinking
    # into that action instead of inserting a standalone assistant message.
    # Otherwise the ShareGPT sequence would become ``gpt -> gpt -> observation``
    # even though the plain-text renderer merely makes this easier to notice.
    if inserted and inserted[0].get("from") == "gpt":
        inserted[0] = {
            "from": "gpt",
            "value": _merge_assistant_reasoning(before_verification, str(inserted[0].get("value", ""))),
        }
        replacement = [*inserted, after_message]
    elif inserted:
        replacement = [before_message, *inserted, after_message]
    else:
        # Defensive fallback for an empty verifier trace: do not create two
        # adjacent assistant messages in this degenerate case either.
        replacement = [_replace_content(original_message, _merge_assistant_reasoning(before_verification, after))]
    messages[message_index : message_index + 1] = replacement
    return len(replacement)


def _optimize_record(
    record: dict[str, Any],
    record_index: int,
    model_alias: str | WriterAliasPool,
    max_tokens: int | None,
    verification_model_alias: str,
    verification_max_tokens: int,
    verification_max_turns: int,
    debug_output: str,
    verification_debug_turns: int,
    verbose: bool,
    compare: bool = False,
    verification_workdir: Path | str | None = None,
    visual_vqa_model_alias: str = "",
    terminal_image_resolver: VqaTerminalImageResolver | None = None,
    visual_evidence_workdir: Path | str | None = None,
    history_window_turns: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    output = copy.deepcopy(record)
    key = _get_conversation_key(output)
    messages = output[key]
    changes: list[dict[str, Any]] = []
    assistant_count = 0
    verification_count = 0
    integrated_verification_count = 0
    visual_evidence_check_count = 0
    visual_evidence_trajectory_quality_complaint_count = 0
    visual_evidence_writer_cannot_read_count = 0
    visual_evidence_replacement_count = 0
    visual_evidence_filter_count = 0
    seen_verification_keys: set[tuple[str, str]] = set()
    verification_ranges: list[tuple[int, int]] = []
    normalized_debug_output = "silence" if debug_output == "slience" else debug_output

    try:
        message_index = 0
        while message_index < len(messages):
            message = messages[message_index]
            role, original_response = _role_and_content(message)
            if not _is_assistant(role):
                message_index += 1
                continue
            assistant_count += 1
            # output[key][:message_index] contains the already optimized
            # prefix.  The current response and all later observations are
            # intentionally excluded from the causal history.
            full_history = output[key][:message_index]
            history, history_image_offset, earlier_history_omitted = _window_history_by_assistant_turns(
                full_history,
                history_window_turns,
            )
            decision = _rewrite_turn(
                # Previous integrated verification runs may have appended local
                # images.  Use the evolving record so later turns can map each
                # <image> observation back to the correct image payload.
                output,
                record_index,
                message_index,
                history,
                original_response,
                model_alias,
                max_tokens,
                history_image_offset=history_image_offset,
                earlier_history_omitted=earlier_history_omitted,
            )
            if decision.action == "visual_evidence_check":
                visual_evidence_check_count += 1
                if decision.visual_evidence_reason == "trajectory_quality_complaint":
                    visual_evidence_trajectory_quality_complaint_count += 1
                elif decision.visual_evidence_reason == "writer_cannot_read":
                    visual_evidence_writer_cannot_read_count += 1
                if not visual_vqa_model_alias or terminal_image_resolver is None:
                    changes.append(
                        {
                            "message_index": message_index,
                            "editor_note": "visual_evidence_check_skipped_missing_configuration",
                            "visual_evidence_check": True,
                        }
                    )
                    message_index += 1
                    continue
                outcome = run_terminal_visual_check(
                    record=output,
                    messages=messages,
                    final_assistant_index=message_index,
                    original_final_response=original_response,
                    editor_model_alias=_next_writer_alias(model_alias),
                    vqa_model_alias=visual_vqa_model_alias,
                    resolver=terminal_image_resolver,
                    image_content_part=_image_content_part,
                    workdir=Path(visual_evidence_workdir or (ROOT / "synthesis/.ignore/terminal_visual_evidence")),
                    max_tokens=max_tokens,
                    metadata=_model_worker_metadata("post_process_terminal_visual_evidence"),
                )
                # Keep the initiating condition alongside the terminal-image
                # outcome.  This is useful for compact per-record debug
                # summaries without dumping a whole trajectory.
                outcome.audit["trigger_reason"] = decision.visual_evidence_reason or "unknown"
                if outcome.status == "filter":
                    visual_evidence_filter_count += 1
                    changes.append(
                        {
                            "message_index": message_index,
                            "editor_note": "terminal_visual_evidence_unsupported_filter_recommended",
                            "filter_recommended": True,
                            "terminal_visual_evidence": outcome.audit,
                        }
                    )
                    message_index += 1
                    continue
                if outcome.status == "error":
                    changes.append(
                        {
                            "message_index": message_index,
                            "editor_note": "terminal_visual_evidence_check_error",
                            "terminal_visual_evidence": outcome.audit,
                        }
                    )
                    message_index += 1
                    continue
                if outcome.status == "replaced":
                    if outcome.image_index is None or not outcome.replacement_path:
                        changes.append(
                            {
                                "message_index": message_index,
                                "editor_note": "terminal_visual_evidence_replacement_invalid",
                                "terminal_visual_evidence": outcome.audit,
                            }
                        )
                        message_index += 1
                        continue
                    output.setdefault("images", [])[outcome.image_index] = outcome.replacement_path
                    visual_evidence_replacement_count += 1
                    final_prompt = VISUAL_REPAIRED_FINAL_SYSTEM_PROMPT
                    editor_note = "terminal_visual_evidence_replaced_and_rewritten"
                else:
                    final_prompt = VISUAL_VALIDATED_FINAL_SYSTEM_PROMPT
                    editor_note = "terminal_visual_evidence_matched_and_rewritten"
                final_decision = _rewrite_turn(
                    output,
                    record_index,
                    message_index,
                    history,
                    original_response,
                    model_alias,
                    max_tokens,
                    system_prompt=final_prompt,
                    history_image_offset=history_image_offset,
                    earlier_history_omitted=earlier_history_omitted,
                )
                if final_decision.action != "rewrite":
                    changes.append(
                        {
                            "message_index": message_index,
                            "editor_note": "terminal_visual_evidence_rewrite_invalid",
                            "terminal_visual_evidence": outcome.audit,
                        }
                    )
                    message_index += 1
                    continue
                rewritten = final_decision.rewritten or original_response
                output[key][message_index] = _replace_content(message, rewritten)
                changes.append(
                    {
                        "message_index": message_index,
                        "original": original_response,
                        "rewritten": rewritten,
                        "editor_note": editor_note,
                        "terminal_visual_evidence": outcome.audit,
                        "replacement_image_path": outcome.replacement_path,
                        "replacement_image_index": outcome.image_index,
                    }
                )
                if verbose and normalized_debug_output == "full":
                    _print_turn_rewrite(_record_id(record, record_index), message_index, original_response, rewritten)
                message_index += 1
                continue
            if decision.action == "rewrite":
                rewritten = decision.rewritten or ""
                output[key][message_index] = _replace_content(message, rewritten)
                if verbose and normalized_debug_output == "full":
                    _print_turn_rewrite(
                        _record_id(record, record_index),
                        message_index,
                        original_response,
                        rewritten,
                    )
                if rewritten != original_response or decision.editor_note.startswith("verification_request_rejected"):
                    changes.append(
                        {
                            "message_index": message_index,
                            "original": original_response,
                            "rewritten": rewritten,
                            "editor_note": decision.editor_note,
                            **(
                                {"verification_request": decision.verification_request}
                                if decision.verification_request is not None
                                else {}
                            ),
                        }
                    )
                message_index += 1
                continue

            request_payload = decision.verification_request or {}
            request_key = _verification_request_key(request_payload)
            if request_key in seen_verification_keys:
                changes.append(
                    {
                        "message_index": message_index,
                        "editor_note": "verification_request_rejected_duplicate",
                        "verification_request": request_payload,
                    }
                )
                message_index += 1
                continue
            seen_verification_keys.add(request_key)
            verification_count += 1
            if not verification_model_alias:
                changes.append(
                    {
                        "message_index": message_index,
                        "editor_note": "verification_skipped_no_model_alias",
                        "verification_request": request_payload,
                    }
                )
                message_index += 1
                continue

            try:
                verification_result = _run_verification(
                    record=output,
                    record_index=record_index,
                    assistant_index=message_index,
                    history=history,
                    original_response=original_response,
                    request=request_payload,
                    model_alias=verification_model_alias,
                    max_tokens=verification_max_tokens,
                    max_turns=verification_max_turns,
                    workdir=verification_workdir,
                )
                verification_messages, verification_image_paths = _verification_messages_to_trajectory(verification_result)
            except Exception as exc:
                # A transient tool/model failure must not discard rewrites that
                # were already completed for earlier turns of this record.
                changes.append(
                    {
                        "message_index": message_index,
                        "editor_note": "verification_execution_failed",
                        "verification_request": request_payload,
                        "error": repr(exc),
                    }
                )
                message_index += 1
                continue
            verification_audit = {
                "request": request_payload,
                "final_text": str(getattr(verification_result, "final_text", "") or ""),
                "normalized_final_text": _normalized_verification_final(
                    getattr(verification_result, "final_text", "")
                ),
                "generation_complete": bool((getattr(verification_result, "metadata", {}) or {}).get("generation_complete")),
                "tool_call_count": len(getattr(verification_result, "tool_results", []) or []),
                "usable_evidence_count": _verification_evidence_count(verification_result),
                "usable": _verification_succeeded(verification_result),
            }
            if (
                verification_audit["generation_complete"]
                and verification_audit["normalized_final_text"] == "YES"
                and verification_audit["usable_evidence_count"] > 0
            ):
                # A supported YES means the unselected candidate may itself be
                # the referent.  Do not insert it; leave a clear audit marker
                # so downstream filtering can remove or review this sample.
                changes.append(
                    {
                        "message_index": message_index,
                        "editor_note": "verification_affirmed_candidate",
                        "filter_recommended": True,
                        "verification": verification_audit,
                    }
                )
                message_index += 1
                continue
            if not _verification_succeeded(verification_result):
                changes.append(
                    {
                        "message_index": message_index,
                        "editor_note": "verification_not_integrated",
                        "verification": verification_audit,
                    }
                )
                message_index += 1
                continue

            try:
                # ``messages`` still represents the original ordering at this
                # insertion point.  Record the image offset before splicing so
                # the verifier's downloaded images are mapped to the newly
                # inserted observation messages rather than to later turns.
                image_insert_index = _image_placeholder_count(messages[:message_index])
                before_verification, polished_verification_messages, after_verification = _integration_turn(
                    record=output,
                    record_index=record_index,
                    assistant_index=message_index,
                    history=history,
                    original_response=original_response,
                    verification_request=request_payload,
                    verification_messages=verification_messages,
                    model_alias=model_alias,
                    max_tokens=max_tokens,
                    history_image_offset=history_image_offset,
                    earlier_history_omitted=earlier_history_omitted,
                )
                splice_message_count = _splice_verified_turn(
                    messages=messages,
                    message_index=message_index,
                    original_message=message,
                    before_verification=before_verification,
                    after_verification=after_verification,
                    verification_messages=polished_verification_messages,
                )
            except Exception as exc:
                # Keep the original current turn and carry on with later ones.
                # The verifier trace is intentionally not inserted unless the
                # integration writer produced a fully valid splice.
                changes.append(
                    {
                        "message_index": message_index,
                        "editor_note": "verification_integration_failed",
                        "verification": verification_audit,
                        "error": repr(exc),
                    }
                )
                message_index += 1
                continue
            if verification_image_paths:
                image_list = output.setdefault("images", [])
                image_insert_index = min(image_insert_index, len(image_list))
                image_list[image_insert_index:image_insert_index] = verification_image_paths
            integrated_verification_count += 1
            if verbose and normalized_debug_output == "full":
                integration_preview = before_verification
                integration_preview += "\n" + "\n".join(
                    item["value"]
                    for item in polished_verification_messages
                    if item.get("from") == "gpt"
                )
                integration_preview += "\n" + after_verification
                _print_turn_rewrite(
                    _record_id(record, record_index),
                    message_index,
                    original_response,
                    integration_preview,
                )
            verification_ranges.append((message_index, message_index + splice_message_count))
            changes.append(
                {
                    "message_index": message_index,
                    "editor_note": "verification_integrated",
                    "before_verification": before_verification,
                    "after_verification": after_verification,
                    "verification": verification_audit,
                    "inserted_messages": len(verification_messages),
                    "inserted_image_count": len(verification_image_paths),
                    "image_insert_index": image_insert_index,
                }
            )
            # Skip the complete replacement span (the merged first verifier
            # assistant, the remaining verifier messages, and the rewritten
            # post-verification assistant). The next pass starts at the
            # original next message, normally the main tool observation.
            message_index += splice_message_count
        if not compare:
            _print_completed_trajectory(
                output,
                _record_id(record, record_index),
                normalized_debug_output,
                verification_ranges,
                verification_debug_turns,
            )
        return output, {
            "record_index": record_index,
            "record_id": _record_id(record, record_index),
            "status": "ok",
            "assistant_turns": assistant_count,
            "changed_turns": len(changes),
            "verification_requests": verification_count,
            "integrated_verifications": integrated_verification_count,
            "terminal_visual_evidence_checks": visual_evidence_check_count,
            "terminal_visual_evidence_trajectory_quality_complaint_checks": visual_evidence_trajectory_quality_complaint_count,
            "terminal_visual_evidence_writer_cannot_read_checks": visual_evidence_writer_cannot_read_count,
            "terminal_visual_evidence_replacements": visual_evidence_replacement_count,
            "terminal_visual_evidence_filters": visual_evidence_filter_count,
            "history_window_turns": history_window_turns,
            "changes": changes,
        }
    except Exception as exc:
        return copy.deepcopy(record), {
            "record_index": record_index,
            "record_id": _record_id(record, record_index),
            "status": "error",
            "assistant_turns_attempted": assistant_count,
            "changed_turns": 0,
            "history_window_turns": history_window_turns,
            "error": repr(exc),
        }


def _checkpoint_dir(args: argparse.Namespace) -> Path:
    if args.state_dir is not None:
        return args.state_dir
    anchor = args.audit_jsonl or args.output or args.input
    return anchor.parent / f".{anchor.stem}_rewrite_state"


def _rewrite_run_config(args: argparse.Namespace, offset: int) -> dict[str, Any]:
    """Configuration that must not change across a resumable rewrite run."""

    return {
        "schema_version": 1,
        "input": str(args.input.resolve()),
        "model_alias": args.model_alias,
        # None keeps checkpoints created before --model-alias-2 compatible
        # with a single-alias resume.
        "model_alias_2": args.model_alias_2 or None,
        "verification_model_alias": args.verification_model_alias,
        "visual_vqa_model_alias": args.visual_vqa_model_alias,
        "max_tokens": args.max_tokens,
        "verification_max_tokens": args.verification_max_tokens,
        "verification_max_turns": args.verification_max_turns,
        "history_window_turns": args.history_window_turns,
        "offset": offset,
        "limit": args.limit,
        "dry_run": bool(args.dry_run),
    }


def _is_expandable_rewrite_resume_config(actual: dict[str, Any], expected: dict[str, Any]) -> bool:
    """Return whether a checkpointed rewrite prefix can safely be extended."""

    # All behavior-affecting options must remain identical.  ``limit`` is the
    # sole permitted change, and it may only expand the selected source range.
    for key in (
        "schema_version",
        "input",
        "model_alias",
        "model_alias_2",
        "verification_model_alias",
        "visual_vqa_model_alias",
        "max_tokens",
        "verification_max_tokens",
        "verification_max_turns",
        "history_window_turns",
        "offset",
        "dry_run",
    ):
        if actual.get(key) != expected.get(key):
            return False

    old_limit = actual.get("limit")
    new_limit = expected.get("limit")
    if old_limit is None:
        return new_limit is None  # An unbounded run cannot safely shrink.
    if new_limit is None:
        return True  # A bounded prefix may be extended to the end.
    try:
        return int(new_limit) >= int(old_limit)
    except (TypeError, ValueError):
        return False


def _prepare_rewrite_state(args: argparse.Namespace, offset: int) -> Path:
    state_dir = _checkpoint_dir(args)
    config_path = state_dir / "run_config.json"
    expected = _rewrite_run_config(args, offset)
    if args.resume:
        if args.force_overwrite_state:
            raise SystemExit("--force-overwrite-state cannot be combined with --resume.")
        if not config_path.is_file():
            raise SystemExit(f"--resume requires existing checkpoint configuration: {config_path}")
        try:
            actual = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise SystemExit(f"Unable to read checkpoint configuration {config_path}: {exc}") from exc
        if actual != expected:
            # A rewrite checkpoint is keyed by immutable source record index
            # and record ID.  Intentionally permit a caller to change any
            # runtime configuration on resume (alias pool, history window,
            # selected range, etc.) and reuse that completed prefix.  This is
            # useful when recovering a long run after changing an endpoint or
            # increasing the range.  The trade-off is explicit in the log:
            # old and new records can have been produced under different
            # settings.
            config_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print(
                "WARNING: resuming rewrite checkpoint while ignoring prior run configuration; "
                "completed records will be reused under the new settings.",
                file=sys.stderr,
                flush=True,
            )
    else:
        if state_dir.exists() and any(state_dir.iterdir()):
            if not args.force_overwrite_state:
                raise SystemExit(f"Checkpoint state already exists: {state_dir}; use --resume or a new --state-dir.")
            # Do not remove the directory itself: HDFS/FUSE can retain a
            # directory entry briefly after unlink.  Truncating only the two
            # state files is sufficient and leaves no stale completed rows.
            completed_path = state_dir / "completed_records.jsonl"
            if completed_path.exists():
                with completed_path.open("w", encoding="utf-8") as handle:
                    handle.flush()
                    try:
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
            print(f"Resetting rewrite checkpoint state: {state_dir}", file=sys.stderr, flush=True)
        state_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return state_dir / "completed_records.jsonl"


def main() -> int:
    args = _parse_args()
    if not args.no_action and not args.model_alias:
        raise SystemExit("--model-alias is required (or set POST_PROCESS_MODEL_ALIAS/TEXT_PROCESS_MODEL).")
    writer_alias_pool = WriterAliasPool([args.model_alias, args.model_alias_2]) if not args.no_action else None
    if not args.dry_run and not args.output:
        raise SystemExit("--output is required unless --dry-run is set.")
    offset = args.offset if args.offset is not None else (args.start or 0)
    if offset < 0 or args.limit is not None and args.limit < 0:
        raise SystemExit("--offset/--start and --limit must be non-negative.")
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1.")
    if args.verification_max_turns < 1:
        raise SystemExit("--verification-max-turns must be at least 1.")
    if args.verification_debug_turns < 0:
        raise SystemExit("--verification-debug-turns must be non-negative.")
    if args.history_window_turns is not None and args.history_window_turns < 1:
        raise SystemExit("--history-window-turns must be at least 1 when provided.")
    if args.checkpoint_every < 1:
        raise SystemExit("--checkpoint-every must be at least 1.")
    if args.no_action and args.resume:
        raise SystemExit("--resume is not supported together with --no-action.")
    if bool(args.vqa_dir) != bool(args.visual_vqa_model_alias):
        raise SystemExit("--vqa-dir and --visual-vqa-model-alias must be provided together.")
    if args.vqa_dir is not None and not args.vqa_dir.is_dir():
        raise SystemExit(f"--vqa-dir does not exist or is not a directory: {args.vqa_dir}")
    if args.graph_dir is not None and not args.graph_dir.is_dir():
        raise SystemExit(f"--graph-dir does not exist or is not a directory: {args.graph_dir}")
    terminal_image_resolver = (
        VqaTerminalImageResolver(
            args.vqa_dir,
            args.graph_dir,
            cache_dir=args.visual_evidence_workdir,
        )
        if args.vqa_dir is not None
        else None
    )

    requested_sample_ids = {
        item.strip()
        for raw_value in args.sample_ids
        for item in raw_value.split(",")
        if item.strip()
    }
    iterator, is_array = _iter_records(args.input)
    selected: list[tuple[int, dict[str, Any]]] = []
    matched_sample_ids: set[str] = set()
    if requested_sample_ids:
        matching_offset = 0
        for index, record in enumerate(iterator):
            record_id = _record_id(record, index)
            if record_id not in requested_sample_ids:
                continue
            matched_sample_ids.add(record_id)
            if matching_offset < offset:
                matching_offset += 1
                continue
            if args.limit is not None and len(selected) >= args.limit:
                continue
            selected.append((index, record))
        if not selected:
            raise SystemExit(
                "None of the requested sample IDs were selected. "
                f"requested={sorted(requested_sample_ids)} "
                f"matched={sorted(matched_sample_ids)}"
            )
    else:
        for index, record in enumerate(iterator):
            if index < offset:
                continue
            if args.limit is not None and len(selected) >= args.limit:
                break
            selected.append((index, record))

    checkpoint_path: Path | None = None
    resumed = 0
    optimized: dict[int, dict[str, Any]] = {}
    audits: dict[int, dict[str, Any]] = {}
    if not args.no_action:
        checkpoint_path = _prepare_rewrite_state(args, offset)
        completed: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
        if args.resume:
            for row in _read_jsonl(checkpoint_path):
                index = row.get("record_index")
                record_id = row.get("record_id")
                output = row.get("output")
                audit = row.get("audit")
                if (
                    isinstance(index, int)
                    and isinstance(record_id, str)
                    and isinstance(output, dict)
                    and isinstance(audit, dict)
                    and audit.get("status") == "ok"
                ):
                    completed[(index, record_id)] = (output, audit)
        todo: list[tuple[int, dict[str, Any]]] = []
        for index, record in selected:
            prior = completed.get((index, _record_id(record, index)))
            if prior is None:
                todo.append((index, record))
            else:
                optimized[index], audits[index] = prior
                resumed += 1
    else:
        todo = selected

    pending_checkpoints: list[dict[str, Any]] = []

    def checkpoint(index: int, record: dict[str, Any], output: dict[str, Any], audit: dict[str, Any]) -> None:
        if checkpoint_path is None or audit.get("status") != "ok":
            return
        pending_checkpoints.append(
            {
                "record_index": index,
                "record_id": _record_id(record, index),
                "output": output,
                "audit": audit,
            }
        )
        if len(pending_checkpoints) >= args.checkpoint_every:
            _append_jsonl(checkpoint_path, pending_checkpoints)
            if args.audit_jsonl:
                _append_jsonl(
                    args.audit_jsonl,
                    [row["audit"] for row in pending_checkpoints],
                )
            pending_checkpoints.clear()

    progress_enabled = args.debug_output not in {"silence", "slience"}
    if args.no_action:
        records_to_process = tqdm(
            todo,
            total=len(todo),
            desc="Debugging original trajectories",
            unit="trajectory",
            disable=not progress_enabled,
        )
        for index, record in records_to_process:
            output = copy.deepcopy(record)
            optimized[index] = output
            audits[index] = {
                "record_index": index,
                "record_id": _record_id(record, index),
                "status": "ok",
                "mode": "no_action",
                "assistant_turns": sum(
                    1 for message in record[_get_conversation_key(record)]
                    if _is_assistant(_role_and_content(message)[0])
                ),
                "changed_turns": 0,
                "verification_requests": 0,
                "integrated_verifications": 0,
                "changes": [],
            }
            if args.compare:
                _print_compare_trajectories(
                    record,
                    output,
                    _record_id(record, index),
                    args.debug_output,
                )
            else:
                _print_completed_trajectory(
                    output,
                    _record_id(record, index),
                    args.debug_output,
                    [],
                    args.verification_debug_turns,
                )
    elif args.workers == 1:
        records_to_process = tqdm(
            todo,
            total=len(selected),
            desc="Expanding trajectories",
            unit="trajectory",
            disable=not progress_enabled,
        )
        verification_total = 0
        integrated_verification_total = 0
        final_image_replacement_total = 0
        if resumed:
            records_to_process.update(resumed)
        for index, record in records_to_process:
            output, audit = _optimize_record(
                record,
                index,
                writer_alias_pool or args.model_alias,
                args.max_tokens,
                args.verification_model_alias,
                args.verification_max_tokens,
                args.verification_max_turns,
                args.debug_output,
                args.verification_debug_turns,
                args.verbose,
                args.compare,
                args.verification_workdir,
                args.visual_vqa_model_alias,
                terminal_image_resolver,
                args.visual_evidence_workdir,
                args.history_window_turns,
            )
            optimized[index] = output
            audits[index] = audit
            checkpoint(index, record, output, audit)
            if args.compare:
                _print_compare_trajectories(
                    record,
                    output,
                    _record_id(record, index),
                    args.debug_output,
                )
            verification_total += int(audit.get("verification_requests", 0))
            integrated_verification_total += int(audit.get("integrated_verifications", 0))
            final_image_replacement_total += int(
                audit.get("terminal_visual_evidence_replacements", 0)
            )
            records_to_process.set_postfix(
                verification=verification_total,
                integrated=integrated_verification_total,
                final_image_replacements=final_image_replacement_total,
                refresh=False,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(
                    _optimize_record,
                    record,
                    index,
                    writer_alias_pool or args.model_alias,
                    args.max_tokens,
                    args.verification_model_alias,
                    args.verification_max_tokens,
                    args.verification_max_turns,
                    args.debug_output,
                    args.verification_debug_turns,
                    args.verbose,
                    args.compare,
                    args.verification_workdir,
                    args.visual_vqa_model_alias,
                    terminal_image_resolver,
                    args.visual_evidence_workdir,
                    args.history_window_turns,
                ): index
                for index, record in todo
            }
            completed_futures = tqdm(
                as_completed(futures),
                total=len(selected),
                desc="Expanding trajectories",
                unit="trajectory",
                disable=not progress_enabled,
            )
            verification_total = 0
            integrated_verification_total = 0
            final_image_replacement_total = 0
            if resumed:
                completed_futures.update(resumed)
            for future in completed_futures:
                index = futures[future]
                output, audit = future.result()
                optimized[index] = output
                audits[index] = audit
                original_record = next(original for original_index, original in todo if original_index == index)
                checkpoint(index, original_record, output, audit)
                if args.compare:
                    _print_compare_trajectories(
                        original_record,
                        output,
                        _record_id(original_record, index),
                        args.debug_output,
                    )
                verification_total += int(audit.get("verification_requests", 0))
                integrated_verification_total += int(audit.get("integrated_verifications", 0))
                final_image_replacement_total += int(
                    audit.get("terminal_visual_evidence_replacements", 0)
                )
                completed_futures.set_postfix(
                    verification=verification_total,
                    integrated=integrated_verification_total,
                    final_image_replacements=final_image_replacement_total,
                    refresh=False,
                )

    if checkpoint_path is not None:
        _append_jsonl(checkpoint_path, pending_checkpoints)
        if args.audit_jsonl:
            _append_jsonl(args.audit_jsonl, [row["audit"] for row in pending_checkpoints])
        pending_checkpoints.clear()

    ordered_audits = [audits[index] for index, _ in selected]
    ok_count = sum(audit["status"] == "ok" for audit in ordered_audits)
    error_count = len(ordered_audits) - ok_count
    changed_turns = sum(int(audit.get("changed_turns", 0)) for audit in ordered_audits)
    verification_requests = sum(int(audit.get("verification_requests", 0)) for audit in ordered_audits)
    integrated_verifications = sum(int(audit.get("integrated_verifications", 0)) for audit in ordered_audits)
    terminal_visual_evidence_checks = sum(int(audit.get("terminal_visual_evidence_checks", 0)) for audit in ordered_audits)
    terminal_visual_evidence_trajectory_quality_complaint_checks = sum(
        int(audit.get("terminal_visual_evidence_trajectory_quality_complaint_checks", 0))
        for audit in ordered_audits
    )
    terminal_visual_evidence_writer_cannot_read_checks = sum(
        int(audit.get("terminal_visual_evidence_writer_cannot_read_checks", 0))
        for audit in ordered_audits
    )
    terminal_visual_evidence_replacements = sum(int(audit.get("terminal_visual_evidence_replacements", 0)) for audit in ordered_audits)
    terminal_visual_evidence_filters = sum(int(audit.get("terminal_visual_evidence_filters", 0)) for audit in ordered_audits)

    if args.audit_jsonl:
        args.audit_jsonl.parent.mkdir(parents=True, exist_ok=True)
        with args.audit_jsonl.open("w", encoding="utf-8") as handle:
            for audit in ordered_audits:
                handle.write(json.dumps(audit, ensure_ascii=False) + "\n")

    written = 0
    if not args.dry_run and args.output:
        records = [optimized[index] for index, _ in selected]
        if is_array:
            written = _write_json_array(args.output, records)
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("w", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    written += 1

    summary = {
        "input": str(args.input),
        "model_alias": args.model_alias,
        "model_alias_2": args.model_alias_2 or None,
        "writer_aliases": list(writer_alias_pool.aliases) if writer_alias_pool is not None else [],
        "history_window_turns": args.history_window_turns,
        "resume": bool(args.resume),
        "resumed": resumed,
        "checkpoint_state": str(_checkpoint_dir(args)) if not args.no_action else None,
        "verification_model_alias": args.verification_model_alias or None,
        "visual_vqa_model_alias": args.visual_vqa_model_alias or None,
        "vqa_dir": str(args.vqa_dir) if args.vqa_dir else None,
        "graph_dir": str(args.graph_dir) if args.graph_dir else None,
        "verification_max_turns": args.verification_max_turns,
        "debug_output": "silence" if args.debug_output == "slience" else args.debug_output,
        "verbose": bool(args.verbose),
        "compare": bool(args.compare),
        "no_action": bool(args.no_action),
        "dry_run": bool(args.dry_run),
        "selected": len(selected),
        "requested_sample_ids": sorted(requested_sample_ids) if requested_sample_ids else None,
        "matched_sample_ids": sorted(matched_sample_ids) if requested_sample_ids else None,
        "missing_sample_ids": (
            sorted(requested_sample_ids - matched_sample_ids) if requested_sample_ids else []
        ),
        "successful_trajectories": ok_count,
        "failed_trajectories": error_count,
        "changed_assistant_turns": changed_turns,
        "verification_requests": verification_requests,
        "integrated_verifications": integrated_verifications,
        "terminal_visual_evidence_checks": terminal_visual_evidence_checks,
        "terminal_visual_evidence_trajectory_quality_complaint_checks": terminal_visual_evidence_trajectory_quality_complaint_checks,
        "terminal_visual_evidence_writer_cannot_read_checks": terminal_visual_evidence_writer_cannot_read_checks,
        "terminal_visual_evidence_replacements": terminal_visual_evidence_replacements,
        "terminal_visual_evidence_filters": terminal_visual_evidence_filters,
        "written": written,
        "audit_jsonl": str(args.audit_jsonl) if args.audit_jsonl else None,
        "output": str(args.output) if args.output and not args.dry_run else None,
    }
    if args.debug_output not in {"silence", "slience"}:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if error_count == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
