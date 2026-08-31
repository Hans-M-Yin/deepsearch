"""Validate terminal visual evidence before polishing an SFT final answer.

This module is intentionally conservative.  It is only called after the
reasoning writer explicitly flags an end-of-trajectory image as too small or
unclear.  A vision-only answer is first obtained from that downloaded image.
If it disagrees with the stored final answer, the terminal image node from the
original VQA construction graph is tried.  A replacement is allowed only when
that graph image independently supports the stored answer.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

import requests

from synthesis.model_worker import LLM_WORKER, ModelMessage, ModelRequest


LOCAL_QUESTION_SYSTEM_PROMPT = """You turn a multi-hop question into one short
visual-only question for an image-quality check.  Keep only the final property
that must be read directly from the target image.  Remove all entity names,
historical chains, and facts that would let someone answer from memory.  Do
not supply or imply an answer.  Return exactly one natural-language question.
"""

VQA_SYSTEM_PROMPT = """Answer the user's question using only visibly readable
content in the supplied image.  Do not use outside knowledge, search, the
image filename, or a remembered answer.  If the requested feature is too
small, blurred, cropped out, or otherwise not visually identifiable, output
exactly UNREADABLE.  Otherwise output only a concise answer, with no reasoning.
"""

SEMANTIC_MATCH_SYSTEM_PROMPT = """Decide whether a visual-only candidate answer
and a stored trajectory answer give the same answer to the supplied visual
question. Judge only the property asked by that question. The stored answer
may include a long multi-hop explanation, names, dates, and other facts that
are irrelevant to this visual property; ignore all of that surrounding
material. Ignore formatting, articles, and harmless wording differences, but
do not accept a broader, weaker, guessed, contradictory, or merely related
answer. Output exactly MATCH or MISMATCH.
"""

_LOW_QUALITY_PATTERNS = (
    r"\b(?:low[- ]?resolution|low[- ]?res|thumbnail|tiny|too small|small image)\b",
    r"\b(?:blurry|blurred|out of focus|pixelated|indistinct|illegible)\b",
    r"\b(?:cannot|can't|unable to)\s+(?:see|read|make out|identify|inspect)\b",
    r"\b(?:not clear|unclear|insufficient (?:visual )?(?:detail|resolution|quality))\b",
)


@dataclass(frozen=True, slots=True)
class TerminalImageNode:
    node_id: str
    image_source: str
    source: str


@dataclass(slots=True)
class VisualEvidenceOutcome:
    status: str
    audit: dict[str, Any]
    replacement_path: str | None = None
    image_index: int | None = None


def final_answer_text(text: str) -> str:
    match = re.search(r"<answer\b[^>]*>(.*?)</answer>", str(text or ""), re.I | re.S)
    return (match.group(1) if match else str(text or "")).strip()


def has_low_quality_complaint(*texts: str) -> bool:
    joined = "\n".join(str(text or "") for text in texts)
    return any(re.search(pattern, joined, re.I) for pattern in _LOW_QUALITY_PATTERNS)


def terminal_observation_image_index(messages: list[Any], final_assistant_index: int) -> int | None:
    """Return the positional ``images`` index for the observation before answer."""

    image_index = 0
    previous_image_index: int | None = None
    previous_role = ""
    for index, message in enumerate(messages[:final_assistant_index]):
        role = str(message.get("from") or message.get("role") or "").lower() if isinstance(message, dict) else ""
        content = ""
        if isinstance(message, dict):
            content = str(message.get("value") or message.get("content") or message.get("response_text") or "")
        count = content.count("<image>")
        if count:
            if role in {"observation", "tool"}:
                previous_image_index = image_index + count - 1
                previous_role = role
            image_index += count
    # Only an immediately preceding visual observation is a valid terminal
    # evidence target.  Earlier images may be question inputs or old searches.
    if final_assistant_index <= 0:
        return None
    immediate = messages[final_assistant_index - 1]
    immediate_role = str(immediate.get("from") or immediate.get("role") or "").lower() if isinstance(immediate, dict) else ""
    immediate_content = str((immediate.get("value") or immediate.get("content") or immediate.get("response_text") or "") if isinstance(immediate, dict) else "")
    if immediate_role not in {"observation", "tool"} or "<image>" not in immediate_content:
        return None
    return previous_image_index if previous_role in {"observation", "tool"} else None


class VqaTerminalImageResolver:
    """Resolve a question/path id to the terminal graph image node.

    ``questions.jsonl`` is optional metadata; ``samples.jsonl`` contains the
    path and terminal-image-node id.  When the sample does not retain a direct
    URL, ``nodes.jsonl`` in ``graph_dir`` is the authoritative fallback.
    """

    def __init__(
        self,
        vqa_dir: Path | str,
        graph_dir: Path | str | None = None,
        *,
        cache_dir: Path | str | None = None,
    ) -> None:
        self.vqa_dir = Path(vqa_dir)
        self.graph_dir = Path(graph_dir) if graph_dir else None
        self.cache_dir = Path(cache_dir) if cache_dir else self.vqa_dir / ".terminal_visual_evidence_cache"
        self._samples: dict[str, dict[str, str]] = {}
        self._nodes: dict[str, str] | None = None
        self._lock = threading.Lock()
        self._load_samples()

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if not path.is_file():
            return records
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict):
                    records.append(payload)
        return records

    @staticmethod
    def _fingerprint(path: Path) -> dict[str, Any]:
        stat = path.stat()
        return {"path": str(path.resolve()), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}

    def _read_cached_index(self, path: Path, fingerprint: dict[str, Any]) -> dict[str, Any] | None:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        if not isinstance(payload, dict) or payload.get("version") != 1:
            return None
        if payload.get("fingerprint") != fingerprint or not isinstance(payload.get("entries"), dict):
            return None
        return payload["entries"]

    def _write_cached_index(self, path: Path, fingerprint: dict[str, Any], entries: dict[str, Any]) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{threading.get_ident()}")
        temporary.write_text(
            json.dumps({"version": 1, "fingerprint": fingerprint, "entries": entries}, ensure_ascii=False),
            encoding="utf-8",
        )
        temporary.replace(path)

    def _load_samples(self) -> None:
        """Build/load a compact index instead of retaining full VQA samples.

        Raw samples are intentionally verbose (candidate evaluations and model
        traces make each line large).  The visual-repair branch needs only the
        path id, terminal node id, and an optional unambiguous terminal image
        source, so retain exactly those fields.
        """

        source_path = self.vqa_dir / "samples.jsonl"
        if not source_path.is_file():
            raise FileNotFoundError(f"VQA samples file not found: {source_path}")
        fingerprint = self._fingerprint(source_path)
        # v2 additionally preserves the construction-time, image-validated
        # terminal ask_target.  Keep a separate cache name so a pre-existing
        # v1 index cannot silently drop that crucial visual constraint.
        index_path = self.cache_dir / "vqa_terminal_image_index_v2.json"
        cached = self._read_cached_index(index_path, fingerprint)
        if cached is not None:
            self._samples = {
                str(key): {str(name): str(value) for name, value in item.items()}
                for key, item in cached.items()
                if isinstance(item, dict)
            }
            return
        entries: dict[str, dict[str, str]] = {}
        for record in self._load_jsonl(source_path):
            node_id = self._terminal_node_id(record)
            if not node_id:
                continue
            image_source = ""
            for container in (
                record.get("question_terminal_bridge"),
                record.get("image_target_terminal_normalization"),
            ):
                if isinstance(container, dict):
                    image_source = self._image_source(container)
                    if image_source:
                        break
            path = record.get("path") if isinstance(record.get("path"), dict) else {}
            value = {
                "node_id": node_id,
                "image_source": image_source,
                "terminal_ask_target": self._terminal_ask_target(record),
            }
            for key in (record.get("sample_id"), path.get("path_id")):
                normalized = str(key or "").strip()
                if normalized:
                    entries[normalized] = value
        self._samples = entries
        self._write_cached_index(index_path, fingerprint, entries)

    def _load_nodes(self) -> dict[str, str]:
        with self._lock:
            if self._nodes is not None:
                return self._nodes
            if self.graph_dir is None:
                self._nodes = {}
                return self._nodes
            nodes_path = self.graph_dir / "nodes.jsonl"
            if not nodes_path.is_file():
                raise FileNotFoundError(f"graph nodes file not found: {nodes_path}")
            fingerprint = self._fingerprint(nodes_path)
            index_path = self.cache_dir / "graph_image_node_index_v1.json"
            cached = self._read_cached_index(index_path, fingerprint)
            if cached is not None:
                self._nodes = {str(key): str(value) for key, value in cached.items() if str(value).strip()}
                return self._nodes
            image_nodes: dict[str, str] = {}
            for node in self._load_jsonl(nodes_path):
                if str(node.get("node_type") or "").lower() != "image":
                    continue
                node_id = str(node.get("node_id") or "").strip()
                source = self._image_source(node)
                if node_id and source:
                    image_nodes[node_id] = source
            self._nodes = image_nodes
            self._write_cached_index(index_path, fingerprint, image_nodes)
            return self._nodes

    @staticmethod
    def _terminal_node_id(sample: dict[str, Any]) -> str:
        for container_key in ("question_terminal_bridge", "image_target_terminal_normalization"):
            container = sample.get(container_key)
            if isinstance(container, dict):
                for key in ("terminal_image_node_id", "image_node_id"):
                    value = str(container.get(key) or "").strip()
                    if value:
                        return value
        path = sample.get("path") if isinstance(sample.get("path"), dict) else {}
        node_ids = path.get("node_ids") if isinstance(path.get("node_ids"), list) else []
        node_types = path.get("node_types") if isinstance(path.get("node_types"), list) else []
        if node_ids and node_types and len(node_ids) == len(node_types) and str(node_types[-1]).lower() == "image":
            return str(node_ids[-1]).strip()
        return ""

    @staticmethod
    def _terminal_ask_target(sample: dict[str, Any]) -> str:
        """Return the already image-validated terminal visual question.

        VQA construction deliberately records this question before converting
        the graph into a multi-hop natural-language task.  It preserves
        spatial qualifiers such as "besides the raised red card" which a
        generic writer-generated summary can easily lose.
        """

        containers: list[Any] = [
            sample.get("question_terminal_bridge"),
            sample.get("question_target_ask"),
            sample.get("target_ask"),
        ]
        terminal_normalization = sample.get("image_target_terminal_normalization")
        if isinstance(terminal_normalization, dict):
            containers.append(terminal_normalization.get("question_target_ask"))
        for container in containers:
            if not isinstance(container, dict):
                continue
            # raw_ask_target is the pre-bridging formulation and therefore
            # carries the most exact visual disambiguators.
            for key in ("raw_ask_target", "ask_target", "rewritten_ask_target"):
                value = str(container.get(key) or "").strip()
                if value:
                    return value
        return ""

    @staticmethod
    def _image_source(payload: dict[str, Any]) -> str:
        for key in ("terminal_image_url", "image_url", "oss_uri", "local_path", "path", "file_path", "thumb_oss_uri"):
            value = str(payload.get(key) or "").strip()
            if value:
                return value
        return ""

    def resolve(self, record: dict[str, Any]) -> TerminalImageNode | None:
        sample = self._sample_for_record(record)
        if not sample:
            return None
        node_id = str(sample.get("node_id") or "").strip()
        if not node_id:
            return None
        source = str(sample.get("image_source") or "").strip()
        if source:
            return TerminalImageNode(node_id=node_id, image_source=source, source="vqa_metadata")
        source = self._load_nodes().get(node_id)
        if not source:
            return None
        return TerminalImageNode(node_id=node_id, image_source=source, source="graph_node")

    def terminal_ask_target(self, record: dict[str, Any]) -> str | None:
        """Get the VQA construction-time terminal question for this record."""

        sample = self._sample_for_record(record)
        value = str((sample or {}).get("terminal_ask_target") or "").strip()
        return value or None

    def _sample_for_record(self, record: dict[str, Any]) -> dict[str, str] | None:
        keys = [str(record.get(key) or "").strip() for key in ("sample_id", "path_id", "question_id", "id")]
        return next((self._samples[key] for key in keys if key in self._samples), None)


def _request_text(
    model_alias: str,
    system: str,
    user_content: Any,
    *,
    max_tokens: int | None,
    metadata: dict[str, str],
) -> str:
    response = LLM_WORKER.generate(
        ModelRequest(
            model=model_alias,
            messages=[ModelMessage(role="system", content=system), ModelMessage(role="user", content=user_content)],
            metadata=metadata,
        )
    )
    text = str(getattr(response, "content", "") or "").strip()
    if not text:
        raise ValueError("model returned empty content")
    return text


def _auxiliary_max_tokens(max_tokens: int | None, cap: int) -> int:
    """Bound short auxiliary calls when the parent writer has no cap."""
    return cap if max_tokens is None else min(max_tokens, cap)


def _semantic_match(
    *,
    model_alias: str,
    visual_question: str,
    stored_answer: str,
    candidate_answer: str,
    max_tokens: int | None,
    metadata: dict[str, str],
) -> bool:
    result = _request_text(
        model_alias,
        SEMANTIC_MATCH_SYSTEM_PROMPT,
        "Visual question (the only target for comparison):\n"
        + visual_question
        + "\n\nStored trajectory answer (may contain irrelevant explanation):\n"
        + stored_answer
        + "\n\nVisual-only candidate answer:\n"
        + candidate_answer,
        max_tokens=max_tokens,
        metadata=metadata,
    ).strip().upper()
    if result == "MATCH":
        return True
    if result == "MISMATCH":
        return False
    raise ValueError(f"semantic judge must return MATCH or MISMATCH, got {result!r}")


def _download_replacement(source: str, workdir: Path) -> str:
    if not source.startswith(("http://", "https://")):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"terminal image node path does not exist: {source}")
        # The rewritten ShareGPT dataset may later be trained from a different
        # working directory (for example ``SFT/``).  A repository-relative
        # cache path such as ``data/.../terminal_image_node.png`` would then
        # become invalid.  Persist an absolute path for any externally supplied
        # local terminal image instead.
        return str(path.resolve())
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    suffix = Path(urlparse(source).path).suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        suffix = ".img"
    destination = workdir / f"terminal_image_node_{digest}{suffix}"
    workdir.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return str(destination.resolve())
    response = requests.get(source, timeout=60)
    response.raise_for_status()
    if not response.content:
        raise ValueError(f"terminal image node download was empty: {source}")
    temporary = destination.with_suffix(destination.suffix + f".tmp-{os.getpid()}")
    temporary.write_bytes(response.content)
    temporary.replace(destination)
    return str(destination.resolve())


def run_terminal_visual_check(
    *,
    record: dict[str, Any],
    messages: list[Any],
    final_assistant_index: int,
    original_final_response: str,
    editor_model_alias: str,
    vqa_model_alias: str,
    resolver: VqaTerminalImageResolver,
    image_content_part: Callable[[str], dict[str, Any] | None],
    workdir: Path,
    max_tokens: int | None,
    metadata: dict[str, str],
) -> VisualEvidenceOutcome:
    image_index = terminal_observation_image_index(messages, final_assistant_index)
    images = [str(item) for item in (record.get("images") or [])]
    stored_answer = final_answer_text(original_final_response)
    audit: dict[str, Any] = {
        "stored_answer": stored_answer,
        "terminal_image_index": image_index,
    }
    if image_index is None or image_index >= len(images):
        return VisualEvidenceOutcome("error", {**audit, "error": "terminal_observation_image_not_found"})
    current_image = images[image_index]
    question = str(record.get("question") or "")
    if not question:
        for message in messages:
            if isinstance(message, dict) and str(message.get("from") or message.get("role") or "").lower() in {"human", "user"}:
                question = str(message.get("value") or message.get("content") or "")
                break
    try:
        local_question = resolver.terminal_ask_target(record)
        if local_question:
            audit["local_visual_question_source"] = "vqa_terminal_ask_target"
        else:
            local_question = _request_text(
                editor_model_alias,
                LOCAL_QUESTION_SYSTEM_PROMPT,
                "Original question:\n" + question,
                max_tokens=_auxiliary_max_tokens(max_tokens, 1024),
                metadata=metadata,
            )
            audit["local_visual_question_source"] = "writer_generated_fallback"
        audit["local_visual_question"] = local_question
        current_part = image_content_part(current_image)
        if current_part is None:
            return VisualEvidenceOutcome("error", {**audit, "error": "terminal_image_unavailable"}, image_index=image_index)
        current_answer = _request_text(
            vqa_model_alias,
            VQA_SYSTEM_PROMPT,
            [{"type": "text", "text": local_question}, current_part],
            max_tokens=_auxiliary_max_tokens(max_tokens, 512),
            metadata=metadata,
        )
        audit["downloaded_image_answer"] = current_answer
        if current_answer.strip().upper() != "UNREADABLE" and _semantic_match(
            # The same dedicated visual model both answers the local question
            # and judges semantic agreement.  The writer is deliberately kept
            # out of this label-validation decision.
            model_alias=vqa_model_alias,
            visual_question=local_question,
            stored_answer=stored_answer,
            candidate_answer=current_answer,
            max_tokens=_auxiliary_max_tokens(max_tokens, 256),
            metadata=metadata,
        ):
            return VisualEvidenceOutcome("matched_original", audit, image_index=image_index)

        node = resolver.resolve(record)
        if node is None:
            return VisualEvidenceOutcome("filter", {**audit, "error": "terminal_image_node_unavailable"}, image_index=image_index)
        audit["terminal_image_node_id"] = node.node_id
        audit["terminal_image_node_source"] = node.source
        replacement_path = _download_replacement(node.image_source, workdir)
        replacement_part = image_content_part(replacement_path)
        if replacement_part is None:
            return VisualEvidenceOutcome("filter", {**audit, "error": "terminal_image_node_not_usable"}, image_index=image_index)
        replacement_answer = _request_text(
            vqa_model_alias,
            VQA_SYSTEM_PROMPT,
            [{"type": "text", "text": local_question}, replacement_part],
            max_tokens=_auxiliary_max_tokens(max_tokens, 512),
            metadata=metadata,
        )
        audit["terminal_node_image_answer"] = replacement_answer
        if replacement_answer.strip().upper() != "UNREADABLE" and _semantic_match(
            model_alias=vqa_model_alias,
            visual_question=local_question,
            stored_answer=stored_answer,
            candidate_answer=replacement_answer,
            max_tokens=_auxiliary_max_tokens(max_tokens, 256),
            metadata=metadata,
        ):
            return VisualEvidenceOutcome("replaced", audit, replacement_path=replacement_path, image_index=image_index)
        return VisualEvidenceOutcome("filter", audit, image_index=image_index)
    except Exception as exc:
        return VisualEvidenceOutcome("error", {**audit, "error": repr(exc)}, image_index=image_index)
