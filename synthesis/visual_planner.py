"""Visual search planning objects and interfaces."""

from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .evidence import Evidence, EvidenceType
from .model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelResponse, ModelWorkerClient


PROMPT_VISUAL_SEARCH_PLANNER = """You are planning image searches for a multimodal deep-search data synthesis graph.

Task:
Given a Wikipedia text node split into numbered passages, find textual facts
inside the passages that can be grounded to visually unique existing evidence.
Do not brainstorm images from the page title alone. First inspect the passages,
then extract or lightly summarize only facts that are supported by a specific
source quote.

A valid visual-groundable fact must define one unambiguous visual evidence
target: a specific event, artifact, version, scene, document, object, named
place, appearance, cover, figure, official visual identity, or other visual
fact. Different acceptable images may vary by angle, crop, or resolution, but
they must depict the same target and let a human decide whether a candidate
image matches the description.

Procedure:
1. Read the numbered passages, not just the title.
2. Identify passages that mention a concrete fact with visual evidence.
3. Reject facts that cannot be tied to an exact source quote.
4. For each accepted fact, explain why the visual target is unique.
5. Rewrite the fact into 2 to 4 precise image-search queries.

Keep targets that:
- are explicitly supported by source_quote from the passages;
- are likely searchable on Wikimedia Commons or image search;
- have enough constraints such as year, event name, award ceremony, edition,
  version, title, location, named artifact, named document, product model,
  landmark, outfit, or moment;
- can become a reliable intermediate image node in a multi-hop question.

Reject targets if:
- they are based only on the page title or general knowledge;
- they are future, unreleased, speculative, or unstable;
- many visually different images would satisfy the description;
- the target mixes alternatives using wording like "or", "such as", "logo/banner/screenshot";
- it is a generic portrait/photo/game/celebration/building/city skyline without
  a specific event, date, object, version, landmark, or viewpoint;
- it is a pure text fact with no useful visual search target.

Good target examples:
- Source quote mentions a singer's debut album title.
  Good target: the debut album cover, because the cover is a stable artifact.
- Source quote mentions an actor appearing at a named award ceremony in a
  specific year.
  Good target: that on-stage appearance, because person, event, outfit, and
  context constrain the image.
- Source quote mentions a specific film.
  Good target: the official theatrical poster for that film, if the poster is
  a stable visual artifact.
- Source quote mentions a named landmark, product model, scientific figure,
  legal document, map, official logo, book cover, or document page.
  Good target: that exact visual artifact/object, if uniquely determined.

Bad target examples:
- "photo of Justin Bieber": too many unrelated images satisfy it.
- "Messi winning the Champions League": this happened multiple times and is
  not unique unless the year/match/moment is specified.
- "Los Angeles Lakers championship celebration": many seasons and moments
  qualify unless the exact Finals/year/moment is specified.
- "South Korea city skyline": too broad without city, viewpoint, landmark, or date.
- "2026-27 Los Angeles Lakers team photo": future/unreleased visual evidence
  is unstable and may not exist.

Output at most 3 targets. If no passage supports a suitable target, output nothing.
Each target must contain 2 to 4 queries.
Do not output markdown, JSON, explanations, or extra text.

Output format:
<target>
source_passage: P3
source_quote: exact quote or close excerpt from the provided passage
description: concrete visual evidence target
uniqueness: why this points to one unambiguous visual target
reason: short reason this target is useful for multimodal multi-hop data
expected_visual: what the image should visibly contain, including uniqueness constraints
query: image search query 1
query: image search query 2
</target>
"""


def _stable_hash(*parts: object, length: int = 16) -> str:
    payload = "||".join("" if part is None else str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()[:length]


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


class VisualTargetType(str, Enum):
    EVENT_PHOTO = "event_photo"
    POSTER = "poster"
    FIGURE = "figure"
    ARTWORK = "artwork"
    PRODUCT = "product"
    LOGO = "logo"
    MAP = "map"
    DOCUMENT = "document"
    GROUP_PHOTO = "group_photo"
    OBJECT_DETAIL = "object_detail"
    SCREENSHOT = "screenshot"
    OTHER = "other"


class DownstreamUse(str, Enum):
    ANSWER_EVIDENCE = "answer_evidence"
    ROUTING_CLUE = "routing_clue"
    GROUNDING = "grounding"
    DISTRACTOR = "distractor"


@dataclass(slots=True)
class SearchQuerySpec:
    """One text-to-image query proposed for a visual target."""

    query_id: str
    query: str
    target_evidence_id: str
    intent: str | None = None
    expected_visual: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    @classmethod
    def create(
        cls,
        query: str,
        target_evidence_id: str,
        *,
        intent: str | None = None,
        expected_visual: str | None = None,
        source: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "SearchQuerySpec":
        return cls(
            query_id=f"query_{_stable_hash(target_evidence_id, query, intent)}",
            query=query,
            target_evidence_id=target_evidence_id,
            intent=intent,
            expected_visual=expected_visual,
            source=source,
            metadata=metadata or {},
        )


@dataclass(slots=True)
class VisualSearchPlan:
    """MLLM-produced visual target plus the queries used to search for it."""

    plan_id: str
    target: Evidence
    queries: list[SearchQuerySpec] = field(default_factory=list)
    source_node_id: str | None = None
    source_evidence_ids: list[str] = field(default_factory=list)
    planner: str | None = None
    raw_model_asset_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = _jsonify(asdict(self))
        data["target"] = self.target.to_dict()
        data["queries"] = [query.to_dict() for query in self.queries]
        return data

    @classmethod
    def create(
        cls,
        target: Evidence,
        *,
        queries: list[SearchQuerySpec] | None = None,
        source_node_id: str | None = None,
        source_evidence_ids: list[str] | None = None,
        planner: str | None = None,
        raw_model_asset_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "VisualSearchPlan":
        return cls(
            plan_id=f"visual_plan_{_stable_hash(target.evidence_id)}",
            target=target,
            queries=queries or [],
            source_node_id=source_node_id,
            source_evidence_ids=source_evidence_ids or [],
            planner=planner,
            raw_model_asset_id=raw_model_asset_id,
            metadata=metadata or {},
        )


class VisualSearchPlanner(Protocol):
    """Plan visual targets and image-search queries from a text node."""

    model_client: ModelWorkerClient

    def plan(
        self,
        *,
        node: dict[str, Any],
        page_text: str,
        source_evidence_ids: list[str] | None = None,
        run_id: str | None = None,
    ) -> list[VisualSearchPlan]:
        """Return target evidences together with their image-search queries."""


class LLMVisualSearchPlanner:
    """LLM-backed visual target and image query planner."""

    planner_name = "llm_visual_search_planner"

    def __init__(
        self,
        *,
        model_client: ModelWorkerClient | None = None,
        model_alias: str | None = None,
        max_targets: int = 3,
        max_queries_per_target: int = 4,
        min_query_terms: int = 3,
    ) -> None:
        self.model_client = model_client or LLM_WORKER
        self.model_alias = model_alias
        self.max_targets = max_targets
        self.max_queries_per_target = max_queries_per_target
        self.min_query_terms = min_query_terms

    def plan(
        self,
        *,
        node: dict[str, Any],
        page_text: str,
        source_evidence_ids: list[str] | None = None,
        run_id: str | None = None,
    ) -> list[VisualSearchPlan]:
        model_alias = self.model_alias or os.environ.get("VISUAL_PLANNER_MODEL") or os.environ.get("TEXT_PROCESS_MODEL")
        if not model_alias:
            raise ValueError("VISUAL_PLANNER_MODEL or TEXT_PROCESS_MODEL is required for visual planning.")

        response = self.model_client.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=PROMPT_VISUAL_SEARCH_PLANNER),
                    ModelMessage(role="user", content=self._prompt_input(node, page_text)),
                ],
                temperature=0.0,
            )
        )
        candidates = self._parse_targets(response.content)
        plans: list[VisualSearchPlan] = []
        for candidate in candidates:
            if len(plans) >= self.max_targets:
                break
            plan = self._candidate_to_plan(
                candidate,
                node=node,
                source_evidence_ids=source_evidence_ids or [],
                raw_output=response.content,
                run_id=run_id,
            )
            if plan is not None:
                plans.append(plan)
        return plans

    @staticmethod
    def _prompt_input(node: dict[str, Any], page_text: str) -> str:
        title = node.get("title") or ""
        attributes = node.get("attributes") or {}
        return (
            f"Title:\n{title}\n\n"
            f"Attributes:\n{attributes}\n\n"
            "Numbered passages:\n"
            f"{LLMVisualSearchPlanner._numbered_passages(page_text)}"
        )

    @staticmethod
    def _numbered_passages(page_text: str, *, max_passages: int = 80, max_chars_per_passage: int = 1200) -> str:
        blocks = [block.strip() for block in re.split(r"\n\s*\n", page_text or "") if block.strip()]
        passages: list[str] = []
        for block in blocks:
            compact = re.sub(r"[ \t]+", " ", block).strip()
            if not compact:
                continue
            if len(compact) > max_chars_per_passage:
                compact = compact[:max_chars_per_passage].rstrip() + " ..."
            passages.append(compact)
            if len(passages) >= max_passages:
                break
        if not passages:
            return "P1: "
        return "\n\n".join(f"P{index}: {passage}" for index, passage in enumerate(passages, start=1))

    @classmethod
    def _parse_targets(cls, text: str) -> list[dict[str, Any]]:
        candidates: list[dict[str, Any]] = []
        for block in re.findall(r"<target>(.*?)</target>", text, flags=re.DOTALL | re.IGNORECASE):
            fields = cls._parse_target_block(block)
            if fields:
                candidates.append(fields)
        return candidates

    @staticmethod
    def _parse_target_block(block: str) -> dict[str, Any]:
        fields: dict[str, Any] = {"queries": []}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not value:
                continue
            if key == "query":
                fields["queries"].append(value)
            else:
                fields[key] = value
        return fields

    def _candidate_to_plan(
        self,
        candidate: dict[str, Any],
        *,
        node: dict[str, Any],
        source_evidence_ids: list[str],
        raw_output: str,
        run_id: str | None,
    ) -> VisualSearchPlan | None:
        description = candidate.get("description")
        source_quote = candidate.get("source_quote")
        uniqueness = candidate.get("uniqueness")
        queries = self._filter_queries(candidate.get("queries") or [], node_title=node.get("title"))
        if not description or not source_quote or not uniqueness or not queries:
            return None

        target_type = VisualTargetType.OTHER
        downstream_use = DownstreamUse.ROUTING_CLUE
        source_node_id = node.get("node_id")
        target = Evidence.create(
            EvidenceType.VISUAL_TARGET,
            content=description,
            node_ids=[source_node_id] if source_node_id else [],
            extractor=self.planner_name,
            confidence=None,
            metadata={
                "target_type": target_type.value,
                "downstream_use": downstream_use.value,
                "source_passage": candidate.get("source_passage"),
                "source_quote": source_quote,
                "uniqueness": uniqueness,
                "reason": candidate.get("reason"),
                "expected_visual": candidate.get("expected_visual"),
                "source_evidence_ids": source_evidence_ids,
                "run_id": run_id,
            },
            evidence_key=f"{source_node_id}:{description}",
        )
        query_specs = [
            SearchQuerySpec.create(
                query,
                target.evidence_id,
                intent=target_type.value,
                expected_visual=candidate.get("expected_visual"),
                source=self.planner_name,
                metadata={
                    "downstream_use": downstream_use.value,
                    "source_passage": candidate.get("source_passage"),
                    "source_quote": source_quote,
                    "uniqueness": uniqueness,
                    "reason": candidate.get("reason"),
                },
            )
            for query in queries
        ]
        return VisualSearchPlan.create(
            target,
            queries=query_specs,
            source_node_id=source_node_id,
            source_evidence_ids=source_evidence_ids,
            planner=self.planner_name,
            metadata={
                "raw_model_output_preview": raw_output[:2000],
                "target_type": target_type.value,
                "downstream_use": downstream_use.value,
                "source_passage": candidate.get("source_passage"),
                "source_quote": source_quote,
                "uniqueness": uniqueness,
            },
        )

    def _filter_queries(self, queries: list[str], *, node_title: str | None) -> list[str]:
        seen: set[str] = set()
        filtered: list[str] = []
        title = (node_title or "").strip().lower()
        for query in queries:
            normalized = re.sub(r"\s+", " ", query).strip()
            if not normalized:
                continue
            key = normalized.lower()
            if key in seen:
                continue
            if title and key == title:
                continue
            if len(normalized.split()) < self.min_query_terms:
                continue
            seen.add(key)
            filtered.append(normalized)
            if len(filtered) >= self.max_queries_per_target:
                break
        return filtered

    @staticmethod
    def _target_type(value: str | None) -> VisualTargetType:
        try:
            return VisualTargetType((value or "").strip())
        except ValueError:
            return VisualTargetType.OTHER

    @staticmethod
    def _downstream_use(value: str | None) -> DownstreamUse:
        try:
            return DownstreamUse((value or "").strip())
        except ValueError:
            return DownstreamUse.ROUTING_CLUE


def _smoke_test() -> None:
    class MockModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            assert request.model == "mock_planner"
            return ModelResponse(
                content="""<target>
description: Kobe Bryant final game jersey photo
source_passage: P1
source_quote: Kobe Bryant played his final game in 2016.
uniqueness: The final NBA game in 2016 is a specific event and uniform appearance.
reason: iconic visual clue
expected_visual: Kobe Bryant wearing his final game uniform
query: Kobe Bryant final game jersey photo
query: Kobe Bryant 2016 final game uniform
</target>"""
            )

    planner = LLMVisualSearchPlanner(model_client=MockModel(), model_alias="mock_planner")
    plans = planner.plan(
        node={"node_id": "text_1", "title": "Kobe Bryant", "attributes": {"team": "Lakers"}},
        page_text="Kobe Bryant played his final game in 2016.",
        source_evidence_ids=["evidence_1"],
        run_id="run_smoke",
    )
    assert len(plans) == 1
    assert plans[0].source_node_id == "text_1"
    assert len(plans[0].queries) == 2
    assert plans[0].target.metadata["source_passage"] == "P1"
    assert "final game" in plans[0].target.metadata["source_quote"]
    prompt_input = planner._prompt_input(
        {"node_id": "text_1", "title": "Kobe Bryant", "attributes": {}},
        "First paragraph.\n\nSecond paragraph.",
    )
    assert "P1: First paragraph." in prompt_input
    assert "P2: Second paragraph." in prompt_input
    print("visual_planner smoke test passed")


if __name__ == "__main__":
    _smoke_test()
