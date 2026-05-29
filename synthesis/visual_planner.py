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


PROMPT_VISUAL_SEARCH_PLANNER = """You are adding illustrative images to a Wikipedia page.

You are given a Wikipedia text node split into numbered passages, containing a brief introduction, biography, and important events related to the entity. Your task is to identify passages that are worth illustrating with images, and then search internet image search engines for images corresponding to those passages.

A passage is considered worth illustrating if it has **clear visual evidence** and can correspond completely to **one specific image**. In other words, the text should be visually equivalent to the image you are trying to retrieve.

For example:
- If a passage mentions a singer’s album, the passage can explicitly point to the album cover, so the album cover can be used as the illustration for that passage.
- If a passage about Shou Chew mentions his appearance at a U.S. congressional hearing, the passage clearly points to photographs of Shou Chew during that hearing.

# Requirements

1. The image must be **unique**.  
The image must be the only image that truly satisfies the passage. In simple terms, different people reading the same passage should retrieve the same image target.

For passages that do not naturally correspond to a unique image, you may rewrite or refine the passage so that it points to only one unique image target.

Some categories of images are considered inherently unique.  
Examples include paintings, buildings, album covers, or specific moments from historical events. Even if different photographers captured different photos, the underlying visual content is effectively the same, so this still satisfies uniqueness.

However, the event itself must be **specific and unambiguous**.

For example:
- “Los Angeles Lakers championship parade” is ambiguous because it could refer to multiple years.
- “Photo of the 2008 Los Angeles Lakers championship parade” clearly points to Lakers players celebrating on parade buses in 2008.

Another example:
- “1960 Los Angeles Lakers vs Boston Celtics game” is still ambiguous because many completely different moments from the game could satisfy the description, even though everyone would retrieve images from the same game.

2. The image must actually exist on the internet. The supplied URL shoule be existing.
For example:
- “The final shot of the 1960 Lakers vs Boston Celtics game” may point to a unique historical moment, but there may be no surviving image of that exact moment online.

3. Naive or trivial images should be ignored.
Do not output images that are too visually simple or too semantically shallow to support useful multi-hop reasoning. In particular, avoid pure logos, wordmarks, icons, generic portraits, UI screenshots, plain document scans, text-dominant posters, flags, simple maps, default profile photos, standard ID-style headshots, and plain white-background product shots. These images are usually only useful for basic identity recognition and do not provide rich enough visual content for follow-up reasoning.X
The query must not contain any explicit URL, domain name, filename, image identifier, or other direct locator of the image. It must contain only semantic information. The query should stand on its own: a user who searches by the query alone, without seeing the URL, should be able to retrieve the same image or an equivalent depiction of the same unique visual target. Do not rely on the URL itself to make the target appear unique.

# Goals

1. You may examine the provided Wikipedia passages one by one, analyze whether the event or object described in each passage is visually unique, and then rewrite the passage into a concise and precise form suitable for image search.

2. You may also use your own knowledge about the subject to propose additional specific events or related objects that are not explicitly mentioned in the Wikipedia text, and rewrite them into search-ready passages.

3. The number of unique image materials corresponding to the subject is uncertain. If you believe no suitable image exists, you may output nothing. Otherwise, output at most 4 passages.

4. We will directly use your rewritten text for image search. Please strictly follow the format below:

The query must not contain any explicit URL, domain name, filename, image identifier, or other direct locator of the image. It must contain only semantic information. The query should stand on its own: a user who searches by the query alone, without seeing the URL, should be able to retrieve the same image or an equivalent depiction of the same unique visual target. Do not rely on the URL itself to make the target appear unique.

```text
<query>Your rewritten text</query><url>The image URL that satisfy your query</url>
<reason>Explain why this text satisfies the requirements, including how it fulfills the three conditions above.</reason>
```

Repeat the format for multiple results.

# Example

Entity: Lionel Messi
Content: ...(Emit here)...

<query>Argentina national team lifting the trophy after winning the 2022 FIFA World Cup final</query>
<url>https://lh4.googleusercontent.com/proxy/qRa-xkqyCyZHdrq6zfRapSwk9Tk-bW6KLmZ3YL_dBCz1rBDty7HCI2_LNExqebZO0D-2yuQwECRA6rMkANAKelDBuEgCFAdSoUdIkog-kuMTTPCahEG4x16pBuAuqOWzC7arSZC8bw</url>
<reason>
1. This passage points to one unique historical moment: Argentina winning the 2022 World Cup and lifting the trophy. Even though different photographers may have taken different images, the visual content is fundamentally consistent: the Argentina team on the award stage, with Lionel Messi at the front holding the trophy. Therefore, it satisfies the uniqueness requirement.
2. This was a globally covered event with extensive media photography, so many matching images exist online. The provided URL is valid and accessible.
3. This image does not belong to any trivial case.
</reason>

<query>Photo of Lionel Messi sleeping while holding the FIFA World Cup trophy</query>
<url>https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQJg3YxEnhdWa8iUEpGPOgaflKltqz41y2QfA&s</url>
<reason>
1. Although this query does not explicitly specify a date, internet evidence shows that it clearly refers to the famous photo Messi posted on Instagram after winning the 2022 FIFA World Cup. There are no other widely known or competing images matching the description “Messi sleeping while holding the World Cup trophy.” The only images satisfying this query originate from Messi’s official post, so the visual target is effectively unique. Even if the image is reposted, cropped, or compressed by different media sources, the visual content remains the same.
2. The image genuinely exists online, was officially published by Messi himself, and is widely documented across news outlets and social media. The purpose of this query is specifically to retrieve that exact image through image search.
3. The image is quite unique, and does not belong to any trivial case.
</reason>

Now, strictly follow all the requirements and goals above to complete the following person.
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
        max_targets: int = 4,
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
            "Complete the following person.\n\n"
            f"Entity: {title}\n"
            f"Attributes: {attributes}\n\n"
            "Content:\n"
            f"{LLMVisualSearchPlanner._numbered_passages(page_text)}\n\n"
            "Output requirements:\n"
            "- Output only repeated <query>...</query><reason>...</reason><url></url> blocks.\n"
            "- Do not output markdown fences, bullets, JSON, headings, or any extra text.\n"
            "- Each <query> must be a single rewritten search query.\n"
            "- Each <reason> must explain uniqueness and likely online existence."
            "- Each <url> must wrap a URL of an image that satisfy the requirement."
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
        query_matches = list(re.finditer(r"<query>(.*?)</query>", text, flags=re.DOTALL | re.IGNORECASE))
        reason_matches = list(re.finditer(r"<reason>(.*?)</reason>", text, flags=re.DOTALL | re.IGNORECASE))
        pair_count = min(len(query_matches), len(reason_matches))
        for index in range(pair_count):
            query = re.sub(r"\s+", " ", query_matches[index].group(1)).strip()
            reason = re.sub(r"\s+", " ", reason_matches[index].group(1)).strip()
            if not query:
                continue
            candidates.append({"query": query, "reason": reason})
        return candidates

    def _candidate_to_plan(
        self,
        candidate: dict[str, Any],
        *,
        node: dict[str, Any],
        source_evidence_ids: list[str],
        raw_output: str,
        run_id: str | None,
    ) -> VisualSearchPlan | None:
        query = candidate.get("query")
        reason = candidate.get("reason")
        queries = self._filter_queries([query] if query else [], node_title=node.get("title"))
        if not query or not queries:
            return None

        target_type = VisualTargetType.OTHER
        downstream_use = DownstreamUse.ROUTING_CLUE
        source_node_id = node.get("node_id")
        target = Evidence.create(
            EvidenceType.VISUAL_TARGET,
            content=query,
            node_ids=[source_node_id] if source_node_id else [],
            extractor=self.planner_name,
            confidence=None,
            metadata={
                "target_type": target_type.value,
                "downstream_use": downstream_use.value,
                "query": query,
                "reason": reason,
                "expected_visual": query,
                "source_evidence_ids": source_evidence_ids,
                "run_id": run_id,
            },
            evidence_key=f"{source_node_id}:{query}",
        )
        query_specs = [
            SearchQuerySpec.create(
                normalized_query,
                target.evidence_id,
                intent=target_type.value,
                expected_visual=query,
                source=self.planner_name,
                metadata={
                    "downstream_use": downstream_use.value,
                    "reason": reason,
                },
            )
            for normalized_query in queries
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
                "query": query,
                "reason": reason,
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
                content="""<query>Kobe Bryant final game in 2016 photo</query>
<reason>The passage points to one specific event, Kobe Bryant's final NBA game in 2016. Photos of that game are widely available online, and different matching images still depict the same uniquely identified event.</reason>"""
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
    assert len(plans[0].queries) == 1
    assert plans[0].target.content == "Kobe Bryant final game in 2016 photo"
    assert "specific event" in (plans[0].target.metadata["reason"] or "")
    prompt_input = planner._prompt_input(
        {"node_id": "text_1", "title": "Kobe Bryant", "attributes": {}},
        "First paragraph.\n\nSecond paragraph.",
    )
    assert "Complete the following person." in prompt_input
    assert "P1: First paragraph." in prompt_input
    assert "P2: Second paragraph." in prompt_input
    print("visual_planner smoke test passed")


if __name__ == "__main__":
    _smoke_test()
