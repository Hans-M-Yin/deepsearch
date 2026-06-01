"""Low-cost Wikipedia entity resolution helpers for image-grounded entities."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _normalize_label(text: str | None) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", (text or "")).lower()).strip()


@dataclass(slots=True)
class WikiEntityCandidate:
    title: str
    url: str
    canonical_id: str
    snippet: str | None = None
    qid: str | None = None
    score: float = 0.0
    source: str = "wikipedia_search"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class WikiEntityResolver:
    """Resolve ambiguous entity names to Wikipedia URLs with cheap official APIs."""

    def __init__(
        self,
        *,
        api_url: str = "https://en.wikipedia.org/w/api.php",
        timeout_s: float = 20.0,
        language: str = "en",
    ) -> None:
        self.api_url = api_url
        self.timeout_s = timeout_s
        self.language = language

    def resolve(
        self,
        label: str,
        *,
        entity_type: str | None = None,
        source_title: str | None = None,
        context: str | None = None,
        limit: int = 5,
    ) -> WikiEntityCandidate | None:
        label = (label or "").strip()
        if not label:
            return None

        queries = self._build_queries(label, entity_type=entity_type, source_title=source_title, context=context)
        candidates: dict[str, WikiEntityCandidate] = {}
        for query in queries:
            for candidate in self._search(query, limit=limit):
                existing = candidates.get(candidate.url)
                if existing is None or candidate.score > existing.score:
                    candidates[candidate.url] = candidate

        ranked = sorted(candidates.values(), key=lambda item: item.score, reverse=True)
        if not ranked:
            return None
        if len(ranked) == 1 and ranked[0].score >= 2.5:
            return ranked[0]
        top = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        if top.score >= 4.0 and (runner_up is None or top.score - runner_up.score >= 0.75):
            return top
        return None

    def _build_queries(
        self,
        label: str,
        *,
        entity_type: str | None,
        source_title: str | None,
        context: str | None,
    ) -> list[str]:
        queries = [label]
        type_hint = self._entity_type_hint(entity_type)
        if type_hint:
            queries.append(f"{label} {type_hint}")
        if source_title:
            queries.append(f"{label} {source_title}")
        if context:
            hint = " ".join(context.split()[:8]).strip()
            if hint:
                queries.append(f"{label} {hint}")
        deduped: list[str] = []
        seen: set[str] = set()
        for query in queries:
            normalized = re.sub(r"\s+", " ", query).strip()
            if not normalized or normalized.lower() in seen:
                continue
            seen.add(normalized.lower())
            deduped.append(normalized)
        return deduped[:4]

    @staticmethod
    def _entity_type_hint(entity_type: str | None) -> str | None:
        mapping = {
            "person": "person",
            "team": "sports team",
            "organization": "organization",
            "event": "event",
            "movie": "film",
            "book": "book",
            "album": "album",
            "brand": "brand",
            "product": "product",
            "landmark": "landmark",
            "document": "document",
        }
        normalized = _normalize_label(entity_type)
        return mapping.get(normalized, entity_type.strip() if entity_type else None)

    def _search(self, query: str, *, limit: int) -> list[WikiEntityCandidate]:
        params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": query,
            "srlimit": max(1, min(int(limit), 10)),
            "srprop": "snippet",
        }
        request = Request(
            f"{self.api_url}?{urlencode(params)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "deepsearch-synthesis/0.1",
            },
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        search_results = payload.get("query", {}).get("search") or []
        if not isinstance(search_results, list):
            return []

        page_ids = [str(item.get("pageid")) for item in search_results if item.get("pageid") is not None]
        page_details = self._page_details(page_ids)
        candidates: list[WikiEntityCandidate] = []
        for rank, item in enumerate(search_results, start=1):
            if not isinstance(item, dict):
                continue
            pageid = str(item.get("pageid")) if item.get("pageid") is not None else None
            detail = page_details.get(pageid or "", {})
            title = detail.get("title") or item.get("title")
            fullurl = detail.get("fullurl") or self._title_to_url(title)
            qid = ((detail.get("pageprops") or {}).get("wikibase_item") if isinstance(detail, dict) else None)
            candidate = WikiEntityCandidate(
                title=title,
                url=fullurl,
                canonical_id=f"wikidata:{qid}" if qid else f"wikipedia:{title}" if title else "wikipedia:unknown",
                snippet=self._strip_html(item.get("snippet")),
                qid=qid,
                score=self._score_candidate(item.get("title"), item.get("snippet"), query=query, rank=rank),
            )
            candidates.append(candidate)
        return candidates

    def _page_details(self, page_ids: list[str]) -> dict[str, dict[str, Any]]:
        if not page_ids:
            return {}
        params = {
            "action": "query",
            "format": "json",
            "prop": "info|pageprops",
            "inprop": "url",
            "pageids": "|".join(page_ids),
        }
        request = Request(
            f"{self.api_url}?{urlencode(params)}",
            headers={
                "Accept": "application/json",
                "User-Agent": "deepsearch-synthesis/0.1",
            },
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = json.loads(response.read().decode("utf-8"))
        pages = payload.get("query", {}).get("pages") or {}
        return pages if isinstance(pages, dict) else {}

    @staticmethod
    def _title_to_url(title: str | None) -> str:
        safe_title = (title or "").replace(" ", "_")
        return f"https://en.wikipedia.org/wiki/{safe_title}"

    @staticmethod
    def _score_candidate(
        title: str | None,
        snippet: str | None,
        *,
        query: str,
        rank: int,
    ) -> float:
        normalized_title = _normalize_label(title)
        normalized_query = _normalize_label(query)
        score = 0.0
        if normalized_title and normalized_title == normalized_query:
            score += 5.0
        elif normalized_title and normalized_query:
            title_tokens = set(normalized_title.split())
            query_tokens = set(normalized_query.split())
            overlap = len(title_tokens & query_tokens)
            if overlap:
                score += 1.0 + 0.5 * overlap
            if normalized_title in normalized_query or normalized_query in normalized_title:
                score += 1.0

        snippet_text = _normalize_label(WikiEntityResolver._strip_html(snippet))
        if snippet_text and normalized_query:
            query_tokens = set(normalized_query.split())
            snippet_tokens = set(snippet_text.split())
            score += min(1.5, 0.2 * len(query_tokens & snippet_tokens))
        score -= 0.15 * max(0, rank - 1)
        return score

    @staticmethod
    def _strip_html(text: str | None) -> str | None:
        if not text:
            return text
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()
