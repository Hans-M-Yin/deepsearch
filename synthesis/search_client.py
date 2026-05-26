"""Search client abstractions for text and image discovery."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Protocol
from urllib.parse import urlencode
from urllib.request import Request, urlopen


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


@dataclass(slots=True)
class TextSearchResult:
    title: str | None = None
    url: str | None = None
    snippet: str | None = None
    source: str | None = None
    rank: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class ImageSearchResult:
    title: str | None = None
    image_url: str | None = None
    source_page_url: str | None = None
    thumbnail_url: str | None = None
    snippet: str | None = None
    source: str | None = None
    width: int | None = None
    height: int | None = None
    rank: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class SearchResponse:
    query: str
    engine: str
    results: list[TextSearchResult] | list[ImageSearchResult]
    raw_response: dict[str, Any] = field(default_factory=dict)
    status_code: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


class SearchClient(Protocol):
    """Minimal search interface used by graph builders."""

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        """Return text/web search results."""

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        """Return image search results."""


class OpenSerpSearchClient:
    """HTTP client for an OpenSERP-compatible server."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:7000",
        *,
        text_engine: str = "google",
        image_engine: str = "bing",
        use_mega: bool = False,
        mega_engines: str = "google,bing",
        mega_mode: str = "balanced",
        timeout_s: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.text_engine = text_engine
        self.image_engine = image_engine
        self.use_mega = use_mega
        self.mega_engines = mega_engines
        self.mega_mode = mega_mode
        self.timeout_s = timeout_s

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        engine = "mega" if self.use_mega else self.text_engine
        raw = self._get_json(f"/{engine}/search", query, limit, kwargs)
        return SearchResponse(
            query=query,
            engine=f"openserp:{engine}:search",
            results=self._parse_text_results(raw),
            raw_response=raw,
            status_code=200,
        )

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        engine = "mega" if self.use_mega else self.image_engine
        raw = self._get_json(f"/{engine}/image", query, limit, kwargs)
        return SearchResponse(
            query=query,
            engine=f"openserp:{engine}:image",
            results=self._parse_image_results(raw),
            raw_response=raw,
            status_code=200,
        )

    def _get_json(
        self,
        path: str,
        query: str,
        limit: int,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        request_params = self._openserp_params(query, limit, params)
        url = f"{self.base_url}{path}?{urlencode(request_params, doseq=True)}"
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = response.read().decode("utf-8")
        return json.loads(payload)

    def _openserp_params(self, query: str, limit: int, params: dict[str, Any]) -> dict[str, Any]:
        request_params: dict[str, Any] = {
            "text": query,
            "limit": max(1, min(int(limit), 100)),
            "start": self._start(params, limit),
            "format": "json",
        }

        lang = params.get("lang") or params.get("hl")
        region = params.get("region") or params.get("gl")
        if lang:
            request_params["lang"] = str(lang).upper()
        if region:
            request_params["region"] = str(region).upper()
        for key in ("date", "site", "file", "filter", "answers"):
            if params.get(key) is not None:
                request_params[key] = params[key]

        if self.use_mega:
            request_params["mode"] = params.get("mode") or self.mega_mode
            request_params["engines"] = params.get("engines") or self.mega_engines
            request_params["dedupe"] = str(params.get("dedupe", True)).lower()
            request_params["merge"] = str(params.get("merge", True)).lower()
        return request_params

    @staticmethod
    def _start(params: dict[str, Any], limit: int) -> int:
        if params.get("start") is not None:
            try:
                return max(0, int(params["start"]))
            except (TypeError, ValueError):
                return 0
        try:
            page = int(params.get("page") or 1)
        except (TypeError, ValueError):
            page = 1
        return max(0, page - 1) * max(1, limit)

    @staticmethod
    def _result_items(raw: dict[str, Any]) -> list[dict[str, Any]]:
        for key in ("results", "organic", "items", "images"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        if isinstance(raw.get("data"), list):
            return raw["data"]
        if isinstance(raw.get("data"), dict):
            return OpenSerpSearchClient._result_items(raw["data"])
        return []

    @classmethod
    def _parse_text_results(cls, raw: dict[str, Any]) -> list[TextSearchResult]:
        results = []
        for rank, item in enumerate(cls._result_items(raw), start=1):
            if item.get("type") not in (None, "organic", "answer"):
                continue
            results.append(
                TextSearchResult(
                    title=item.get("title") or item.get("name"),
                    url=item.get("url") or item.get("link"),
                    snippet=item.get("snippet") or item.get("description") or item.get("text"),
                    source=item.get("source") or item.get("engine"),
                    rank=cls._position(item, rank),
                    raw=item,
                )
            )
        return results

    @classmethod
    def _parse_image_results(cls, raw: dict[str, Any]) -> list[ImageSearchResult]:
        results = []
        for rank, item in enumerate(cls._result_items(raw), start=1):
            if item.get("type") not in (None, "image"):
                continue
            results.append(
                ImageSearchResult(
                    title=item.get("title") or item.get("name"),
                    image_url=cls._image_url(item),
                    source_page_url=cls._source_page_url(item),
                    thumbnail_url=cls._thumbnail_url(item),
                    snippet=item.get("snippet") or item.get("description") or item.get("text"),
                    source=cls._source_name(item),
                    width=item.get("width"),
                    height=item.get("height"),
                    rank=cls._position(item, rank),
                    raw=item,
                )
            )
        return results

    @staticmethod
    def _position(item: dict[str, Any], fallback: int) -> int:
        position = item.get("rank") or item.get("position")
        if isinstance(position, dict):
            position = position.get("absolute")
        try:
            return int(position)
        except (TypeError, ValueError):
            return fallback

    @staticmethod
    def _source_name(item: dict[str, Any]) -> str | None:
        source = item.get("source")
        if isinstance(source, dict):
            return source.get("name") or source.get("engine")
        return source or item.get("engine")

    @staticmethod
    def _image_url(item: dict[str, Any]) -> str | None:
        image = item.get("image")
        if isinstance(image, dict):
            return image.get("url") or image.get("imageUrl")
        if isinstance(image, str):
            return image
        return item.get("imageUrl") or item.get("image_url") or item.get("url")

    @staticmethod
    def _thumbnail_url(item: dict[str, Any]) -> str | None:
        image = item.get("image")
        if isinstance(image, dict):
            return image.get("thumbnail") or image.get("thumbnailUrl")
        return item.get("thumbnailUrl") or item.get("thumbnail_url") or item.get("thumbnail")

    @staticmethod
    def _source_page_url(item: dict[str, Any]) -> str | None:
        source = item.get("source")
        if isinstance(source, dict):
            return source.get("page_url") or source.get("url")
        return item.get("source_page_url") or item.get("source_url") or item.get("link") or item.get("page_url")


class SerperAdapterSearchClient:
    """Client for the local Serper-compatible OpenSERP adapter."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:7001",
        *,
        api_key: str = "local-openserp",
        timeout_s: float = 60.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout_s = timeout_s

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        body = self._serper_body(query, limit, kwargs)
        raw = self._post_json("/search", body)
        return SearchResponse(
            query=query,
            engine="serper_adapter:search",
            results=OpenSerpSearchClient._parse_text_results(raw),
            raw_response=raw,
            status_code=200,
        )

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        body = self._serper_body(query, limit, kwargs)
        raw = self._post_json("/images", body)
        return SearchResponse(
            query=query,
            engine="serper_adapter:images",
            results=OpenSerpSearchClient._parse_image_results(raw),
            raw_response=raw,
            status_code=200,
        )

    @staticmethod
    def _serper_body(query: str, limit: int, params: dict[str, Any]) -> dict[str, Any]:
        body = {"q": query, "num": max(1, min(int(limit), 100))}
        for src_key, dst_key in (
            ("hl", "hl"),
            ("lang", "hl"),
            ("gl", "gl"),
            ("region", "gl"),
            ("page", "page"),
            ("start", "start"),
            ("date", "date"),
            ("site", "site"),
            ("file", "file"),
            ("mode", "mode"),
            ("engines", "engines"),
            ("dedupe", "dedupe"),
            ("merge", "merge"),
        ):
            if params.get(src_key) is not None:
                body[dst_key] = params[src_key]
        return body

    def _post_json(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(body).encode("utf-8")
        request = Request(
            f"{self.base_url}{path}",
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "X-API-KEY": self.api_key,
            },
            method="POST",
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            response_payload = response.read().decode("utf-8")
        return json.loads(response_payload)


class MockSearchClient:
    """Deterministic search client for tests and offline development."""

    def __init__(
        self,
        *,
        text_results: dict[str, list[TextSearchResult]] | None = None,
        image_results: dict[str, list[ImageSearchResult]] | None = None,
    ) -> None:
        self.text_results = text_results or {}
        self.image_results = image_results or {}

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        results = self.text_results.get(query, [])[:limit]
        return SearchResponse(query=query, engine="mock:text", results=results)

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        results = self.image_results.get(query, [])[:limit]
        return SearchResponse(query=query, engine="mock:image", results=results)
