"""Trajectory-local duplicate-search reward for DeepResearch rollouts.

The model never sees the URL trace used here. The trace is retained only in
``Episode.info`` after a completed ``t2t_search`` so we can distinguish a
genuine semantic loop from a useful query refinement that retrieves new pages.

The reward is deliberately binary at trajectory level: detecting one or more
confirmed duplicate searches carries the same penalty.  It is intended to
replace the former MAX_TURNS terminal penalty without making a long trajectory
receive an increasingly large auxiliary reward signal.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import math
import os
import re
import time
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
import weakref

import httpx


_EMBEDDING_INSTRUCTION = (
    "Given a web search query, represent its information need for semantic "
    "similarity and duplicate-search detection."
)
_TRACKING_QUERY_KEYS = {"fbclid", "gclid", "srsltid"}
_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?:\.\d+)?(?![A-Za-z])")
_TICKER_RE = re.compile(r"\b[A-Z]{2,5}\b")
_WHITESPACE_RE = re.compile(r"\s+")

# A rollout process has one asyncio loop, but test workers can create several.
# Use a limiter per loop so a burst of completed trajectories cannot create an
# unbounded number of concurrent TEI requests.
_LIMITERS: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, tuple[int, asyncio.Semaphore]
] = weakref.WeakKeyDictionary()

# One compact configuration diagnostic per rollout process proves the *remote
# worker's* effective configuration without flooding logs or exposing
# queries/URLs.  In contrast, the caller intentionally asked for one result
# diagnostic per completed trajectory, so score-result diagnostics are not
# deduplicated.
_RUNTIME_CONFIG_LOGGED = False
_DEBUG_LOG_PREFIX = "##############3"


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    try:
        return max(minimum, float(os.environ.get(name, default)))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class SearchRedundancyConfig:
    """Environment-configurable policy for duplicate-search penalties.

    ``off`` and ``shadow`` are both observation-only modes: they issue the
    embedding request and report metrics, but produce an effective penalty of
    zero. Only ``reward`` changes the trajectory reward.
    """

    mode: str = "off"  # off/observe | shadow/observe | reward/apply
    embedding_url: str = ""
    embedding_timeout_s: float = 10.0
    embedding_concurrency: int = 4
    semantic_threshold: float = 0.90
    url_overlap_threshold: float = 0.40
    # ``event_penalty`` is retained as the environment/telemetry field name
    # for compatibility.  It is a *single trajectory penalty*, not a
    # per-confirmed-event penalty: one or more events cost this amount once.
    event_penalty: float = 0.10
    # A defensive cap for custom environment settings.  At the default it is
    # equal to ``event_penalty`` and therefore never changes the binary rule.
    trajectory_penalty_cap: float = 0.10
    url_top_k: int = 5
    min_urls_per_search: int = 3

    @classmethod
    def from_env(cls) -> "SearchRedundancyConfig":
        mode = str(os.environ.get("SEARCH_REDUNDANCY_MODE", "off")).strip().lower()
        # ``on``/``off`` is the convenient public switch for experiments;
        # retain the original explicit names for backwards compatibility.
        if mode in {"on", "true", "1", "apply"}:
            mode = "reward"
        elif mode in {"observe", "metrics", "metrics_only"}:
            mode = "off"
        if mode not in {"off", "shadow", "reward"}:
            mode = "off"
        return cls(
            mode=mode,
            embedding_url=(
                os.environ.get("SEARCH_REDUNDANCY_EMBEDDING_URL")
                or os.environ.get("SEARCH_SIMILARITY_EMBEDDING_URL")
                or ""
            ).strip(),
            embedding_timeout_s=_env_float(
                "SEARCH_REDUNDANCY_EMBED_TIMEOUT_S", 10.0, minimum=0.1
            ),
            embedding_concurrency=_env_int(
                "SEARCH_REDUNDANCY_EMBED_CONCURRENCY", 4, minimum=1
            ),
            semantic_threshold=_env_float(
                "SEARCH_REDUNDANCY_SEMANTIC_THRESHOLD", 0.90
            ),
            url_overlap_threshold=_env_float(
                "SEARCH_REDUNDANCY_URL_OVERLAP_THRESHOLD", 0.40
            ),
            event_penalty=_env_float(
                "SEARCH_REDUNDANCY_EVENT_PENALTY", 0.10
            ),
            trajectory_penalty_cap=_env_float(
                "SEARCH_REDUNDANCY_TRAJECTORY_CAP", 0.10
            ),
            url_top_k=_env_int("SEARCH_REDUNDANCY_URL_TOP_K", 5, minimum=1),
            min_urls_per_search=_env_int(
                "SEARCH_REDUNDANCY_MIN_URLS_PER_SEARCH", 3, minimum=1
            ),
        )


def normalize_query(query: Any) -> str:
    return _WHITESPACE_RE.sub(" ", str(query or "").strip())


def canonicalize_url(value: Any) -> str:
    """Drop only tracking syntax; preserve parameters that may alter content."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if not parsed.scheme or not parsed.netloc:
        return raw
    kept_query = [
        (key, item)
        for key, item in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_")
        and key.lower() not in _TRACKING_QUERY_KEYS
    ]
    return urlunsplit(
        (
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path or "/",
            urlencode(sorted(kept_query), doseq=True),
            "",
        )
    )


def _canonical_urls(event: Mapping[str, Any], top_k: int) -> list[str]:
    raw_urls = event.get("result_urls", [])
    if not isinstance(raw_urls, list):
        return []
    urls: list[str] = []
    seen: set[str] = set()
    for raw_url in raw_urls:
        url = canonicalize_url(raw_url)
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
        if len(urls) >= top_k:
            break
    return urls


def _hard_constraints(query: str) -> set[str]:
    """Extract intentionally conservative structured refinements."""

    values = {f"number:{match.group(0)}" for match in _NUMBER_RE.finditer(query)}
    values.update(f"ticker:{match.group(0)}" for match in _TICKER_RE.finditer(query))
    return values


def _has_new_hard_constraint(current: str, previous: str) -> bool:
    return bool(_hard_constraints(current) - _hard_constraints(previous))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or not left:
        return None
    numerator = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for left_value, right_value in zip(left, right, strict=True):
        try:
            left_float = float(left_value)
            right_float = float(right_value)
        except (TypeError, ValueError):
            return None
        numerator += left_float * right_float
        left_norm += left_float * left_float
        right_norm += right_float * right_float
    if left_norm <= 0.0 or right_norm <= 0.0:
        return None
    return numerator / math.sqrt(left_norm * right_norm)


def _url_overlap(current_urls: Sequence[str], previous_urls: Sequence[str]) -> float:
    if not current_urls or not previous_urls:
        return 0.0
    return len(set(current_urls) & set(previous_urls)) / min(
        len(set(current_urls)), len(set(previous_urls))
    )


def _eligible_events(
    search_trace: Sequence[Mapping[str, Any]], config: SearchRedundancyConfig
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for trace_index, raw_event in enumerate(search_trace):
        if not isinstance(raw_event, Mapping):
            continue
        if str(raw_event.get("tool_name") or "") != "t2t_search":
            continue
        query = normalize_query(raw_event.get("query"))
        urls = _canonical_urls(raw_event, config.url_top_k)
        if not bool(raw_event.get("ok")) or not query:
            continue
        if len(urls) < config.min_urls_per_search:
            continue
        events.append(
            {
                "trace_index": trace_index,
                "round": raw_event.get("round"),
                "query": query,
                "query_key": query.casefold(),
                "urls": urls,
            }
        )
    return events


def _empty_result(
    config: SearchRedundancyConfig, *, reason: str, search_trace_count: int
) -> dict[str, Any]:
    return {
        "mode": config.mode,
        "scored": False,
        "reason": reason,
        "search_trace_count": search_trace_count,
        "eligible_search_count": 0,
        "embedded_query_count": 0,
        # ``embedding_request_executed`` means this trajectory reached the
        # HTTP embedding call; it remains true even if that call fails.  The
        # separate response flag prevents a failed request being mistaken for
        # a successful score.
        "embedding_request_executed": False,
        "embedding_response_valid": False,
        "embedding_request_s": 0.0,
        "embedding_queue_wait_s": 0.0,
        "pair_count": 0,
        "hard_constraint_exempt_pair_count": 0,
        "confirmed_event_count": 0,
        "raw_penalty": 0.0,
        "penalty": 0.0,
        # Keep the effective configuration with the result for diagnostics and
        # aggregate metrics. These values never contain prompts, queries, or
        # URLs, so they are safe to propagate to training telemetry.
        "semantic_threshold": config.semantic_threshold,
        "url_overlap_threshold": config.url_overlap_threshold,
        "event_penalty": config.event_penalty,
        "trajectory_penalty_cap": config.trajectory_penalty_cap,
        "events": [],
    }


def compute_redundancy_from_embeddings(
    search_trace: Sequence[Mapping[str, Any]],
    embeddings_by_query: Mapping[str, Sequence[float]],
    config: SearchRedundancyConfig,
) -> dict[str, Any]:
    """Apply the deterministic reward rule to precomputed query embeddings."""

    trace_count = len(search_trace)
    events = _eligible_events(search_trace, config)
    result = _empty_result(config, reason="scored", search_trace_count=trace_count)
    result["eligible_search_count"] = len(events)
    result["embedded_query_count"] = len(embeddings_by_query)
    if len(events) < 2:
        result["reason"] = "fewer_than_two_eligible_searches"
        return result

    pair_count = 0
    hard_constraint_exempt_pair_count = 0
    confirmed: list[dict[str, Any]] = []
    for current_index, current in enumerate(events):
        current_embedding = embeddings_by_query.get(current["query_key"])
        if current_embedding is None:
            continue
        best_match: dict[str, Any] | None = None
        for previous in events[:current_index]:
            previous_embedding = embeddings_by_query.get(previous["query_key"])
            if previous_embedding is None:
                continue
            pair_count += 1
            if _has_new_hard_constraint(current["query"], previous["query"]):
                hard_constraint_exempt_pair_count += 1
                continue
            similarity = _cosine(current_embedding, previous_embedding)
            if similarity is None:
                continue
            overlap = _url_overlap(current["urls"], previous["urls"])
            if (
                similarity < config.semantic_threshold
                or overlap < config.url_overlap_threshold
            ):
                continue
            candidate = {
                "current_trace_index": current["trace_index"],
                "previous_trace_index": previous["trace_index"],
                "current_round": current["round"],
                "previous_round": previous["round"],
                "similarity": similarity,
                "url_overlap": overlap,
                "penalty": config.event_penalty,
            }
            if best_match is None or (
                candidate["similarity"], candidate["url_overlap"]
            ) > (best_match["similarity"], best_match["url_overlap"]):
                best_match = candidate
        if best_match is not None:
            confirmed.append(best_match)

    # A trajectory is penalized once if it contains at least one confirmed
    # duplicate.  Do not accumulate by event count: that would let the
    # auxiliary signal dominate reward differences in groups whose rollouts
    # are all answer-incorrect.
    raw_penalty = config.event_penalty if confirmed else 0.0
    effective_penalty = min(raw_penalty, config.trajectory_penalty_cap)
    result.update(
        {
            "scored": True,
            "reason": "scored",
            "pair_count": pair_count,
            "hard_constraint_exempt_pair_count": hard_constraint_exempt_pair_count,
            "confirmed_event_count": len(confirmed),
            "raw_penalty": raw_penalty,
            "penalty": effective_penalty if config.mode == "reward" else 0.0,
            "events": confirmed,
        }
    )
    return result


def _get_limiter(limit: int) -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    existing = _LIMITERS.get(loop)
    if existing is None or existing[0] != limit:
        limiter = asyncio.Semaphore(limit)
        _LIMITERS[loop] = (limit, limiter)
        return limiter
    return existing[1]


def _log_runtime_config_once(config: SearchRedundancyConfig) -> None:
    """Log only safe, process-local scorer configuration diagnostics."""

    global _RUNTIME_CONFIG_LOGGED
    if _RUNTIME_CONFIG_LOGGED:
        return
    _RUNTIME_CONFIG_LOGGED = True
    print(
        f"{_DEBUG_LOG_PREFIX} [SearchRedundancy][WorkerConfig] "
        f"mode={config.mode} "
        f"embedding_url_configured={bool(config.embedding_url)} "
        f"semantic_threshold={config.semantic_threshold:.3f} "
        f"url_overlap_threshold={config.url_overlap_threshold:.3f} "
        f"min_urls_per_search={config.min_urls_per_search}",
        flush=True,
    )


def _log_score_result(result: Mapping[str, Any]) -> None:
    """Log one safe result summary for every completed trajectory."""

    print(
        f"{_DEBUG_LOG_PREFIX} [SearchRedundancy][WorkerResult] "
        f"reason={str(result.get('reason') or 'unknown')} "
        f"scored={bool(result.get('scored'))} "
        f"search_trace_count={int(result.get('search_trace_count', 0) or 0)} "
        f"eligible_search_count={int(result.get('eligible_search_count', 0) or 0)} "
        f"embedded_query_count={int(result.get('embedded_query_count', 0) or 0)} "
        f"embedding_request_executed="
        f"{bool(result.get('embedding_request_executed'))} "
        f"embedding_response_valid="
        f"{bool(result.get('embedding_response_valid'))} "
        f"confirmed_event_count={int(result.get('confirmed_event_count', 0) or 0)}",
        flush=True,
    )


async def _request_embeddings(
    inputs: list[str], config: SearchRedundancyConfig
) -> tuple[list[Sequence[float]], float, float]:
    queued_at = time.perf_counter()
    async with _get_limiter(config.embedding_concurrency):
        request_started_at = time.perf_counter()
        timeout = httpx.Timeout(config.embedding_timeout_s)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(config.embedding_url, json={"inputs": inputs})
            response.raise_for_status()
            vectors = response.json()
        request_elapsed_s = time.perf_counter() - request_started_at
    if not isinstance(vectors, list) or len(vectors) != len(inputs):
        actual_length = len(vectors) if isinstance(vectors, list) else "n/a"
        raise ValueError(
            "Unexpected embedding response length: "
            f"expected={len(inputs)} actual={actual_length}"
        )
    return vectors, request_elapsed_s, request_started_at - queued_at


async def score_search_redundancy(
    search_trace: Sequence[Mapping[str, Any]],
    config: SearchRedundancyConfig | None = None,
) -> dict[str, Any]:
    """Embed one completed trajectory and return a fail-open reward record."""

    config = config or SearchRedundancyConfig.from_env()
    _log_runtime_config_once(config)
    if not isinstance(search_trace, Sequence) or isinstance(search_trace, (str, bytes)):
        search_trace = []
    trace_count = len(search_trace)
    if not config.embedding_url:
        result = _empty_result(
            config, reason="missing_embedding_url", search_trace_count=trace_count
        )
        _log_score_result(result)
        return result

    eligible = _eligible_events(search_trace, config)
    if len(eligible) < 2:
        result = _empty_result(
            config, reason="fewer_than_two_eligible_searches", search_trace_count=trace_count
        )
        result["eligible_search_count"] = len(eligible)
        _log_score_result(result)
        return result

    unique_queries: dict[str, str] = {}
    for event in eligible:
        unique_queries.setdefault(event["query_key"], event["query"])
    encoded_inputs = [
        f"Instruct: {_EMBEDDING_INSTRUCTION}\nQuery: {query}"
        for query in unique_queries.values()
    ]
    try:
        vectors, request_elapsed_s, queue_wait_s = await _request_embeddings(
            encoded_inputs, config
        )
    except Exception as exc:  # noqa: BLE001 - reward augmentation must fail open.
        result = _empty_result(
            config, reason="embedding_request_failed", search_trace_count=trace_count
        )
        result.update(
            {
                "eligible_search_count": len(eligible),
                "embedded_query_count": len(unique_queries),
                "embedding_request_executed": True,
                "embedding_error_type": type(exc).__name__,
                "embedding_error": str(exc),
            }
        )
        _log_score_result(result)
        return result

    embeddings_by_query = dict(zip(unique_queries.keys(), vectors, strict=True))
    result = compute_redundancy_from_embeddings(
        search_trace, embeddings_by_query, config
    )
    result["embedding_request_s"] = request_elapsed_s
    result["embedding_queue_wait_s"] = queue_wait_s
    result["embedding_request_executed"] = True
    result["embedding_response_valid"] = True
    _log_score_result(result)
    return result


__all__ = [
    "SearchRedundancyConfig",
    "canonicalize_url",
    "compute_redundancy_from_embeddings",
    "normalize_query",
    "score_search_redundancy",
]
