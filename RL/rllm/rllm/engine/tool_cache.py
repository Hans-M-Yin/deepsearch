"""Small, process-local tool cache used by RL rollouts.

The cache is deliberately kept outside ``synthesis.sft``.  SFT/inference
callers do not construct these objects, so the shared tool implementation has
no cache state unless an RL caller explicitly injects one.
"""

from __future__ import annotations

import copy
import json
import threading
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit


def _canonical_url(value: Any) -> str:
    """Canonicalize only inconsequential URL differences for hard matching."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urlsplit(raw)
    except Exception:
        return raw
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return raw
    hostname = (parsed.hostname or "").lower()
    netloc = hostname
    if parsed.port is not None and not (
        (scheme == "http" and parsed.port == 80)
        or (scheme == "https" and parsed.port == 443)
    ):
        netloc = f"{hostname}:{parsed.port}"
    return urlunsplit(
        (
            scheme,
            netloc,
            parsed.path or "/",
            parsed.query,
            "",
        )
    )


def _canonical_arguments(arguments: dict[str, Any]) -> str:
    """Serialize normalized arguments for exact cache matching."""

    normalized = dict(arguments or {})
    for key in ("query", "q"):
        if isinstance(normalized.get(key), str):
            # This is still a hard match; it only removes whitespace/case noise.
            normalized[key] = " ".join(normalized[key].split()).casefold()
    if isinstance(normalized.get("url"), str):
        normalized["url"] = _canonical_url(normalized["url"])
    if isinstance(normalized.get("URL"), str):
        normalized["URL"] = _canonical_url(normalized["URL"])
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)


class SampleToolCache:
    """Thread-safe cache shared by all rollouts of one sample."""

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], Any] = {}
        self._inflight: dict[tuple[str, str], threading.Lock] = {}
        self._state_lock = threading.Lock()
        self._stats_callback: Callable[[str, bool], None] | None = None
        self.hits = 0
        self.misses = 0

    def set_stats_callback(
        self, callback: Callable[[str, bool], None] | None
    ) -> None:
        """Attach an optional observer without coupling RL cache to SFT."""

        self._stats_callback = callback

    def _notify_stats(self, tool_name: str, hit: bool) -> None:
        callback = self._stats_callback
        if callback is None:
            return
        try:
            callback(str(tool_name), bool(hit))
        except Exception:
            # Cache accounting must never make a tool call fail.
            return

    @staticmethod
    def _clone(value: Any) -> Any:
        # Search/document payloads are dictionaries.  Returning copies prevents
        # trajectory-specific postprocessing from mutating the shared value.
        return copy.deepcopy(value)

    def get_or_compute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        compute: Callable[[], Any],
    ) -> Any:
        """Return a cached successful backend result, deduplicating concurrent misses."""

        key = (str(tool_name), _canonical_arguments(arguments))
        with self._state_lock:
            cached = self._entries.get(key)
            if key in self._entries:
                self.hits += 1
                self._notify_stats(tool_name, True)
                return self._clone(cached)
            lock = self._inflight.setdefault(key, threading.Lock())

        with lock:
            try:
                with self._state_lock:
                    cached = self._entries.get(key)
                    if key in self._entries:
                        self.hits += 1
                        self._notify_stats(tool_name, True)
                        return self._clone(cached)
                    self.misses += 1
                    self._notify_stats(tool_name, False)

                value = compute()
                # Do not retain known tool failures.  A transient failure in
                # one rollout should not poison all sibling rollouts.
                if not (
                    isinstance(value, dict) and value.get("ok") is False
                ):
                    with self._state_lock:
                        self._entries[key] = self._clone(value)
                return value
            finally:
                with self._state_lock:
                    if self._inflight.get(key) is lock:
                        self._inflight.pop(key, None)

    def fetch_document(
        self,
        backend: str,
        url: str,
        compute: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        """Cache a raw Enhanced Reader/Firecrawl document by backend and URL."""

        # If an earlier call had to fall back to Firecrawl, reuse that raw
        # document immediately on the next call instead of probing Enhanced
        # Reader again.  The shared read_url code can still summarize it using
        # the current goal.
        if backend == "enhanced_reader":
            fallback_key = (
                "read_url_raw",
                _canonical_arguments(
                    {"backend": "firecrawl", "url": _canonical_url(url)}
                ),
            )
            with self._state_lock:
                if fallback_key in self._entries:
                    self.hits += 1
                    self._notify_stats("read_url_raw", True)
                    return self._clone(self._entries[fallback_key])

        return self.get_or_compute(
            "read_url_raw",
            {"backend": backend, "url": _canonical_url(url)},
            compute,
        )


class EpochToolCache:
    """Container for per-sample caches, normally discarded after one epoch."""

    def __init__(self) -> None:
        self._samples: dict[str, SampleToolCache] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _explicit_sample_id(task: dict[str, Any]) -> str | None:
        sources = [task]
        nested = task.get("extra_info") if isinstance(task, dict) else None
        if isinstance(nested, dict):
            sources.append(nested)
        for source in sources:
            for field in (
                "sample_id",
                "source_sample_id",
                "question_id",
                "source_question_id",
                "problem_id",
                "data_id",
                "id",
            ):
                value = source.get(field)
                if value is not None and str(value).strip():
                    return str(value).strip()
        return None

    def for_sample(self, task: dict[str, Any], fallback_id: str) -> SampleToolCache:
        """Get a sample cache, preferring a stable dataset-provided ID."""

        sample_id = self._explicit_sample_id(task) or f"task:{fallback_id}"
        with self._lock:
            return self._samples.setdefault(sample_id, SampleToolCache())

    def clear(self) -> None:
        """Release all raw search/page payloads held for the epoch."""

        with self._lock:
            self._samples.clear()


__all__ = ["EpochToolCache", "SampleToolCache"]
