"""Search client abstractions for text and image discovery."""

from __future__ import annotations

import hashlib
import http.client
import json
import os
import re
import socket
import sys
import tempfile
import threading
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlencode, urlparse
from urllib.error import HTTPError, URLError
from urllib.request import HTTPHandler, HTTPSHandler, Request, build_opener, urlopen

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


_FIXED_SERPER_KEYS_FILE = Path(__file__).resolve().parent / "serper_keys.txt.example"
_FIXED_SERPER_POOL_STATE_FILE = (
    Path(__file__).resolve().parent / "ignore" / "serper_pool_state.json"
)
_FIXED_SERPER_POOL_MIN_REMAINING = 100
_FIXED_SERPER_POOL_DEFAULT_CREDITS = 2500
_FIXED_SERPER_KEY_POOL: "SerperApiKeyPool | None" = None

# QPM is intentionally tracked per Python process.  Serper requests can be
# issued concurrently by several worker threads, so access to the rolling
# window must be synchronized.  This is a debug metric, not a rate limiter.
_SERPER_QPM_WINDOW_S = 60.0
_SERPER_QPM_LOCK = threading.Lock()
_SERPER_SUCCESS_TIMES: deque[float] = deque()

# ``fcntl.flock`` coordinates separate processes, but it is not sufficient to
# protect all callers in the same process/thread pool.  Serialize the local
# read-modify-write operation as well.
_SERPER_POOL_THREAD_LOCK = threading.RLock()


def _create_ipv6_connection(address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT, source_address=None):
    """Create a TCP connection using only AF_INET6 DNS results."""

    host, port = address
    errors: list[OSError] = []
    infos = socket.getaddrinfo(host, port, socket.AF_INET6, socket.SOCK_STREAM)
    for family, socktype, proto, _canonname, sockaddr in infos:
        sock = socket.socket(family, socktype, proto)
        try:
            if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(timeout)
            if source_address:
                sock.bind(source_address)
            sock.connect(sockaddr)
            return sock
        except OSError as exc:
            errors.append(exc)
            sock.close()
    if errors:
        raise errors[-1]
    raise OSError(f"No IPv6 address found for {host!r}")


class _SerperIPv6HTTPConnection(http.client.HTTPConnection):
    _create_connection = staticmethod(_create_ipv6_connection)


class _SerperIPv6HTTPSConnection(http.client.HTTPSConnection):
    _create_connection = staticmethod(_create_ipv6_connection)


class _SerperIPv6HTTPHandler(HTTPHandler):
    def http_open(self, req):
        return self.do_open(_SerperIPv6HTTPConnection, req)


class _SerperIPv6HTTPSHandler(HTTPSHandler):
    def https_open(self, req):
        return self.do_open(
            _SerperIPv6HTTPSConnection,
            req,
            context=getattr(self, "_context", None),
        )


def _serper_ipv6_enabled(value: bool | None) -> bool:
    if value is not None:
        return bool(value)
    raw = os.environ.get("SERPER_IPV6_ONLY") or os.environ.get("SFT_SERPER_IPV6_ONLY")
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _local_lock_path_for_state(state_path: Path) -> Path:
    state_path_str = str(state_path.resolve())
    state_hash = hashlib.sha256(state_path_str.encode("utf-8")).hexdigest()[:16]
    lock_name = f"serper_pool_state_{state_hash}.lock"
    return Path(tempfile.gettempdir()) / lock_name


def _augment_query_with_site_exclusion(query: str, domain: str) -> str:
    normalized_query = str(query or "").strip()
    normalized_domain = str(domain or "").strip().lower()
    if not normalized_query or not normalized_domain:
        return normalized_query
    exclusion = f"-site:{normalized_domain}"
    if exclusion.lower() in normalized_query.lower():
        return normalized_query
    return f"{normalized_query} {exclusion}".strip()


def _augment_query_with_literal_exclusion(query: str, exclusion: str) -> str:
    normalized_query = str(query or "").strip()
    normalized_exclusion = str(exclusion or "").strip()
    if not normalized_query or not normalized_exclusion:
        return normalized_query
    if normalized_exclusion.lower() in normalized_query.lower():
        return normalized_query
    return f"{normalized_query} {normalized_exclusion}".strip()


def _augment_text_query(query: str) -> str:
    # Keep provider queries free of advanced operators such as `-site:`.
    # Serper free accounts reject those patterns; callers that want to exclude
    # domains should fetch extra results and filter URLs locally.
    return str(query or "").strip()


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


def _log_serper_key(*, url: str, api_key: str, metadata: dict[str, Any] | None = None) -> None:
    """Log the exact Serper API key selected for an outbound request."""
    del url, api_key, metadata
    return


def _log_serper_request(*, url: str, body: dict[str, Any]) -> None:
    """Log an outbound Serper request before network I/O begins."""
    del url, body
    return


def _serper_debug_enabled() -> bool:
    raw = os.environ.get("SFT_SERPER_DEBUG") or os.environ.get("SERPER_DEBUG")
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _serper_debug(event: str, **kwargs: Any) -> None:
    # Suppress normal request lifecycle output. Keep only failure diagnostics
    # so failed calls retain their request/response context.
    if event not in {"http_error", "url_error", "key_credits_exhausted"}:
        return
    details = " ".join(f"{key}={value!r}" for key, value in kwargs.items())
    suffix = f" {details}" if details else ""
    print(f"[serper-debug] {event}{suffix}", file=sys.stderr, flush=True)


def _serper_qpm_debug_enabled() -> bool:
    """Return whether per-success Serper QPM debug output is enabled.

    This output is enabled by default because it is useful for diagnosing the
    effective request rate.  Set ``SFT_SERPER_QPM_DEBUG=0`` to silence it.
    """

    raw = os.environ.get("SFT_SERPER_QPM_DEBUG")
    if raw is None or not str(raw).strip():
        return True
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _record_serper_success_qpm() -> int:
    """Record one successful request and return successes in the last minute."""

    now = time.monotonic()
    cutoff = now - _SERPER_QPM_WINDOW_S
    with _SERPER_QPM_LOCK:
        while _SERPER_SUCCESS_TIMES and _SERPER_SUCCESS_TIMES[0] <= cutoff:
            _SERPER_SUCCESS_TIMES.popleft()
        _SERPER_SUCCESS_TIMES.append(now)
        return len(_SERPER_SUCCESS_TIMES)


def _log_serper_success_qpm(
    *,
    url: str,
    status_code: int,
    elapsed_s: float,
    response_chars: int,
) -> None:
    """Print the rolling process-local QPM after a successful Serper call."""

    current_qpm = _record_serper_success_qpm()
    # Success-path QPM diagnostics are intentionally disabled. Keep recording
    # the rolling count above for in-process rate accounting.
    del url, status_code, elapsed_s, response_chars, current_qpm


def _serper_dns_debug(url: str) -> dict[str, Any] | None:
    if str(os.environ.get("SFT_SERPER_DEBUG_DNS") or "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    parsed = urlparse(url)
    host = parsed.hostname or ""
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return {"host": host, "error": "missing_host"}
    started = time.perf_counter()
    try:
        infos = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    except Exception as exc:  # noqa: BLE001 - debug path only
        return {
            "host": host,
            "port": port,
            "elapsed_s": round(time.perf_counter() - started, 3),
            "error_type": exc.__class__.__name__,
            "error": str(exc),
        }
    addresses = []
    for info in infos[:5]:
        sockaddr = info[4]
        if sockaddr:
            addresses.append(str(sockaddr[0]))
    return {
        "host": host,
        "port": port,
        "elapsed_s": round(time.perf_counter() - started, 3),
        "addresses": addresses,
        "count": len(infos),
    }


def _is_serper_credit_exhausted(*, status_code: int, response_body: str) -> bool:
    """Recognize Serper's credit-exhaustion errors without treating QPM as exhaustion."""
    if status_code == 429:
        return False
    message = response_body.lower()
    return any(
        marker in message
        for marker in (
            "insufficient credits",
            "not enough credits",
            "credit balance",
            "credits exhausted",
            "credit limit exceeded",
            "quota exhausted",
        )
    )


def _log_serper_raw_response(*, url: str, status_code: int, raw: dict[str, Any]) -> None:
    """Log Serper's unmodified parsed JSON response before result parsing."""

    # print(
    #     "[serper-raw-response]"
    #     f" url={url!r}"
    #     f" status_code={status_code}"
    #     f" raw={json.dumps(raw, ensure_ascii=False, default=str)}",
    #     file=sys.stderr,
    #     flush=True,
    # )


def _log_serper_results(response: SearchResponse) -> None:
    """Print the parsed Serper response so search failures are visible in terminal traces."""
    del response
    return


class SearchClient(Protocol):
    """Minimal search interface used by graph builders."""

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        """Return text/web search results."""

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        """Return image search results."""


class SerperApiKeyPool:
    """Simple cross-process credit-tracked key pool for Serper."""

    def __init__(
        self,
        *,
        keys: list[str],
        state_path: str | Path | None = None,
        default_credits: int = 2500,
        min_remaining: int = 100,
    ) -> None:
        cleaned = list(dict.fromkeys(key.strip() for key in keys if key and key.strip()))
        if not cleaned:
            raise ValueError("Serper API key pool requires at least one key.")
        self.keys = cleaned
        self.default_credits = max(1, int(default_credits))
        self.min_remaining = max(0, int(min_remaining))
        configured_path = state_path or os.environ.get("SERPER_API_POOL_STATE_FILE")
        self.state_path = (
            Path(configured_path)
            if configured_path
            else Path(__file__).resolve().parent / "ignore" / "serper_pool_state.json"
        )
        if not self.state_path.is_absolute():
            self.state_path = Path.cwd() / self.state_path
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_id(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_env(cls) -> "SerperApiKeyPool" | None:
        keys = cls._load_keys_from_env()
        if not keys:
            return None
        return cls(
            keys=keys,
            state_path=os.environ.get("SERPER_API_POOL_STATE_FILE"),
            default_credits=int(os.environ.get("SERPER_API_POOL_DEFAULT_CREDITS") or 2500),
            min_remaining=int(os.environ.get("SERPER_API_POOL_MIN_REMAINING") or 100),
        )

    @classmethod
    def from_fixed_pool(cls) -> "SerperApiKeyPool":
        """Return the process-wide fixed pool, loading its key file once."""
        global _FIXED_SERPER_KEY_POOL
        if _FIXED_SERPER_KEY_POOL is not None:
            return _FIXED_SERPER_KEY_POOL
        _FIXED_SERPER_KEY_POOL = cls(
            keys=cls._load_keys_from_file(_FIXED_SERPER_KEYS_FILE),
            state_path=_FIXED_SERPER_POOL_STATE_FILE,
            default_credits=_FIXED_SERPER_POOL_DEFAULT_CREDITS,
            min_remaining=_FIXED_SERPER_POOL_MIN_REMAINING,
        )
        return _FIXED_SERPER_KEY_POOL

    @staticmethod
    def _load_keys_from_env() -> list[str]:
        if os.environ.get("SERPER_API_KEYS"):
            return [item.strip() for item in os.environ["SERPER_API_KEYS"].split(",") if item.strip()]
        keys_file = os.environ.get("SERPER_API_KEYS_FILE")
        if not keys_file:
            return []
        path = Path(keys_file)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Serper API keys file does not exist: {path}")
        keys: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            keys.append(line)
        return keys

    @staticmethod
    def _load_keys_from_file(path_like: str | Path) -> list[str]:
        path = Path(path_like)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Serper API keys file does not exist: {path}")
        keys: list[str] = []
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            keys.append(line)
        return keys

    def acquire_key(self) -> tuple[str, dict[str, Any]]:
        state = self._with_locked_state(self._acquire_from_state)
        key = state.pop("_selected_key")
        metadata = state.pop("_selected_metadata")
        return key, metadata

    def status(self) -> dict[str, Any]:
        """Return the shared pool's currently usable keys and estimated credits."""
        state = self._with_locked_state(self._initialize_state)
        return self._pool_status(state)

    def mark_credits_exhausted(self, key_id: str, *, reason: str | None = None) -> dict[str, Any]:
        """Disable a key after Serper explicitly reports that its credits are gone."""
        def update(state: dict[str, Any]) -> dict[str, Any]:
            state = self._initialize_state(state)
            record = dict((state.get("keys") or {}).get(key_id) or {})
            if record:
                record["remaining_credits"] = 0
                record["disabled"] = True
                record["disabled_reason"] = reason or "credits_exhausted"
                record["disabled_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
                state["keys"][key_id] = record
            self._store_pool_status(state)
            return state

        return self._pool_status(self._with_locked_state(update))

    def _initialize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        pool = dict(state.get("keys") or {})
        ordered_ids = []
        for key in self.keys:
            key_id = self.key_id(key)
            ordered_ids.append(key_id)
            record = dict(pool.get(key_id) or {})
            record.setdefault("remaining_credits", self.default_credits)
            record.setdefault("initial_credits", self.default_credits)
            record.setdefault("disabled", False)
            record.setdefault("masked_key", self._mask_key(key))
            pool[key_id] = record
        state["keys"] = pool
        state["key_order"] = ordered_ids
        state["default_credits"] = self.default_credits
        state["min_remaining"] = self.min_remaining
        self._store_pool_status(state)
        return state

    def _pool_status(self, state: dict[str, Any]) -> dict[str, Any]:
        pool = dict(state.get("keys") or {})
        records = [dict(pool.get(self.key_id(key)) or {}) for key in self.keys]
        available = [
            record
            for record in records
            if not bool(record.get("disabled"))
            and int(record.get("remaining_credits") or 0) > self.min_remaining
        ]
        return {
            "remaining_credits_total": sum(max(0, int(record.get("remaining_credits") or 0)) for record in records),
            "available_key_count": len(available),
            "total_key_count": len(records),
            "min_remaining": self.min_remaining,
        }

    def _store_pool_status(self, state: dict[str, Any]) -> None:
        state["pool_status"] = self._pool_status(state)

    def _acquire_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        state = self._initialize_state(state)
        pool = dict(state["keys"])
        ordered_ids = list(state["key_order"])

        selected_key: str | None = None
        selected_id: str | None = None
        previous_id = str(state.get("last_selected_key_id") or "")
        start_index = (ordered_ids.index(previous_id) + 1) % len(ordered_ids) if previous_id in ordered_ids else 0
        for offset in range(len(ordered_ids)):
            key_id = ordered_ids[(start_index + offset) % len(ordered_ids)]
            record = pool[key_id]
            remaining = int(record.get("remaining_credits") or 0)
            disabled = bool(record.get("disabled"))
            if disabled:
                continue
            if remaining <= self.min_remaining:
                continue
            selected_key = self.keys[ordered_ids.index(key_id)]
            selected_id = key_id
            break

        if selected_key is None or selected_id is None:
            raise RuntimeError(
                "No Serper API key in the pool has enough remaining credits. "
                f"Minimum remaining threshold: {self.min_remaining}."
            )

        selected = pool[selected_id]
        selected["remaining_credits"] = max(0, int(selected.get("remaining_credits") or 0) - 1)
        selected["last_used_at"] = now
        pool[selected_id] = selected
        state["keys"] = pool
        state["last_selected_key_id"] = selected_id
        state["updated_at"] = now
        self._store_pool_status(state)
        state["_selected_key"] = selected_key
        state["_selected_metadata"] = {
            "key_id": selected_id,
            "masked_key": selected.get("masked_key"),
            "remaining_credits": selected.get("remaining_credits"),
            "initial_credits": selected.get("initial_credits"),
            **self._pool_status(state),
        }
        return state

    def _with_locked_state(self, callback) -> dict[str, Any]:
        lock_path = _local_lock_path_for_state(self.state_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with _SERPER_POOL_THREAD_LOCK:
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                if fcntl is not None:
                    fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                try:
                    state = self._read_state()
                    updated = callback(state)
                    self._write_state(updated)
                    return updated
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {}
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        persisted = {key: value for key, value in state.items() if not key.startswith("_")}
        tmp_path: Path | None = None
        try:
            # Use a unique temporary file.  A fixed ``.tmp`` path can be
            # replaced by another worker before the first worker calls
            # ``os.replace``.
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(self.state_path.parent),
                prefix=f".{self.state_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as tmp_handle:
                tmp_path = Path(tmp_handle.name)
                tmp_handle.write(json.dumps(persisted, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
                tmp_handle.flush()
                os.fsync(tmp_handle.fileno())
            os.replace(tmp_path, self.state_path)
            tmp_path = None
        finally:
            if tmp_path is not None:
                tmp_path.unlink(missing_ok=True)

    @staticmethod
    def _mask_key(key: str) -> str:
        if len(key) <= 8:
            return key
        return f"{key[:4]}...{key[-4:]}"


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
        body = self._serper_body(_augment_text_query(query), limit, kwargs)
        raw = self._post_json("/search", body)
        response = SearchResponse(
            query=query,
            engine="serper_adapter:search",
            results=OpenSerpSearchClient._parse_text_results(raw),
            raw_response=raw,
            status_code=200,
        )
        _log_serper_results(response)
        return response

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        body = self._serper_body(query, limit, kwargs)
        raw = self._post_json("/images", body)
        response = SearchResponse(
            query=query,
            engine="serper_adapter:images",
            results=OpenSerpSearchClient._parse_image_results(raw),
            raw_response=raw,
            status_code=200,
        )
        _log_serper_results(response)
        return response

    @staticmethod
    def _serper_body(query: str, limit: int, params: dict[str, Any]) -> dict[str, Any]:
        body = {
            "q": query,
            "num": max(1, min(int(limit), 100)),
        }
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
        url = f"{self.base_url}{path}"
        _log_serper_key(url=url, api_key=self.api_key, metadata={"client": self.__class__.__name__})
        _log_serper_request(url=url, body=body)
        payload = json.dumps(body).encode("utf-8")
        request = Request(
            url,
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
            status_code = response.getcode()
        raw = json.loads(response_payload)
        _log_serper_raw_response(url=url, status_code=status_code, raw=raw)
        return raw


class SerperSearchClient:
    """Direct client for the official Serper.dev API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        api_keys: list[str] | None = None,
        search_url: str | None = None,
        images_url: str | None = None,
        timeout_s: float = 60.0,
        pool_state_path: str | Path | None = None,
        pool_default_credits: int | None = None,
        pool_min_remaining: int | None = None,
        ipv6_only: bool | None = None,
    ) -> None:
        self.api_key = None
        self.key_pool: SerperApiKeyPool | None = None
        if api_keys:
            self.key_pool = SerperApiKeyPool(
                keys=api_keys,
                state_path=pool_state_path if pool_state_path is not None else _FIXED_SERPER_POOL_STATE_FILE,
                default_credits=(
                    pool_default_credits
                    if pool_default_credits is not None
                    else _FIXED_SERPER_POOL_DEFAULT_CREDITS
                ),
                min_remaining=(
                    pool_min_remaining
                    if pool_min_remaining is not None
                    else _FIXED_SERPER_POOL_MIN_REMAINING
                ),
            )
        else:
            self.key_pool = SerperApiKeyPool.from_fixed_pool()
        self.search_url = search_url or os.environ.get("SERPER_SEARCH_URL") or "https://google.serper.dev/search"
        self.images_url = images_url or os.environ.get("SERPER_IMAGES_URL") or "https://google.serper.dev/images"
        self.timeout_s = timeout_s
        self.ipv6_only = _serper_ipv6_enabled(ipv6_only)
        self._url_opener = (
            build_opener(_SerperIPv6HTTPHandler(), _SerperIPv6HTTPSHandler())
            if self.ipv6_only
            else None
        )

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        raw, status_code, metadata = self._post_json(self.search_url, self._serper_body(_augment_text_query(query), limit, kwargs))
        response = SearchResponse(
            query=query,
            engine="serper:search",
            results=self._parse_text_results(raw),
            raw_response=raw,
            status_code=status_code,
            metadata=metadata,
        )
        _log_serper_results(response)
        return response

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        raw, status_code, metadata = self._post_json(self.images_url, self._serper_body(query, limit, kwargs))
        response = SearchResponse(
            query=query,
            engine="serper:images",
            results=self._parse_image_results(raw),
            raw_response=raw,
            status_code=status_code,
            metadata=metadata,
        )
        _log_serper_results(response)
        return response

    def _post_json(self, url: str, body: dict[str, Any]) -> tuple[dict[str, Any], int, dict[str, Any]]:
        api_key = self.api_key
        pool_metadata: dict[str, Any] = {}
        if self.key_pool is not None:
            api_key, pool_metadata = self.key_pool.acquire_key()
        if not api_key:
            raise ValueError(
                "Serper API key is required. Populate the fixed pool file "
                f"at {_FIXED_SERPER_KEYS_FILE}."
            )

        _log_serper_key(url=url, api_key=api_key, metadata=pool_metadata)
        _log_serper_request(url=url, body=body)
        parsed_url = urlparse(url)
        _serper_debug(
            "request_start",
            url=url,
            host=parsed_url.hostname,
            timeout_s=self.timeout_s,
            body=body,
            key_pool=pool_metadata if pool_metadata else None,
            dns=_serper_dns_debug(url),
        )
        payload = json.dumps(body).encode("utf-8")
        request_headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "X-API-KEY": api_key,
        }
        relay_token = os.environ.get("SERPER_RELAY_TOKEN")
        if relay_token:
            request_headers["X-Serper-Relay-Token"] = relay_token
        request = Request(
            url,
            data=payload,
            headers=request_headers,
            method="POST",
        )
        started_at = time.perf_counter()
        try:
            open_request = self._url_opener.open if self._url_opener is not None else urlopen
            with open_request(request, timeout=self.timeout_s) as response:
                response_payload = response.read().decode("utf-8")
                status_code = response.getcode()
            _serper_debug(
                "request_done",
                url=url,
                status_code=status_code,
                elapsed_s=round(time.perf_counter() - started_at, 3),
                response_chars=len(response_payload),
            )
        except HTTPError as exc:
            elapsed_s = round(time.perf_counter() - started_at, 3)
            error_payload = ""
            try:
                error_payload = exc.read().decode("utf-8", errors="replace")
            except Exception:
                error_payload = ""
            if self.key_pool is not None and _is_serper_credit_exhausted(
                status_code=exc.code,
                response_body=error_payload,
            ):
                pool_status = self.key_pool.mark_credits_exhausted(
                    str(pool_metadata.get("key_id") or ""),
                    reason=f"serper_http_{exc.code}_credits_exhausted",
                )
                _serper_debug(
                    "key_credits_exhausted",
                    key_id=pool_metadata.get("key_id"),
                    status_code=exc.code,
                    key_pool=pool_status,
                )
            _serper_debug(
                "http_error",
                url=url,
                status_code=exc.code,
                elapsed_s=elapsed_s,
                body=body,
                response_preview=error_payload[:500],
            )
            message = (
                f"Serper request failed with HTTP {exc.code} for {url}. "
                f"Request body: {json.dumps(body, ensure_ascii=False)}"
            )
            if error_payload:
                message += f" Response body: {error_payload}"
            raise RuntimeError(message) from exc
        except URLError as exc:
            elapsed_s = round(time.perf_counter() - started_at, 3)
            reason = getattr(exc, "reason", exc)
            _serper_debug(
                "url_error",
                url=url,
                host=parsed_url.hostname,
                elapsed_s=elapsed_s,
                timeout_s=self.timeout_s,
                reason_type=reason.__class__.__name__,
                reason=str(reason),
                body=body,
                key_pool=pool_metadata if pool_metadata else None,
            )
            raise RuntimeError(
                "Serper request failed before receiving an HTTP response "
                f"for {url}. host={parsed_url.hostname!r} timeout_s={self.timeout_s} "
                f"elapsed_s={elapsed_s} reason={reason.__class__.__name__}: {reason}. "
                f"Request body: {json.dumps(body, ensure_ascii=False)}"
            ) from exc
        raw = json.loads(response_payload)
        _log_serper_success_qpm(
            url=url,
            status_code=status_code,
            elapsed_s=time.perf_counter() - started_at,
            response_chars=len(response_payload),
        )
        _log_serper_raw_response(url=url, status_code=status_code, raw=raw)
        self._log_raw_response(url=url, body=body, status_code=status_code, raw=raw)
        metadata = {
            "serper_key_pool": pool_metadata if pool_metadata else None,
        }
        return raw, status_code, metadata

    @staticmethod
    def _log_raw_response(
        *,
        url: str,
        body: dict[str, Any],
        status_code: int,
        raw: dict[str, Any],
    ) -> None:
        return

    @staticmethod
    def _serper_body(query: str, limit: int, params: dict[str, Any]) -> dict[str, Any]:
        body = {
            "q": query,
            "num": max(1, min(int(limit), 100)),
        }
        for src_key, dst_key in (
            ("hl", "hl"),
            ("lang", "hl"),
            ("gl", "gl"),
            ("region", "gl"),
            ("location", "location"),
            ("page", "page"),
            ("num", "num"),
            ("autocorrect", "autocorrect"),
            ("safe", "safe"),
            ("tbs", "tbs"),
            ("type", "type"),
        ):
            if params.get(src_key) is not None:
                body[dst_key] = params[src_key]
        return body

    @staticmethod
    def _parse_text_results(raw: dict[str, Any]) -> list[TextSearchResult]:
        results: list[TextSearchResult] = []
        organic = raw.get("organic") or []
        if not isinstance(organic, list):
            return results
        for rank, item in enumerate(organic, start=1):
            if not isinstance(item, dict):
                continue
            results.append(
                TextSearchResult(
                    title=item.get("title"),
                    url=item.get("link"),
                    snippet=item.get("snippet"),
                    source=item.get("source") or item.get("domain"),
                    rank=SerperSearchClient._position(item, rank),
                    raw=item,
                )
            )
        return results

    @staticmethod
    def _parse_image_results(raw: dict[str, Any]) -> list[ImageSearchResult]:
        results: list[ImageSearchResult] = []
        images = raw.get("images") or []
        if not isinstance(images, list):
            return results
        for rank, item in enumerate(images, start=1):
            if not isinstance(item, dict):
                continue
            results.append(
                ImageSearchResult(
                    title=item.get("title"),
                    image_url=item.get("imageUrl"),
                    source_page_url=item.get("link"),
                    thumbnail_url=item.get("thumbnailUrl"),
                    snippet=item.get("snippet"),
                    source=item.get("source") or item.get("domain"),
                    width=item.get("imageWidth"),
                    height=item.get("imageHeight"),
                    rank=SerperSearchClient._position(item, rank),
                    raw=item,
                )
            )
        return results

    @staticmethod
    def _position(item: dict[str, Any], fallback: int) -> int:
        try:
            return int(item.get("position"))
        except (TypeError, ValueError):
            return fallback


def acquire_serper_api_key() -> tuple[str, dict[str, Any]]:
    """Acquire one Serper API key from the fixed cross-process pool."""

    return SerperApiKeyPool.from_fixed_pool().acquire_key()


class SerpApiSearchClient:
    """Client for SerpApi-compatible Google Search and Google Images APIs.

    It supports both the public SerpApi endpoint and ByteDance's AIDP SERP
    proxy. The public endpoint uses `api_key`; the AIDP proxy uses `ak`.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        auth_param: str | None = None,
        timeout_s: float = 60.0,
        text_engine: str = "google",
        image_engine: str = "google_images",
    ) -> None:
        self.base_url = base_url or os.environ.get("SERPAPI_BASE_URL") or "https://serpapi.com/search.json"
        self.auth_param = auth_param or os.environ.get("SERPAPI_AUTH_PARAM") or self._default_auth_param(self.base_url)
        self.api_key = (
            api_key
            or os.environ.get("SERPAPI_AK")
            or os.environ.get("AIDP_SERP_AK")
            or os.environ.get("SERPAPI_API_KEY")
            or os.environ.get("SERP_API_KEY")
        )
        self.timeout_s = timeout_s
        self.text_engine = text_engine
        self.image_engine = image_engine

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        raw, status_code = self._get_json(
            self._serpapi_params(query, limit, kwargs, engine=self.text_engine)
        )
        return SearchResponse(
            query=query,
            engine=f"serpapi:{self.text_engine}",
            results=self._parse_text_results(raw),
            raw_response=raw,
            status_code=status_code,
        )

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        raw, status_code = self._get_json(
            self._serpapi_params(query, limit, kwargs, engine=self.image_engine)
        )
        return SearchResponse(
            query=query,
            engine=f"serpapi:{self.image_engine}",
            results=self._parse_image_results(raw),
            raw_response=raw,
            status_code=status_code,
        )

    def _serpapi_params(
        self,
        query: str,
        limit: int,
        params: dict[str, Any],
        *,
        engine: str,
    ) -> dict[str, Any]:
        if not self.api_key:
            raise ValueError(
                "SerpApi credential is required. Set SERPAPI_AK/AIDP_SERP_AK for the "
                "AIDP proxy, or SERPAPI_API_KEY for public SerpApi."
            )

        request_params: dict[str, Any] = {
            "engine": engine,
            "q": query,
            self.auth_param: self.api_key,
            "num": max(1, min(int(limit), 100)),
        }
        self._apply_default_params(request_params)
        for key in (
            "hl",
            "gl",
            "location",
            "google_domain",
            "safe",
            "device",
            "tbs",
            "tbm",
            "ijn",
            "start",
        ):
            if params.get(key) is not None:
                request_params[key] = params[key]

        if params.get("lang") is not None and "hl" not in request_params:
            request_params["hl"] = params["lang"]
        if params.get("region") is not None and "gl" not in request_params:
            request_params["gl"] = params["region"]
        if params.get("page") is not None:
            page = self._int_or_default(params["page"], 1)
            if engine == self.image_engine and "ijn" not in request_params:
                request_params["ijn"] = max(0, page - 1)
            elif "start" not in request_params:
                request_params["start"] = max(0, page - 1) * max(1, int(limit))
        return request_params

    @staticmethod
    def _default_auth_param(base_url: str) -> str:
        if "aidp-i18ntt" in base_url:
            return "ak"
        return "api_key"

    @staticmethod
    def _apply_default_params(request_params: dict[str, Any]) -> None:
        defaults = {
            "location": os.environ.get("SERPAPI_LOCATION"),
            "google_domain": os.environ.get("SERPAPI_GOOGLE_DOMAIN"),
            "hl": os.environ.get("SERPAPI_HL"),
            "gl": os.environ.get("SERPAPI_GL"),
        }
        for key, value in defaults.items():
            if value and request_params.get(key) is None:
                request_params[key] = value

    def _get_json(self, params: dict[str, Any]) -> tuple[dict[str, Any], int]:
        url = f"{self.base_url}?{urlencode(params, doseq=True)}"
        request = Request(url, headers={"Accept": "application/json"})
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = response.read().decode("utf-8")
            status_code = response.getcode()
        return json.loads(payload), status_code

    @staticmethod
    def _parse_text_results(raw: dict[str, Any]) -> list[TextSearchResult]:
        results: list[TextSearchResult] = []
        organic = raw.get("organic_results") or []
        if not isinstance(organic, list):
            return results
        for rank, item in enumerate(organic, start=1):
            if not isinstance(item, dict):
                continue
            results.append(
                TextSearchResult(
                    title=item.get("title"),
                    url=item.get("link") or item.get("redirect_link"),
                    snippet=item.get("snippet") or item.get("snippet_highlighted_words"),
                    source=item.get("source") or item.get("displayed_link"),
                    rank=SerpApiSearchClient._position(item, rank),
                    raw=item,
                )
            )
        return results

    @staticmethod
    def _parse_image_results(raw: dict[str, Any]) -> list[ImageSearchResult]:
        results: list[ImageSearchResult] = []
        images = raw.get("images_results") or []
        if not isinstance(images, list):
            return results
        for rank, item in enumerate(images, start=1):
            if not isinstance(item, dict):
                continue
            results.append(
                ImageSearchResult(
                    title=item.get("title"),
                    image_url=item.get("original") or item.get("image") or item.get("link"),
                    source_page_url=item.get("link") or item.get("source_link"),
                    thumbnail_url=item.get("thumbnail"),
                    snippet=item.get("snippet"),
                    source=item.get("source"),
                    width=item.get("original_width") or item.get("width"),
                    height=item.get("original_height") or item.get("height"),
                    rank=SerpApiSearchClient._position(item, rank),
                    raw=item,
                )
            )
        return results

    @staticmethod
    def _position(item: dict[str, Any], fallback: int) -> int:
        return SerpApiSearchClient._int_or_default(
            item.get("position") or item.get("block_position"),
            fallback,
        )

    @staticmethod
    def _int_or_default(value: Any, default: int) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return default


class CommonsImageSearchClient:
    """Image search client backed by Wikimedia Commons.

    It uses the structured MediaWiki API instead of scraping Special:MediaSearch
    pages. This is more stable while returning the same core information needed
    by the graph pipeline: file title, original image URL, thumbnail URL, and
    Commons file page URL.
    """

    def __init__(
        self,
        *,
        api_url: str = "https://commons.wikimedia.org/w/api.php",
        timeout_s: float = 30.0,
        thumb_width: int = 320,
    ) -> None:
        self.api_url = api_url
        self.timeout_s = timeout_s
        self.thumb_width = thumb_width

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        del kwargs
        return SearchResponse(
            query=query,
            engine="commons:text_unsupported",
            results=[],
            metadata={"reason": "Wikimedia Commons client only supports image search."},
        )

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        request_params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrsearch": query,
            "gsrnamespace": 6,
            "gsrlimit": max(1, min(int(limit), 50)),
            "prop": "info|imageinfo",
            "inprop": "url",
            "iiprop": "url|mime|size|extmetadata",
            "iiurlwidth": kwargs.get("thumb_width", self.thumb_width),
        }
        url = f"{self.api_url}?{urlencode(request_params, doseq=True)}"
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "deepsearch-synthesis/0.1",
            },
        )
        with urlopen(request, timeout=self.timeout_s) as response:
            payload = response.read().decode("utf-8")
        raw = json.loads(payload)
        return SearchResponse(
            query=query,
            engine="commons:image",
            results=self._parse_commons_results(raw),
            raw_response=raw,
            status_code=200,
        )

    @classmethod
    def _parse_commons_results(cls, raw: dict[str, Any]) -> list[ImageSearchResult]:
        pages = raw.get("query", {}).get("pages", {})
        if not isinstance(pages, dict):
            return []

        results: list[ImageSearchResult] = []
        sorted_pages = sorted(
            pages.values(),
            key=lambda page: page.get("index", page.get("pageid", 0)),
        )
        for rank, page in enumerate(sorted_pages, start=1):
            imageinfo = page.get("imageinfo") or []
            info = imageinfo[0] if imageinfo else {}
            extmetadata = info.get("extmetadata") or {}
            title = page.get("title")
            results.append(
                ImageSearchResult(
                    title=title,
                    image_url=info.get("url"),
                    source_page_url=page.get("fullurl") or cls._commons_file_url(title),
                    thumbnail_url=info.get("thumburl"),
                    snippet=cls._metadata_text(extmetadata),
                    source="wikimedia_commons",
                    width=info.get("width"),
                    height=info.get("height"),
                    rank=rank,
                    raw=page,
                )
            )
        return results

    @staticmethod
    def _commons_file_url(title: str | None) -> str | None:
        if not title:
            return None
        return "https://commons.wikimedia.org/wiki/" + title.replace(" ", "_")

    @classmethod
    def _metadata_text(cls, extmetadata: dict[str, Any]) -> str | None:
        for key in ("ImageDescription", "ObjectName", "Credit", "Artist"):
            value = extmetadata.get(key, {})
            if isinstance(value, dict) and value.get("value"):
                return cls._strip_html(str(value["value"]))
        return None

    @staticmethod
    def _strip_html(text: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", text)).strip()


class CommonsSerpApiSearchClient:
    """Single backend that tries Wikimedia Commons first, then SerpApi.

    This keeps the caller-facing API to one backend choice while preserving the
    pragmatic "Commons first, web fallback second" retrieval behavior.
    """

    def __init__(
        self,
        *,
        commons_client: SearchClient | None = None,
        serpapi_client: SearchClient | None = None,
        min_commons_results: int = 1,
    ) -> None:
        self.commons_client = commons_client or CommonsImageSearchClient()
        self.serpapi_client = serpapi_client or SerpApiSearchClient()
        self.min_commons_results = min_commons_results

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        return self.serpapi_client.search_text(query, limit=limit, **kwargs)

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        commons_response = self.commons_client.search_image(query, limit=limit, **kwargs)
        commons_response.metadata.update(
            {
                "fallback_used": False,
                "backend": "commons_serpapi",
                "attempted_engines": [commons_response.engine],
            }
        )
        if len(commons_response.results) >= self.min_commons_results:
            return commons_response

        serp_response = self.serpapi_client.search_image(query, limit=limit, **kwargs)
        serp_response.metadata.update(
            {
                "fallback_used": True,
                "backend": "commons_serpapi",
                "attempted_engines": [commons_response.engine, serp_response.engine],
                "primary_engine": commons_response.engine,
                "primary_result_count": len(commons_response.results),
            }
        )
        return serp_response


class FallbackImageSearchClient:
    """Try one image search client first and fall back if too few results appear."""

    def __init__(
        self,
        primary: SearchClient,
        fallback: SearchClient,
        *,
        min_primary_results: int = 1,
    ) -> None:
        self.primary = primary
        self.fallback = fallback
        self.min_primary_results = min_primary_results

    def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        return self.fallback.search_text(query, limit=limit, **kwargs)

    def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
        primary_response = self.primary.search_image(query, limit=limit, **kwargs)
        if len(primary_response.results) >= self.min_primary_results:
            primary_response.metadata["fallback_used"] = False
            return primary_response

        fallback_response = self.fallback.search_image(query, limit=limit, **kwargs)
        fallback_response.metadata.update(
            {
                "fallback_used": True,
                "primary_engine": primary_response.engine,
                "primary_result_count": len(primary_response.results),
            }
        )
        return fallback_response


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


def _smoke_test() -> None:
    mock = MockSearchClient(
        text_results={
            "coffee": [TextSearchResult(title="Coffee", url="https://example.com/coffee")]
        },
        image_results={
            "coffee": [
                ImageSearchResult(
                    title="Coffee image",
                    image_url="https://example.com/coffee.jpg",
                    width=320,
                    height=240,
                )
            ]
        },
    )
    assert mock.search_text("coffee").results[0].title == "Coffee"
    assert mock.search_image("coffee").results[0].image_url.endswith(".jpg")

    serp = SerpApiSearchClient(
        api_key="dummy",
        base_url="https://aidp-i18ntt-sg.byteintl.net/api/modelhub/online/v2/crawl/serp",
    )
    params = serp._serpapi_params(
        "Coffee",
        10,
        {"location": "Austin, Texas, United States", "hl": "en", "gl": "us"},
        engine="google",
    )
    assert params["ak"] == "dummy"
    assert params["q"] == "Coffee"

    serper_body = SerperSearchClient._serper_body(
        _augment_text_query("Coffee"),
        5,
        {"hl": "en", "gl": "us", "location": "Austin, Texas, United States"},
    )
    assert serper_body["q"] == "Coffee"
    assert serper_body["num"] == 5
    assert serper_body["hl"] == "en"
    assert serper_body["gl"] == "us"

    raw_serper_images = {
        "images": [
            {
                "title": "Coffee cup",
                "imageUrl": "https://example.com/coffee.jpg",
                "imageWidth": 640,
                "imageHeight": 480,
                "thumbnailUrl": "https://example.com/thumb.jpg",
                "source": "Example",
                "domain": "example.com",
                "link": "https://example.com/page",
                "position": 1,
            }
        ]
    }
    parsed_serper = SerperSearchClient._parse_image_results(raw_serper_images)
    assert parsed_serper[0].image_url == "https://example.com/coffee.jpg"
    assert parsed_serper[0].source_page_url == "https://example.com/page"

    raw_commons = {
        "query": {
            "pages": {
                "1": {
                    "title": "File:Coffee.jpg",
                    "fullurl": "https://commons.wikimedia.org/wiki/File:Coffee.jpg",
                    "imageinfo": [
                        {
                            "url": "https://upload.wikimedia.org/coffee.jpg",
                            "thumburl": "https://upload.wikimedia.org/thumb/coffee.jpg",
                            "width": 640,
                            "height": 480,
                            "extmetadata": {"ImageDescription": {"value": "<p>Coffee cup</p>"}},
                        }
                    ],
                }
            }
        }
    }
    parsed = CommonsImageSearchClient._parse_commons_results(raw_commons)
    assert parsed[0].title == "File:Coffee.jpg"
    assert parsed[0].snippet == "Coffee cup"

    composite = CommonsSerpApiSearchClient(
        commons_client=MockSearchClient(
            image_results={
                "coffee": [],
            }
        ),
        serpapi_client=MockSearchClient(
            image_results={
                "coffee": [
                    ImageSearchResult(
                        title="Coffee via fallback",
                        image_url="https://example.com/fallback.jpg",
                    )
                ]
            }
        ),
    )
    composite_response = composite.search_image("coffee")
    assert composite_response.metadata["fallback_used"] is True
    assert composite_response.results[0].image_url == "https://example.com/fallback.jpg"
    print("search_client smoke test passed")


if __name__ == "__main__":
    _smoke_test()
