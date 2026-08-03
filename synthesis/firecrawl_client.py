"""Standalone Firecrawl URL-scraping backend with a shared API-key pool.

The backend is deliberately not wired into the existing readers.  Call
``FirecrawlClient.scrape`` directly when Firecrawl's managed browser is needed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows does not provide fcntl.
    fcntl = None


_MODULE_DIR = Path(__file__).resolve().parent
_FIXED_FIRECRAWL_KEYS_FILE = _MODULE_DIR / "firecrawl_keys.txt"
_FIXED_FIRECRAWL_POOL_STATE_FILE = _MODULE_DIR / "firecrawl_state.json"
_FIXED_FIRECRAWL_POOL_DEFAULT_CREDITS = 10000
_FIXED_FIRECRAWL_KEY_POOL: "FirecrawlApiKeyPool | None" = None


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _local_lock_path_for_state(state_path: Path) -> Path:
    """Keep the lock off the shared state directory used by concurrent workers."""
    state_hash = hashlib.sha256(str(state_path.resolve()).encode("utf-8")).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"firecrawl_pool_state_{state_hash}.lock"


class FirecrawlApiKeyPool:
    """Round-robin, cross-process Firecrawl key pool backed by JSON state."""

    def __init__(
        self,
        *,
        keys: list[str],
        state_path: str | Path | None = None,
        default_credits: int = _FIXED_FIRECRAWL_POOL_DEFAULT_CREDITS,
    ) -> None:
        self.keys = list(dict.fromkeys(key.strip() for key in keys if key and key.strip()))
        if not self.keys:
            raise ValueError("Firecrawl API key pool requires at least one key.")
        self.default_credits = max(1, int(default_credits))
        configured_path = state_path or os.environ.get("FIRECRAWL_API_POOL_STATE_FILE")
        self.state_path = Path(configured_path) if configured_path else _FIXED_FIRECRAWL_POOL_STATE_FILE
        if not self.state_path.is_absolute():
            self.state_path = Path.cwd() / self.state_path
        self.lock_path = _local_lock_path_for_state(self.state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key_id(key: str) -> str:
        return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]

    @classmethod
    def from_fixed_pool(cls) -> "FirecrawlApiKeyPool":
        global _FIXED_FIRECRAWL_KEY_POOL
        if _FIXED_FIRECRAWL_KEY_POOL is None:
            _FIXED_FIRECRAWL_KEY_POOL = cls(keys=cls._load_keys(_FIXED_FIRECRAWL_KEYS_FILE))
        return _FIXED_FIRECRAWL_KEY_POOL

    @classmethod
    def from_env(cls) -> "FirecrawlApiKeyPool | None":
        raw_keys = os.environ.get("FIRECRAWL_API_KEYS", "")
        if raw_keys:
            keys = [key.strip() for key in raw_keys.split(",") if key.strip()]
        else:
            path = os.environ.get("FIRECRAWL_API_KEYS_FILE")
            if not path:
                return None
            keys = cls._load_keys(Path(path))
        return cls(
            keys=keys,
            state_path=os.environ.get("FIRECRAWL_API_POOL_STATE_FILE"),
            default_credits=int(
                os.environ.get("FIRECRAWL_API_POOL_DEFAULT_CREDITS")
                or _FIXED_FIRECRAWL_POOL_DEFAULT_CREDITS
            ),
        )

    @staticmethod
    def _load_keys(path_like: str | Path) -> list[str]:
        path = Path(path_like)
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            raise FileNotFoundError(f"Firecrawl API keys file does not exist: {path}")
        return [
            line
            for raw_line in path.read_text(encoding="utf-8").splitlines()
            if (line := raw_line.strip()) and not line.startswith("#")
        ]

    def acquire_key(self) -> tuple[str, dict[str, Any]]:
        state = self._with_locked_state(self._acquire_from_state)
        return state.pop("_selected_key"), state.pop("_selected_metadata")

    def record_result(
        self,
        key_id: str,
        *,
        success: bool,
        error: str | None = None,
        credits_used: int = 0,
        status_code: int | None = None,
    ) -> dict[str, Any]:
        """Persist the result of one Firecrawl call for the selected key."""
        def update(state: dict[str, Any]) -> dict[str, Any]:
            state = self._initialize_state(state)
            record = dict(state["keys"].get(key_id) or {})
            if not record:
                return state
            now = _utc_now()
            record["last_result"] = "success" if success else "failure"
            record["last_result_at"] = now
            record["last_credits_used"] = max(0, int(credits_used))
            if status_code is not None:
                record["last_status_code"] = status_code
            if credits_used:
                record["remaining_credits"] = max(
                    0, int(record.get("remaining_credits") or 0) - max(0, int(credits_used))
                )
                record["credits_consumed"] = int(record.get("credits_consumed") or 0) + max(0, int(credits_used))
            if success:
                record["successful_requests"] = int(record.get("successful_requests") or 0) + 1
                record["consecutive_failures"] = 0
                record.pop("last_error", None)
            else:
                record["failed_requests"] = int(record.get("failed_requests") or 0) + 1
                record["consecutive_failures"] = int(record.get("consecutive_failures") or 0) + 1
                record["last_error"] = (error or "unknown Firecrawl error")[:2000]
                if self._is_terminal_key_error(error or ""):
                    record["state"] = "disabled"
                    record["disabled"] = True
                    if "credit" in (error or "").lower() or "quota" in (error or "").lower():
                        record["remaining_credits"] = 0
                    record["disabled_at"] = now
                    record["disabled_reason"] = "firecrawl_key_rejected_or_exhausted"
            if int(record.get("remaining_credits") or 0) == 0:
                record["state"] = "disabled"
                record["disabled"] = True
                record.setdefault("disabled_at", now)
                record.setdefault("disabled_reason", "estimated_credits_exhausted")
            state["keys"][key_id] = record
            state["updated_at"] = now
            self._store_pool_status(state)
            return state

        return self._pool_status(self._with_locked_state(update))

    def status(self) -> dict[str, Any]:
        return self._pool_status(self._with_locked_state(self._initialize_state))

    def _initialize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        pool = dict(state.get("keys") or {})
        ordered_ids: list[str] = []
        for key in self.keys:
            key_id = self.key_id(key)
            ordered_ids.append(key_id)
            record = dict(pool.get(key_id) or {})
            record.setdefault("masked_key", self._mask_key(key))
            record.setdefault("state", "active")
            record.setdefault("disabled", False)
            record.setdefault("initial_credits", self.default_credits)
            record.setdefault("remaining_credits", self.default_credits)
            record.setdefault("credits_consumed", 0)
            record.setdefault("total_requests", 0)
            record.setdefault("successful_requests", 0)
            record.setdefault("failed_requests", 0)
            record.setdefault("consecutive_failures", 0)
            pool[key_id] = record
        state["keys"] = pool
        state["key_order"] = ordered_ids
        state["default_credits"] = self.default_credits
        self._store_pool_status(state)
        return state

    def _acquire_from_state(self, state: dict[str, Any]) -> dict[str, Any]:
        state = self._initialize_state(state)
        ordered_ids = list(state["key_order"])
        previous_id = str(state.get("last_selected_key_id") or "")
        start = (ordered_ids.index(previous_id) + 1) % len(ordered_ids) if previous_id in ordered_ids else 0
        selected_id: str | None = None
        for offset in range(len(ordered_ids)):
            candidate_id = ordered_ids[(start + offset) % len(ordered_ids)]
            record = state["keys"][candidate_id]
            if not bool(record.get("disabled")) and int(record.get("remaining_credits") or 0) > 0:
                selected_id = candidate_id
                break
        if selected_id is None:
            raise RuntimeError("No active Firecrawl API key is available in the pool.")
        record = dict(state["keys"][selected_id])
        now = _utc_now()
        record["total_requests"] = int(record.get("total_requests") or 0) + 1
        record["last_used_at"] = now
        state["keys"][selected_id] = record
        state["last_selected_key_id"] = selected_id
        state["updated_at"] = now
        self._store_pool_status(state)
        state["_selected_key"] = self.keys[ordered_ids.index(selected_id)]
        state["_selected_metadata"] = {
            "key_id": selected_id,
            "masked_key": record["masked_key"],
            "state": record["state"],
            "pool_status": self._pool_status(state),
        }
        return state

    def _pool_status(self, state: dict[str, Any]) -> dict[str, Any]:
        records = [dict((state.get("keys") or {}).get(self.key_id(key)) or {}) for key in self.keys]
        return {
            "available_key_count": sum(
                not bool(record.get("disabled")) and int(record.get("remaining_credits") or 0) > 0
                for record in records
            ),
            "total_key_count": len(records),
            "remaining_credits_total": sum(max(0, int(record.get("remaining_credits") or 0)) for record in records),
            "credits_consumed": sum(int(record.get("credits_consumed") or 0) for record in records),
            "default_credits": self.default_credits,
            "successful_requests": sum(int(record.get("successful_requests") or 0) for record in records),
            "failed_requests": sum(int(record.get("failed_requests") or 0) for record in records),
        }

    def _store_pool_status(self, state: dict[str, Any]) -> None:
        state["pool_status"] = self._pool_status(state)

    def _with_locked_state(self, callback: Callable[[dict[str, Any]], dict[str, Any]]) -> dict[str, Any]:
        # Keep the lock on local storage: concurrent creates on a shared mount
        # can transiently report ENOENT even though its parent exists.
        lock_handle = self._open_lock_file()
        with lock_handle:
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

    def _open_lock_file(self):
        for attempt in range(2):
            self.lock_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                return self.lock_path.open("a+", encoding="utf-8")
            except FileNotFoundError:
                if attempt:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    def _read_state(self) -> dict[str, Any]:
        try:
            return json.loads(self.state_path.read_text(encoding="utf-8")) if self.state_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def _write_state(self, state: dict[str, Any]) -> None:
        persisted = {key: value for key, value in state.items() if not key.startswith("_")}
        temporary = self.state_path.with_name(f"{self.state_path.name}.tmp")
        temporary.write_text(json.dumps(persisted, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, self.state_path)

    @staticmethod
    def _is_terminal_key_error(error: str) -> bool:
        message = error.lower()
        return any(marker in message for marker in (
            "invalid api key", "invalid_api_key", "api key is invalid", "unauthorized",
            "authentication failed", "api key not found", "insufficient credits",
            "credits exhausted", "credit balance", "quota exhausted",
        ))

    @staticmethod
    def _mask_key(key: str) -> str:
        return key if len(key) <= 8 else f"{key[:4]}...{key[-4:]}"


class FirecrawlClient:
    """Synchronous Firecrawl scraper that rotates keys before every request."""

    def __init__(
        self,
        *,
        api_keys: list[str] | None = None,
        pool_state_path: str | Path | None = None,
        app_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.key_pool = (
            FirecrawlApiKeyPool(keys=api_keys, state_path=pool_state_path)
            if api_keys is not None
            else FirecrawlApiKeyPool.from_fixed_pool()
        )
        self._app_factory = app_factory
        self.last_pool_metadata: dict[str, Any] | None = None

    def scrape(
        self,
        url: str,
        *,
        only_main_content: bool = True,
        max_age: int | None = 172800000,
        parsers: list[str] | None = None,
        formats: list[str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Scrape ``url`` and return Firecrawl's unmodified response dictionary."""
        api_key, pool_metadata = self.key_pool.acquire_key()
        self.last_pool_metadata = pool_metadata
        request_kwargs: dict[str, Any] = {"only_main_content": only_main_content}
        if max_age is not None:
            request_kwargs["max_age"] = max_age
        if parsers is not None:
            request_kwargs["parsers"] = parsers
        if formats is not None:
            request_kwargs["formats"] = formats
        request_kwargs.update(kwargs)
        print(f"[firecrawl] scrape_start url={url}", file=sys.stderr, flush=True)
        try:
            result = self._app(api_key).scrape(url, **request_kwargs)
            raw = self._response_as_dict(result)
        except Exception as exc:  # SDK transport and provider exceptions are recorded identically.
            self.key_pool.record_result(pool_metadata["key_id"], success=False, error=str(exc))
            raise RuntimeError(f"Firecrawl scrape failed for {url}: {exc}") from exc
        response_metadata = self._response_metadata(raw)
        credits_used = self._non_negative_int(
            response_metadata.get("creditsUsed", response_metadata.get("credits_used"))
        ) or 0
        status_code = self._non_negative_int(
            response_metadata.get("statusCode", response_metadata.get("status_code"))
        )
        print(
            f"[firecrawl] scrape_done url={url} markdown_chars={self._markdown_length(raw)} "
            f"credits_used={credits_used} status_code={status_code}",
            file=sys.stderr,
            flush=True,
        )
        if status_code is not None and status_code != 200:
            error = self._status_code_error(raw, response_metadata, status_code)
            self.key_pool.record_result(
                pool_metadata["key_id"],
                success=False,
                error=error,
                credits_used=credits_used,
                status_code=status_code,
            )
            return {"error": error}
        if raw.get("success") is True or (
            "success" not in raw and self._is_direct_success_response(raw)
        ):
            self.key_pool.record_result(
                pool_metadata["key_id"],
                success=True,
                credits_used=credits_used,
                status_code=status_code,
            )
            return raw
        error = str(raw.get("error") or "Firecrawl returned an unsuccessful response")
        self.key_pool.record_result(
            pool_metadata["key_id"],
            success=False,
            error=error,
            credits_used=credits_used,
            status_code=status_code,
        )
        return raw

    def _app(self, api_key: str) -> Any:
        if self._app_factory is not None:
            return self._app_factory(api_key=api_key)
        try:
            from firecrawl import Firecrawl
        except ImportError as exc:  # Keep the backend importable without the optional SDK.
            raise RuntimeError(
                "Firecrawl SDK import failed. Install the firecrawl package in the Python "
                f"environment running this process. Original ImportError: {exc}"
            ) from exc
        app = Firecrawl(api_key=api_key)
        if not callable(getattr(app, "scrape", None)):
            raise RuntimeError(
                "Installed Firecrawl SDK is incompatible: Firecrawl(api_key=...) has no callable "
                ".scrape method. Upgrade the firecrawl package."
            )
        return app

    @staticmethod
    def _response_as_dict(response: Any) -> dict[str, Any]:
        normalized = FirecrawlClient._normalize_response_value(response)
        if not isinstance(normalized, dict):
            raise TypeError(f"Unexpected Firecrawl response type: {type(response).__name__}")
        return normalized

    @staticmethod
    def _normalize_response_value(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: FirecrawlClient._normalize_response_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [FirecrawlClient._normalize_response_value(item) for item in value]
        if hasattr(value, "model_dump"):
            return FirecrawlClient._normalize_response_value(value.model_dump())
        if hasattr(value, "dict"):
            return FirecrawlClient._normalize_response_value(value.dict())
        return value

    @staticmethod
    def _response_metadata(response: dict[str, Any]) -> dict[str, Any]:
        payload = FirecrawlClient._content_payload(response)
        if isinstance(payload.get("metadata"), dict):
            return payload["metadata"]
        metadata = response.get("metadata")
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _content_payload(response: dict[str, Any]) -> dict[str, Any]:
        data = response.get("data")
        return data if isinstance(data, dict) else response

    @staticmethod
    def _markdown_length(response: dict[str, Any]) -> int:
        markdown = FirecrawlClient._content_payload(response).get("markdown")
        return len(markdown) if isinstance(markdown, str) else 0

    @staticmethod
    def _is_direct_success_response(response: dict[str, Any]) -> bool:
        """Support SDK versions that return the scrape document without `success/data`."""
        return "error" not in response and isinstance(response.get("metadata"), dict)

    @staticmethod
    def _non_negative_int(value: Any) -> int | None:
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _status_code_error(
        response: dict[str, Any], metadata: dict[str, Any], status_code: int,
    ) -> str:
        provider_error = metadata.get("error") or response.get("error")
        if provider_error:
            return f"Firecrawl scrape returned statusCode {status_code}: {provider_error}"
        source_url = metadata.get("sourceURL") or metadata.get("url")
        suffix = f" for {source_url}" if source_url else ""
        return f"Firecrawl scrape returned statusCode {status_code}{suffix}."


def acquire_firecrawl_api_key() -> tuple[str, dict[str, Any]]:
    """Acquire one key from the process-wide Firecrawl pool."""
    return FirecrawlApiKeyPool.from_fixed_pool().acquire_key()
