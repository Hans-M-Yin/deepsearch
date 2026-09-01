"""Process-local retry/backoff monitoring for concurrent inference runs.

The monitor is deliberately small and dependency-free.  ``run_infer`` installs
one instance for the duration of a batch, while lower-level model/tool modules
call :func:`tracked_sleep`.  When no monitor is installed, the helper behaves
like the caller's original ``time.sleep`` function, so standalone synthesis
jobs keep their existing behaviour.
"""

from __future__ import annotations

from collections import Counter
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import itertools
from pathlib import Path
import re
import sys
import threading
import time
from typing import Iterator, Any, Callable


@dataclass(frozen=True, slots=True)
class RetryContext:
    """Identity of the benchmark sample executing on the current thread."""

    case_id: str
    case_idx: int


@dataclass(frozen=True, slots=True)
class _ActiveWait:
    token: int
    case_id: str
    case_idx: int
    kind: str
    reason: str
    tool: str
    requested_s: float
    started_at: float
    url: str
    error_type: str
    error: str


_CASE_CONTEXT: ContextVar[RetryContext | None] = ContextVar(
    "retry_monitor_case_context", default=None
)
_ACTIVE_MONITOR: "RetryMonitor | None" = None
_ACTIVE_MONITOR_LOCK = threading.Lock()


def set_case_context(case_id: object, case_idx: int) -> Any:
    """Set the current sample identity and return a token for resetting it."""

    return _CASE_CONTEXT.set(RetryContext(str(case_id), int(case_idx)))


def reset_case_context(token: Any) -> None:
    _CASE_CONTEXT.reset(token)


@contextmanager
def case_context(case_id: object, case_idx: int) -> Iterator[None]:
    token = set_case_context(case_id, case_idx)
    try:
        yield
    finally:
        reset_case_context(token)


def install_monitor(monitor: "RetryMonitor | None") -> None:
    """Install the process-local monitor used by :func:`tracked_sleep`."""

    global _ACTIVE_MONITOR
    with _ACTIVE_MONITOR_LOCK:
        _ACTIVE_MONITOR = monitor


def get_monitor() -> "RetryMonitor | None":
    with _ACTIVE_MONITOR_LOCK:
        return _ACTIVE_MONITOR


def retry_reason_from_exception(exc: BaseException, *, default: str) -> str:
    """Return a compact, stable reason label for retry diagnostics."""

    message = str(exc or "").lower()
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None)
    if status is None:
        # urllib.error.HTTPError exposes the response code as ``code`` rather
        # than ``status_code``.  Enhanced Reader uses urllib in its client.
        status = getattr(exc, "code", None)
    if status is None and response is not None:
        status = getattr(response, "status_code", None)
    if status is None:
        status_match = re.search(
            r"\b(?:http(?:\s+error)?|status(?:_code)?)\s*[:=]?\s*(\d{3})\b",
            message,
        )
        if status_match:
            status = status_match.group(1)
    try:
        status = int(status) if status is not None else None
    except (TypeError, ValueError):
        status = None

    if "name or service not known" in message or "temporary failure in name resolution" in message:
        return "dns_resolution"
    if status == 429 or "429" in message or "too many requests" in message:
        return "http_429"
    if status is not None and 500 <= status <= 599:
        return f"http_{status}"
    if "timeout" in message or "timed out" in message:
        return "timeout"
    if status is not None and 400 <= status <= 499:
        return f"http_{status}"
    return default


def _error_details(error: object | None) -> str:
    """Return an uncropped error description, including an HTTP response body."""

    if error is None:
        return ""
    if isinstance(error, BaseException):
        details = str(error) or repr(error)
        response = getattr(error, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            try:
                response_body = str(getattr(response, "text", "") or "")
            except Exception as response_exc:  # pragma: no cover - defensive
                response_body = f"<unable to read response body: {response_exc!r}>"
            if response_body:
                details += (
                    f"\nresponse_status_code={status_code!r}"
                    f"\nresponse_body:\n{response_body}"
                )
        return details
    return str(error)


def tracked_sleep(
    seconds: float,
    *,
    reason: str,
    tool: str,
    error: object | None = None,
    url: object | None = None,
    error_type: str | None = None,
    kind: str = "retry",
    fallback_sleep: Callable[[float], None] | None = None,
) -> None:
    """Sleep while registering the wait with the active monitor, if any."""

    duration = max(0.0, float(seconds))
    monitor = get_monitor()
    if monitor is None:
        (fallback_sleep or time.sleep)(duration)
        return
    monitor.sleep(
        duration,
        reason=reason,
        tool=tool,
        error=error,
        url=url,
        error_type=error_type,
        kind=kind,
    )


class RetryMonitor:
    """Thread-safe process-local monitor for retry and resource waits."""

    def __init__(
        self,
        *,
        interval_s: float = 10.0,
        error_log_path: str | Path | None = None,
    ) -> None:
        self.interval_s = max(0.5, float(interval_s))
        self.error_log_path = Path(error_log_path) if error_log_path else None
        self._error_log_lock = threading.Lock()
        self._error_log_failed = False
        self._lock = threading.Lock()
        self._active: dict[int, _ActiveWait] = {}
        self._next_token = itertools.count(1)
        self._total_events = 0
        self._total_retry_events = 0
        self._total_resource_wait_events = 0
        self._total_sleep_s = 0.0
        self._total_retry_sleep_s = 0.0
        self._total_resource_wait_s = 0.0
        self._totals_by_reason: Counter[str] = Counter()
        self._totals_by_tool: Counter[str] = Counter()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._report_loop,
            name="retry-monitor",
            daemon=True,
        )
        self._thread.start()
        print(
            f"[retry-status] event=monitor_started interval_s={self.interval_s:g}",
            file=sys.stderr,
            flush=True,
        )

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(1.0, self.interval_s + 0.5))
        self._emit("final")
        self._thread = None

    def sleep(
        self,
        seconds: float,
        *,
        reason: str,
        tool: str,
        error: object | None = None,
        url: object | None = None,
        error_type: str | None = None,
        kind: str = "retry",
    ) -> None:
        duration = max(0.0, float(seconds))
        if duration <= 0.0:
            return
        context = _CASE_CONTEXT.get()
        record = _ActiveWait(
            token=next(self._next_token),
            case_id=context.case_id if context is not None else "-",
            case_idx=context.case_idx if context is not None else -1,
            kind=str(kind or "retry"),
            reason=str(reason or "unknown"),
            tool=str(tool or "unknown"),
            requested_s=duration,
            started_at=time.monotonic(),
            url=str(url or ""),
            error_type=str(error_type or (type(error).__name__ if error is not None else "")),
            error=_error_details(error),
        )
        with self._lock:
            self._active[record.token] = record
            self._total_events += 1
            if record.kind == "retry":
                self._total_retry_events += 1
            else:
                self._total_resource_wait_events += 1
            self._totals_by_reason[record.reason] += 1
            self._totals_by_tool[record.tool] += 1
        if record.kind == "retry":
            self._write_error_record(record)
        try:
            time.sleep(duration)
        finally:
            elapsed = max(0.0, time.monotonic() - record.started_at)
            with self._lock:
                self._active.pop(record.token, None)
                self._total_sleep_s += elapsed
                if record.kind == "retry":
                    self._total_retry_sleep_s += elapsed
                else:
                    self._total_resource_wait_s += elapsed

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            active = list(self._active.values())
            return {
                "active": active,
                "total_events": self._total_events,
                "total_retry_events": self._total_retry_events,
                "total_resource_wait_events": self._total_resource_wait_events,
                "total_sleep_s": self._total_sleep_s,
                "total_retry_sleep_s": self._total_retry_sleep_s,
                "total_resource_wait_s": self._total_resource_wait_s,
                "totals_by_reason": dict(self._totals_by_reason),
                "totals_by_tool": dict(self._totals_by_tool),
            }

    def _report_loop(self) -> None:
        while not self._stop_event.wait(self.interval_s):
            self._emit("periodic")

    def _emit(self, event: str) -> None:
        snapshot = self.snapshot()
        active = snapshot["active"]
        active_retry = [item for item in active if item.kind == "retry"]
        active_wait = [item for item in active if item.kind != "retry"]
        active_reasons = Counter(item.reason for item in active_retry)
        active_tools = Counter(item.tool for item in active_retry)
        active_cases = sorted({item.case_id for item in active})
        details = {
            "event": event,
            "active_sleep": len(active_retry),
            "active_resource_wait": len(active_wait),
            "active_cases": len(active_cases),
            "active_reasons": dict(active_reasons),
            "active_tools": dict(active_tools),
            "total_wait_events": snapshot["total_events"],
            "total_retry_events": snapshot["total_retry_events"],
            "total_resource_wait_events": snapshot["total_resource_wait_events"],
            "total_sleep_s": round(snapshot["total_sleep_s"], 3),
            "total_retry_sleep_s": round(snapshot["total_retry_sleep_s"], 3),
            "total_resource_wait_s": round(snapshot["total_resource_wait_s"], 3),
            "total_by_reason": snapshot["totals_by_reason"],
        }
        if active_cases:
            details["active_case_ids"] = active_cases[:20]
        suffix = " ".join(f"{key}={value!r}" for key, value in details.items())
        print(f"[retry-status] {suffix}", file=sys.stderr, flush=True)

    def _write_error_record(self, record: _ActiveWait) -> None:
        """Append the complete retry error without affecting inference on I/O failure."""

        if self.error_log_path is None:
            return
        try:
            self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._error_log_lock:
                with self.error_log_path.open("a", encoding="utf-8") as handle:
                    handle.write("===== retry_error =====\n")
                    handle.write(f"timestamp_utc={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n")
                    handle.write(f"case_id={record.case_id!r}\n")
                    handle.write(f"case_idx={record.case_idx}\n")
                    handle.write(f"kind={record.kind!r}\n")
                    handle.write(f"tool={record.tool!r}\n")
                    handle.write(f"reason={record.reason!r}\n")
                    handle.write(f"sleep_seconds={record.requested_s!r}\n")
                    handle.write(f"url={record.url!r}\n")
                    handle.write(f"error_type={record.error_type!r}\n")
                    handle.write("error_details_begin\n")
                    handle.write(record.error)
                    if not record.error.endswith("\n"):
                        handle.write("\n")
                    handle.write("error_details_end\n\n")
        except OSError as exc:
            if not self._error_log_failed:
                self._error_log_failed = True
                print(
                    f"[retry-status] error_log_write_failed path={str(self.error_log_path)!r} "
                    f"error={str(exc)!r}",
                    file=sys.stderr,
                    flush=True,
                )
