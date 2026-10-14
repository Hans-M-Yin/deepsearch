"""Wall-clock telemetry; nested stage durations are inclusive, not additive."""

from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
from time import perf_counter, monotonic
from uuid import uuid4
from threading import RLock, Thread
from queue import Queue, Empty
from pathlib import Path
import os

_active = ContextVar("fix_timing", default=None)
_writers_lock = RLock()
_writers = {}
_STOP = object()


class AsyncLogWriter:
    """One persistent file handle; producers never perform filesystem I/O."""

    def __init__(self, path, batch_size=128, flush_interval=1.0):
        self.path = Path(path)
        self.queue = Queue()
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.error = None
        self.closed = False
        self.lock = RLock()
        self.thread = Thread(target=self._run, name="i2i-timing-writer", daemon=True)
        self.thread.start()

    def emit(self, event):
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with self.lock:
            self.check_error()
            if self.closed:
                raise RuntimeError("Timing writer is closed")
            self.queue.put_nowait(line)

    def check_error(self):
        if self.error is not None:
            raise RuntimeError(f"Timing log write failed: {self.path}") from self.error

    def _run(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as handle:
                pending = []
                last_flush = monotonic()
                while True:
                    try:
                        item = self.queue.get(
                            timeout=max(
                                0, self.flush_interval - (monotonic() - last_flush)
                            )
                        )
                    except Empty:
                        item = None
                    if item is not None and item is not _STOP:
                        pending.append(item)
                    if (
                        item is None
                        or item is _STOP
                        or len(pending) >= self.batch_size
                        or monotonic() - last_flush >= self.flush_interval
                    ):
                        if pending:
                            handle.write("".join(pending))
                            handle.flush()
                            pending.clear()
                        last_flush = monotonic()
                    if item is _STOP:
                        break
        except BaseException as exc:
            self.error = exc

    def close(self, timeout=30):
        with self.lock:
            if not self.closed:
                self.closed = True
                self.queue.put_nowait(_STOP)
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise RuntimeError(
                "Timing log drain timed out; queued events may not be persisted"
            )
        self.check_error()


def writer_for(path):
    key = os.path.abspath(path)
    with _writers_lock:
        if key not in _writers:
            _writers[key] = AsyncLogWriter(path)
        return _writers[key]


def close_writer(path):
    if path is None:
        return
    key = os.path.abspath(path)
    with _writers_lock:
        writer = _writers.get(key)
    if writer is not None:
        try:
            writer.close()
        finally:
            if not writer.thread.is_alive():
                with _writers_lock:
                    if _writers.get(key) is writer:
                        del _writers[key]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def summarize(events, elapsed_s):
    stages = defaultdict(
        lambda: {
            "calls": 0,
            "elapsed_s": 0.0,
            "errors": 0,
            "cache_hits": 0,
            "cache_misses": 0,
        }
    )
    for event in events:
        row = stages[event["stage"]]
        row["calls"] += 1
        row["elapsed_s"] += event["elapsed_s"]
        row["errors"] += event["status"] != "ok"
        if "cache_hit" in event:
            row["cache_hits" if event["cache_hit"] else "cache_misses"] += 1
    for row in stages.values():
        row["elapsed_s"] = round(row["elapsed_s"], 6)
    return {
        "elapsed_s": round(elapsed_s, 6),
        "stages": dict(stages),
        "durations_are_inclusive": True,
    }


class Timer:
    def __init__(self, path=None, **context):
        self.path, self.context = path, context
        self.events = []
        self._lock = RLock()
        self.started_at = utc_now()
        self.started = perf_counter()

    def emit(self, event):
        if self.path is not None:
            writer_for(self.path).emit({**self.context, **event})

    def close(self):
        """Call after all producers sharing this log have finished."""
        close_writer(self.path)

    @contextmanager
    def activate(self):
        token = _active.set(self)
        try:
            yield self
        finally:
            _active.reset(token)

    @contextmanager
    def measure(self, stage, **metadata):
        identity, start = uuid4().hex, perf_counter()
        self.emit(
            {
                "event": "start",
                "event_id": identity,
                "stage": stage,
                "at": utc_now(),
                **metadata,
            }
        )
        status, error_type = "ok", None
        try:
            yield
        except BaseException as exc:
            status, error_type = "error", type(exc).__name__
            raise
        finally:
            event = {
                "event": "end",
                "event_id": identity,
                "stage": stage,
                "at": utc_now(),
                "elapsed_s": round(perf_counter() - start, 6),
                "status": status,
                **metadata,
            }
            if error_type:
                event["error_type"] = error_type
            with self._lock:
                self.events.append(event)
            self.emit(event)

    def summary(self):
        return {
            **summarize(self.events, perf_counter() - self.started),
            "started_at": self.started_at,
            "finished_at": utc_now(),
            **self.context,
        }


@contextmanager
def measure(stage, **metadata):
    timer = _active.get()
    if timer is None:
        yield
    else:
        with timer.measure(stage, **metadata):
            yield


def timed_call(stage, function, *args, **kwargs):
    with measure(stage):
        return function(*args, **kwargs)


def mark():
    timer = _active.get()
    return perf_counter(), len(timer.events) if timer else 0


def since(marker):
    start, offset = marker
    timer = _active.get()
    return summarize(timer.events[offset:] if timer else [], perf_counter() - start)
