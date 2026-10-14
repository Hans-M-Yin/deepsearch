"""Publish independently judged base difficulty cases into the teacher queue."""

from __future__ import annotations

from pathlib import Path
import json
import re
from typing import Any, Iterable, Mapping

from .mining import mine_base_hard_cases
from .schema import CanonicalCase
from .task_queue import SqliteTaskQueue


class BaseTrajectoryCostIndex:
    """Incrementally index immutable ``run_infer`` trajectory cost metadata.

    A follow-mode router polls frequently.  Reading every full trajectory JSON
    on each poll would be expensive, especially because source rows can embed
    image bytes.  This index scans names every time but reads only new or
    modified trajectory files.
    """

    def __init__(self, trajectory_dir: str | Path):
        self.trajectory_dir = Path(trajectory_dir)
        self._seen_mtimes: dict[Path, int] = {}
        self.tool_call_counts: dict[str, int] = {}

    _CASE_ID_RE = re.compile(r'"case_id"\s*:\s*"((?:\\.|[^"\\])*)"')
    _TOOL_CALL_COUNT_RE = re.compile(r'"tool_call_count"\s*:\s*(-?\d+)')

    @classmethod
    def _read_cost_metadata(cls, path: Path) -> tuple[str, int]:
        """Read only the two top-level fields needed for routing.

        ``run_infer`` records embed source image bytes.  A full ``json.loads``
        would allocate the entire record solely to recover an integer at its
        tail, so scan the file in bounded chunks instead.
        """

        case_id: str | None = None
        tool_call_count: int | None = None
        tail = ""
        with path.open("r", encoding="utf-8") as handle:
            while True:
                chunk = handle.read(64 * 1024)
                if not chunk:
                    break
                text = tail + chunk
                if case_id is None:
                    match = cls._CASE_ID_RE.search(text)
                    if match:
                        case_id = str(json.loads(f'"{match.group(1)}"')).strip()
                if tool_call_count is None:
                    match = cls._TOOL_CALL_COUNT_RE.search(text)
                    if match:
                        tool_call_count = int(match.group(1))
                if case_id is not None and tool_call_count is not None:
                    break
                tail = text[-512:]
        if not case_id or tool_call_count is None or tool_call_count < 0:
            raise ValueError("missing valid case_id or tool_call_count")
        return case_id, tool_call_count

    def refresh(self) -> dict[str, int]:
        if not self.trajectory_dir.is_dir():
            raise FileNotFoundError(f"Base trajectory directory does not exist: {self.trajectory_dir}")
        counts = {"discovered": 0, "indexed": 0, "malformed": 0}
        for path in self.trajectory_dir.glob("*_trajectory.json"):
            try:
                mtime_ns = path.stat().st_mtime_ns
            except FileNotFoundError:
                continue
            if self._seen_mtimes.get(path) == mtime_ns:
                continue
            counts["discovered"] += 1
            try:
                case_id, tool_call_count = self._read_cost_metadata(path)
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                # Do not mark an incomplete atomic write as seen: a later poll
                # must be able to retry it.
                counts["malformed"] += 1
                continue
            self._seen_mtimes[path] = mtime_ns
            self.tool_call_counts[case_id] = tool_call_count
            counts["indexed"] += 1
        return counts


def publish_base_failures(
    cases: Iterable[CanonicalCase],
    evaluation_rows: Iterable[dict[str, Any]],
    *,
    queue: SqliteTaskQueue,
    queue_name: str = "teacher_collect",
    max_attempts: int = 3,
    max_cases: int | None = None,
    max_ready_tasks: int = 0,
    base_tool_call_counts: Mapping[str, int] | None = None,
    min_correct_tool_calls: int = 0,
) -> dict[str, int]:
    """Idempotently enqueue wrong plus correct-but-expensive base cases.

    The queue payload is the PublicCase view. Gold answers and evaluator text
    remain outside the teacher task, as do base answers and cost metadata.
    """

    # Read the full observed evaluator prefix.  Applying the cap inside the
    # miner makes it a per-poll cap: if an evaluator append/rewrite changes
    # ordering between polls, newly earlier failures can leak past the desired
    # total.  The cap below is relative to durable queue state instead.
    selected, _manifest, mining_counts = mine_base_hard_cases(
        cases,
        evaluation_rows,
        base_tool_call_counts=base_tool_call_counts,
        min_correct_tool_calls=min_correct_tool_calls,
    )
    if max_ready_tasks < 0:
        raise ValueError("max_ready_tasks must be nonnegative")
    if max_cases is not None and max_cases < 0:
        raise ValueError("max_cases must be nonnegative or None")
    queue_stats = queue.stats(queue_name)
    ready = int(queue_stats.get("pending", 0)) + int(queue_stats.get("leased", 0))
    total = int(queue_stats.get("total", 0))
    published = 0
    already_present = 0
    backpressure_stopped = 0
    hard_cap_stopped = 0
    for case in selected:
        if max_cases is not None and total >= max_cases:
            hard_cap_stopped += 1
            continue
        if max_ready_tasks and ready >= max_ready_tasks:
            backpressure_stopped += 1
            continue
        inserted = queue.enqueue(
            queue_name,
            case.case_id,
            case.public_case().to_dict(),
            max_attempts=max_attempts,
        )
        if inserted:
            published += 1
            ready += 1
            total += 1
        else:
            already_present += 1
    return {
        **mining_counts,
        "published": published,
        "already_present": already_present,
        "backpressure_stopped": backpressure_stopped,
        "hard_cap_stopped": hard_cap_stopped,
    }
