"""Lightweight JSONL-backed graph store for synthesis development."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time
import sys
from threading import RLock
from typing import Any, Callable, Iterable

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .edges import Edge
from .evidence import Asset, Evidence, SearchSnapshot
from .nodes import Node


JsonRecord = dict[str, Any]


class JsonlGraphStore:
    """Small JSONL graph store with in-memory indexes.

    This store is intended for early pipeline development. It keeps graph
    records as JSONL tables on disk, loads them into memory at startup, and
    rewrites touched tables atomically on flush.
    """

    TABLE_FILES = {
        "nodes": "nodes.jsonl",
        "edges": "edges.jsonl",
        "assets": "assets.jsonl",
        "evidence": "evidence.jsonl",
        "search_snapshots": "search_snapshots.jsonl",
    }

    DELTA_TABLE_FILES = {
        table: f"{file_name[:-6]}.delta.jsonl"
        for table, file_name in TABLE_FILES.items()
    }

    TABLE_KEYS = {
        "nodes": "node_id",
        "edges": "edge_id",
        "assets": "asset_id",
        "evidence": "evidence_id",
        "search_snapshots": "snapshot_id",
    }

    def __init__(
        self,
        root_dir: str | Path,
        *,
        auto_flush: bool = False,
        flush_record_threshold: int = 1,
        flush_interval_s: float = 0.0,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.auto_flush = auto_flush
        self.flush_record_threshold = max(1, int(flush_record_threshold))
        self.flush_interval_s = max(0.0, float(flush_interval_s))
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self._tables: dict[str, dict[str, JsonRecord]] = {
            table: {} for table in self.TABLE_FILES
        }
        self._pending_records: dict[str, list[JsonRecord]] = {
            table: [] for table in self.TABLE_FILES
        }
        self._delta_record_counts: dict[str, int] = {
            table: 0 for table in self.TABLE_FILES
        }
        self._dirty: set[str] = set()
        self._lock = RLock()
        self._pending_write_count = 0
        self._last_flush_monotonic = time.monotonic()
        # Incremented whenever one or more pending records are durably
        # appended to delta JSONL files. Runners use this to notice flushes
        # that happened inside builders and checkpoint their queue state.
        self._flush_generation = 0

        # These are derived indexes.  The JSONL tables remain the source of
        # truth and all of these structures can be rebuilt by load().  Lists
        # are used instead of sets so lookup results preserve the insertion
        # order of the original JSONL table.
        self._record_order: dict[str, dict[str, int]] = {
            table: {} for table in self.TABLE_FILES
        }
        self._next_record_order: dict[str, int] = {
            table: 0 for table in self.TABLE_FILES
        }
        self._node_ids_by_source_url: dict[str, list[str]] = {}
        self._evidence_ids_by_node_id: dict[str, list[str]] = {}
        self._evidence_ids_by_url: dict[str, list[str]] = {}
        self._edge_ids_by_src_node_id: dict[str, list[str]] = {}
        self._edge_ids_by_dst_node_id: dict[str, list[str]] = {}
        self._node_type_counts: dict[str, int] = {}
        self._latest_node_id: str | None = None
        self._latest_node_created_at = ""
        self.load()

    def load(self) -> None:
        with self._lock:
            for table, file_name in self.TABLE_FILES.items():
                self._tables[table] = self._read_table(table, self.root_dir / file_name)
                delta_path = self.root_dir / self.DELTA_TABLE_FILES[table]
                self._delta_record_counts[table] = self._apply_delta_table(
                    table,
                    self._tables[table],
                    delta_path,
                )
            self._rebuild_indexes_locked()
            self._pending_records = {table: [] for table in self.TABLE_FILES}
            self._dirty.clear()
            self._pending_write_count = 0
            self._last_flush_monotonic = time.monotonic()

    def flush(self) -> bool:
        return self.maybe_flush(force=True)

    def maybe_flush(self, *, force: bool = False) -> bool:
        with self._lock:
            if not self._dirty:
                return False
            if not force and not self._should_flush_locked():
                return False

            flushed = False
            for table in list(self._dirty):
                records = self._pending_records[table]
                if not records:
                    self._dirty.discard(table)
                    continue
                self._append_delta_table(
                    table,
                    self.root_dir / self.DELTA_TABLE_FILES[table],
                    records,
                )
                self._delta_record_counts[table] += len(records)
                self._pending_records[table] = []
                self._dirty.discard(table)
                self._pending_write_count -= len(records)
                flushed = True
            if flushed:
                self._flush_generation += 1
            self._last_flush_monotonic = time.monotonic()
            return flushed

    def has_pending_writes(self) -> bool:
        with self._lock:
            return bool(self._dirty)

    @property
    def flush_generation(self) -> int:
        """Return the number of successful delta flush generations."""

        with self._lock:
            return self._flush_generation

    def pending_write_count(self) -> int:
        with self._lock:
            return self._pending_write_count

    def delta_stats(self) -> dict[str, dict[str, int]]:
        """Return append-log record and byte counts for operational monitoring."""

        with self._lock:
            return {
                table: {
                    "records": self._delta_record_counts[table],
                    "bytes": self._delta_path_size_locked(table),
                }
                for table in self.TABLE_FILES
            }

    def compact(self, tables: Iterable[str] | None = None) -> bool:
        """Merge delta JSONL records into canonical tables atomically.

        Normal graph expansion never needs to call this on every flush.  It is
        intended for maintenance checkpoints or after a run completes.  The
        canonical table is replaced before its delta is removed, so a crash
        between those operations is recoverable: replaying already-compacted
        records is idempotent because records are keyed by ID.
        """

        self.flush()
        with self._lock:
            selected_tables = list(tables) if tables is not None else list(self.TABLE_FILES)
            unknown_tables = set(selected_tables) - set(self.TABLE_FILES)
            if unknown_tables:
                raise ValueError(f"Unknown graph tables for compact: {sorted(unknown_tables)}")

            compacted = False
            for table in selected_tables:
                delta_path = self.root_dir / self.DELTA_TABLE_FILES[table]
                if not delta_path.exists() or self._delta_record_counts[table] <= 0:
                    continue
                self._write_table(table, self.root_dir / self.TABLE_FILES[table])
                # Remove the delta only after the canonical snapshot is in
                # place.  If removal fails, a later load safely replays it.
                delta_path.unlink()
                self._delta_record_counts[table] = 0
                compacted = True
            return compacted

    def upsert_node(self, node: Node | JsonRecord) -> JsonRecord:
        return self._upsert("nodes", node)

    def upsert_edge(self, edge: Edge | JsonRecord) -> JsonRecord:
        return self._upsert("edges", edge)

    def upsert_asset(self, asset: Asset | JsonRecord) -> JsonRecord:
        return self._upsert("assets", asset)

    def upsert_evidence(self, evidence: Evidence | JsonRecord) -> JsonRecord:
        return self._upsert("evidence", evidence)

    def upsert_search_snapshot(self, snapshot: SearchSnapshot | JsonRecord) -> JsonRecord:
        return self._upsert("search_snapshots", snapshot)

    def get_node(self, node_id: str) -> JsonRecord | None:
        with self._lock:
            record = self._tables["nodes"].get(node_id)
            return dict(record) if record is not None else None

    def get_edge(self, edge_id: str) -> JsonRecord | None:
        with self._lock:
            record = self._tables["edges"].get(edge_id)
            return dict(record) if record is not None else None

    def get_asset(self, asset_id: str) -> JsonRecord | None:
        with self._lock:
            record = self._tables["assets"].get(asset_id)
            return dict(record) if record is not None else None

    def get_evidence(self, evidence_id: str) -> JsonRecord | None:
        with self._lock:
            record = self._tables["evidence"].get(evidence_id)
            return dict(record) if record is not None else None

    def get_search_snapshot(self, snapshot_id: str) -> JsonRecord | None:
        with self._lock:
            record = self._tables["search_snapshots"].get(snapshot_id)
            return dict(record) if record is not None else None

    def list_nodes(self) -> list[JsonRecord]:
        with self._lock:
            return [dict(record) for record in self._tables["nodes"].values()]

    def list_edges(self) -> list[JsonRecord]:
        with self._lock:
            return [dict(record) for record in self._tables["edges"].values()]

    def list_assets(self) -> list[JsonRecord]:
        with self._lock:
            return [dict(record) for record in self._tables["assets"].values()]

    def list_evidence(self) -> list[JsonRecord]:
        with self._lock:
            return [dict(record) for record in self._tables["evidence"].values()]

    def list_search_snapshots(self) -> list[JsonRecord]:
        with self._lock:
            return [dict(record) for record in self._tables["search_snapshots"].values()]

    def iter_nodes(self) -> Iterable[JsonRecord]:
        return iter(self.list_nodes())

    def iter_edges(self) -> Iterable[JsonRecord]:
        return iter(self.list_edges())

    def find_nodes(self, predicate: Callable[[JsonRecord], bool]) -> list[JsonRecord]:
        return [record for record in self.iter_nodes() if predicate(record)]

    def find_edges(self, predicate: Callable[[JsonRecord], bool]) -> list[JsonRecord]:
        return [record for record in self.iter_edges() if predicate(record)]

    def edges_from(self, node_id: str) -> list[JsonRecord]:
        with self._lock:
            edge_ids = self._edge_ids_by_src_node_id.get(node_id, [])
            return self._ordered_records_locked("edges", edge_ids)

    def edges_to(self, node_id: str) -> list[JsonRecord]:
        with self._lock:
            edge_ids = self._edge_ids_by_dst_node_id.get(node_id, [])
            return self._ordered_records_locked("edges", edge_ids)

    def find_nodes_by_source_url(
        self,
        url: str,
        *,
        node_type: str | None = None,
    ) -> list[JsonRecord]:
        """Return nodes with an exact source URL without scanning all nodes."""

        if not url:
            return []
        with self._lock:
            records = self._ordered_records_locked(
                "nodes",
                self._node_ids_by_source_url.get(url, []),
            )
            if node_type is None:
                return records
            return [record for record in records if record.get("node_type") == node_type]

    def find_nodes_by_source_urls(
        self,
        urls: Iterable[str],
        *,
        node_type: str | None = None,
    ) -> list[JsonRecord]:
        """Return URL-matched nodes in the original node-table order."""

        url_set = {url for url in urls if url}
        if not url_set:
            return []
        with self._lock:
            node_ids: list[str] = []
            for url in url_set:
                node_ids.extend(self._node_ids_by_source_url.get(url, []))
            records = self._ordered_records_locked("nodes", node_ids)
            if node_type is None:
                return records
            return [record for record in records if record.get("node_type") == node_type]

    def find_node_by_source_url(
        self,
        url: str,
        *,
        node_type: str | None = None,
    ) -> JsonRecord | None:
        records = self.find_nodes_by_source_url(url, node_type=node_type)
        return records[0] if records else None

    def find_evidence(
        self,
        *,
        node_id: str | None = None,
        url: str | None = None,
        evidence_type: str | None = None,
    ) -> list[JsonRecord]:
        """Find evidence using node and/or URL indexes.

        The node_id and URL predicates intentionally have OR semantics to
        match the historical WikiTextBuilder lookup behavior.
        """

        with self._lock:
            evidence_ids: list[str] = []
            seen_ids: set[str] = set()
            if node_id:
                for evidence_id in self._evidence_ids_by_node_id.get(node_id, []):
                    if evidence_id not in seen_ids:
                        seen_ids.add(evidence_id)
                        evidence_ids.append(evidence_id)
            if url:
                for evidence_id in self._evidence_ids_by_url.get(url, []):
                    if evidence_id not in seen_ids:
                        seen_ids.add(evidence_id)
                        evidence_ids.append(evidence_id)

            records = self._ordered_records_locked("evidence", evidence_ids)
            if evidence_type is None:
                return records
            return [
                record
                for record in records
                if record.get("evidence_type") == evidence_type
            ]

    def count_nodes(self, node_type: str | None = None) -> int:
        with self._lock:
            if node_type is None:
                return len(self._tables["nodes"])
            return self._node_type_counts.get(node_type, 0)

    def node_type_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._node_type_counts)

    def latest_node(self) -> JsonRecord | None:
        with self._lock:
            if self._latest_node_id is None:
                return None
            record = self._tables["nodes"].get(self._latest_node_id)
            return dict(record) if record is not None else None

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {table: len(records) for table, records in self._tables.items()}

    def _upsert(self, table: str, record_or_obj: Any) -> JsonRecord:
        with self._lock:
            record = self._to_record(record_or_obj)
            key_name = self.TABLE_KEYS[table]
            record_id = record.get(key_name)
            if not record_id:
                raise ValueError(f"Missing required key {key_name!r} for table {table!r}")

            old_record = self._tables[table].get(record_id)
            old_was_latest_node = (
                table == "nodes" and record_id == self._latest_node_id
            )
            if old_record is not None:
                self._deindex_record_locked(table, old_record)
            self._tables[table][record_id] = record
            if old_record is None:
                self._record_order[table][record_id] = self._next_record_order[table]
                self._next_record_order[table] += 1
            self._index_record_locked(table, record)
            if old_was_latest_node:
                self._recompute_latest_node_locked()
            self._dirty.add(table)
            self._pending_records[table].append(dict(record))
            self._pending_write_count += 1
            if self.auto_flush:
                self.flush()
            return dict(record)

    def _should_flush_locked(self) -> bool:
        if self._pending_write_count >= self.flush_record_threshold:
            return True
        if self.flush_interval_s > 0 and (time.monotonic() - self._last_flush_monotonic) >= self.flush_interval_s:
            return True
        return False

    @staticmethod
    def _to_record(record_or_obj: Any) -> JsonRecord:
        if isinstance(record_or_obj, dict):
            return dict(record_or_obj)
        if hasattr(record_or_obj, "to_dict"):
            return record_or_obj.to_dict()
        raise TypeError(f"Object is not JSON record-like: {type(record_or_obj)!r}")

    def _read_table(self, table: str, path: Path) -> dict[str, JsonRecord]:
        key_name = self.TABLE_KEYS[table]
        records: dict[str, JsonRecord] = {}
        if not path.exists():
            return records

        with path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                stripped = line.strip()
                if not stripped:
                    continue
                record = json.loads(stripped)
                record_id = record.get(key_name)
                if not record_id:
                    raise ValueError(f"{path}:{line_no} missing key {key_name!r}")
                records[record_id] = record
        return records

    def _apply_delta_table(
        self,
        table: str,
        records: dict[str, JsonRecord],
        path: Path,
    ) -> int:
        """Replay an append-only table delta and repair a partial final line."""

        if not path.exists():
            return 0

        key_name = self.TABLE_KEYS[table]
        applied_count = 0
        with path.open("rb") as handle:
            while True:
                line_start = handle.tell()
                line = handle.readline()
                if not line:
                    break
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    if line.endswith(b"\n"):
                        raise ValueError(f"Malformed JSON in delta table {path}")
                    # A process can die after writing only part of the final
                    # record.  Discard only that incomplete tail; all prior
                    # newline-terminated records remain valid.
                    with path.open("r+b") as repair_handle:
                        repair_handle.truncate(line_start)
                    break
                record_id = record.get(key_name)
                if not record_id:
                    raise ValueError(f"{path} missing key {key_name!r}")
                records[record_id] = record
                applied_count += 1
        return applied_count

    @staticmethod
    def _append_delta_table(
        table: str,
        path: Path,
        records: list[JsonRecord],
    ) -> None:
        del table
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("ab") as handle:
            for record in records:
                payload = json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                ).encode("utf-8")
                handle.write(payload)
                handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _delta_path_size_locked(self, table: str) -> int:
        path = self.root_dir / self.DELTA_TABLE_FILES[table]
        try:
            return path.stat().st_size
        except FileNotFoundError:
            return 0

    def _rebuild_indexes_locked(self) -> None:
        self._record_order = {table: {} for table in self.TABLE_FILES}
        self._next_record_order = {table: 0 for table in self.TABLE_FILES}
        self._node_ids_by_source_url = {}
        self._evidence_ids_by_node_id = {}
        self._evidence_ids_by_url = {}
        self._edge_ids_by_src_node_id = {}
        self._edge_ids_by_dst_node_id = {}
        self._node_type_counts = {}
        self._latest_node_id = None
        self._latest_node_created_at = ""

        for table, records in self._tables.items():
            for record_id, record in records.items():
                self._record_order[table][record_id] = self._next_record_order[table]
                self._next_record_order[table] += 1
                self._index_record_locked(table, record)

    def _index_record_locked(self, table: str, record: JsonRecord) -> None:
        record_id = record[self.TABLE_KEYS[table]]
        if table == "nodes":
            node_type = record.get("node_type")
            if node_type:
                self._node_type_counts[node_type] = self._node_type_counts.get(node_type, 0) + 1
            source = record.get("source")
            source_url = source.get("url") if isinstance(source, dict) else None
            if source_url:
                self._append_index_value(self._node_ids_by_source_url, source_url, record_id)
            created_at = str(record.get("created_at") or "")
            if (
                self._latest_node_id is None
                or created_at >= self._latest_node_created_at
            ):
                self._latest_node_id = record_id
                self._latest_node_created_at = created_at
            return

        if table == "evidence":
            for node_id in set(record.get("node_ids") or []):
                self._append_index_value(self._evidence_ids_by_node_id, node_id, record_id)
            url = record.get("url")
            if url:
                self._append_index_value(self._evidence_ids_by_url, url, record_id)
            return

        if table == "edges":
            src_node_id = record.get("src_node_id")
            dst_node_id = record.get("dst_node_id")
            if src_node_id:
                self._append_index_value(
                    self._edge_ids_by_src_node_id,
                    src_node_id,
                    record_id,
                )
            if dst_node_id:
                self._append_index_value(
                    self._edge_ids_by_dst_node_id,
                    dst_node_id,
                    record_id,
                )

    def _deindex_record_locked(self, table: str, record: JsonRecord) -> None:
        record_id = record[self.TABLE_KEYS[table]]
        if table == "nodes":
            node_type = record.get("node_type")
            if node_type:
                count = self._node_type_counts.get(node_type, 0) - 1
                if count > 0:
                    self._node_type_counts[node_type] = count
                else:
                    self._node_type_counts.pop(node_type, None)
            source = record.get("source")
            source_url = source.get("url") if isinstance(source, dict) else None
            if source_url:
                self._remove_index_value(self._node_ids_by_source_url, source_url, record_id)
            if record_id == self._latest_node_id:
                self._latest_node_id = None
                self._latest_node_created_at = ""
            return

        if table == "evidence":
            for node_id in set(record.get("node_ids") or []):
                self._remove_index_value(self._evidence_ids_by_node_id, node_id, record_id)
            url = record.get("url")
            if url:
                self._remove_index_value(self._evidence_ids_by_url, url, record_id)
            return

        if table == "edges":
            src_node_id = record.get("src_node_id")
            dst_node_id = record.get("dst_node_id")
            if src_node_id:
                self._remove_index_value(
                    self._edge_ids_by_src_node_id,
                    src_node_id,
                    record_id,
                )
            if dst_node_id:
                self._remove_index_value(
                    self._edge_ids_by_dst_node_id,
                    dst_node_id,
                    record_id,
                )

    @staticmethod
    def _append_index_value(
        index: dict[str, list[str]],
        key: str,
        value: str,
    ) -> None:
        # A record is indexed once per key during rebuild/upsert.  Evidence
        # node_ids are deduplicated by the caller, so avoid a linear
        # membership check here; high-degree adjacency lists can be large.
        index.setdefault(key, []).append(value)

    @staticmethod
    def _remove_index_value(
        index: dict[str, list[str]],
        key: str,
        value: str,
    ) -> None:
        values = index.get(key)
        if not values:
            return
        try:
            values.remove(value)
        except ValueError:
            return
        if not values:
            index.pop(key, None)

    def _ordered_records_locked(
        self,
        table: str,
        record_ids: Iterable[str],
    ) -> list[JsonRecord]:
        order = self._record_order[table]
        unique_ids = set(record_ids)
        ordered_ids = sorted(unique_ids, key=lambda record_id: order.get(record_id, 0))
        return [
            dict(self._tables[table][record_id])
            for record_id in ordered_ids
            if record_id in self._tables[table]
        ]

    def _recompute_latest_node_locked(self) -> None:
        self._latest_node_id = None
        self._latest_node_created_at = ""
        for record_id, record in self._tables["nodes"].items():
            created_at = str(record.get("created_at") or "")
            if (
                self._latest_node_id is None
                or created_at >= self._latest_node_created_at
            ):
                self._latest_node_id = record_id
                self._latest_node_created_at = created_at

    def _write_table(self, table: str, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        records = self._tables[table]
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record_id in sorted(records):
                json.dump(records[record_id], handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)


def _smoke_test() -> None:
    import tempfile

    from .edges import Edge, EdgeType
    from .evidence import Evidence, EvidenceType
    from .nodes import TextNode

    with tempfile.TemporaryDirectory() as tmpdir:
        store = JsonlGraphStore(tmpdir)
        node = TextNode.from_webpage("https://example.com/a", title="A")
        evidence = Evidence.create(EvidenceType.WEB_TEXT, content="hello", node_ids=[node.node_id])
        edge = Edge.create(node.node_id, node.node_id, edge_type=EdgeType.DERIVED, relation="self")
        store.upsert_node(node)
        store.upsert_evidence(evidence)
        store.upsert_edge(edge)
        store.flush()

        reloaded = JsonlGraphStore(tmpdir)
        assert reloaded.get_node(node.node_id)["title"] == "A"
        assert reloaded.get_evidence(evidence.evidence_id)["content"] == "hello"
        assert reloaded.stats()["edges"] == 1
    print("store smoke test passed")


if __name__ == "__main__":
    _smoke_test()
