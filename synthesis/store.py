"""Lightweight JSONL-backed graph store for synthesis development."""

from __future__ import annotations

import json
import os
from pathlib import Path
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

    TABLE_KEYS = {
        "nodes": "node_id",
        "edges": "edge_id",
        "assets": "asset_id",
        "evidence": "evidence_id",
        "search_snapshots": "snapshot_id",
    }

    def __init__(self, root_dir: str | Path, *, auto_flush: bool = False) -> None:
        self.root_dir = Path(root_dir)
        self.auto_flush = auto_flush
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self._tables: dict[str, dict[str, JsonRecord]] = {
            table: {} for table in self.TABLE_FILES
        }
        self._dirty: set[str] = set()
        self._lock = RLock()
        self.load()

    def load(self) -> None:
        with self._lock:
            for table, file_name in self.TABLE_FILES.items():
                self._tables[table] = self._read_table(table, self.root_dir / file_name)
            self._dirty.clear()

    def flush(self) -> None:
        with self._lock:
            for table in list(self._dirty):
                self._write_table(table, self.root_dir / self.TABLE_FILES[table])
            self._dirty.clear()

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
        return self.find_edges(lambda edge: edge.get("src_node_id") == node_id)

    def edges_to(self, node_id: str) -> list[JsonRecord]:
        return self.find_edges(lambda edge: edge.get("dst_node_id") == node_id)

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

            self._tables[table][record_id] = record
            self._dirty.add(table)
            if self.auto_flush:
                self.flush()
            return dict(record)

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

    def _write_table(self, table: str, path: Path) -> None:
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        records = self._tables[table]
        with tmp_path.open("w", encoding="utf-8") as handle:
            for record_id in sorted(records):
                json.dump(records[record_id], handle, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
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
