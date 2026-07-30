"""Apply checkpointed image-text verification results to a graph.

The input JSONL is produced by ``verify_image_text_edges.py --results-jsonl``.
For each edge ID, the latest valid JSONL record is written to the graph edge's
``metadata.post_verify_image_text``. Contradicting edges are soft-deleted by
default, so they remain restorable.

Typical apply: write all verdicts into ``edges.jsonl`` and soft-delete only
``contradict`` edges. This is reversible with the restore commands in
``verify_image_text_edges.py``:

  python synthesis/post_process/apply_image_edge_verification_results.py \
    --graph-dir runs/my_graph \
    --results-jsonl synthesis/ignore/verify_full.results.jsonl \
    --drop-on contradict \
    --pretty

Preview without changing the graph:

  python synthesis/post_process/apply_image_edge_verification_results.py \
    --graph-dir runs/my_graph \
    --results-jsonl synthesis/ignore/verify_full.results.jsonl \
    --drop-on contradict \
    --dry-run \
    --pretty

Write verification metadata but retain every active edge:

  python synthesis/post_process/apply_image_edge_verification_results.py \
    --graph-dir runs/my_graph \
    --results-jsonl synthesis/ignore/verify_full.results.jsonl \
    --drop-on never \
    --pretty

``--hard-delete`` permanently removes matching edges and is not recommended.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.post_process.verify_image_text_edges import (
    _delete_edge,
    _soft_delete_verified_edge,
    _should_drop,
)
from synthesis.store import JsonlGraphStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True, help="Directory containing graph JSONL tables.")
    parser.add_argument("--results-jsonl", type=Path, required=True, help="Checkpoint JSONL written by verify_image_text_edges.py.")
    parser.add_argument(
        "--drop-on",
        default="contradict",
        choices=["contradict", "contradict_or_insufficient", "never"],
        help="Which verdicts should soft-delete their graph edge. Default: contradict.",
    )
    parser.add_argument(
        "--hard-delete",
        action="store_true",
        help="Physically delete matching edges instead of the default reversible soft delete.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview mutations without changing edges.jsonl.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print the result JSON.")
    return parser.parse_args()


def _load_latest_records(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Read a checkpoint, retaining the final valid record for every edge."""
    records: dict[str, dict[str, Any]] = {}
    stats: Counter[str] = Counter()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                stats["blank_lines"] += 1
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats["invalid_json_lines"] += 1
                print(
                    f"[apply_image_edge_verification_results] ignoring invalid JSON on line {line_number}: {path}",
                    file=sys.stderr,
                )
                continue
            if not isinstance(record, dict):
                stats["non_object_lines"] += 1
                continue
            edge_id = str(record.get("edge_id") or "").strip()
            if not edge_id:
                stats["missing_edge_id_lines"] += 1
                continue
            if edge_id in records:
                stats["superseded_records"] += 1
            records[edge_id] = record
            stats["valid_records"] += 1
    stats["unique_edge_records"] = len(records)
    return records, dict(stats)


def _graph_verification_payload(record: dict[str, Any]) -> dict[str, Any]:
    """Keep the graph-level verification metadata compact and stable."""
    return {
        "decision": record.get("decision"),
        "error_type": record.get("error_type"),
        "confidence": record.get("confidence"),
        "reason": record.get("reason"),
        "evidence_for": record.get("evidence_for") or [],
        "evidence_against": record.get("evidence_against") or [],
        "judge_model_alias": record.get("judge_model_alias"),
        "prepare_model_alias": record.get("prepare_model_alias"),
        "kept_reference_image_count": record.get("kept_reference_image_count"),
        "verified_at_unix": record.get("verified_at_unix"),
        "applied_from_results_jsonl": True,
    }


def apply_results(
    *,
    graph_dir: Path,
    results_jsonl: Path,
    drop_on: str,
    hard_delete: bool,
    dry_run: bool,
) -> dict[str, Any]:
    graph_dir = graph_dir.expanduser().resolve()
    results_jsonl = results_jsonl.expanduser().resolve()
    if not results_jsonl.exists():
        raise FileNotFoundError(f"results JSONL does not exist: {results_jsonl}")

    records, input_stats = _load_latest_records(results_jsonl)
    store = JsonlGraphStore(graph_dir)
    counters: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    affected_edges: list[dict[str, Any]] = []

    for edge_id, record in records.items():
        decision = str(record.get("decision") or "insufficient")
        decisions[decision] += 1
        edge = store.get_edge(edge_id)
        if edge is None:
            counters["missing_graph_edge"] += 1
            affected_edges.append({"edge_id": edge_id, "status": "missing_graph_edge"})
            continue
        if str(edge.get("status") or "active").lower() != "active":
            counters["skipped_inactive_graph_edge"] += 1
            affected_edges.append(
                {
                    "edge_id": edge_id,
                    "status": "skipped_inactive_graph_edge",
                    "edge_status": edge.get("status"),
                }
            )
            continue

        metadata = dict(edge.get("metadata") or {})
        metadata["post_verify_image_text"] = _graph_verification_payload(record)
        edge["metadata"] = metadata
        should_drop = _should_drop(decision, drop_on)

        if dry_run:
            counters["would_write_verification"] += 1
            if should_drop:
                counters["would_hard_delete" if hard_delete else "would_soft_delete"] += 1
            affected_edges.append(
                {
                    "edge_id": edge_id,
                    "status": "would_hard_delete" if should_drop and hard_delete else (
                        "would_soft_delete" if should_drop else "would_write_verification"
                    ),
                    "decision": decision,
                }
            )
            continue

        if should_drop:
            if hard_delete:
                _delete_edge(store, edge_id)
                counters["hard_deleted"] += 1
                status = "hard_deleted"
            else:
                store.upsert_edge(_soft_delete_verified_edge(edge, decision=decision))
                counters["soft_deleted"] += 1
                status = "soft_deleted"
        else:
            store.upsert_edge(edge)
            counters["verification_written"] += 1
            status = "verification_written"
        affected_edges.append({"edge_id": edge_id, "status": status, "decision": decision})

    if not dry_run and store.has_pending_writes():
        store.flush()

    return {
        "graph_dir": str(graph_dir),
        "results_jsonl": str(results_jsonl),
        "dry_run": dry_run,
        "drop_on": drop_on,
        "drop_mode": "hard_delete" if hard_delete else "soft_delete",
        "input_stats": input_stats,
        "decision_counts": dict(decisions),
        "apply_counts": dict(counters),
        "affected_edges": affected_edges,
    }


def main() -> int:
    args = parse_args()
    try:
        payload = apply_results(
            graph_dir=args.graph_dir,
            results_jsonl=args.results_jsonl,
            drop_on=args.drop_on,
            hard_delete=bool(args.hard_delete),
            dry_run=bool(args.dry_run),
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
