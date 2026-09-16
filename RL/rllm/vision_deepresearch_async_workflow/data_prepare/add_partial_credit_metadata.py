#!/usr/bin/env python3
"""Attach compact statement-level partial-credit metadata to an RL parquet.

The input RL dataset contains source provenance in
``extra_info.source_run_dir`` and ``extra_info.source_sample_id`` for synthetic
VQA records.  This script joins locally available source runs to the source
run's
``samples.jsonl``, extracts only eligible question-facing merged statements,
and writes them to ``extra_info.partial_credit``. Benchmark and legacy rows
whose source runs are not locally available are intentionally left unchanged.

The output is a new parquet and, when ``--register-name`` is supplied, a new
DatasetRegistry entry.  Existing source parquets and registered datasets are
never overwritten unless ``--overwrite`` is explicitly passed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

# Make this utility directly runnable from the repository root, just as the
# training launchers do.  ``rllm`` imports synthesis helpers during package
# initialization, so it needs both roots rather than only ``RL/rllm``.
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_RLLM_ROOT = _PROJECT_ROOT / "RL" / "rllm"
for _path in (str(_PROJECT_ROOT), str(_RLLM_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from rllm.data.dataset import DatasetRegistry


PARTIAL_CREDIT_VERSION = 3


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return False


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}


def _to_builtin(value: Any) -> Any:
    """Convert pandas/NumPy values so pyarrow can round-trip them."""

    if isinstance(value, np.ndarray):
        return [_to_builtin(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    if _is_missing(value):
        return None
    return value


def _write_parquet_atomically(dataframe: pd.DataFrame, output_path: Path) -> None:
    """Serialize locally, validate, then atomically publish to HDFS/FUSE.

    Direct parquet writes to the shared FUSE mount expose a zero-byte target
    while pyarrow is still serializing.  A stalled or killed writer can then
    look like a completed dataset.  Keep construction local and publish only
    a fully readable temporary file.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, local_name = tempfile.mkstemp(
        prefix=f"{output_path.name}.", suffix=".parquet", dir="/tmp"
    )
    os.close(descriptor)
    local_path = Path(local_name)
    staged_path = output_path.with_name(
        f".{output_path.name}.{os.getpid()}.partial"
    )
    try:
        dataframe.to_parquet(local_path, index=False)
        verified = pd.read_parquet(local_path)
        if len(verified) != len(dataframe):
            raise RuntimeError(
                f"Parquet validation row mismatch: {len(verified)} != {len(dataframe)}"
            )
        shutil.copyfile(local_path, staged_path)
        os.replace(staged_path, output_path)
    finally:
        local_path.unlink(missing_ok=True)
        staged_path.unlink(missing_ok=True)


def _record_source_key(record: Mapping[str, Any]) -> tuple[str, str] | None:
    extra_info = _as_mapping(record.get("extra_info"))
    run_dir = str(extra_info.get("source_run_dir") or "").strip()
    sample_id = str(extra_info.get("source_sample_id") or "").strip()
    if not run_dir and not sample_id:
        return None
    if not run_dir or not sample_id:
        return (run_dir, sample_id)
    return (str(Path(run_dir).expanduser()), sample_id)


def _extract_statements(source: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Extract all non-empty, distinct question-facing merged VQA hops.

    The ``source == '-'`` entry hop represents an input-image fact.  Although
    its source is not another textual node, the policy still has to inspect
    and verify that fact while solving the task, so it is deliberately kept.
    Final answers are normally encoded in ``target_ask`` rather than
    ``merged_hops``; therefore no positional or target-answer filtering is
    applied here.
    """

    counts = {
        "source_hop_count": 0,
        "entry_hop_count": 0,
        "duplicate_statement_skipped_count": 0,
        "empty_statement_skipped_count": 0,
        "target_object_missing_count": 0,
        "non_entry_source_object_missing_count": 0,
    }
    compose = source.get("compose")
    payload = compose.get("payload") if isinstance(compose, Mapping) else None
    merged_hops = payload.get("merged_hops") if isinstance(payload, Mapping) else None
    if not isinstance(merged_hops, list):
        return [], counts

    statements: list[dict[str, Any]] = []
    seen: set[str] = set()
    for position, hop in enumerate(merged_hops):
        if not isinstance(hop, Mapping):
            continue
        counts["source_hop_count"] += 1
        statement = _normalize_text(hop.get("statement"))
        if not statement:
            counts["empty_statement_skipped_count"] += 1
            continue
        dedupe_key = statement.casefold()
        if dedupe_key in seen:
            counts["duplicate_statement_skipped_count"] += 1
            continue
        seen.add(dedupe_key)
        hop_index = hop.get("hop_index")
        statement_id = (
            f"hop_{hop_index}" if hop_index is not None else f"hop_pos_{position}"
        )
        raw_source_object = _normalize_text(hop.get("source"))
        is_entry_hop = raw_source_object == "-"
        if is_entry_hop:
            counts["entry_hop_count"] += 1
            source_object: str | None = None
        else:
            source_object = raw_source_object or None
            if source_object is None:
                counts["non_entry_source_object_missing_count"] += 1
        target_object = _normalize_text(hop.get("target"))
        if not target_object:
            counts["target_object_missing_count"] += 1
        statements.append(
            {
                "statement_id": statement_id,
                "statement": statement,
                # ``source_object`` is deliberately null for hop_0 / image
                # entry hops. The reward-side hard gate then requires only
                # target evidence for that first discovery, while later hops
                # require both chain endpoints in cited tool observations.
                "source_object": source_object,
                "target_object": target_object,
            }
        )
    return statements, counts


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if isinstance(record, dict):
                yield record


def _collect_source_payloads(
    needed: Mapping[str, Mapping[str, Any]],
    *,
    strict_source: bool,
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, int]]:
    """Stream source runs and retain only samples used by the input parquet."""

    payloads: dict[tuple[str, str], dict[str, Any]] = {}
    stats: dict[str, int] = defaultdict(int)
    by_run: dict[str, set[str]] = defaultdict(set)
    for key in needed:
        by_run[key[0]].add(key[1])

    for run_dir_text, wanted_ids in by_run.items():
        samples_path = Path(run_dir_text) / "samples.jsonl"
        if not samples_path.is_file():
            message = f"Missing source samples file: {samples_path}"
            if strict_source:
                raise FileNotFoundError(message)
            print(f"[partial-credit] warning: {message}")
            stats["missing_source_runs"] += 1
            continue
        found_ids: set[str] = set()
        for source in _iter_jsonl(samples_path):
            sample_id = str(source.get("sample_id") or "").strip()
            if sample_id not in wanted_ids:
                continue
            key = (run_dir_text, sample_id)
            statements, extraction_counts = _extract_statements(source)
            payloads[key] = {
                "version": PARTIAL_CREDIT_VERSION,
                "statements": statements,
            }
            found_ids.add(sample_id)
            stats["matched_source_samples"] += 1
            stats["eligible_statements"] += len(statements)
            for name, count in extraction_counts.items():
                stats[name] += count
        missing_ids = wanted_ids.difference(found_ids)
        if missing_ids:
            message = (
                f"{len(missing_ids)} sample IDs in {samples_path} were not found"
            )
            if strict_source:
                raise KeyError(message)
            print(f"[partial-credit] warning: {message}")
            stats["missing_source_samples"] += len(missing_ids)
    return payloads, dict(stats)


def _register_parquet(
    *,
    parquet_path: Path,
    records: list[dict[str, Any]],
    register_name: str,
    split: str,
    overwrite: bool,
) -> Path:
    """Register a fresh parquet without changing existing registered names."""

    dataset_dir = Path(DatasetRegistry._DATASET_DIR) / register_name
    registered_path = dataset_dir / f"{split}.parquet"
    verl_path = dataset_dir / f"{split}_verl.parquet"
    registry = DatasetRegistry._load_registry()
    if (
        (registered_path.exists() or registered_path.is_symlink())
        or (verl_path.exists() or verl_path.is_symlink())
        or split in registry.get(register_name, {})
    ) and not overwrite:
        raise FileExistsError(
            f"Dataset '{register_name}' split '{split}' already exists. "
            "Use a new name or pass --overwrite."
        )

    dataset_dir.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for path in (registered_path, verl_path):
            if path.is_symlink() or path.exists():
                path.unlink()
    verl_data = DatasetRegistry.apply_verl_postprocessing(records)
    _write_parquet_atomically(pd.DataFrame(verl_data), verl_path)
    os.symlink(parquet_path, registered_path)
    registry.setdefault(register_name, {})[split] = str(registered_path)
    DatasetRegistry._save_registry(registry)
    return verl_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach source-derived partial-credit statements to an RL parquet."
    )
    parser.add_argument("--input-parquet", required=True)
    parser.add_argument("--output-parquet", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--register-name", default="")
    parser.add_argument("--split", default="train")
    parser.add_argument(
        "--strict-source",
        action="store_true",
        help="Fail instead of emitting a partial dataset when synthetic provenance is broken.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Permit replacing the named output parquet and/or registry split.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print coverage without writing parquet, report, or registry state.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_parquet).expanduser().resolve()
    output_path = Path(args.output_parquet).expanduser().resolve()
    report_path = Path(args.report_json).expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if output_path == input_path:
        raise ValueError("--output-parquet must differ from --input-parquet")
    if (output_path.exists() or report_path.exists()) and not args.overwrite:
        raise FileExistsError("Output already exists; use --overwrite or a new path")

    dataframe = pd.read_parquet(input_path)
    required_columns = {"question", "answer", "images", "extra_info"}
    missing_columns = sorted(required_columns.difference(dataframe.columns))
    if missing_columns:
        raise ValueError(f"Input parquet is missing columns: {missing_columns}")
    records = [
        {str(key): _to_builtin(value) for key, value in row.items()}
        for row in dataframe.to_dict("records")
    ]

    needed: dict[tuple[str, str], dict[str, Any]] = {}
    rows_without_source_provenance = 0
    malformed_provenance_rows = 0
    for record in records:
        source_key = _record_source_key(record)
        if source_key is None:
            rows_without_source_provenance += 1
            continue
        if not source_key[0] or not source_key[1]:
            malformed_provenance_rows += 1
            continue
        needed[source_key] = record

    if malformed_provenance_rows and args.strict_source:
        raise ValueError(
            f"{malformed_provenance_rows} rows have incomplete source provenance"
        )
    payloads, extraction_stats = _collect_source_payloads(
        needed, strict_source=args.strict_source
    )

    enriched_rows = 0
    no_eligible_statement_rows = 0
    unmatched_source_rows = 0
    for record in records:
        source_key = _record_source_key(record)
        if source_key is None or not source_key[0] or not source_key[1]:
            continue
        payload = payloads.get(source_key)
        if payload is None:
            unmatched_source_rows += 1
            continue
        extra_info = _as_mapping(record.get("extra_info"))
        extra_info["partial_credit"] = payload
        record["extra_info"] = extra_info
        enriched_rows += 1
        if not payload["statements"]:
            no_eligible_statement_rows += 1

    report = {
        "version": PARTIAL_CREDIT_VERSION,
        "input_parquet": str(input_path),
        "output_parquet": str(output_path),
        "row_count": len(records),
        "rows_without_source_provenance": rows_without_source_provenance,
        "rows_without_local_source_samples": unmatched_source_rows,
        "malformed_provenance_rows": malformed_provenance_rows,
        "synthetic_source_keys": len(needed),
        "synthetic_rows_enriched": enriched_rows,
        "rows_with_source_provenance_but_unmatched": unmatched_source_rows,
        "synthetic_rows_with_no_eligible_statements": no_eligible_statement_rows,
        "registered_name": args.register_name or None,
        "split": args.split,
        "extraction": extraction_stats,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    if args.dry_run:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_parquet_atomically(pd.DataFrame(records), output_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.register_name:
        verl_path = _register_parquet(
            parquet_path=output_path,
            records=records,
            register_name=args.register_name,
            split=args.split,
            overwrite=args.overwrite,
        )
        print(f"registered_verl_parquet={verl_path}")
    print(f"output_parquet={output_path}")
    print(f"report_json={report_path}")


if __name__ == "__main__":
    main()
