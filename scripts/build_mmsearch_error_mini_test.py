#!/usr/bin/env python3
"""Build reproducible MMSearch error-focused mini test subsets.

The default sources are the three SFT12 MMSearch runs analyzed in September
2026.  A sample is in the ``error_union`` subset when it was either judged
incorrect or failed before judging in at least one source run.  The
``persistent_errors`` subset keeps samples that met that condition in at least
two source runs, which makes it a less retrieval- and judge-sensitive signal
for quick regressions.

The companion JSONL manifest records the exact source-run evidence used for
each sample.  It intentionally does not ingest ``mmsearch_0901/failures``:
that directory contains stale artifacts for all 300 cases rather than failures
from the evaluated 0901 run.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class RunSource:
    name: str
    directory: Path
    include_failure_files: bool


DEFAULT_RUNS = (
    RunSource(
        name="sft12_0901",
        directory=Path(
            "opensearch_infer/outputs/"
            "Qwen3-VL-8B-SFT12-gpt54-only-epoch5/mmsearch_0901"
        ),
        include_failure_files=False,
    ),
    RunSource(
        name="sft12_0914_sampling07",
        directory=Path(
            "opensearch_infer/outputs/"
            "Qwen3-VL-8B-SFT12-gpt54-only/mmsearch_0914_reproduce"
        ),
        include_failure_files=True,
    ),
    RunSource(
        name="sft12_0914_sampling0",
        directory=Path(
            "opensearch_infer/outputs/"
            "Qwen3-VL-8B-SFT12-gpt54-only-epoch5/"
            "mmsearch_0914_reproduce_2"
        ),
        include_failure_files=True,
    ),
)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def collect_membership(run_sources: tuple[RunSource, ...]) -> dict[str, list[dict[str, Any]]]:
    """Return per-case evidence for evaluated errors and genuine failures."""
    membership: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in run_sources:
        details_path = source.directory / "llm_eval_report_details.jsonl"
        if not details_path.exists():
            raise FileNotFoundError(f"Missing evaluation details: {details_path}")

        for row in _load_jsonl(details_path):
            if int(row["acc"]) == 1:
                continue
            membership[str(row["case_id"])].append(
                {
                    "run": source.name,
                    "outcome": "judged_incorrect",
                    "judge_prompt_style": row.get("judge_prompt_style"),
                }
            )

        if not source.include_failure_files:
            continue
        for failure_path in sorted((source.directory / "failures").glob("*_failure.json")):
            failure = json.loads(failure_path.read_text(encoding="utf-8"))
            if failure.get("status") != "failed":
                continue
            membership[str(failure["case_id"])].append(
                {
                    "run": source.name,
                    "outcome": "inference_failed",
                    "failure_kind": failure.get("failure_kind"),
                    "failure_reason": failure.get("failure_reason"),
                    "failed_turn": failure.get("failed_turn"),
                }
            )
    return dict(membership)


def _select_rows(
    source_df: pd.DataFrame, case_ids: set[str], subset_name: str
) -> pd.DataFrame:
    if "data_id" not in source_df.columns:
        raise ValueError("Source parquet must contain a data_id column")
    normalized_ids = source_df["data_id"].astype(str)
    if normalized_ids.duplicated().any():
        duplicate_ids = normalized_ids[normalized_ids.duplicated()].unique().tolist()
        raise ValueError(f"Source parquet has duplicate data_id values: {duplicate_ids[:10]}")
    missing = sorted(case_ids - set(normalized_ids))
    if missing:
        raise ValueError(f"Subset IDs missing from source parquet: {missing}")

    result = source_df.loc[normalized_ids.isin(case_ids)].copy()
    result["mini_test_subset"] = subset_name
    return result


def _write_manifest(
    path: Path,
    source_df: pd.DataFrame,
    membership: dict[str, list[dict[str, Any]]],
    min_occurrences: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_by_id = source_df.set_index(source_df["data_id"].astype(str), drop=False)
    selected_ids = {
        case_id
        for case_id, evidence in membership.items()
        if len(evidence) >= min_occurrences
    }
    for case_id in source_df["data_id"].astype(str):
        if case_id not in selected_ids:
            continue
        row = source_by_id.loc[case_id]
        evidence = membership[case_id]
        rows.append(
            {
                "data_id": case_id,
                "category": row.get("category"),
                "question": row.get("question"),
                "error_occurrence_count": len(evidence),
                "source_runs": [item["run"] for item in evidence],
                "evidence": evidence,
            }
        )
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return rows


def build_subsets(
    source_data: Path, output_dir: Path, subset_names: tuple[str, ...]
) -> dict[str, Any]:
    source_df = pd.read_parquet(source_data)
    membership = collect_membership(DEFAULT_RUNS)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_subsets = {
        "error_union": 1,
        "persistent_errors": 2,
    }
    unknown = set(subset_names) - set(all_subsets)
    if unknown:
        raise ValueError(f"Unknown subset(s): {sorted(unknown)}")
    subsets = {name: all_subsets[name] for name in subset_names}
    summaries: dict[str, Any] = {}
    for subset_name, min_occurrences in subsets.items():
        case_ids = {
            case_id
            for case_id, evidence in membership.items()
            if len(evidence) >= min_occurrences
        }
        subset = _select_rows(source_df, case_ids, subset_name)
        parquet_path = output_dir / f"mmsearch_sft12_{subset_name}_mini_test.parquet"
        manifest_path = output_dir / f"mmsearch_sft12_{subset_name}_mini_test_manifest.jsonl"
        subset.to_parquet(parquet_path, index=False)
        manifest_rows = _write_manifest(
            manifest_path, source_df, membership, min_occurrences
        )
        if len(subset) != len(case_ids) or len(manifest_rows) != len(case_ids):
            raise AssertionError(f"Unexpected output size for {subset_name}")
        summaries[subset_name] = {
            "rows": len(subset),
            "parquet": str(parquet_path),
            "manifest": str(manifest_path),
            "category_counts": Counter(subset["category"]).most_common(),
        }

    summary_path = output_dir / "mmsearch_sft12_error_mini_test_summary.json"
    summary = {
        "source_data": str(source_data),
        "source_runs": [
            {
                "name": source.name,
                "directory": str(source.directory),
                "include_failure_files": source.include_failure_files,
            }
            for source in DEFAULT_RUNS
        ],
        "membership_frequency": dict(sorted(Counter(map(len, membership.values())).items())),
        "subsets": summaries,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-data",
        type=Path,
        default=Path("../data/benchmarks/mmsearch/end2end/data.parquet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("opensearch_vl/data/benchmarks/mmsearch"),
    )
    parser.add_argument(
        "--subset",
        action="append",
        choices=("error_union", "persistent_errors"),
        help="Subset to build; repeat to build both. Defaults to both.",
    )
    args = parser.parse_args()
    selected_subsets = tuple(args.subset or ("error_union", "persistent_errors"))
    summary = build_subsets(args.source_data, args.output_dir, selected_subsets)
    for name, result in summary["subsets"].items():
        print(f"{name}: {result['rows']} rows -> {result['parquet']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
