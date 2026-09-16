#!/usr/bin/env python3
"""Build an RL dataset augmented with benchmark questions selected by eval difficulty.

Difficulty is estimated from the available evaluation results of SFT3, SFT7,
SFT8, SFT9, SFT10, and SFT12.  Each model version contributes at most one
binary result per benchmark case.  When one version has several partially
overlapping evaluation directories, later directories fill missing cases and
override earlier results for duplicate cases.

The default difficulty labels are deliberately simple and auditable:

* ``medium``: at least one version answered correctly and at least one failed;
* ``hard``: every observed version failed;
* ``easy``: every observed version answered correctly.

The selected benchmark subset is exactly 60% medium and 40% hard.  If no
target size is supplied, the largest possible 3:2 subset is produced.  The
script preserves the existing train/test split, adds benchmark examples only
to train, reconstructs benchmark images from the saved inference trajectories,
and can register the result as a new rLLM dataset without modifying the source
dataset.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import re
import shutil
import sys
import tempfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RLLM_ROOT = ROOT / "RL" / "rllm"


# Directory order is significant: within one model version, later entries
# override duplicate case IDs from earlier runs.  This treats reruns as one
# policy rollout while still recovering cases from incomplete runs.
EVAL_SOURCES: dict[str, dict[str, tuple[str, ...]]] = {
    "SFT3": {
        "browsecomp_vl": ("Qwen3-VL-8B-SFT3/browsecomp_vl_0831",),
        "mmsearch": (
            "Qwen3-VL-8B-SFT3/mmsearch_full_0819_tools_sampling_params",
            "Qwen3-VL-8B-SFT3/mmsearch_0831",
        ),
        "vdr_testmini": ("Qwen3-VL-8B-SFT3/vdr_testmini_0831",),
    },
    "SFT7": {
        "browsecomp_vl": ("Qwen3-VL-8B-SFT7-rewrited-7k/browsecomp_vl_full_0823",),
        "mmsearch": ("Qwen3-VL-8B-SFT7-rewrited-7k/mmsearch_full_0823",),
    },
    "SFT8": {
        "browsecomp_vl": ("Qwen3-VL-8B-SFT8-rewrited-15k/browsecomp_0824",),
        "mmsearch": ("Qwen3-VL-8B-SFT8-rewrited-15k/mmsearch_0824",),
    },
    "SFT9": {
        "browsecomp_vl": ("Qwen3-VL-8B-SFT9-rewrited-24k/browsecomp_vl_0825",),
        "mmsearch": ("Qwen3-VL-8B-SFT9-rewrited-24k/mmsearch_0825",),
        "vdr_testmini": ("Qwen3-VL-8B-SFT9-rewrited-24k/vdr_testmini_0827",),
    },
    "SFT10": {
        "browsecomp_vl": (
            "Qwen3-VL-8B-SFT10-gpt54-only/browsecomp_vl_0827",
            "Qwen3-VL-8B-SFT10-gpt54-only/browsecomp_vl_0828",
            "Qwen3-VL-8B-SFT10-gpt54-only/browsecomp_vl_0830",
            "Qwen3-VL-8B-SFT10-gpt54-only/browsecomp_vl_0903_no_repeat",
        ),
        "mmsearch": (
            "Qwen3-VL-8B-SFT10-gpt54-only/mmsearch_0827",
            "Qwen3-VL-8B-SFT10-gpt54-only/mmsearch_0828",
            "Qwen3-VL-8B-SFT10-gpt54-only/mmsearch_0831",
            "Qwen3-VL-8B-SFT10-gpt54-only/mmsearch_0907_add_fallback",
        ),
        "fvqa": ("Qwen3-VL-8B-SFT10-gpt54-only/fvqa",),
        "vdr_testmini": ("Qwen3-VL-8B-SFT10-gpt54-only/vdr_testmini_0828",),
    },
    "SFT12": {
        "browsecomp_vl": (
            "Qwen3-VL-8B-SFT12-gpt54-only-epoch5/browsecomp_vl_0901",
            "Qwen3-VL-8B-SFT12-gpt54-only-epoch5/browsecomp_vl_full_0904",
        ),
        "mmsearch": ("Qwen3-VL-8B-SFT12-gpt54-only-epoch5/mmsearch_0901",),
        "vdr_testmini": ("Qwen3-VL-8B-SFT12-gpt54-only-epoch5/vdr_testmini_0901",),
    },
}


@dataclass(frozen=True)
class EvalVote:
    version: str
    benchmark: str
    case_id: str
    acc: int
    question: str
    answer: str
    report_dir: Path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            records.append(value)
    return records


def _write_jsonl(path: Path, records: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            count += 1
    return count


def _normalise_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _parse_acc(value: Any) -> int | None:
    if value is True or value == 1 or value == "1":
        return 1
    if value is False or value == 0 or value == "0":
        return 0
    return None


def load_eval_votes(
    outputs_root: Path,
) -> tuple[dict[tuple[str, str], dict[str, EvalVote]], dict[str, Any]]:
    votes: dict[tuple[str, str], dict[str, EvalVote]] = defaultdict(dict)
    report_stats: list[dict[str, Any]] = []
    duplicate_overrides = 0
    invalid_rows = 0

    for version, benchmark_sources in EVAL_SOURCES.items():
        for benchmark, relative_dirs in benchmark_sources.items():
            for relative_dir in relative_dirs:
                report_dir = outputs_root / relative_dir
                details_path = report_dir / "llm_eval_report_details.jsonl"
                if not details_path.exists():
                    report_stats.append(
                        {
                            "version": version,
                            "benchmark": benchmark,
                            "report_dir": str(report_dir),
                            "status": "missing",
                        }
                    )
                    continue

                row_count = 0
                valid_count = 0
                for row in _read_jsonl(details_path):
                    row_count += 1
                    case_id = str(row.get("case_id") or "").strip()
                    acc = _parse_acc(row.get("acc"))
                    if not case_id or acc is None:
                        invalid_rows += 1
                        continue
                    valid_count += 1
                    key = (benchmark, case_id)
                    if version in votes[key]:
                        duplicate_overrides += 1
                    votes[key][version] = EvalVote(
                        version=version,
                        benchmark=benchmark,
                        case_id=case_id,
                        acc=acc,
                        question=str(row.get("question") or "").strip(),
                        answer=str(row.get("correct_answer") or "").strip(),
                        report_dir=report_dir,
                    )
                report_stats.append(
                    {
                        "version": version,
                        "benchmark": benchmark,
                        "report_dir": str(report_dir),
                        "status": "loaded",
                        "rows": row_count,
                        "valid_rows": valid_count,
                    }
                )

    return votes, {
        "reports": report_stats,
        "duplicate_version_case_overrides": duplicate_overrides,
        "invalid_detail_rows": invalid_rows,
    }


def build_difficulty_records(
    votes: dict[tuple[str, str], dict[str, EvalVote]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    records: list[dict[str, Any]] = []
    conflicts = Counter()

    for (benchmark, case_id), version_votes in sorted(votes.items()):
        ordered_votes = [
            version_votes[version] for version in EVAL_SOURCES if version in version_votes
        ]
        questions = {_normalise_text(v.question) for v in ordered_votes if v.question}
        answers = {_normalise_text(v.answer) for v in ordered_votes if v.answer}
        conflict_reasons: list[str] = []
        if len(questions) > 1:
            conflicts["question"] += 1
            conflict_reasons.append("question_conflict")
        if len(answers) > 1:
            conflicts["answer"] += 1
            conflict_reasons.append("answer_conflict")

        observed_count = len(ordered_votes)
        correct_count = sum(v.acc for v in ordered_votes)
        if correct_count == 0:
            difficulty = "hard"
        elif correct_count == observed_count:
            difficulty = "easy"
        else:
            difficulty = "medium"

        latest = ordered_votes[-1]
        records.append(
            {
                "benchmark": benchmark,
                "case_id": case_id,
                "question": latest.question,
                "answer": latest.answer,
                "observed_count": observed_count,
                "correct_count": correct_count,
                "correct_rate": correct_count / observed_count,
                "difficulty": difficulty,
                "coverage_confidence": "single_observation" if observed_count == 1 else "multi_version",
                "conflicts": conflict_reasons,
                "version_results": {
                    vote.version: {
                        "acc": vote.acc,
                        "report_dir": str(vote.report_dir),
                    }
                    for vote in ordered_votes
                },
            }
        )

    return records, dict(conflicts)


def _stable_rank(record: dict[str, Any], seed: int) -> str:
    payload = f"{seed}|{record['benchmark']}|{record['case_id']}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _allocate_strata(records: list[dict[str, Any]], target: int) -> dict[str, int]:
    counts = Counter(str(record["benchmark"]) for record in records)
    if target > sum(counts.values()):
        raise ValueError("stratified target exceeds available records")
    if target == 0:
        return {key: 0 for key in counts}

    total = sum(counts.values())
    exact = {key: target * count / total for key, count in counts.items()}
    allocated = {key: min(counts[key], math.floor(value)) for key, value in exact.items()}
    remaining = target - sum(allocated.values())
    order = sorted(counts, key=lambda key: (-(exact[key] - allocated[key]), key))
    while remaining:
        progressed = False
        for key in order:
            if allocated[key] < counts[key]:
                allocated[key] += 1
                remaining -= 1
                progressed = True
                if not remaining:
                    break
        if not progressed:
            raise RuntimeError("could not allocate all stratified samples")
    return allocated


def select_medium_hard(
    records: list[dict[str, Any]],
    *,
    target_count: int,
    seed: int,
) -> list[dict[str, Any]]:
    usable = [record for record in records if not record["conflicts"]]
    medium = [record for record in usable if record["difficulty"] == "medium"]
    hard = [record for record in usable if record["difficulty"] == "hard"]

    if target_count:
        if target_count % 5:
            raise ValueError("--target-count must be divisible by 5 for an exact 60:40 split")
        units = target_count // 5
        if 3 * units > len(medium) or 2 * units > len(hard):
            raise ValueError(
                f"target requires {3 * units} medium/{2 * units} hard, but only "
                f"{len(medium)}/{len(hard)} are available"
            )
    else:
        units = min(len(medium) // 3, len(hard) // 2)

    targets = {"medium": 3 * units, "hard": 2 * units}
    selected: list[dict[str, Any]] = []
    for difficulty, pool in (("medium", medium), ("hard", hard)):
        allocations = _allocate_strata(pool, targets[difficulty])
        for benchmark, amount in sorted(allocations.items()):
            benchmark_pool = [record for record in pool if record["benchmark"] == benchmark]
            # Prefer cases observed by more independent model versions.  The
            # hash randomises ties reproducibly and prevents case-ID bias.
            benchmark_pool.sort(
                key=lambda record: (-int(record["observed_count"]), _stable_rank(record, seed))
            )
            selected.extend(benchmark_pool[:amount])

    selected.sort(key=lambda record: (record["difficulty"], record["benchmark"], record["case_id"]))
    return selected


def _trajectory_candidates(record: dict[str, Any]) -> list[Path]:
    candidates: list[Path] = []
    for version in reversed(tuple(EVAL_SOURCES)):
        result = record["version_results"].get(version)
        if result:
            candidates.append(Path(result["report_dir"]) / f"{record['case_id']}_trajectory.json")
    return candidates


def _decode_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str) and value.lstrip().startswith(("b'", 'b"')):
        decoded = ast.literal_eval(value)
        if isinstance(decoded, bytes):
            return decoded
    raise ValueError(f"unsupported image bytes representation: {type(value).__name__}")


def _image_extension(data: bytes, declared_format: Any = None) -> str:
    aliases = {"jpeg": ".jpg", "jpg": ".jpg", "png": ".png", "webp": ".webp", "gif": ".gif"}
    declared = str(declared_format or "").strip().lower()
    if declared in aliases:
        return aliases[declared]
    with Image.open(io.BytesIO(data)) as image:
        detected = str(image.format or "").lower()
        image.verify()
    return aliases.get(detected, f".{detected}" if detected else ".img")


def _materialise_image(image: dict[str, Any], images_dir: Path) -> Path:
    data = _decode_bytes(image.get("bytes"))
    extension = _image_extension(data, image.get("format"))
    digest = hashlib.sha256(data).hexdigest()
    output_path = images_dir / f"{digest}{extension}"
    if not output_path.exists():
        output_path.write_bytes(data)
    return output_path.resolve()


def build_benchmark_row(
    record: dict[str, Any],
    *,
    images_dir: Path,
) -> dict[str, Any]:
    trajectory_path: Path | None = None
    trajectory: dict[str, Any] | None = None
    for candidate in _trajectory_candidates(record):
        try:
            with candidate.open("r", encoding="utf-8") as handle:
                trajectory = json.load(handle)
            trajectory_path = candidate
            break
        except FileNotFoundError:
            continue
    if trajectory_path is None:
        raise FileNotFoundError(
            f"no trajectory found for {record['benchmark']}:{record['case_id']}"
        )
    assert trajectory is not None
    original = trajectory.get("original_data")
    if not isinstance(original, dict):
        raise ValueError(f"trajectory has no original_data object: {trajectory_path}")

    question = str(original.get("question") or record["question"] or "").strip()
    answer = str(original.get("answer") or record["answer"] or "").strip()
    if not question or not answer:
        raise ValueError(f"empty question/answer in {trajectory_path}")
    raw_images = original.get("images") or []
    if not isinstance(raw_images, list):
        raise ValueError(f"benchmark row images is not a list: {trajectory_path}")
    image_paths = [str(_materialise_image(image, images_dir)) for image in raw_images]
    primary_image_path = image_paths[0] if image_paths else ""

    source_metadata = original.get("source_metadata")
    evidence = {
        "benchmark": record["benchmark"],
        "case_id": record["case_id"],
        "difficulty": record["difficulty"],
        "observed_count": record["observed_count"],
        "correct_count": record["correct_count"],
        "correct_rate": record["correct_rate"],
        "coverage_confidence": record["coverage_confidence"],
        "version_results": record["version_results"],
    }
    identifier = f"benchmark__{record['benchmark']}__{record['case_id']}"
    return {
        "id": identifier,
        "question": question,
        "answer": answer,
        "images": image_paths,
        "extra_info": {
            "source_run_dir": str(trajectory_path.parent),
            "source_index": trajectory.get("case_idx"),
            "source_question_id": str(original.get("question_id") or record["case_id"]),
            "source_sample_id": str(original.get("sample_id") or record["case_id"]),
            "source_path_id": "",
            "source_status": "benchmark_eval_selected",
            "source_image_url": primary_image_path,
            "drafted_question": question,
            "enhanced_question": question,
            "final_question": question,
            "image_mode": "trajectory_bytes_extracted",
            "sample_metadata": {
                "input_image_url": primary_image_path,
                "status": "benchmark_eval_selected",
                "created_at": "",
                "updated_at": "",
            },
            "benchmark_source_metadata": str(source_metadata or ""),
            "benchmark_difficulty_evidence": json.dumps(evidence, ensure_ascii=False, sort_keys=True),
        },
    }


def _read_parquet(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_parquet(path).to_dict("records")


def _write_parquet(path: Path, records: list[dict[str, Any]]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="benchmark-rl-parquet-", dir="/tmp"))
    temp_path = temp_dir / path.name
    try:
        pd.DataFrame(records).to_parquet(temp_path, index=False)
        shutil.move(str(temp_path), str(path))
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _split_filtered_base(
    base_records: list[dict[str, Any]],
    original_train: list[dict[str, Any]],
    original_test: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    train_ids = {str(record.get("id")) for record in original_train}
    test_ids = {str(record.get("id")) for record in original_test}
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(f"original train/test IDs overlap: {len(overlap)}")

    train: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    unknown = 0
    for record in base_records:
        record_id = str(record.get("id") or "")
        if record_id in train_ids:
            train.append(record)
        elif record_id in test_ids:
            test.append(record)
        else:
            train.append(record)
            unknown += 1
    return train, test, unknown


def _count_nested(records: list[dict[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(record[field]) for record in records).items()))


def _register_dataset(name: str, train: list[dict[str, Any]], test: list[dict[str, Any]]) -> None:
    if str(RLLM_ROOT) not in sys.path:
        sys.path.insert(0, str(RLLM_ROOT))
    from rllm.data.dataset import DatasetRegistry

    if DatasetRegistry.dataset_exists(name):
        raise FileExistsError(
            f"registered dataset already exists: {name}; choose a new --register-name"
        )
    DatasetRegistry.register_dataset(name, train, split="train")
    if test:
        DatasetRegistry.register_dataset(name, test, split="test")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=ROOT / "opensearch_infer" / "outputs",
    )
    parser.add_argument(
        "--base-jsonl",
        type=Path,
        default=ROOT
        / "runs/0712_multi_seed_visual_test_8192_8/vqa/"
        "rl_combined_hard_quality_audit_with_uniqueness/rl_dataset_drop_filtered.jsonl",
        help="Filtered base RL records to preserve.",
    )
    parser.add_argument(
        "--original-train-parquet",
        type=Path,
        default=RLLM_ROOT
        / "rllm/data/datasets/Vision-DeepResearch-QA-rl-combined-5871/train.parquet",
    )
    parser.add_argument(
        "--original-test-parquet",
        type=Path,
        default=RLLM_ROOT
        / "rllm/data/datasets/Vision-DeepResearch-QA-rl-combined-5871/test.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT
        / "runs/0712_multi_seed_visual_test_8192_8/vqa/"
        "rl_audited_benchmark_medium_hard_augmented",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=0,
        help="Exact benchmark additions; must be divisible by 5. Zero maximises the 60:40 subset.",
    )
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument(
        "--image-workers",
        type=int,
        default=8,
        help="Parallel workers used to read trajectory files and materialise images.",
    )
    parser.add_argument(
        "--register-name",
        default="",
        help="Optionally register the merged train/test data under this new rLLM dataset name.",
    )
    parser.add_argument(
        "--all-train",
        action="store_true",
        help="Put every filtered base and selected benchmark record in train and omit test.",
    )
    args = parser.parse_args()

    for path in (args.outputs_root, args.base_jsonl, args.original_train_parquet, args.original_test_parquet):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.target_count < 0:
        parser.error("--target-count must be non-negative")
    if args.image_workers < 1:
        parser.error("--image-workers must be positive")
    if args.output_dir.exists():
        existing = {path.name for path in args.output_dir.iterdir()}
        unexpected = existing - {"images"}
        if unexpected:
            raise FileExistsError(
                f"output directory already contains completed artifacts: {sorted(unexpected)}"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = args.output_dir / "images"

    votes, load_stats = load_eval_votes(args.outputs_root)
    difficulty_records, conflict_counts = build_difficulty_records(votes)
    base_records = _read_jsonl(args.base_jsonl)
    base_question_keys = {_normalise_text(record.get("question")) for record in base_records}
    novel_records = [
        record
        for record in difficulty_records
        if _normalise_text(record.get("question")) not in base_question_keys
    ]
    base_duplicate_count = len(difficulty_records) - len(novel_records)
    selected = select_medium_hard(
        novel_records,
        target_count=args.target_count,
        seed=args.seed,
    )

    stage_root = Path(tempfile.mkdtemp(prefix="benchmark-rl-images-", dir="/tmp"))
    stage_images_dir = stage_root / "images"
    stage_images_dir.mkdir(parents=True, exist_ok=True)
    benchmark_rows: list[dict[str, Any] | None] = [None] * len(selected)
    try:
        with ThreadPoolExecutor(max_workers=args.image_workers) as executor:
            futures = {
                executor.submit(build_benchmark_row, record, images_dir=stage_images_dir): index
                for index, record in enumerate(selected)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                benchmark_rows[futures[future]] = future.result()
                if completed % 50 == 0 or completed == len(selected):
                    print(
                        f"[trajectory-images] completed={completed}/{len(selected)}",
                        flush=True,
                    )

        images_dir.mkdir(parents=True, exist_ok=True)
        staged_images = sorted(stage_images_dir.iterdir())
        for completed, source_path in enumerate(staged_images, start=1):
            shutil.copy2(source_path, images_dir / source_path.name)
            if completed % 100 == 0 or completed == len(staged_images):
                print(
                    f"[publish-images] completed={completed}/{len(staged_images)}",
                    flush=True,
                )
    finally:
        shutil.rmtree(stage_root, ignore_errors=True)
    if any(record is None for record in benchmark_rows):
        raise RuntimeError("one or more benchmark rows were not materialised")
    benchmark_rows = [record for record in benchmark_rows if record is not None]
    published_root = images_dir.resolve()
    for benchmark_row in benchmark_rows:
        published_paths = [
            str(published_root / Path(path).name) for path in benchmark_row["images"]
        ]
        benchmark_row["images"] = published_paths
        primary_image_path = published_paths[0] if published_paths else ""
        benchmark_row["extra_info"]["source_image_url"] = primary_image_path
        benchmark_row["extra_info"]["sample_metadata"]["input_image_url"] = primary_image_path
    benchmark_ids = [str(record["id"]) for record in benchmark_rows]
    if len(benchmark_ids) != len(set(benchmark_ids)):
        raise ValueError("selected benchmark IDs are not unique")

    original_train = _read_parquet(args.original_train_parquet)
    original_test = _read_parquet(args.original_test_parquet)
    base_train, base_test, unknown_split_count = _split_filtered_base(
        base_records, original_train, original_test
    )
    original_base_train_count = len(base_train)
    original_base_test_count = len(base_test)
    if args.all_train:
        base_train.extend(base_test)
        base_test = []
    merged_train = base_train + benchmark_rows
    merged_all = merged_train + base_test

    _write_jsonl(args.output_dir / "difficulty_manifest.jsonl", difficulty_records)
    _write_jsonl(args.output_dir / "selected_benchmark_rl.jsonl", benchmark_rows)
    _write_jsonl(args.output_dir / "merged_rl_dataset.jsonl", merged_all)
    _write_parquet(args.output_dir / "train.parquet", merged_train)
    if base_test:
        _write_parquet(args.output_dir / "test.parquet", base_test)

    if args.register_name:
        _register_dataset(args.register_name, merged_train, base_test)

    candidate_counts = {
        difficulty: _count_nested(
            [record for record in novel_records if record["difficulty"] == difficulty],
            "benchmark",
        )
        for difficulty in ("medium", "hard", "easy")
    }
    selected_counts = {
        difficulty: _count_nested(
            [record for record in selected if record["difficulty"] == difficulty],
            "benchmark",
        )
        for difficulty in ("medium", "hard")
    }
    report = {
        "format": "eval_difficulty_augmented_rl_dataset_v1",
        "difficulty_definition": {
            "medium": "at least one correct and at least one incorrect result across distinct SFT versions",
            "hard": "all available distinct SFT versions incorrect",
            "easy": "all available distinct SFT versions correct",
            "duplicate_policy": "one vote per SFT version and case; later configured report overrides",
        },
        "models": list(EVAL_SOURCES),
        "load_stats": load_stats,
        "conflict_counts": conflict_counts,
        "base_question_duplicates_excluded": base_duplicate_count,
        "candidate_counts": candidate_counts,
        "candidate_coverage": dict(
            sorted(Counter(record["observed_count"] for record in novel_records).items())
        ),
        "selection": {
            "seed": args.seed,
            "target_count_argument": args.target_count,
            "total": len(selected),
            "medium": sum(record["difficulty"] == "medium" for record in selected),
            "hard": sum(record["difficulty"] == "hard" for record in selected),
            "counts_by_difficulty_and_benchmark": selected_counts,
            "single_observation": sum(
                record["coverage_confidence"] == "single_observation" for record in selected
            ),
        },
        "base": {
            "source_jsonl": str(args.base_jsonl.resolve()),
            "total": len(base_records),
            "train": len(base_train),
            "test": len(base_test),
            "original_train_before_all_train": original_base_train_count,
            "original_test_before_all_train": original_base_test_count,
            "split_policy": "all_train" if args.all_train else "preserve_original",
            "unknown_original_split_assigned_to_train": unknown_split_count,
        },
        "output": {
            "directory": str(args.output_dir.resolve()),
            "train": len(merged_train),
            "test": len(base_test),
            "total": len(merged_all),
            "registered_name": args.register_name or None,
        },
    }
    report_path = args.output_dir / "selection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report["selection"], ensure_ascii=False, indent=2))
    print(f"Merged dataset: {args.output_dir}")
    if args.register_name:
        print(f"Registered dataset: {args.register_name}")


if __name__ == "__main__":
    main()
