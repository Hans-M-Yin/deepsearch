#!/usr/bin/env python3
"""Sample OpenSearch-VL-SFT-36K subsets into run_infer-ready parquet files.

Example:
    python scripts/build_opensearch_sft_eval_subsets.py
    python scripts/build_opensearch_sft_eval_subsets.py --sft-root /path/to/Search-VL-SFT-36K
    python scripts/build_opensearch_sft_eval_subsets.py --subsets wiki_en,wiki_zh,fvqa --sample-size 100
"""

from __future__ import annotations

import argparse
import io
import json
import random
import re
from pathlib import Path
from typing import Any


DEFAULT_OUTPUT_ROOT = Path("data/opensearch_vl_sft")

SUBSET_SPECS = {
    "fvqa": {
        "json_relpath": Path("fvqa/fvqa_llama_factory_clean.json"),
    },
    "palace": {
        "json_relpath": Path("palace/palace_llama_factory_filtered.json"),
    },
    "webqa": {
        "json_relpath": Path("webqa/webqa_llama_factory_filtered.json"),
    },
    "livevqa": {
        "json_relpath": Path("livevqa/livevqa_llama_factory_filtered.json"),
    },
    "wiki_art": {
        "json_relpath": Path("wiki_art/wiki_art_llama_factory_filtered.json"),
    },
    "wiki_en": {
        "json_relpath": Path("wiki_en/wiki_en_llama_factory_filtered.json"),
    },
    "wiki_zh": {
        "json_relpath": Path("wiki_zh/wiki_zh_llama_factory_filtered.json"),
    },
}

IMAGE_TOKEN_RE = re.compile(r"^\s*<image>\s*", flags=re.IGNORECASE)
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", flags=re.IGNORECASE | re.DOTALL)
TOOL_BLOCK_RE = re.compile(
    r"<tool_call>.*?</tool_call>", flags=re.IGNORECASE | re.DOTALL
)
TAG_RE = re.compile(r"</?(response|observation)>", flags=re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\n{3,}")


def _normalize_image_reference(image_path: str) -> str:
    if image_path.startswith(("http://", "https://", "data:", "file://")):
        return image_path
    return Path(image_path).expanduser().resolve().as_uri()


def _strip_question_prefix(text: str) -> str:
    return IMAGE_TOKEN_RE.sub("", text or "").strip()


def _clean_answer_text(text: str) -> str:
    cleaned = THINK_BLOCK_RE.sub("", text or "")
    cleaned = TOOL_BLOCK_RE.sub("", cleaned)
    cleaned = TAG_RE.sub("", cleaned)
    cleaned = cleaned.strip()
    cleaned = WHITESPACE_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def _load_json_array(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level JSON array in {path}")
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} in {path} is not a JSON object")
        rows.append(item)
    return rows


def _extract_question(record: dict[str, Any]) -> str:
    conversations = record.get("conversations") or []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("from", "")).strip().lower() == "human":
            return _strip_question_prefix(str(turn.get("value", "")))
    return ""


def _extract_answer(record: dict[str, Any]) -> str:
    conversations = record.get("conversations") or []
    assistant_turns: list[str] = []
    for turn in conversations:
        if not isinstance(turn, dict):
            continue
        if str(turn.get("from", "")).strip().lower() == "gpt":
            assistant_turns.append(str(turn.get("value", "")))

    for text in reversed(assistant_turns):
        cleaned = _clean_answer_text(text)
        if cleaned:
            return cleaned
    return ""


def _resolve_image_path(
    image_ref: str,
    *,
    json_path: Path,
    sft_root: Path,
) -> Path:
    candidate = Path(image_ref)
    if candidate.is_absolute():
        resolved = candidate.resolve()
        if resolved.exists():
            return resolved
        raise FileNotFoundError(
            f"Absolute image path referenced by {json_path.name} does not exist: {resolved}"
        )

    candidates = [
        (sft_root / candidate).resolve(),
        (json_path.parent / candidate).resolve(),
    ]

    # Some records redundantly prefix the subset folder, e.g. `fvqa/images/...`.
    # If the path is already rooted at the subset dir, dropping the first segment
    # recovers the actual relative image location under the JSON directory.
    if len(candidate.parts) > 1:
        candidates.append((json_path.parent / Path(*candidate.parts[1:])).resolve())

    seen: set[str] = set()
    for path in candidates:
        marker = str(path)
        if marker in seen:
            continue
        seen.add(marker)
        if path.exists():
            return path

    attempted = ", ".join(seen)
    raise FileNotFoundError(
        f"Image referenced by {json_path.name} does not exist. Tried: {attempted}"
    )


def _resolve_images(
    record: dict[str, Any],
    json_path: Path,
    *,
    sft_root: Path,
) -> list[dict[str, str]]:
    resolved: list[dict[str, str]] = []
    for item in record.get("images") or []:
        if not isinstance(item, str):
            continue
        candidate = _resolve_image_path(item, json_path=json_path, sft_root=sft_root)
        resolved.append({"url": _normalize_image_reference(str(candidate))})
    return resolved


def _source_metadata(
    subset_name: str, source_index: int, json_path: Path, record: dict[str, Any]
) -> str:
    payload = {
        "subset": subset_name,
        "source_json": str(json_path),
        "source_index": source_index,
        "images": record.get("images", []),
        "system": record.get("system", ""),
    }
    return json.dumps(payload, ensure_ascii=False)


def _record_to_row(
    *,
    subset_name: str,
    source_index: int,
    json_path: Path,
    sft_root: Path,
    record: dict[str, Any],
) -> dict[str, object]:
    question = _extract_question(record)
    answer = _extract_answer(record)
    images = _resolve_images(record, json_path, sft_root=sft_root)
    if not question:
        raise ValueError(
            f"Could not find a human question for {subset_name} index={source_index}"
        )
    if not images:
        raise ValueError(
            f"Could not find image paths for {subset_name} index={source_index}"
        )

    case_id = f"{subset_name}_{source_index:06d}"
    return {
        "data_id": case_id,
        "question_id": case_id,
        "sample_id": case_id,
        "category": subset_name,
        "data_source": "OpenSearch-VL-SFT-36K",
        "question": question,
        "prompt": [{"content": question}],
        "images": images,
        "answer": answer,
        "source_metadata": _source_metadata(
            subset_name=subset_name,
            source_index=source_index,
            json_path=json_path,
            record=record,
        ),
    }


def _write_parquet(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
    except ModuleNotFoundError:
        pd = None

    if pd is not None:
        df = pd.DataFrame(rows)
        buffer = io.BytesIO()
        try:
            df.to_parquet(buffer, index=False)
        except Exception as exc:
            raise RuntimeError(
                "Failed to write parquet. Make sure pyarrow or fastparquet is installed."
            ) from exc
        buffer.seek(0)
        output_path.write_bytes(buffer.getvalue())
        return

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Writing parquet requires either pandas or pyarrow to be installed."
        ) from exc

    table = pa.Table.from_pylist(rows)
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    buffer.seek(0)
    output_path.write_bytes(buffer.getvalue())


def _write_manifest(payload: dict[str, Any], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _resolve_subset_json_path(
    subset_name: str,
    sft_root: Path,
) -> Path:
    spec = SUBSET_SPECS[subset_name]
    json_path = (sft_root / spec["json_relpath"]).resolve()
    if json_path.exists():
        return json_path
    raise FileNotFoundError(
        f"Subset JSON not found for {subset_name}: {json_path}. "
        "Pass --sft-root to the extracted Search-VL-SFT-36K directory."
    )


def _parse_subsets(raw: str) -> list[str]:
    requested = [item.strip() for item in raw.split(",") if item.strip()]
    if not requested:
        raise ValueError("No subsets were requested.")
    invalid = [item for item in requested if item not in SUBSET_SPECS]
    if invalid:
        raise ValueError(
            f"Unknown subsets: {', '.join(invalid)}. "
            f"Expected one of: {', '.join(SUBSET_SPECS)}"
        )
    return requested


def build_subset_rows(
    *,
    subset_name: str,
    json_path: Path,
    sft_root: Path,
    sample_size: int,
    seed: int,
) -> tuple[list[dict[str, object]], dict[str, Any]]:
    records = _load_json_array(json_path)
    total = len(records)
    if total == 0:
        raise ValueError(f"Subset {subset_name} is empty: {json_path}")

    rng = random.Random(f"{seed}:{subset_name}")
    sample_count = min(sample_size, total)
    chosen_indices = sorted(rng.sample(range(total), sample_count))

    rows: list[dict[str, object]] = []
    for source_index in chosen_indices:
        row = _record_to_row(
            subset_name=subset_name,
            source_index=source_index,
            json_path=json_path,
            sft_root=sft_root,
            record=records[source_index],
        )
        rows.append(row)

    manifest = {
        "subset": subset_name,
        "source_json": str(json_path),
        "source_size": total,
        "sample_size": sample_count,
        "requested_sample_size": sample_size,
        "seed": seed,
        "sampled_indices": chosen_indices,
    }
    return rows, manifest


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sft-root",
        type=str,
        default="SFT/data",
        help=(
            "Root directory containing the extracted Search-VL-SFT-36K subsets. "
            "Defaults to SFT/data."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory where per-subset parquet files will be written.",
    )
    parser.add_argument(
        "--subsets",
        type=str,
        default=",".join(SUBSET_SPECS.keys()),
        help="Comma-separated subset names to sample.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Maximum number of examples to sample from each subset.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible subset sampling.",
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    subsets = _parse_subsets(args.subsets)
    sft_root = Path(args.sft_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()

    for subset_name in subsets:
        json_path = _resolve_subset_json_path(subset_name, sft_root)
        rows, manifest = build_subset_rows(
            subset_name=subset_name,
            json_path=json_path,
            sft_root=sft_root,
            sample_size=args.sample_size,
            seed=args.seed,
        )

        subset_dir = output_root / subset_name
        parquet_path = subset_dir / f"{subset_name}_sample_{len(rows)}.parquet"
        manifest_path = subset_dir / "manifest.json"

        _write_parquet(rows, parquet_path)
        _write_manifest(manifest, manifest_path)
        print(
            f"[{subset_name}] wrote {len(rows)} rows "
            f"(from {manifest['source_size']}) -> {parquet_path}"
        )


if __name__ == "__main__":
    main()
