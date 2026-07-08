#!/usr/bin/env python3
"""Build a unified OpenSearch-VL dataset from benchmark-specific sources.

Example:
    python scripts/build_benchmark_dataset.py --benchmark mmsearch
    python scripts/build_benchmark_dataset.py --benchmark mmsearch_plus --split train
    python scripts/build_benchmark_dataset.py --benchmark vdr_bench_testmini --output data/benchmarks/vdr/testmini.parquet
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Optional


def _normalize_image_reference(image_url: str) -> str:
    """Convert local absolute paths to ``file://`` URLs for API backends."""

    if image_url.startswith(("http://", "https://", "data:", "file://")):
        return image_url

    candidate = Path(image_url).expanduser()
    if candidate.is_absolute():
        return candidate.resolve().as_uri()
    return image_url


def _optional_numpy() -> Any:
    try:
        import numpy as np  # type: ignore
    except ModuleNotFoundError:
        return None
    return np


def _maybe_decode_base64_image(value: str) -> Optional[bytes]:
    stripped = value.strip()
    if not stripped:
        return None
    if stripped.startswith(("http://", "https://", "file://", "data:")):
        return None
    if "\n" in stripped or "\r" in stripped:
        stripped = "".join(stripped.split())
    try:
        decoded = base64.b64decode(stripped, validate=True)
    except (binascii.Error, ValueError):
        return None

    if decoded.startswith(
        (
            b"\xff\xd8\xff",  # jpeg
            b"\x89PNG\r\n\x1a\n",  # png
            b"GIF87a",
            b"GIF89a",
            b"RIFF",  # webp container, validated below
        )
    ):
        if decoded.startswith(b"RIFF") and decoded[8:12] != b"WEBP":
            return None
        return decoded
    return None


def _numeric_sequence_to_bytes(value: Any) -> Optional[bytes]:
    if not isinstance(value, list) or not value:
        return None
    if not all(isinstance(item, int) and 0 <= item <= 255 for item in value):
        return None
    return bytes(value)


def _numpy_array_to_bytes(value: Any) -> Optional[bytes]:
    np = _optional_numpy()
    if np is None or not isinstance(value, np.ndarray):
        return None

    if value.ndim == 1:
        try:
            return bytes(value.astype("uint8").tolist())
        except Exception as exc:
            raise TypeError(f"Failed to convert 1D numpy image buffer: {exc}") from exc

    if value.ndim in (2, 3):
        try:
            from PIL import Image
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Converting numpy pixel arrays to encoded images requires Pillow."
            ) from exc

        array = value
        if str(array.dtype) != "uint8":
            array = array.astype("uint8")
        image = Image.fromarray(array)
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    return None


def _write_parquet(rows: list[dict[str, object]], output_path: Path) -> None:
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Writing parquet requires pandas plus a parquet backend such as "
            "pyarrow or fastparquet."
        ) from exc

    df = pd.DataFrame(rows)
    buffer = io.BytesIO()
    try:
        df.to_parquet(buffer, index=False)
    except Exception as exc:
        raise RuntimeError(
            "Failed to write parquet. Make sure pyarrow or fastparquet is installed."
        ) from exc
    buffer.seek(0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(buffer.getvalue())


def _write_json(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _stringify_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(parts)
    return str(value).strip()


def _question_text(record: dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _first_non_empty(record: dict[str, Any], candidates: Iterable[str]) -> str:
    for key in candidates:
        value = record.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    # Handle PIL images or dataset image wrappers without importing PIL eagerly.
    if hasattr(value, "size") and hasattr(value, "mode"):
        size = getattr(value, "size", None)
        mode = getattr(value, "mode", None)
        return {"_type": "PIL.Image", "size": size, "mode": mode}

    return str(value)


def _image_bytes_from_object(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, str):
        decoded = _maybe_decode_base64_image(value)
        if decoded is not None:
            return decoded
        return base64.b64decode(value)
    numeric_bytes = _numeric_sequence_to_bytes(value)
    if numeric_bytes is not None:
        return numeric_bytes
    numpy_bytes = _numpy_array_to_bytes(value)
    if numpy_bytes is not None:
        return numpy_bytes
    if isinstance(value, dict):
        if isinstance(value.get("bytes"), (bytes, bytearray)):
            return bytes(value["bytes"])
        if isinstance(value.get("data"), (bytes, bytearray)):
            return bytes(value["data"])
        if isinstance(value.get("bytes"), str):
            return base64.b64decode(value["bytes"])
        if isinstance(value.get("data"), str):
            return base64.b64decode(value["data"])
    if hasattr(value, "save"):
        buffer = io.BytesIO()
        value.save(buffer, format="PNG")
        return buffer.getvalue()
    raise TypeError(f"Unsupported image payload type: {type(value)!r}")


def _image_entry_from_value(value: Any) -> Optional[dict[str, object]]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        decoded = _maybe_decode_base64_image(stripped)
        if decoded is not None:
            return {"bytes": decoded}
        return {"url": _normalize_image_reference(stripped)}
    if isinstance(value, dict):
        if "url" in value and str(value["url"]).strip():
            return {"url": _normalize_image_reference(str(value["url"]).strip())}
        if "path" in value and str(value["path"]).strip():
            return {"url": _normalize_image_reference(str(value["path"]).strip())}
        if any(key in value for key in ("bytes", "data")):
            return {"bytes": _image_bytes_from_object(value)}
    return {"bytes": _image_bytes_from_object(value)}


def _collect_images(record: dict[str, Any], image_keys: Iterable[str]) -> list[dict[str, object]]:
    images: list[dict[str, object]] = []
    seen: set[str] = set()
    for key in image_keys:
        if key not in record:
            continue
        value = record.get(key)
        if value is None:
            continue
        candidates = value if isinstance(value, list) else [value]
        for item in candidates:
            image_entry = _image_entry_from_value(item)
            if not image_entry:
                continue
            dedupe_key = json.dumps(_json_safe(image_entry), ensure_ascii=False, sort_keys=True)
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            images.append(image_entry)
    return images


@dataclass(frozen=True)
class BenchmarkAdapter:
    benchmark: str
    default_dataset: str
    default_split: str
    data_source: str
    question_keys: tuple[str, ...]
    answer_keys: tuple[str, ...]
    id_keys: tuple[str, ...]
    sample_id_keys: tuple[str, ...]
    category_keys: tuple[str, ...]
    image_keys: tuple[str, ...]
    dataset_split: str = "train"
    row_builder: Optional[Callable[[dict[str, Any], "BenchmarkAdapter"], dict[str, object]]] = None


def _default_row_builder(
    record: dict[str, Any],
    adapter: BenchmarkAdapter,
) -> dict[str, object]:
    question = _question_text(record, adapter.question_keys)
    answer = ""
    for key in adapter.answer_keys:
        if key in record:
            answer = _stringify_answer(record.get(key))
            if answer:
                break

    data_id = _first_non_empty(record, adapter.id_keys) or _first_non_empty(
        record, adapter.sample_id_keys
    )
    sample_id = _first_non_empty(record, adapter.sample_id_keys) or data_id
    question_id = _first_non_empty(record, ("question_id", "qid", "id")) or data_id
    category = _first_non_empty(record, adapter.category_keys) or "unknown"
    images = _collect_images(record, adapter.image_keys)

    source_record = _json_safe(record)
    row: dict[str, object] = {
        "data_id": data_id,
        "question_id": question_id,
        "sample_id": sample_id,
        "category": category,
        "data_source": adapter.data_source,
        "question": question,
        "prompt": [{"content": question}],
        "images": images,
        "answer": answer,
        "source_metadata": json.dumps(source_record, ensure_ascii=False),
    }
    return row


def _infer_vdr_category(sample_id: str) -> str:
    parts = sample_id.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def _build_vdr_row(record: dict[str, Any], adapter: BenchmarkAdapter) -> dict[str, object]:
    row = _default_row_builder(record, adapter)
    sample_id = str(record.get("id") or row["data_id"])
    row["data_id"] = sample_id
    row["question_id"] = str(record.get("question_id") or sample_id)
    row["sample_id"] = sample_id
    row["category"] = _infer_vdr_category(sample_id)
    return row


def _build_mmsearch_plus_row(
    record: dict[str, Any],
    adapter: BenchmarkAdapter,
) -> dict[str, object]:
    row = _default_row_builder(record, adapter)

    num_images = record.get("num_images")
    extra_images: list[dict[str, object]] = []
    if isinstance(num_images, int) and num_images > 0:
        for index in range(1, num_images + 1):
            image_key = f"img_{index}"
            if image_key not in record:
                continue
            image_entry = _image_entry_from_value(record.get(image_key))
            if image_entry:
                extra_images.append(image_entry)
    else:
        for key, value in record.items():
            if key.startswith("img_"):
                image_entry = _image_entry_from_value(value)
                if image_entry:
                    extra_images.append(image_entry)

    if extra_images:
        seen = {
            json.dumps(_json_safe(item), ensure_ascii=False, sort_keys=True)
            for item in row.get("images", [])
        }
        merged = list(row.get("images", []))
        for item in extra_images:
            token = json.dumps(_json_safe(item), ensure_ascii=False, sort_keys=True)
            if token in seen:
                continue
            seen.add(token)
            merged.append(item)
        row["images"] = merged

    return row


ADAPTERS: dict[str, BenchmarkAdapter] = {
    "mmsearch": BenchmarkAdapter(
        benchmark="mmsearch",
        default_dataset="CaraJ/MMSearch",
        default_split="end2end",
        dataset_split="train",
        data_source="MMSearch",
        question_keys=("question", "query", "prompt"),
        answer_keys=("answer", "answers", "gt_answer"),
        id_keys=("id", "question_id", "sample_id"),
        sample_id_keys=("sample_id", "id", "question_id"),
        category_keys=("category", "subfield", "field", "domain"),
        image_keys=("images", "image"),
    ),
    "mmsearch_plus": BenchmarkAdapter(
        benchmark="mmsearch_plus",
        default_dataset="Cie1/MMSearch-Plus",
        default_split="train",
        dataset_split="train",
        data_source="MMSearch-Plus",
        question_keys=("question", "query", "prompt"),
        answer_keys=("answer", "answers", "gt_answer", "acceptable_answers"),
        id_keys=("id", "question_id", "sample_id"),
        sample_id_keys=("sample_id", "id", "question_id"),
        category_keys=("category", "primary_category", "secondary_category", "domain"),
        image_keys=("images", "image"),
        row_builder=_build_mmsearch_plus_row,
    ),
    "vdr_bench": BenchmarkAdapter(
        benchmark="vdr_bench",
        default_dataset="Osilly/VDR-Bench",
        default_split="train",
        dataset_split="train",
        data_source="VDR-Bench",
        question_keys=("question", "query", "prompt"),
        answer_keys=("answer", "answers"),
        id_keys=("id", "question_id", "sample_id"),
        sample_id_keys=("sample_id", "id", "question_id"),
        category_keys=("category",),
        image_keys=("image", "images"),
        row_builder=_build_vdr_row,
    ),
    "vdr_bench_testmini": BenchmarkAdapter(
        benchmark="vdr_bench_testmini",
        default_dataset="Osilly/VDR-Bench-testmini",
        default_split="train",
        dataset_split="train",
        data_source="VDR-Bench-testmini",
        question_keys=("question", "query", "prompt"),
        answer_keys=("answer", "answers"),
        id_keys=("id", "question_id", "sample_id"),
        sample_id_keys=("sample_id", "id", "question_id"),
        category_keys=("category",),
        image_keys=("image", "images"),
        row_builder=_build_vdr_row,
    ),
}


def _load_dataset_records(
    dataset_name: str,
    split: str,
    dataset_split: str = "train",
    cache_dir: Optional[str] = None,
    benchmark: Optional[str] = None,
    mmsearch_plus_decrypt_script: Optional[str] = None,
    mmsearch_plus_canary: Optional[str] = None,
) -> list[dict[str, Any]]:
    from datasets import load_dataset

    kwargs: dict[str, Any] = {"split": dataset_split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if benchmark == "mmsearch":
        dataset = load_dataset(dataset_name, split, **kwargs)
    else:
        dataset = load_dataset(dataset_name, **kwargs)

    if benchmark == "mmsearch_plus":
        decrypt_script = (
            Path(mmsearch_plus_decrypt_script).expanduser().resolve()
            if mmsearch_plus_decrypt_script
            else None
        )
        if decrypt_script:
            dataset = _decrypt_mmsearch_plus_dataset(
                dataset=dataset,
                decrypt_script=decrypt_script,
                canary=mmsearch_plus_canary or "MMSearch-Plus",
            )

    return [dict(item) for item in dataset]


def _decrypt_mmsearch_plus_dataset(
    dataset: Any,
    decrypt_script: Path,
    canary: str,
) -> Any:
    if not decrypt_script.exists():
        raise FileNotFoundError(
            f"MMSearch-Plus decrypt script does not exist: {decrypt_script}"
        )

    spec = importlib.util.spec_from_file_location(
        "mmsearch_plus_decrypt_module",
        decrypt_script,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load decrypt script: {decrypt_script}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    decrypt_dataset = getattr(module, "decrypt_dataset", None)
    if decrypt_dataset is None:
        raise AttributeError(
            f"decrypt_dataset() not found in decrypt script: {decrypt_script}"
        )

    return decrypt_dataset(
        encrypted_dataset=dataset,
        canary=canary,
    )


def _build_rows(
    records: list[dict[str, Any]],
    adapter: BenchmarkAdapter,
    limit: int = 0,
) -> list[dict[str, object]]:
    builder = adapter.row_builder or _default_row_builder
    rows: list[dict[str, object]] = []
    for index, record in enumerate(records):
        if limit > 0 and index >= limit:
            break
        row = builder(record, adapter)
        if not row.get("question"):
            raise ValueError(
                f"Missing question text for benchmark={adapter.benchmark} "
                f"record index={index}"
            )
        rows.append(row)
    return rows


def _default_output_path(
    benchmark: str,
    split: str,
    output_format: str,
) -> Path:
    suffix = "json" if output_format == "json" else "parquet"
    return Path("data") / "benchmarks" / benchmark / f"{split}.{suffix}"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=sorted(ADAPTERS.keys()),
        help="Benchmark adapter to use.",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        help="Override the Hugging Face dataset name/path.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help=(
            "For MMSearch, this selects the dataset config "
            "(end2end/rerank/summarization). For other benchmarks, this is the "
            "dataset split to load."
        ),
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output dataset path. Defaults to "
            "data/benchmarks/<benchmark>/<split>.parquet"
        ),
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "json"],
        default="parquet",
        help="Dataset format to write.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optionally convert only the first N records.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache dir.",
    )
    parser.add_argument(
        "--mmsearch-plus-decrypt-script",
        default=None,
        help=(
            "Path to MMSearch-Plus decrypt_after_load.py. "
            "Needed when converting the encrypted HF dataset."
        ),
    )
    parser.add_argument(
        "--mmsearch-plus-canary",
        default="MMSearch-Plus",
        help=(
            "Canary string for MMSearch-Plus decryption. "
            "The dataset card hints this is the repo name without username."
        ),
    )
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    adapter = ADAPTERS[args.benchmark]
    dataset_name = args.dataset or adapter.default_dataset
    split = args.split or adapter.default_split
    dataset_split = adapter.dataset_split
    if args.benchmark != "mmsearch":
        dataset_split = split
    output_path = (
        Path(args.output).expanduser()
        if args.output
        else _default_output_path(args.benchmark, split, args.format)
    )

    records = _load_dataset_records(
        dataset_name=dataset_name,
        split=split,
        dataset_split=dataset_split,
        cache_dir=args.cache_dir,
        benchmark=args.benchmark,
        mmsearch_plus_decrypt_script=args.mmsearch_plus_decrypt_script,
        mmsearch_plus_canary=args.mmsearch_plus_canary,
    )
    rows = _build_rows(records, adapter, limit=args.limit)

    if args.format == "json":
        _write_json(rows, output_path)
    else:
        _write_parquet(rows, output_path)

    print(
        f"Wrote {len(rows)} rows to {output_path} "
        f"(benchmark={args.benchmark}, dataset={dataset_name}, split={split})"
    )


if __name__ == "__main__":
    main()
