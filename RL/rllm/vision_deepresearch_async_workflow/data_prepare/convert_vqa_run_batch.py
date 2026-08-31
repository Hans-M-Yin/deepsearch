#!/usr/bin/env python3
"""Convert ``synthesis.vqa.run_batch`` output to an rLLM input JSONL.

The VQA batch runner writes a directory containing, among other files,
``questions.jsonl`` and ``samples.jsonl``.  The question file is the compact
source of truth for RL training.  This converter normalizes it to the fields
used by :class:`DeepResearchWorkflow`::

    {
        "id": "q_000001",
        "question": "...",
        "answer": "...",
        "images": ["/absolute/path/to/image.png"],
        "extra_info": {...}
    }

The Registry subsequently places the whole record in Verl's runtime
``extra_info`` field.  Keeping a nested ``extra_info`` here gives us a stable
place for future per-example metadata without changing the required task
fields.

Record selection uses zero-based line indices from ``questions.jsonl``.  For
example, ``--offset 100 --limit 50`` converts records 100 through 149.  An
``--indices-file`` contains one zero-based index per non-empty line and takes
precedence as an explicit selection; it cannot be combined with a non-default
offset or limit.

By default remote ``image_url`` values are downloaded.  This is intentional:
the current RL workflow can open local paths and data URIs, but does not
download HTTP URLs while constructing the initial PIL image.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import sys
import time
from io import BytesIO
from pathlib import Path
from collections.abc import Iterable, Iterator
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image


_IMAGE_SUFFIXES = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tif",
}


def _iter_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            yield value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return list(_iter_jsonl(path))


def _safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or "image"


def _suffix_from_response(url: str, content_type: str | None) -> str:
    if content_type:
        media_type = content_type.split(";", 1)[0].strip().lower()
        if media_type in _IMAGE_SUFFIXES:
            return _IMAGE_SUFFIXES[media_type]

    suffix = Path(urlparse(url).path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tif", ".tiff"}:
        return ".jpg" if suffix == ".jpeg" else suffix

    guessed = mimetypes.guess_extension(content_type or "")
    return guessed if guessed else ".png"


def _verify_image(image_bytes: bytes, source: str) -> None:
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Downloaded content is not a valid image: {source}: {exc}") from exc


def _download_image(
    session: requests.Session,
    url: str,
    output_dir: Path,
    sample_id: str,
    timeout: float,
    retries: int,
) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    url_hash = hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]
    prefix = _safe_filename(sample_id)

    # The destination name is content-addressed by the source URL.  Reuse a
    # valid existing file so rerunning the converter (or resuming a failed
    # conversion) does not issue another remote request.
    for existing in sorted(output_dir.glob(f"{prefix}_{url_hash}.*")):
        if existing.name.endswith(".part") or not existing.is_file():
            continue
        try:
            with Image.open(existing) as image:
                image.verify()
            return str(existing.resolve())
        except Exception:
            continue

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = session.get(url, timeout=timeout)
            response.raise_for_status()
            image_bytes = response.content
            _verify_image(image_bytes, url)

            suffix = _suffix_from_response(url, response.headers.get("content-type"))
            destination = output_dir / f"{prefix}_{url_hash}{suffix}"
            temporary = destination.with_suffix(destination.suffix + ".part")
            temporary.write_bytes(image_bytes)
            temporary.replace(destination)
            return str(destination.resolve())
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                time.sleep(min(2.0**attempt, 8.0))

    raise RuntimeError(
        f"Failed to download image after {retries + 1} attempts: {url}: {last_error}"
    ) from last_error


def _resolve_image(
    image_value: Any,
    *,
    sample_id: str,
    image_mode: str,
    image_dir: Path,
    session: requests.Session,
    timeout: float,
    retries: int,
) -> list[str]:
    if not image_value:
        return []

    if isinstance(image_value, list):
        values = image_value
    else:
        values = [image_value]

    resolved: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            continue

        if image_mode == "none":
            continue
        if image_mode == "url" or value.startswith("data:image/"):
            resolved.append(value)
            continue

        if value.startswith(("http://", "https://")):
            resolved.append(
                _download_image(
                    session=session,
                    url=value,
                    output_dir=image_dir,
                    sample_id=sample_id,
                    timeout=timeout,
                    retries=retries,
                )
            )
            continue

        local_path = Path(value).expanduser()
        if not local_path.exists():
            raise FileNotFoundError(f"Image path does not exist: {value}")
        resolved.append(str(local_path.resolve()))

    return resolved


def _sample_metadata(
    samples: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Index only small, stable fields from the large samples.jsonl records."""
    indexed: dict[str, dict[str, Any]] = {}
    for sample in samples:
        sample_id = sample.get("sample_id")
        if not sample_id:
            continue
        indexed[str(sample_id)] = {
            "input_image_url": sample.get("input_image_url"),
            "status": sample.get("status"),
            "created_at": sample.get("created_at"),
            "updated_at": sample.get("updated_at"),
        }
    return indexed


def _load_indices_file(path: Path, num_records: int) -> list[int]:
    """Load and validate zero-based record indices, preserving file order."""
    selected: list[int] = []
    seen: set[int] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                index = int(line)
            except ValueError as exc:
                raise ValueError(
                    f"Invalid index at {path}:{line_number}: {line!r}; "
                    "expected one zero-based integer per line"
                ) from exc
            if index < 0 or index >= num_records:
                raise IndexError(
                    f"Index {index} at {path}:{line_number} is outside "
                    f"[0, {num_records})"
                )
            # Avoid accidentally duplicating a training sample when an index
            # is repeated in the selection file, while retaining file order.
            if index not in seen:
                selected.append(index)
                seen.add(index)
    return selected


def _select_indices(
    *,
    num_records: int,
    offset: int,
    limit: int | None,
    indices_file: Path | None,
) -> list[int]:
    if offset < 0:
        raise ValueError(f"offset must be non-negative, got {offset}")
    if limit is not None and limit < 0:
        raise ValueError(f"limit must be non-negative, got {limit}")

    if indices_file is not None:
        if offset != 0 or limit is not None:
            raise ValueError(
                "--indices-file cannot be combined with a non-default "
                "--offset or --limit"
            )
        if not indices_file.exists():
            raise FileNotFoundError(f"Indices file not found: {indices_file}")
        return _load_indices_file(indices_file, num_records)

    end = None if limit is None else offset + limit
    return list(range(offset, min(num_records, end) if end is not None else num_records))


def convert_run_batch(
    *,
    run_dir: Path,
    output_jsonl: Path,
    image_mode: str,
    image_dir: Path,
    system_prompt: str | None,
    include_samples: bool,
    timeout: float,
    retries: int,
    offset: int = 0,
    limit: int | None = None,
    indices_file: Path | None = None,
) -> tuple[int, int]:
    questions_path = run_dir / "questions.jsonl"
    samples_path = run_dir / "samples.jsonl"
    if not questions_path.exists():
        raise FileNotFoundError(f"questions.jsonl not found under {run_dir}")

    questions = _load_jsonl(questions_path)
    selected_indices = _select_indices(
        num_records=len(questions),
        offset=offset,
        limit=limit,
        indices_file=indices_file,
    )
    samples_by_id: dict[str, dict[str, Any]] = {}
    if include_samples and samples_path.exists():
        # Stream the potentially large samples.jsonl instead of retaining all
        # raw trajectories in memory.
        samples_by_id = _sample_metadata(_iter_jsonl(samples_path))

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if image_mode == "download":
        image_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "OpenSearch-VL-RL-data-preprocessor/1.0"})

    image_count = 0
    with output_jsonl.open("w", encoding="utf-8") as handle:
        for index in selected_indices:
            source = questions[index]
            question = (
                source.get("final_question")
                or source.get("question")
                or source.get("enhanced_question")
                or source.get("drafted_question")
            )
            answer = source.get("answer")
            if not isinstance(question, str) or not question.strip():
                raise ValueError(f"Question record {index} has no usable question")
            if answer is None:
                raise ValueError(f"Question record {index} has no answer")

            question_id = str(source.get("question_id") or f"q_{index + 1:06d}")
            sample_id = str(source.get("sample_id") or "")
            source_image_url = source.get("image_url")
            if not source_image_url and sample_id in samples_by_id:
                source_image_url = samples_by_id[sample_id].get("input_image_url")

            images = _resolve_image(
                source_image_url,
                sample_id=sample_id or question_id,
                image_mode=image_mode,
                image_dir=image_dir,
                session=session,
                timeout=timeout,
                retries=retries,
            )
            image_count += len(images)

            metadata: dict[str, Any] = {
                "source_run_dir": str(run_dir.resolve()),
                "source_index": index,
                "source_question_id": source.get("question_id"),
                "source_sample_id": source.get("sample_id"),
                "source_path_id": source.get("path_id"),
                "source_status": source.get("status"),
                "source_image_url": source_image_url,
                "drafted_question": source.get("drafted_question"),
                "enhanced_question": source.get("enhanced_question"),
                "final_question": source.get("final_question"),
                "image_mode": image_mode,
            }
            if sample_id in samples_by_id:
                metadata["sample_metadata"] = samples_by_id[sample_id]
            if system_prompt is not None:
                metadata["system_prompt"] = system_prompt

            record = {
                "id": question_id,
                "question": question.strip(),
                "answer": answer if isinstance(answer, str) else str(answer),
                "images": images,
                "extra_info": metadata,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return len(selected_indices), image_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a synthesis/vqa/run_batch directory to RL JSONL"
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Directory containing questions.jsonl (and optionally samples.jsonl)",
    )
    parser.add_argument(
        "--output_jsonl",
        type=Path,
        required=True,
        help="Output JSONL consumed by register_rl_dataset.py",
    )
    parser.add_argument(
        "--image_mode",
        choices=("download", "url", "none"),
        default="download",
        help="How to populate images; download is recommended for the current RL workflow",
    )
    parser.add_argument(
        "--image_dir",
        type=Path,
        default=None,
        help="Directory for downloaded images (default: <output_stem>_images)",
    )
    parser.add_argument(
        "--system_prompt_file",
        type=Path,
        default=None,
        help="Optional text file; its contents are stored in extra_info.system_prompt",
    )
    parser.add_argument(
        "--include_samples_metadata",
        action="store_true",
        help=(
            "Read samples.jsonl and retain a few small metadata fields; disabled "
            "by default because samples.jsonl can be very large"
        ),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Zero-based start index in questions.jsonl (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of records to convert after offset (default: all)",
    )
    parser.add_argument(
        "--indices-file",
        "--indices_file",
        dest="indices_file",
        type=Path,
        default=None,
        help=(
            "Text file with one zero-based questions.jsonl index per line; "
            "cannot be combined with non-default offset/limit"
        ),
    )
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--retries", type=int, default=3)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_jsonl = args.output_jsonl.expanduser().resolve()
    image_dir = (
        args.image_dir.expanduser().resolve()
        if args.image_dir is not None
        else output_jsonl.with_name(f"{output_jsonl.stem}_images")
    )

    system_prompt = None
    if args.system_prompt_file is not None:
        prompt_path = args.system_prompt_file.expanduser().resolve()
        system_prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not system_prompt:
            raise ValueError(f"System prompt file is empty: {prompt_path}")

    count, image_count = convert_run_batch(
        run_dir=run_dir,
        output_jsonl=output_jsonl,
        image_mode=args.image_mode,
        image_dir=image_dir,
        system_prompt=system_prompt,
        include_samples=args.include_samples_metadata,
        timeout=args.timeout,
        retries=args.retries,
        offset=args.offset,
        limit=args.limit,
        indices_file=(
            args.indices_file.expanduser().resolve()
            if args.indices_file is not None
            else None
        ),
    )
    selection = (
        f"indices_file={args.indices_file}"
        if args.indices_file is not None
        else f"offset={args.offset}, limit={args.limit}"
    )
    print(f"Converted {count} records to {output_jsonl} ({selection})")
    print(f"Resolved {image_count} image(s) using image_mode={args.image_mode}")
    if args.image_mode == "download":
        print(f"Downloaded images are under {image_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
