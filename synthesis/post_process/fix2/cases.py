"""FVQA source normalisation and label-safe case manifests."""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import imghdr
import json
from pathlib import Path
import re
from typing import Any, Iterable

from .io import append_jsonl, iter_json_records
from .schema import CanonicalCase, PublicCase


def normalized_question(question: str) -> str:
    return " ".join(str(question).casefold().split())


def content_hash(value: bytes | str) -> str:
    payload = value.encode("utf-8") if isinstance(value, str) else value
    return hashlib.sha256(payload).hexdigest()


def case_fingerprint(case: CanonicalCase) -> str:
    """Deduplicate by wording and the actual source image, not an arbitrary ID."""

    image = Path(case.image_path)
    image_digest = content_hash(image.read_bytes()) if image.exists() else case.image_path
    return content_hash(f"{normalized_question(case.question)}\n{image_digest}")


def _safe_filename_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-")
    return cleaned[:96] or "case"


def _image_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray)):
        return _image_bytes(value.tolist())
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"Expected exactly one FVQA input image, found {len(value)}")
        return _image_bytes(value[0])
    if isinstance(value, dict):
        for key in ("bytes", "data", "image", "binary"):
            if value.get(key) is not None:
                return _image_bytes(value[key])
    if isinstance(value, str) and value.startswith("data:image") and "," in value:
        import base64

        return base64.b64decode(value.split(",", 1)[1])
    raise ValueError(f"Cannot convert image value of type {type(value).__name__} to bytes")


def _image_suffix(data: bytes) -> str:
    kind = imghdr.what(None, h=data)
    return {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "gif": ".gif", "bmp": ".bmp"}.get(kind or "", ".png")


def _pick(row: dict[str, Any], field: str | None, candidates: tuple[str, ...]) -> Any:
    if field:
        if field not in row:
            raise ValueError(f"Required column {field!r} is absent; available={sorted(row)}")
        return row[field]
    for name in candidates:
        if name in row and row[name] not in (None, ""):
            return row[name]
    raise ValueError(f"None of {candidates!r} exists in row with columns {sorted(row)}")


def _looks_like_test_row(row: dict[str, Any]) -> bool:
    """Block the common train.parquet-is-actually-test split accident."""

    markers = []
    for key in ("data_id", "case_id", "question_id", "sample_id", "category", "data_source"):
        if row.get(key) is not None:
            markers.append(str(row[key]).casefold())
    metadata = row.get("source_metadata")
    if isinstance(metadata, str):
        markers.append(metadata.casefold())
    elif isinstance(metadata, dict):
        markers.append(json.dumps(metadata, ensure_ascii=False).casefold())
    return any("fvqa_test" in value or "/test" in value or "_test_" in value for value in markers)


def _decode_image_for_materialization(image_value: Any) -> tuple[bytes, str, str]:
    """Decode and hash one source image without touching shared manifests."""

    image_data = _image_bytes(image_value)
    return image_data, content_hash(image_data), _image_suffix(image_data)


def _write_image_if_missing(image_path: Path, image_data: bytes) -> None:
    """Write one already-deduplicated image target.

    This deliberately runs outside the manifest coordinator: each accepted
    case owns a unique target path, while the coordinator remains the only
    writer to the append/resume JSONL files.
    """

    if not image_path.exists():
        image_path.write_bytes(image_data)


def prepare_cases_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    output_jsonl: str | Path,
    image_dir: str | Path,
    split: str,
    question_field: str | None = None,
    answer_field: str | None = None,
    image_field: str | None = None,
    id_field: str | None = None,
    source_dataset: str = "fvqa",
    allow_nontrain: bool = False,
    progress: bool = False,
    start: int = 0,
    limit: int | None = None,
    resume: bool = False,
    materialize_workers: int = 1,
) -> dict[str, int]:
    """Materialize source images and write canonical cases.

    ``gold_answer`` remains in the manifest for later evaluation only.  The
    collector consumes :meth:`CanonicalCase.prompt_view`, which has no label.
    """

    if start < 0:
        raise ValueError("start must be nonnegative")
    if limit is not None and limit < 0:
        raise ValueError("limit must be nonnegative or None")
    if materialize_workers < 1:
        raise ValueError("materialize_workers must be positive")
    output = Path(output_jsonl)
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to append ambiguously to existing manifest: {output}")
    target_images = Path(image_dir)
    seen: set[str] = set()
    completed_ids: set[str] = set()
    if output.exists():
        for existing in iter_json_records(output):
            case_id = str(existing.get("case_id") or "").strip()
            if case_id:
                completed_ids.add(case_id)
            metadata = existing.get("source_metadata")
            if isinstance(metadata, dict) and str(metadata.get("case_fingerprint") or ""):
                seen.add(str(metadata["case_fingerprint"]))
    counts = {"input": 0, "written": 0, "duplicate": 0, "skipped_resume": 0}
    progress_bar: Any | None = None
    stop = None if limit is None or limit == 0 else start + limit

    def selected_rows() -> Iterable[tuple[int, dict[str, Any]]]:
        for row_index, raw in enumerate(rows):
            if row_index < start:
                continue
            if stop is not None and row_index >= stop:
                break
            yield row_index, dict(raw)

    total: int | None = None
    try:
        total_rows = len(rows)  # type: ignore[arg-type]
        total = max(0, total_rows - start)
        if limit is not None and limit > 0:
            total = min(total, limit)
    except TypeError:
        pass
    iterable: Iterable[tuple[int, dict[str, Any]]] = selected_rows()
    if progress:
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(total=total, desc="Materializing FVQA cases", unit="case", dynamic_ncols=True)
        except ImportError:  # pragma: no cover - tqdm is a runtime nicety
            print("[fix2-prepare] tqdm is unavailable; continuing without a progress bar", flush=True)

    target_images.mkdir(parents=True, exist_ok=True)
    next_sequence = 0
    next_to_resolve = 0
    next_to_append = 0
    decoded: dict[int, tuple[int, dict[str, Any], str, str, str, bytes, str, str]] = {}
    resolved: dict[int, CanonicalCase | None] = {}
    pending_decode: dict[Any, int] = {}
    pending_write: dict[Any, tuple[int, CanonicalCase]] = {}

    def append_ready() -> None:
        nonlocal next_to_append
        while next_to_append in resolved:
            case = resolved.pop(next_to_append)
            if case is not None:
                append_jsonl(output, case.to_dict())
                counts["written"] += 1
                completed_ids.add(case.case_id)
            next_to_append += 1

    def resolve_ready(write_pool: ThreadPoolExecutor | None) -> None:
        nonlocal next_to_resolve
        while next_to_resolve in decoded:
            row_index, row, case_id, question, answer, image_data, image_hash, image_suffix = decoded.pop(next_to_resolve)
            fingerprint = content_hash(f"{normalized_question(question)}\n{image_hash}")
            if fingerprint in seen:
                counts["duplicate"] += 1
                resolved[next_to_resolve] = None
                next_to_resolve += 1
                continue
            seen.add(fingerprint)
            image_path = target_images / f"{_safe_filename_component(case_id)}_{image_hash[:12]}{image_suffix}"
            metadata = {
                "source_image_sha256": image_hash,
                "source_columns": sorted(row),
                "case_fingerprint": fingerprint,
            }
            case = CanonicalCase(
                case_id=case_id,
                question=question,
                gold_answer=answer,
                image_path=str(image_path.resolve()),
                split=split,
                source_dataset=source_dataset,
                source_index=row_index,
                source_metadata=metadata,
            )
            if write_pool is None:
                _write_image_if_missing(image_path, image_data)
                resolved[next_to_resolve] = case
            else:
                pending_write[write_pool.submit(_write_image_if_missing, image_path, image_data)] = (next_to_resolve, case)
            next_to_resolve += 1
        append_ready()

    def collect_decoded(*, block: bool, write_pool: ThreadPoolExecutor | None) -> None:
        if not pending_decode:
            return
        if block:
            done, _ = wait(pending_decode, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in pending_decode if future.done()}
        for future in done:
            sequence = pending_decode.pop(future)
            decoded[sequence] = future.result()
        resolve_ready(write_pool)

    def collect_written(*, block: bool) -> None:
        if not pending_write:
            return
        if block:
            done, _ = wait(pending_write, return_when=FIRST_COMPLETED)
        else:
            done = {future for future in pending_write if future.done()}
        for future in done:
            sequence, case = pending_write.pop(future)
            future.result()
            resolved[sequence] = case
        append_ready()

    def candidate_payload(row_index: int, row: dict[str, Any], case_id: str, question: str, answer: str) -> tuple[int, dict[str, Any], str, str, str, bytes, str, str]:
        image_value = _pick(row, image_field, ("image", "images", "img", "image_bytes"))
        image_data, image_hash, image_suffix = _decode_image_for_materialization(image_value)
        return row_index, row, case_id, question, answer, image_data, image_hash, image_suffix

    try:
        # Image byte payloads are already resident in the parent (pandas has
        # decoded the Parquet rows). Threads avoid copying these often-large
        # buffers through multiprocessing pipes, while parallelizing hashing
        # and HDFS writes that dominate this stage.
        compute_pool: ThreadPoolExecutor | None = None
        write_pool: ThreadPoolExecutor | None = None
        if materialize_workers > 1:
            compute_pool = ThreadPoolExecutor(max_workers=materialize_workers, thread_name_prefix="fix2-image-decode")
            write_pool = ThreadPoolExecutor(max_workers=materialize_workers, thread_name_prefix="fix2-image-write")
        try:
            for row_index, row in iterable:
                counts["input"] += 1
                if progress_bar is not None:
                    progress_bar.update(1)
                if split == "train" and not allow_nontrain and _looks_like_test_row(row):
                    raise ValueError(
                        "Input row appears to belong to an FVQA test split. "
                        "Use a real source/train split; do not relabel a test row as train."
                    )
                question = str(_pick(row, question_field, ("question", "Question", "query"))).strip()
                answer = str(_pick(row, answer_field, ("answer", "Answer", "label"))).strip()
                if not question or not answer:
                    continue
                case_id_value = _pick(row, id_field, ("data_id", "case_id", "question_id", "id", "qid")) if id_field or any(k in row for k in ("data_id", "case_id", "question_id", "id", "qid")) else f"{split}_{row_index:06d}"
                case_id = str(case_id_value).strip() or f"{split}_{row_index:06d}"
                if case_id in completed_ids:
                    counts["skipped_resume"] += 1
                    continue
                sequence = next_sequence
                next_sequence += 1
                if compute_pool is None:
                    decoded[sequence] = candidate_payload(row_index, row, case_id, question, answer)
                    resolve_ready(None)
                else:
                    pending_decode[compute_pool.submit(candidate_payload, row_index, row, case_id, question, answer)] = sequence
                    if len(pending_decode) >= materialize_workers * 2:
                        collect_decoded(block=True, write_pool=write_pool)
                # Do not let a slower HDFS/FUSE writer retain every decoded
                # source image in memory while the parquet scan continues.
                # One worker-sized batch is enough to keep it saturated.
                if len(pending_write) >= materialize_workers * 2:
                    collect_written(block=True)
                collect_written(block=False)
                if progress_bar is not None and counts["input"] % 32 == 0:
                    progress_bar.set_postfix(
                        written=counts["written"],
                        duplicate=counts["duplicate"],
                        resumed=counts["skipped_resume"],
                    )
            while pending_decode:
                collect_decoded(block=True, write_pool=write_pool)
                collect_written(block=False)
            while pending_write:
                collect_written(block=True)
            append_ready()
        finally:
            if compute_pool is not None:
                compute_pool.shutdown(wait=True, cancel_futures=False)
            if write_pool is not None:
                write_pool.shutdown(wait=True, cancel_futures=False)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return counts


def prepare_cases_from_parquet(
    input_parquet: str | Path,
    **kwargs: Any,
) -> dict[str, int]:
    """Parquet adapter kept optional so deterministic filters have no pandas dependency."""

    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("prepare requires pandas plus a Parquet engine") from error
    frame = pd.read_parquet(input_parquet)
    return prepare_cases_from_rows(frame.to_dict(orient="records"), **kwargs)


def load_cases(path: str | Path) -> list[CanonicalCase]:
    from .io import iter_json_records

    return [CanonicalCase.from_dict(value) for value in iter_json_records(path)]


def write_public_cases(cases: Iterable[CanonicalCase], output_jsonl: str | Path, *, resume: bool = False) -> int:
    """Publish a writer-safe manifest with no answer or construction facts."""

    output = Path(output_jsonl)
    if output.exists() and not resume:
        raise FileExistsError(f"Refusing to overwrite public case manifest: {output}")
    existing_ids: set[str] = set()
    if output.exists():
        existing_ids = {case.case_id for case in load_public_cases(output)}
    count = 0
    for case in cases:
        if case.case_id in existing_ids:
            continue
        append_jsonl(output, case.public_case().to_dict())
        count += 1
        existing_ids.add(case.case_id)
    return count


def load_public_cases(path: str | Path) -> list[PublicCase]:
    from .io import iter_json_records

    return [PublicCase.from_dict(value) for value in iter_json_records(path)]


def private_case_map(path: str | Path) -> dict[str, CanonicalCase]:
    return {case.case_id: case for case in load_cases(path)}
