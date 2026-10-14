"""Direct, label-safe LiveVQA training sampling and image materialization.

Unlike FVQA hard-case mining, this module deliberately does *not* invoke a
base model.  It samples Level 1 and Level 2 training questions reproducibly,
deduplicates only on downloaded image SHA256, and hands the resulting public
manifest directly to the existing teacher queue.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import imghdr
import json
from pathlib import Path
import random
import re
import tempfile
import threading
from typing import Any, Iterable
from urllib.parse import quote
from urllib.request import Request, urlopen
import zipfile

from .cases import content_hash, load_cases, write_public_cases
from .io import append_jsonl, write_json
from .schema import CanonicalCase


DEFAULT_IMAGE_BASE_URL = "https://huggingface.co/datasets/ONE-Lab/LiveVQA-new/resolve/main/"
DEFAULT_DATASET_REPO = "ONE-Lab/LiveVQA-new"


@dataclass(frozen=True)
class _Candidate:
    case_id: str
    question: str
    acceptable_answers: tuple[str, ...]
    image_location: str
    level: int
    source_index: int
    source: str
    topic: str
    original_image_path: str


class _ArchiveImageResolver:
    """Lazily stage LiveVQA's source ZIPs locally and read selected members.

    The public repository stores source pixels only in ``news.zip``,
    ``youtube_images.zip`` and ``paper_images.zip``. It does *not* expose
    ``img_path`` entries as individual Hub files. Archive staging is therefore
    kept on local disk, outside the shared/HDFS result directory.
    """

    def __init__(self, *, cache_dir: str | Path, dataset_repo: str = DEFAULT_DATASET_REPO) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dataset_repo = dataset_repo
        self._paths: dict[str, Path] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @staticmethod
    def archive_name(candidate: _Candidate) -> str | None:
        path = candidate.original_image_path.replace("\\", "/").lstrip("/").casefold()
        if path.startswith("paper_images/"):
            return "paper_images.zip"
        if path.startswith("youtube_images/"):
            return "youtube_images.zip"
        if path.startswith(("news_images/", "news/")):
            return "news.zip"
        source = candidate.source.casefold()
        if "arxiv" in source or "paper" in source:
            return "paper_images.zip"
        if "youtube" in source or "video" in source:
            return "youtube_images.zip"
        if "news" in source or "article" in source:
            return "news.zip"
        return None

    def _lock_for(self, archive_name: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks.setdefault(archive_name, threading.Lock())

    def _archive_path(self, archive_name: str) -> Path:
        existing = self._paths.get(archive_name)
        if existing is not None and existing.is_file():
            return existing
        with self._lock_for(archive_name):
            existing = self._paths.get(archive_name)
            if existing is not None and existing.is_file():
                return existing
            # This stable local name also supports manually prefetched ZIPs;
            # useful on clusters whose Hub proxy needs a separate download job.
            prefetched = self.cache_dir / archive_name
            if prefetched.is_file():
                self._paths[archive_name] = prefetched
                return prefetched
            try:
                from huggingface_hub import hf_hub_download
            except ImportError as error:  # pragma: no cover - environment-specific
                raise RuntimeError("LiveVQA archive extraction requires huggingface_hub") from error
            print(f"[fix2-livevqa] staging {archive_name} under {self.cache_dir}", flush=True)
            downloaded = Path(
                hf_hub_download(
                    repo_id=self.dataset_repo,
                    repo_type="dataset",
                    filename=archive_name,
                    cache_dir=str(self.cache_dir),
                )
            )
            self._paths[archive_name] = downloaded
            return downloaded

    def read(self, candidate: _Candidate) -> bytes | None:
        archive_name = self.archive_name(candidate)
        member = candidate.original_image_path.replace("\\", "/").lstrip("/")
        if not archive_name or not member:
            return None
        archive = self._archive_path(archive_name)
        try:
            with zipfile.ZipFile(archive) as handle:
                # Current release members are stored at img_path. The fallback
                # covers archives packaged with one leading directory.
                choices = (member, member.lstrip("/"), f"{archive.stem}/{member}")
                for name in dict.fromkeys(choices):
                    try:
                        with handle.open(name) as image_handle:
                            return image_handle.read()
                    except KeyError:
                        continue
        except zipfile.BadZipFile as error:
            raise RuntimeError(f"LiveVQA archive is not a valid ZIP: {archive}") from error
        raise FileNotFoundError(f"Image member {member!r} is absent from {archive.name}")


def _as_object(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, bytearray, dict, list, tuple)):
        value = value.tolist()
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def _nonempty_strings(value: Any) -> tuple[str, ...]:
    value = _as_object(value)
    if value is None:
        return ()
    if isinstance(value, (list, tuple)):
        values: list[str] = []
        for item in value:
            values.extend(_nonempty_strings(item))
        return tuple(dict.fromkeys(values))
    if isinstance(value, dict):
        for key in ("answer", "text", "value", "label"):
            if key in value:
                return _nonempty_strings(value[key])
        return ()
    text = str(value).strip()
    return (text,) if text else ()


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = _as_object(row.get("source_metadata"))
    return dict(value) if isinstance(value, dict) else {}


def _level(value: Any) -> int | None:
    match = re.search(r"(?:level\s*)?([12])", str(value or ""), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _first_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def _image_locations(row: dict[str, Any], *, image_base_url: str) -> list[str]:
    """Resolve LiveVQA's benchmark rows and raw Hugging Face rows uniformly."""

    locations: list[str] = []

    def add(value: Any) -> None:
        value = _as_object(value)
        if isinstance(value, dict):
            for key in ("url", "image_url", "path", "image_path", "img_path"):
                if value.get(key):
                    add(value[key])
                    return
            return
        if isinstance(value, (list, tuple)):
            for item in value:
                add(item)
            return
        text = str(value or "").strip()
        if text and text not in locations:
            locations.append(text)

    for key in ("images", "image", "image_url"):
        add(row.get(key))
    for key in ("img_path", "image_path"):
        path = str(row.get(key) or "").strip()
        if not path:
            continue
        if path.startswith(("https://", "http://")) or Path(path).is_file():
            add(path)
        else:
            add(image_base_url.rstrip("/") + "/" + quote(path.lstrip("/"), safe="/"))
    return locations


def _candidate_from_row(
    row: dict[str, Any],
    *,
    source_index: int,
    image_base_url: str,
) -> _Candidate | None:
    level = _level(row.get("level"))
    if level not in {1, 2}:
        return None
    question = _first_text(row, ("question", "Question", "query"))
    answers: tuple[str, ...] = ()
    for key in ("ground_truth_list", "Ground_Truth_List", "Ground_Truth", "answer"):
        answers = _nonempty_strings(row.get(key))
        if answers:
            break
    raw_id = _first_text(row, ("data_id", "id", "question_id", "sample_id"))
    if not raw_id:
        raw_id = f"livevqa_train_{source_index:06d}"
    case_id = raw_id if raw_id.startswith("livevqa_") else f"livevqa_{raw_id}"
    locations = _image_locations(row, image_base_url=image_base_url)
    if not question or not answers or not locations:
        return None
    raw_metadata = _metadata(row)
    return _Candidate(
        case_id=case_id,
        question=question,
        acceptable_answers=answers,
        image_location=locations[0],
        level=level,
        source_index=source_index,
        source=_first_text(row, ("source",)) or str(raw_metadata.get("source") or "unknown"),
        topic=_first_text(row, ("topic", "category")) or str(raw_metadata.get("topic") or "unknown"),
        original_image_path=_first_text(row, ("img_path", "image_path")),
    )


def _suffix(data: bytes, location: str) -> str:
    kind = imghdr.what(None, h=data)
    by_kind = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "gif": ".gif", "bmp": ".bmp"}
    if kind in by_kind:
        return by_kind[kind]
    lowered = location.split("?", 1)[0].lower()
    for extension in (".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"):
        if lowered.endswith(extension):
            return ".jpg" if extension == ".jpeg" else extension
    return ".png"


def _read_image(location: str, *, timeout_s: float, max_bytes: int) -> bytes:
    local = Path(location)
    if local.is_file():
        data = local.read_bytes()
        if len(data) > max_bytes:
            raise ValueError(f"local image exceeds max_bytes={max_bytes}")
        return data
    request = Request(location, headers={"User-Agent": "OpenSearch-VL-LiveVQA-materializer/1.0"})
    with urlopen(request, timeout=timeout_s) as response:
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = response.read(min(1024 * 1024, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"downloaded image exceeds max_bytes={max_bytes}")
            chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise ValueError("downloaded empty image")
    return data


def _download_candidate(
    candidate: _Candidate,
    *,
    timeout_s: float,
    max_bytes: int,
    archive_resolver: _ArchiveImageResolver | None,
) -> tuple[_Candidate, bytes, str, str]:
    data = archive_resolver.read(candidate) if archive_resolver is not None else None
    if data is None:
        data = _read_image(candidate.image_location, timeout_s=timeout_s, max_bytes=max_bytes)
    if len(data) > max_bytes:
        raise ValueError(f"source image exceeds max_bytes={max_bytes}")
    digest = content_hash(data)
    return candidate, data, digest, _suffix(data, candidate.image_location)


def _stable_permutation(values: list[_Candidate], *, seed: int) -> list[_Candidate]:
    """Use an explicit PRNG permutation rather than filesystem/dataset order."""

    shuffled = list(values)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def prepare_livevqa_from_rows(
    rows: Iterable[dict[str, Any]],
    *,
    output_cases: str | Path,
    output_public_cases: str | Path,
    image_dir: str | Path,
    level1_count: int,
    seed: int,
    image_base_url: str = DEFAULT_IMAGE_BASE_URL,
    download_workers: int = 16,
    download_timeout_s: float = 45.0,
    max_image_bytes: int = 32 * 1024 * 1024,
    archive_cache_dir: str | Path | None = None,
    dataset_repo: str = DEFAULT_DATASET_REPO,
    resume: bool = False,
    progress: bool = False,
    selection_report: str | Path | None = None,
) -> dict[str, int]:
    """Randomly sample LiveVQA train with 1:3 L1:L2 and exact-image dedup.

    A Level 1 sample is selected first, followed by three Level 2 samples per
    Level 1 sample. Within each level the ordering is a seeded random
    permutation. SHA256 is intentionally the *only* content filter; no base
    model, topic/source rule, difficulty heuristic, or answer-based selection
    participates in sampling.
    """

    if level1_count < 1:
        raise ValueError("level1_count must be positive")
    if download_workers < 1:
        raise ValueError("download_workers must be positive")
    if download_timeout_s <= 0 or max_image_bytes < 1:
        raise ValueError("download timeout and max image bytes must be positive")
    private_path = Path(output_cases)
    public_path = Path(output_public_cases)
    if private_path.exists() and not resume:
        raise FileExistsError(f"Private manifest exists; use --resume: {private_path}")
    if public_path.exists() and not resume:
        raise FileExistsError(f"Public manifest exists; use --resume: {public_path}")

    candidates_by_level: dict[int, list[_Candidate]] = {1: [], 2: []}
    seen_ids: set[str] = set()
    counts = {
        "input": 0,
        "eligible": 0,
        "eligible_level1": 0,
        "eligible_level2": 0,
        "invalid_or_missing": 0,
        "duplicate_case_id": 0,
        "download_error": 0,
        "duplicate_image_sha256": 0,
        "skipped_resume": 0,
        "written_level1": 0,
        "written_level2": 0,
    }
    for source_index, raw in enumerate(rows):
        counts["input"] += 1
        candidate = _candidate_from_row(dict(raw), source_index=source_index, image_base_url=image_base_url)
        if candidate is None:
            counts["invalid_or_missing"] += 1
            continue
        if candidate.case_id in seen_ids:
            counts["duplicate_case_id"] += 1
            continue
        seen_ids.add(candidate.case_id)
        candidates_by_level[candidate.level].append(candidate)
        counts["eligible"] += 1
        counts[f"eligible_level{candidate.level}"] += 1

    if not candidates_by_level[1] or not candidates_by_level[2]:
        raise ValueError("LiveVQA source has no eligible candidates for one or both required levels")

    existing = load_cases(private_path) if private_path.exists() else []
    existing_ids = {case.case_id for case in existing}
    seen_hashes: set[str] = set()
    selected = {1: 0, 2: 0}
    for case in existing:
        if case.source_dataset != "livevqa_train":
            raise ValueError(f"Refusing to resume a non-LiveVQA manifest: {private_path}")
        metadata = case.source_metadata
        digest = str(metadata.get("source_image_sha256") or "").strip()
        level = _level(metadata.get("livevqa_level"))
        if not digest or level not in {1, 2}:
            raise ValueError(f"Existing LiveVQA case is missing hash/level resume metadata: {case.case_id}")
        prior_seed = metadata.get("sampling_seed")
        try:
            seed_matches = int(prior_seed) == seed
        except (TypeError, ValueError):
            seed_matches = False
        if not seed_matches:
            raise ValueError(
                "LiveVQA expansion requires the same SAMPLE_SEED as the existing manifest; "
                f"case {case.case_id} has sampling_seed={prior_seed!r}, requested={seed}"
            )
        seen_hashes.add(digest)
        selected[level] += 1

    targets = {1: level1_count, 2: level1_count * 3}
    for level in (1, 2):
        if selected[level] > targets[level]:
            raise ValueError(f"Resume manifest already has {selected[level]} Level {level} cases, above target {targets[level]}")

    image_root = Path(image_dir)
    image_root.mkdir(parents=True, exist_ok=True)
    archive_root = (
        Path(archive_cache_dir)
        if archive_cache_dir is not None
        else Path(tempfile.gettempdir()) / "fix2_livevqa_archives"
    )
    archive_resolver = _ArchiveImageResolver(cache_dir=archive_root, dataset_repo=dataset_repo)
    progress_bar: Any | None = None
    if progress:
        try:
            from tqdm.auto import tqdm

            progress_bar = tqdm(
                total=sum(targets.values()),
                initial=sum(selected.values()),
                desc="Sampling LiveVQA images",
                unit="image",
                dynamic_ncols=True,
            )
        except ImportError:  # pragma: no cover - optional UI dependency
            print("[fix2-livevqa] tqdm is unavailable; continuing without a progress bar", flush=True)

    def append_case(candidate: _Candidate, data: bytes, digest: str, suffix: str) -> None:
        path = image_root / f"{digest}{suffix}"
        if not path.exists():
            path.write_bytes(data)
        case = CanonicalCase(
            case_id=candidate.case_id,
            question=candidate.question,
            gold_answer=candidate.acceptable_answers[0],
            image_path=str(path.resolve()),
            split="train",
            source_dataset="livevqa_train",
            source_index=candidate.source_index,
            source_metadata={
                "source_image_sha256": digest,
                "livevqa_level": candidate.level,
                "livevqa_source": candidate.source,
                "livevqa_topic": candidate.topic,
                "livevqa_img_path": candidate.original_image_path,
                "sampling": "seeded_random_by_level_sha256_dedup",
                "sampling_seed": seed,
            },
            acceptable_answers=candidate.acceptable_answers,
        )
        append_jsonl(private_path, case.to_dict())
        existing_ids.add(candidate.case_id)
        seen_hashes.add(digest)
        selected[candidate.level] += 1
        counts[f"written_level{candidate.level}"] += 1
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix(level1=selected[1], level2=selected[2], sha_dup=counts["duplicate_image_sha256"])

    try:
        with ThreadPoolExecutor(max_workers=download_workers, thread_name_prefix="fix2-livevqa-download") as pool:
            # L1 is intentionally materialized first. It guarantees the exact
            # 1:3 final candidate ratio while preserving random order within
            # each level and ensuring a cross-level duplicate only survives once.
            for level in (1, 2):
                shuffled = _stable_permutation(candidates_by_level[level], seed=seed + (0 if level == 1 else 1))
                cursor = 0
                while selected[level] < targets[level] and cursor < len(shuffled):
                    batch = shuffled[cursor : cursor + download_workers * 2]
                    cursor += len(batch)
                    futures = [
                        pool.submit(
                            _download_candidate,
                            candidate,
                            timeout_s=download_timeout_s,
                            max_bytes=max_image_bytes,
                            archive_resolver=archive_resolver,
                        )
                        for candidate in batch
                        if candidate.case_id not in existing_ids
                    ]
                    # Resolve in seeded candidate order, never completion order.
                    for future in futures:
                        try:
                            candidate, data, digest, suffix = future.result()
                        except Exception as error:
                            counts["download_error"] += 1
                            print(f"[fix2-livevqa] download failed: {error.__class__.__name__}: {error}", flush=True)
                            continue
                        if digest in seen_hashes:
                            counts["duplicate_image_sha256"] += 1
                            continue
                        if selected[level] >= targets[level]:
                            break
                        append_case(candidate, data, digest, suffix)
                if selected[level] < targets[level]:
                    raise ValueError(
                        f"Only {selected[level]} unique downloaded Level {level} images are available; "
                        f"need {targets[level]}. See download_error/duplicate_image_sha256 in the report."
                    )
    finally:
        if progress_bar is not None:
            progress_bar.close()

    public_count = write_public_cases(load_cases(private_path), public_path, resume=resume)
    counts["public_cases_written"] = public_count
    counts["written"] = counts["written_level1"] + counts["written_level2"]
    counts["selected_level1"] = selected[1]
    counts["selected_level2"] = selected[2]
    if selection_report is not None:
        write_json(
            selection_report,
            {
                "source_dataset": "ONE-Lab/LiveVQA-new",
                "source_split": "train",
                "base_model_filter_used": False,
                "archive_cache_dir": str(archive_root),
                "selection": {
                    "level1_count": targets[1],
                    "level2_count": targets[2],
                    "ratio_level1_to_level2": "1:3",
                    "seed": seed,
                    "deduplication": "exact downloaded image SHA256 only",
                },
                "counts": counts,
            },
        )
    return counts


def prepare_livevqa_from_parquet(input_parquet: str | Path, **kwargs: Any) -> dict[str, int]:
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - environment-specific
        raise RuntimeError("LiveVQA preparation requires pandas plus a Parquet engine") from error
    frame = pd.read_parquet(input_parquet)
    return prepare_livevqa_from_rows(frame.to_dict(orient="records"), **kwargs)
