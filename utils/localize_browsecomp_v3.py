#!/usr/bin/env python3
"""Download BrowseComp-V3 images and rewrite the parquet to local paths.

The original benchmark stores image URLs in the ``images`` column.  This
utility creates a self-contained benchmark directory containing the original
metadata plus an ``images/`` directory, and replaces each image URL with an
absolute local path that OpenSearch-VL can inline into API requests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from PIL import Image


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if hasattr(value, "tolist"):
        value = value.tolist()
        return value if isinstance(value, list) else [value]
    return [value]


def _extract_url(entry: Any) -> str | None:
    if isinstance(entry, dict):
        value = entry.get("url") or entry.get("image_url") or entry.get("cos_url")
        return str(value) if value else None
    if isinstance(entry, str):
        return entry
    return None


def _filename(url: str, ordinal: int) -> str:
    parsed = urlparse(url)
    basename = Path(parsed.path).name or f"image_{ordinal:04d}.jpg"
    # Avoid collisions between same-named files from different URLs.
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()[:12]
    stem = Path(basename).stem or f"image_{ordinal:04d}"
    suffix = Path(basename).suffix.lower() or ".jpg"
    return f"{stem}__{digest}{suffix}"


def _download_one(
    url: str,
    destination: Path,
    timeout: float,
    retries: int,
) -> tuple[str, str | None, str | None]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        try:
            with Image.open(destination) as image:
                image.verify()
            return url, str(destination), None
        except Exception:
            destination.unlink(missing_ok=True)

    last_error = "unknown download error"
    for attempt in range(1, retries + 1):
        temporary = destination.with_suffix(destination.suffix + ".part")
        temporary.unlink(missing_ok=True)
        try:
            response = requests.get(
                url,
                timeout=(15, timeout),
                headers={"User-Agent": "browsecomp-v3-localizer/1.0"},
                allow_redirects=True,
            )
            response.raise_for_status()
            temporary.write_bytes(response.content)
            with Image.open(temporary) as image:
                image.verify()
            temporary.replace(destination)
            return url, str(destination), None
        except Exception as exc:  # retry transient CDN/network failures
            last_error = f"attempt {attempt}/{retries}: {type(exc).__name__}: {exc}"
            temporary.unlink(missing_ok=True)
    return url, None, last_error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Source parquet")
    parser.add_argument("--output-dir", required=True, help="Output benchmark directory")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_dir = Path(args.output_dir).resolve()
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(input_path)
    url_to_path: dict[str, Path] = {}
    for row_images in frame["images"]:
        for entry in _as_list(row_images):
            url = _extract_url(entry)
            if url and url.startswith(("http://", "https://")):
                url_to_path.setdefault(url, image_dir / _filename(url, len(url_to_path) + 1))

    print(f"rows={len(frame)} unique_image_urls={len(url_to_path)}")
    results: dict[str, str] = {}
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(_download_one, url, path, args.timeout, args.retries): url
            for url, path in url_to_path.items()
        }
        for index, future in enumerate(as_completed(futures), start=1):
            url, local_path, error = future.result()
            if local_path:
                results[url] = local_path
            else:
                failures[url] = error or "unknown error"
            if index % 25 == 0 or index == len(futures):
                print(f"downloaded_or_failed={index}/{len(futures)} ok={len(results)} failed={len(failures)}", flush=True)

    if failures:
        failure_path = output_dir / "download_failures.json"
        failure_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise RuntimeError(
            f"{len(failures)} images failed to download; details written to {failure_path}"
        )

    localized_rows = []
    for row_images in frame["images"]:
        localized = []
        for entry in _as_list(row_images):
            url = _extract_url(entry)
            if not url:
                continue
            if url.startswith(("http://", "https://")):
                local_path = results[url]
            else:
                local_path = url
            localized.append({"url": local_path})
        localized_rows.append(localized)
    frame["images"] = localized_rows

    output_parquet = output_dir / input_path.name
    # pyarrow can create an empty file but fail to stream a parquet writer
    # directly through the HDFS FUSE mount.  Stage locally, then move the
    # completed file to the benchmark directory.
    staging_dir = Path(tempfile.mkdtemp(prefix="browsecomp_v3_parquet_"))
    staged_parquet = staging_dir / output_parquet.name
    try:
        frame.to_parquet(staged_parquet, index=False)
        shutil.move(str(staged_parquet), str(output_parquet))
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)
    manifest = {
        "source_parquet": str(input_path),
        "output_parquet": str(output_parquet),
        "rows": len(frame),
        "unique_image_urls": len(url_to_path),
        "downloaded_images": len(results),
        "images_dir": str(image_dir),
    }
    (output_dir / "localization_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
