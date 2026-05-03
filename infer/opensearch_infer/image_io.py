"""Image download / decode / cache utilities used across the runtime."""

from __future__ import annotations

import base64
import logging
import os
import time
import urllib.parse
import urllib.request
from typing import Optional, Tuple, Union

import requests
from PIL import Image

from . import config


logger = logging.getLogger(__name__)


_BASE64_PREFIXES = ("iVBORw0KGgo", "/9j/4AAQ", "data:image")
_BASE64_ALPHABET = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
)


def looks_like_base64(value: object) -> bool:
    """Heuristically detect a base64 image payload.

    The original codebase routinely passed both file paths and base64
    strings through the same dictionaries; this helper centralises the
    detection logic so we don't accidentally treat a giant string as a
    filesystem path.
    """

    if not isinstance(value, str) or len(value) <= 500:
        return False
    if any(value.startswith(prefix) for prefix in _BASE64_PREFIXES):
        return True
    sample = value[:100]
    if all(ch in _BASE64_ALPHABET for ch in sample):
        try:
            base64.b64decode(sample)
            return True
        except Exception:
            return False
    return False


def image_to_base64(image_bytes: Union[bytes, str]) -> Optional[str]:
    """Encode raw image bytes as base64. Returns the input if already a string."""

    if isinstance(image_bytes, str):
        return image_bytes
    try:
        return base64.b64encode(image_bytes).decode("utf-8")
    except Exception as exc:
        logger.warning("image_to_base64 failed: %s", exc)
        return None


def detect_image_format(base64_image: str) -> str:
    """Return a MIME type by inspecting the image magic bytes."""

    try:
        decode_length = min(100, len(base64_image))
        head = base64.b64decode(base64_image[:decode_length])
    except Exception:
        return "image/jpeg"

    if len(head) >= 8 and head[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(head) >= 3 and head[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if len(head) >= 6 and head[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    if len(head) >= 12 and head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def download_image_bytes(
    url: str, timeout: int = 60, max_retries: int = 3
) -> Optional[bytes]:
    """Download an image with simple exponential backoff."""

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, timeout=(timeout, timeout), stream=True)
            response.raise_for_status()
            return response.content
        except (requests.Timeout, requests.ConnectionError) as exc:
            wait = attempt * 2
            logger.warning(
                "Download attempt %d/%d failed for %s: %s",
                attempt,
                max_retries,
                url[:80],
                exc,
            )
            if attempt < max_retries:
                time.sleep(wait)
        except Exception as exc:
            logger.warning(
                "Download attempt %d/%d errored for %s: %s",
                attempt,
                max_retries,
                url[:80],
                exc,
            )
            if attempt < max_retries:
                time.sleep(attempt * 2)
    return None


def download_to_temp(
    url_or_b64: str, temp_dir: str, filename: Optional[str] = None
) -> Optional[str]:
    """Persist a URL or base64 string to ``temp_dir`` and return the path."""

    os.makedirs(temp_dir, exist_ok=True)

    if looks_like_base64(url_or_b64):
        try:
            payload = url_or_b64
            if "base64," in payload:
                payload = payload.split("base64,", 1)[1]
            data = base64.b64decode(payload)
        except Exception as exc:
            logger.warning("Failed to decode base64 image: %s", exc)
            return None
        out_name = filename or "base64_image.png"
        save_path = os.path.join(temp_dir, out_name)
        if os.path.exists(save_path):
            return save_path
        try:
            with open(save_path, "wb") as fh:
                fh.write(data)
            return save_path
        except OSError as exc:
            logger.warning("Failed to save base64 image: %s", exc)
            return None

    parsed = urllib.parse.urlparse(url_or_b64)
    if not filename:
        filename = os.path.basename(parsed.path)
        if not filename or not os.path.splitext(filename)[1]:
            filename = (filename or "remote") + ".jpg"

    save_path = os.path.join(temp_dir, filename)
    if os.path.exists(save_path):
        return save_path

    try:
        urllib.request.urlretrieve(url_or_b64, save_path)
        return save_path
    except Exception:
        data = download_image_bytes(url_or_b64)
        if not data:
            return None
        try:
            with open(save_path, "wb") as fh:
                fh.write(data)
            return save_path
        except OSError as exc:
            logger.warning("Failed to write downloaded image: %s", exc)
            return None


def cos_url_to_internal(url: str) -> str:
    """Best-effort rewrite of a public COS URL to the matching VPC endpoint.

    Only applied when the user explicitly enabled it via the
    ``COS_USE_INTERNAL`` environment variable. Returns the input unchanged
    when the URL does not look like a COS bucket.
    """

    template = os.environ.get("COS_INTERNAL_URL_TEMPLATE", "").strip()
    if not template:
        return url
    if ".cos." not in url or ".myqcloud.com" not in url:
        return url
    bucket_part, rest = url.split(".cos.", 1)
    bucket = bucket_part.split("//")[-1]
    region, _, path = rest.partition(".myqcloud.com")
    return template.format(bucket=bucket, region=region, path=path)


def ensure_image_local(
    image_ref: str,
    image_paths_dict: dict,
    intermediate_dir: str,
    case_idx: int,
    turn_num: int,
    tool_name: str = "operation",
    case_id: Optional[str] = None,
    filename_prefix: str = "fvqa_train",
) -> Tuple[Optional[Union[bytes, str]], Optional[str]]:
    """Materialize ``image_paths_dict[image_ref]`` to a local resource.

    Returns ``(payload, local_path)`` where ``payload`` is the bytes or
    string the caller can hand to PIL/cv2, and ``local_path`` is set when
    a real file is on disk.
    """

    if image_ref not in image_paths_dict:
        return None, None

    image_data = image_paths_dict[image_ref]

    if isinstance(image_data, str):
        if looks_like_base64(image_data):
            local = _try_local_image(case_id)
            if local:
                image_paths_dict[image_ref] = local
                return local, local
            try:
                payload = image_data.split("base64,", 1)[-1]
                data = base64.b64decode(payload)
                return data, None
            except Exception as exc:
                logger.warning("Failed to decode base64 image: %s", exc)
                return None, None
        if len(image_data) < 500 and os.path.exists(image_data):
            return image_data, image_data

    if isinstance(image_data, str) and image_data.startswith(("http://", "https://")):
        temp_dir = os.path.join(intermediate_dir, "temp_images")
        os.makedirs(temp_dir, exist_ok=True)
        parsed = urllib.parse.urlparse(image_data)
        url_filename = os.path.basename(parsed.path)
        if not url_filename or not os.path.splitext(url_filename)[1]:
            url_filename = (
                f"{filename_prefix}_{case_idx}_trajectory_turn{turn_num}_"
                f"{tool_name}_source.png"
            )
        local_path = download_to_temp(image_data, temp_dir, url_filename)
        if not local_path:
            data = download_image_bytes(image_data)
            if not data:
                return None, None
            local_filename = (
                f"{filename_prefix}_{case_idx}_trajectory_turn{turn_num}_"
                f"{tool_name}_source.png"
            )
            local_path = os.path.join(temp_dir, local_filename)
            try:
                with open(local_path, "wb") as fh:
                    fh.write(data)
            except OSError as exc:
                logger.warning("Failed to save downloaded image: %s", exc)
                return None, None
        image_paths_dict[image_ref] = local_path
        return local_path, local_path

    return image_data, None


def _try_local_image(case_id: Optional[str]) -> Optional[str]:
    """Search the configured local image directory for ``case_id.<ext>``."""

    if not case_id or not config.FVQA_IMAGE_DIR:
        return None
    if not os.path.exists(config.FVQA_IMAGE_DIR):
        return None
    for ext in (".jpg", ".jpeg", ".png", ".gif", ".bmp"):
        candidate = os.path.join(config.FVQA_IMAGE_DIR, f"{case_id}{ext}")
        if os.path.exists(candidate):
            return candidate
    return None


def load_pil_from_any(source: Union[str, bytes, Image.Image]) -> Optional[Image.Image]:
    """Load a PIL image from a path, bytes blob or an existing PIL image."""

    if isinstance(source, Image.Image):
        return source
    try:
        if isinstance(source, bytes):
            from io import BytesIO

            return Image.open(BytesIO(source))
        if isinstance(source, str):
            if looks_like_base64(source):
                payload = source.split("base64,", 1)[-1]
                from io import BytesIO

                return Image.open(BytesIO(base64.b64decode(payload)))
            if os.path.exists(source):
                return Image.open(source)
    except Exception as exc:
        logger.warning("Failed to load PIL image: %s", exc)
    return None
