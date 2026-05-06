"""Lazy bootstrap for an optional, external ``upload.py`` COS uploader.

The internal codebase ships a small ``upload.py`` that wraps the Tencent
COS Python SDK. To keep this package self-contained we look for the
module at runtime in a configurable list of search paths and only call
into it when it is available. When no uploader is found the rest of the
pipeline degrades gracefully: ``image_search`` returns a clear error and
edited images fall back to base64 inlining.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
import tempfile
from datetime import datetime
from typing import Callable, List, Optional, Tuple

from . import config


logger = logging.getLogger(__name__)


_UploadFunc = Callable[..., Tuple[Optional[str], Optional[str]]]
_upload_cos: Optional[_UploadFunc] = None
_attempted_paths: List[str] = []


def _candidate_paths() -> List[str]:
    """Return the ordered list of directories to search for ``upload.py``."""

    candidates: List[str] = []
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.dirname(here))  # parent of opensearch_infer/
    candidates.append(here)
    if config.COS_UPLOAD_PATHS:
        for entry in config.COS_UPLOAD_PATHS.split(":"):
            entry = entry.strip()
            if entry:
                candidates.append(entry)
    # Deduplicate while keeping order.
    seen = set()
    ordered: List[str] = []
    for path in candidates:
        if path and path not in seen:
            ordered.append(path)
            seen.add(path)
    return ordered


def _load_upload_module(path: str) -> Optional[_UploadFunc]:
    upload_file = os.path.join(path, "upload.py")
    if not os.path.exists(upload_file):
        return None
    try:
        spec = importlib.util.spec_from_file_location("upload", upload_file)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        candidate = getattr(module, "upload_cos", None)
        if callable(candidate):
            return candidate
    except Exception as exc:
        logger.warning(
            "Skipping upload.py at %s due to import error: %s",
            upload_file,
            exc,
        )
    # Fallback: add to sys.path and ``import upload`` directly. Some
    # legacy uploaders rely on relative imports inside their package.
    try:
        if path not in sys.path:
            sys.path.insert(0, path)
        import upload  # type: ignore
        candidate = getattr(upload, "upload_cos", None)
        if callable(candidate):
            return candidate
    except Exception as exc:
        logger.warning(
            "Direct import of upload.py at %s failed: %s", upload_file, exc
        )
    return None


def _ensure_loaded() -> None:
    """Discover and bind the external ``upload_cos`` once."""

    global _upload_cos, _attempted_paths

    if _upload_cos is not None or _attempted_paths:
        return

    paths = _candidate_paths()
    for path in paths:
        loaded = _load_upload_module(path)
        if loaded is not None:
            _upload_cos = loaded
            logger.info("Loaded COS uploader from %s", path)
            break
    _attempted_paths = paths

    if _upload_cos is None:
        logger.info(
            "COS uploader not available. Image upload will be skipped. "
            "Searched: %s",
            ", ".join(paths),
        )


def upload_available() -> bool:
    """Return ``True`` when an external ``upload_cos`` was discovered."""

    _ensure_loaded()
    return _upload_cos is not None


def upload_pil_image(
    image_pil,
    filename_prefix: str,
    case_idx: int,
    turn_num: int,
    tool_name: str,
    userid: Optional[str] = None,
) -> Optional[str]:
    """Upload a PIL image to COS and return a public URL.

    Returns ``None`` when the uploader is unavailable or the upload fails.
    Errors are swallowed and logged so the pipeline can fall back to inline
    base64 transmission for visual reasoning.
    """

    _ensure_loaded()
    if _upload_cos is None:
        return None

    user = userid or config.COS_UPLOAD_USERID
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image_pil.save(tmp.name, format="PNG")
            tmp_path = tmp.name

        date_str = datetime.now().strftime("%Y-%m-%d")
        filename = (
            f"{filename_prefix}_{case_idx}_trajectory_turn{turn_num}_{tool_name}.png"
        )
        mode = f"{filename_prefix}_{date_str}_{filename_prefix}"

        cos_key, cos_url = _upload_cos(
            tmp_path,
            filename,
            date_str,
            mode,
            user,
            use_direct_url=True,
        )

        if cos_url:
            return cos_url
        if cos_key and config.COS_BUCKET_HOST_TEMPLATE:
            return config.COS_BUCKET_HOST_TEMPLATE.rstrip("/") + cos_key
        return None
    except Exception as exc:
        logger.warning("Upload to COS failed: %s", exc)
        return None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
