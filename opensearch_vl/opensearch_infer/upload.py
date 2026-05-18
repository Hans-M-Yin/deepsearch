"""Alibaba OSS uploader compatible with OpenSearch-VL's upload bootstrap.

This module exposes ``upload_cos`` because the inference runtime expects an
external uploader with that legacy function name. Despite the name, this
implementation targets Alibaba Cloud OSS.
"""

from __future__ import annotations

import mimetypes
import os
from typing import Optional, Tuple

import oss2


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _normalise_endpoint(endpoint: str) -> Tuple[str, str]:
    endpoint = endpoint.strip()
    if not endpoint:
        raise RuntimeError("OSS_ENDPOINT is empty")

    if endpoint.startswith("http://"):
        scheme = "http"
        host = endpoint[len("http://") :]
    elif endpoint.startswith("https://"):
        scheme = "https"
        host = endpoint[len("https://") :]
    else:
        scheme = "https"
        host = endpoint

    return scheme, host.rstrip("/")


def _content_type_for(filename: str) -> str:
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"


def upload_cos(
    local_path: str,
    filename: str,
    date_str: str,
    mode: str,
    user: str,
    use_direct_url: bool = True,
) -> Tuple[Optional[str], Optional[str]]:
    """Upload a local file to Alibaba OSS.

    Parameters follow OpenSearch-VL's expected uploader ABI. ``mode`` is kept
    for compatibility and used as a folder segment so trajectories from
    different runs do not collide.
    """

    access_key_id = _require_env("OSS_ACCESS_KEY_ID")
    access_key_secret = _require_env("OSS_ACCESS_KEY_SECRET")
    endpoint = _require_env("OSS_ENDPOINT")
    bucket_name = _require_env("OSS_BUCKET_NAME")

    scheme, endpoint_host = _normalise_endpoint(endpoint)
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, f"{scheme}://{endpoint_host}", bucket_name)

    safe_user = (user or "opensearch-vl").strip().replace("/", "_")
    safe_mode = (mode or "default").strip().replace("/", "_")
    safe_filename = os.path.basename(filename)
    object_key = f"vision_deepresearch/{date_str}/{safe_mode}/{safe_user}/{safe_filename}"

    headers = {
        "Content-Type": _content_type_for(safe_filename),
        "Content-Disposition": f'inline; filename="{safe_filename}"',
    }

    with open(local_path, "rb") as file_obj:
        bucket.put_object(object_key, file_obj, headers=headers)

    if use_direct_url:
        public_url = f"{scheme}://{bucket_name}.{endpoint_host}/{object_key}"
        return object_key, public_url
    return object_key, None