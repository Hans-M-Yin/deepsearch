"""Alibaba OSS uploader with an optional worker-to-OSS Relay path.

This module exposes ``upload_cos`` because the inference runtime expects an
external uploader with that legacy function name. When ``OSS_RELAY_URL`` is
configured, the image is sent as base64 to the Relay and the Relay performs
the OSS upload using credentials kept on the egress node. Without that
variable, the historical direct OSS upload path is used.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from typing import Optional, Tuple
from urllib.error import HTTPError
from urllib.request import Request, urlopen


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


def _relay_url() -> str:
    return str(os.environ.get("OSS_RELAY_URL") or "").strip().rstrip("/")


def _relay_upload_url(relay_url: str) -> str:
    normalized = relay_url.rstrip("/")
    return normalized if normalized.endswith("/upload") else f"{normalized}/upload"


def _relay_timeout_s() -> float:
    raw = str(os.environ.get("OSS_RELAY_TIMEOUT_S") or "120").strip()
    try:
        value = float(raw)
    except ValueError:
        return 120.0
    return value if value > 0 else 120.0


def _upload_via_relay(
    local_path: str,
    filename: str,
    date_str: str,
    mode: str,
    user: str,
    use_direct_url: bool,
    relay_url: str,
) -> Tuple[Optional[str], Optional[str]]:
    with open(local_path, "rb") as file_obj:
        image_base64 = base64.b64encode(file_obj.read()).decode("ascii")
    payload = {
        "image_base64": image_base64,
        "filename": os.path.basename(filename),
        "date_str": date_str,
        "mode": mode,
        "user": user,
        "use_direct_url": bool(use_direct_url),
    }
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    relay_token = str(os.environ.get("OSS_RELAY_TOKEN") or "").strip()
    if relay_token:
        headers["X-OSS-Relay-Token"] = relay_token
    request = Request(
        _relay_upload_url(relay_url),
        data=(json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=_relay_timeout_s()) as response:
            response_body = response.read()
            status_code = int(response.getcode() or 200)
    except HTTPError as exc:
        response_body = exc.read(1024 * 1024)
        detail = response_body.decode("utf-8", errors="replace")
        raise RuntimeError(f"OSS Relay returned HTTP {exc.code}: {detail[:2000]}") from exc

    if status_code < 200 or status_code >= 300:
        detail = response_body.decode("utf-8", errors="replace")
        raise RuntimeError(f"OSS Relay returned HTTP {status_code}: {detail[:2000]}")
    try:
        response_payload = json.loads(response_body)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("OSS Relay returned a non-JSON response") from exc
    if not isinstance(response_payload, dict):
        raise RuntimeError("OSS Relay returned a non-object response")
    if not bool(response_payload.get("ok")):
        raise RuntimeError(str(response_payload.get("error") or "OSS Relay upload failed"))
    object_key = response_payload.get("object_key")
    public_url = response_payload.get("url")
    return (
        str(object_key) if object_key else None,
        str(public_url) if public_url else None,
    )


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

    relay_url = _relay_url()
    if relay_url:
        return _upload_via_relay(
            local_path,
            filename,
            date_str,
            mode,
            user,
            use_direct_url,
            relay_url,
        )

    access_key_id = _require_env("OSS_ACCESS_KEY_ID")
    access_key_secret = _require_env("OSS_ACCESS_KEY_SECRET")
    endpoint = _require_env("OSS_ENDPOINT")
    bucket_name = _require_env("OSS_BUCKET_NAME")

    try:
        import oss2
    except ImportError as exc:
        raise RuntimeError(
            "oss2 is required for direct OSS uploads; configure OSS_RELAY_URL "
            "to upload through the OSS Relay instead"
        ) from exc

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
