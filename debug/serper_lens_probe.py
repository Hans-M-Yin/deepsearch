#!/usr/bin/env python3
"""Small, standalone probe for the Serper Lens path used by ``i2i_search``.

The normal i2i path sends ``{"url": image_url}`` to ``SERPER_LENS_URL``.
Serper Lens has no crop/region field, so a supplied region is cropped locally,
uploaded through the project's normal COS uploader, and the resulting public
URL is sent to Lens.  Coordinates use the agent-tool convention: normalized
``[x1, y1, x2, y2]`` on a 0--1000 scale (``xyxy`` by default).

Examples:
  python -u debug/serper_lens_probe.py --image-url 'https://example.com/a.jpg'
  python -u debug/serper_lens_probe.py --image-url 'https://example.com/a.jpg' \
      --region '[100,200,900,800]'
  # Issue 16 concurrent Lens requests for exactly the same URL.
  python -u debug/serper_lens_probe.py --image-url 'https://example.com/a.jpg' \
      --repeat 16
  # Match i2i_search's transient-empty retry policy.
  python -u debug/serper_lens_probe.py --image-url 'https://example.com/a.jpg' \
      --attempts 4 --retry-delay-s 2
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import re
import sys
import time
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_FILE = PROJECT_ROOT / "RL" / ".env_remote"
NORMALIZED_COORD_SCALE = 1000.0
I2I_UPLOAD_MAX_LONG_EDGE = 1000


def _load_simple_env_file(path: Path) -> list[str]:
    """Load simple ``export KEY=value`` entries without evaluating shell code."""

    loaded: list[str] = []
    if not path.is_file():
        return loaded

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key in os.environ:
            continue
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ[key] = value
        loaded.append(key)
    return loaded


def _parse_region(raw_region: str) -> tuple[float, float, float, float]:
    text = raw_region.strip()
    try:
        values = json.loads(text) if text.startswith("[") else [float(part.strip()) for part in text.split(",")]
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise argparse.ArgumentTypeError(
            "region must be four numbers, e.g. '[100,200,900,800]'"
        ) from exc

    if not isinstance(values, list) or len(values) != 4:
        raise argparse.ArgumentTypeError(
            "region must be exactly [x1,y1,x2,y2] with normalized coordinates"
        )
    try:
        coords = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("region values must be numeric") from exc
    if not all(math.isfinite(value) and 0 <= value <= NORMALIZED_COORD_SCALE for value in coords):
        raise argparse.ArgumentTypeError("region values must be finite and within [0,1000]")
    return coords  # type: ignore[return-value]


def _default_coordinate_order() -> str:
    reverse = os.environ.get("REVERSE_IMAGE_CROP_COORDS", "").strip().lower()
    return "yxyx" if reverse in {"1", "true", "yes", "on"} else "xyxy"


def _safe_response_preview(response: requests.Response, limit: int = 2000) -> str:
    try:
        return response.text[:limit]
    except Exception:
        return "<response body unavailable>"


def _probe_remote_url(session: requests.Session, url: str, timeout_s: float) -> dict[str, Any]:
    """Check that the public URL can at least be reached from this host."""

    started_at = time.monotonic()
    response: requests.Response | None = None
    try:
        response = session.get(
            url,
            headers={"Range": "bytes=0-0", "User-Agent": "OpenSearch-VL-Lens-Probe/1.0"},
            stream=True,
            allow_redirects=True,
            timeout=timeout_s,
        )
        return {
            "ok": response.ok,
            "status_code": response.status_code,
            "final_url": response.url,
            "content_type": response.headers.get("content-type", ""),
            "content_length": response.headers.get("content-length", ""),
            "elapsed_s": round(time.monotonic() - started_at, 6),
        }
    except requests.RequestException as exc:
        return {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": round(time.monotonic() - started_at, 6),
        }
    finally:
        if response is not None:
            response.close()


def _download_image(session: requests.Session, url: str, timeout_s: float) -> tuple[Image.Image, dict[str, Any]]:
    started_at = time.monotonic()
    response = session.get(
        url,
        headers={"User-Agent": "OpenSearch-VL-Lens-Probe/1.0"},
        allow_redirects=True,
        timeout=timeout_s,
    )
    response.raise_for_status()
    image_bytes = response.content
    image = Image.open(BytesIO(image_bytes))
    image.load()
    return image, {
        "status_code": response.status_code,
        "final_url": response.url,
        "content_type": response.headers.get("content-type", ""),
        "downloaded_bytes": len(image_bytes),
        "image_size": list(image.size),
        "image_mode": image.mode,
        "elapsed_s": round(time.monotonic() - started_at, 6),
    }


def _crop_normalized_region(
    image: Image.Image,
    region: tuple[float, float, float, float],
    coordinate_order: str,
) -> tuple[Image.Image, dict[str, Any]]:
    coords = region
    if coordinate_order == "yxyx":
        coords = (region[1], region[0], region[3], region[2])
    x1_n, y1_n, x2_n, y2_n = coords
    width, height = image.size
    x1 = min(max(int(round(x1_n / NORMALIZED_COORD_SCALE * width)), 0), width)
    y1 = min(max(int(round(y1_n / NORMALIZED_COORD_SCALE * height)), 0), height)
    x2 = min(max(int(round(x2_n / NORMALIZED_COORD_SCALE * width)), 0), width)
    y2 = min(max(int(round(y2_n / NORMALIZED_COORD_SCALE * height)), 0), height)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"region resolves to an empty crop: {[x1, y1, x2, y2]}")
    return image.crop((x1, y1, x2, y2)), {
        "input_region_normalized": list(region),
        "coordinate_order": coordinate_order,
        "effective_xyxy_normalized": [x1_n, y1_n, x2_n, y2_n],
        "crop_xyxy_pixels": [x1, y1, x2, y2],
        "crop_size": [x2 - x1, y2 - y1],
    }


def _prepare_i2i_upload(image: Image.Image) -> Image.Image:
    """Match the current i2i region-upload preprocessing: PNG-compatible + <=1000px."""

    if image.mode not in {"RGB", "RGBA", "L", "LA"}:
        has_alpha = "A" in image.getbands() or "transparency" in image.info
        image = image.convert("RGBA" if has_alpha else "RGB")
    width, height = image.size
    long_edge = max(width, height)
    if long_edge <= I2I_UPLOAD_MAX_LONG_EDGE:
        return image
    scale = I2I_UPLOAD_MAX_LONG_EDGE / long_edge
    resized = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(resized, resample=Image.Resampling.LANCZOS)


def _image_sha256(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _upload_i2i_image(image: Image.Image) -> str:
    # This is the same lazy COS uploader invoked by the i2i region path.
    from opensearch_vl.opensearch_infer import cos_upload

    image = _prepare_i2i_upload(image)
    uploaded_url = cos_upload.upload_pil_image(
        image,
        filename_prefix="lens_probe",
        case_idx=0,
        turn_num=0,
        tool_name=f"i2i_region_{uuid.uuid4().hex}",
    )
    if not uploaded_url:
        raise RuntimeError("the project COS uploader returned no public URL")
    return uploaded_url


def _call_lens(
    session: requests.Session,
    lens_url: str,
    api_key: str,
    image_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    relay_token = os.environ.get("SERPER_RELAY_TOKEN")
    if relay_token:
        headers["X-Serper-Relay-Token"] = relay_token

    started_at = time.monotonic()
    response: requests.Response | None = None
    try:
        response = session.post(
            lens_url,
            headers=headers,
            json={"url": image_url},
            timeout=timeout_s,
        )
        result: dict[str, Any] = {
            "status_code": response.status_code,
            "ok": response.ok,
            "elapsed_s": round(time.monotonic() - started_at, 6),
        }
        try:
            response_json = response.json()
        except ValueError:
            result["response_text"] = _safe_response_preview(response)
            return result
        result["response_json"] = response_json
        organic = response_json.get("organic", []) if isinstance(response_json, dict) else []
        result["organic_count"] = len(organic) if isinstance(organic, list) else None
        result["top_level_keys"] = sorted(response_json) if isinstance(response_json, dict) else None
        return result
    except requests.RequestException as exc:
        return {
            "ok": False,
            "transport_error": True,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_s": round(time.monotonic() - started_at, 6),
        }
    finally:
        if response is not None:
            response.close()


def _call_lens_in_isolated_session(
    lens_url: str,
    api_key: str,
    image_url: str,
    timeout_s: float,
) -> dict[str, Any]:
    """Perform one Lens call in a worker-private Session.

    ``requests.Session`` is not documented as thread-safe.  The repeat probe
    deliberately runs calls concurrently, so each request gets its own
    session rather than sharing the preflight/crop session from the main
    thread.
    """

    with requests.Session() as session:
        return _call_lens(session, lens_url, api_key, image_url, timeout_s)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-url", required=True, help="Public image URL sent to Serper Lens.")
    parser.add_argument(
        "--region",
        type=_parse_region,
        help="Optional normalized crop [x1,y1,x2,y2] on 0--1000 scale. It is uploaded before Lens.",
    )
    parser.add_argument(
        "--coordinate-order",
        choices=("xyxy", "yxyx"),
        default=None,
        help="Region coordinate order; defaults to the training REVERSE_IMAGE_CROP_COORDS setting.",
    )
    parser.add_argument(
        "--upload-full-image",
        action="store_true",
        help="Upload a freshly re-encoded full image before Lens, for an OSS-vs-source-URL A/B test.",
    )
    parser.add_argument("--lens-url", default=None, help="Override SERPER_LENS_URL.")
    parser.add_argument("--timeout-s", type=float, default=60.0, help="HTTP timeout per request (default: 60).")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help=(
            "Concurrent Lens calls for the same effective URL in each attempt "
            "round (default: 1; each call consumes Serper credits)."
        ),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=1,
        help="Sequential rounds of --repeat parallel calls (default: 1).",
    )
    parser.add_argument("--retry-delay-s", type=float, default=2.0, help="Delay between sequential rounds (default: 2).")
    parser.add_argument(
        "--env-file",
        type=Path,
        default=DEFAULT_ENV_FILE,
        help="Load missing variables from this simple export file (default: RL/.env_remote).",
    )
    parser.add_argument("--skip-url-check", action="store_true", help="Do not preflight GET the original/uploaded URL.")
    args = parser.parse_args()

    if args.timeout_s <= 0 or args.attempts <= 0 or args.repeat <= 0 or args.retry_delay_s < 0:
        parser.error("timeout-s, attempts, and repeat must be positive; retry-delay-s must be non-negative")

    loaded_env_keys = _load_simple_env_file(args.env_file)
    lens_url = args.lens_url or os.environ.get("SERPER_LENS_URL") or "https://google.serper.dev/lens"
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        parser.error("SERPER_API_KEY is not set (set it or add it to --env-file)")

    report: dict[str, Any] = {
        "probe": "serper_lens_i2i",
        "loaded_missing_env_keys": loaded_env_keys,
        "input": {
            "original_image_url": args.image_url,
            "region": list(args.region) if args.region is not None else None,
            "coordinate_order": args.coordinate_order or _default_coordinate_order(),
            "upload_full_image": args.upload_full_image,
        },
        "lens_endpoint": lens_url,
        "lens_payload": None,
        "parallel_repeat": args.repeat,
        "total_planned_lens_calls": args.repeat * args.attempts,
    }
    session = requests.Session()
    effective_image_url = args.image_url
    try:
        if not args.skip_url_check:
            report["original_url_preflight"] = _probe_remote_url(session, args.image_url, args.timeout_s)

        if args.region is not None or args.upload_full_image:
            image, download_info = _download_image(session, args.image_url, args.timeout_s)
            upload_image = image
            report["source_image_download"] = download_info
            if args.region is not None:
                upload_image, crop_info = _crop_normalized_region(
                    image,
                    args.region,
                    args.coordinate_order or _default_coordinate_order(),
                )
                report["crop"] = crop_info
            upload_image = _prepare_i2i_upload(upload_image)
            report["prepared_upload"] = {
                "image_size": list(upload_image.size),
                "image_mode": upload_image.mode,
                "png_sha256": _image_sha256(upload_image),
            }
            effective_image_url = _upload_i2i_image(upload_image)
            report["uploaded_image_url"] = effective_image_url
            if not args.skip_url_check:
                report["uploaded_url_preflight"] = _probe_remote_url(session, effective_image_url, args.timeout_s)

        report["effective_image_url"] = effective_image_url
        report["lens_payload"] = {"url": effective_image_url}
        lens_results: list[dict[str, Any]] = []
        lens_rounds: list[dict[str, Any]] = []
        for attempt in range(1, args.attempts + 1):
            round_started_at = time.monotonic()
            round_results: list[dict[str, Any]] = []
            with ThreadPoolExecutor(max_workers=args.repeat, thread_name_prefix="serper-lens-probe") as executor:
                futures = {
                    executor.submit(
                        _call_lens_in_isolated_session,
                        lens_url,
                        api_key,
                        effective_image_url,
                        args.timeout_s,
                    ): repeat_index
                    for repeat_index in range(1, args.repeat + 1)
                }
                for future in as_completed(futures):
                    repeat_index = futures[future]
                    try:
                        result = future.result()
                    except Exception as exc:  # defensive: request errors are normally handled above
                        result = {
                            "ok": False,
                            "transport_error": True,
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        }
                    result["attempt"] = attempt
                    result["repeat_index"] = repeat_index
                    round_results.append(result)
            round_results.sort(key=lambda item: int(item["repeat_index"]))
            lens_results.extend(round_results)
            lens_rounds.append(
                {
                    "attempt": attempt,
                    "parallel_calls": args.repeat,
                    "elapsed_s": round(time.monotonic() - round_started_at, 6),
                    "successful_http_count": sum(1 for item in round_results if item.get("ok")),
                    "empty_organic_count": sum(
                        1 for item in round_results if item.get("organic_count") == 0
                    ),
                }
            )
            if attempt < args.attempts:
                time.sleep(args.retry_delay_s)
        report["lens_results"] = lens_results
        report["lens_rounds"] = lens_rounds
    except Exception as exc:
        report["fatal_error"] = {"error_type": type(exc).__name__, "error": str(exc)}
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 2
    finally:
        session.close()

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
