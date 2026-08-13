"""Agent tool definitions, parsing helpers and the dispatcher."""

from __future__ import annotations

import io
import json
import logging
import os
import re
import urllib.parse
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from synthesis.sft import tools as sft_tools

from . import cos_upload
from . import image_io
from .image_engines import ImageEnhancementEngine, ImageToolEngine


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON tool schema
# ---------------------------------------------------------------------------


def get_tools_definition() -> str:
    """Return the public OpenAI-style ``tools`` array as a JSON string."""
    return json.dumps(
        sft_tools.get_tool_definitions(),
        indent=2,
        ensure_ascii=False,
    )


# ---------------------------------------------------------------------------
# Tool-call parsing
# ---------------------------------------------------------------------------


_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_FALLBACK_TAGS: Dict[str, Tuple[re.Pattern, str]] = {
    "web_search": (
        re.compile(r"<web_search>\s*(\{.*?\})\s*</web_search>", re.DOTALL),
        "web_search",
    ),
    "t2t_search": (
        re.compile(r"<t2t_search>\s*(\{.*?\})\s*</t2t_search>", re.DOTALL),
        "t2t_search",
    ),
    "t2i_search": (
        re.compile(r"<t2i_search>\s*(\{.*?\})\s*</t2i_search>", re.DOTALL),
        "t2i_search",
    ),
    "i2i_search": (
        re.compile(r"<i2i_search>\s*(\{.*?\})\s*</i2i_search>", re.DOTALL),
        "i2i_search",
    ),
    "read_url": (
        re.compile(r"<read_url>\s*(\{.*?\})\s*</read_url>", re.DOTALL),
        "read_url",
    ),
    "image_search": (
        re.compile(
            r"<(?:image_search|local_image_search|lens_scan)>\s*(\{.*?\})"
            r"\s*</(?:image_search|local_image_search|lens_scan)>",
            re.DOTALL,
        ),
        "image_search",
    ),
    "text_search": (
        re.compile(
            r"<(?:text_search|local_search)>\s*(\{.*?\})"
            r"\s*</(?:text_search|local_search)>",
            re.DOTALL,
        ),
        "text_search",
    ),
    "crop": (re.compile(r"<crop>\s*(\{.*?\})\s*</crop>", re.DOTALL), "crop"),
    "layout_parsing": (
        re.compile(r"<(?:layout_parsing|ocr)>\s*(\{.*?\})\s*</(?:layout_parsing|ocr)>", re.DOTALL),
        "layout_parsing",
    ),
    "perspective_correct": (
        re.compile(r"<perspective_correct>\s*(\{.*?\})\s*</perspective_correct>", re.DOTALL),
        "perspective_correct",
    ),
    "super_resolution": (
        re.compile(r"<super_resolution>\s*(\{.*?\})\s*</super_resolution>", re.DOTALL),
        "super_resolution",
    ),
    "sharpen": (
        re.compile(r"<sharpen>\s*(\{.*?\})\s*</sharpen>", re.DOTALL),
        "sharpen",
    ),
}


def _normalize_search_aliases(name: str, params: dict) -> dict:
    """Normalize aliases to the shared synthesis tool schema."""

    return sft_tools.normalize_tool_arguments(name, params)


def extract_tool_call(text: str) -> Optional[str]:
    """Pull the next tool invocation out of an agent message.

    Returns a JSON string ``{"name": str, "parameters": {...}}`` on success
    and ``None`` when no recognisable tool call is present.
    """

    match = _TOOL_CALL_RE.search(text)
    if match:
        try:
            payload = json.loads(match.group(1).strip())
            name = payload.get("name", "")
            params = payload.get("arguments", payload.get("parameters", {})) or {}
            params = _normalize_search_aliases(name, dict(params))
            if name:
                return json.dumps(
                    {"name": name, "parameters": params}, ensure_ascii=False
                )
        except Exception:
            pass

    for pattern, canonical in _FALLBACK_TAGS.values():
        m = pattern.search(text)
        if not m:
            continue
        raw = m.group(1).strip()
        try:
            params = json.loads(raw)
            if not isinstance(params, dict):
                params = {"value": params}
        except Exception:
            params = {"q": raw} if canonical in {"text_search", "web_search", "t2t_search", "t2i_search"} else {"url": raw}
        params = _normalize_search_aliases(canonical, dict(params))
        return json.dumps(
            {"name": canonical, "parameters": params}, ensure_ascii=False
        )

    return None


def has_response_tag(text: str) -> bool:
    """Return ``True`` once the agent has produced a final response block."""
    return "<answer>" in text and "</answer>" in text


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


ToolResult = Tuple[str, Dict[str, str]]


def _normalize_resource_url(value: object) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    try:
        parsed = urllib.parse.urlsplit(raw)
    except Exception:
        return raw
    if parsed.scheme not in {"http", "https"}:
        return raw
    hostname = (parsed.hostname or "").lower()
    port = parsed.port
    netloc = hostname
    if port and not ((parsed.scheme == "http" and port == 80) or (parsed.scheme == "https" and port == 443)):
        netloc = f"{hostname}:{port}"
    return urllib.parse.urlunsplit(
        (parsed.scheme.lower(), netloc, parsed.path or "/", parsed.query, "")
    )


def _register_search_resources(
    registry: dict[str, sft_tools.UrlResource],
    *,
    tool_name: str,
    output: dict,
) -> None:
    query = str(output.get("query") or "").strip() or None
    raw_results = output.get("results") if tool_name in {"t2t_search", "t2i_search"} else output.get("matches")
    if not isinstance(raw_results, list):
        return
    for raw in raw_results:
        if not isinstance(raw, dict):
            continue
        image_url = str(raw.get("image_url") or raw.get("imageUrl") or "").strip() or None
        thumbnail_url = str(raw.get("thumbnail_url") or raw.get("thumbnailUrl") or "").strip() or None
        source_page_url = str(
            raw.get("source_page_url") or raw.get("link") or raw.get("url") or raw.get("source") or ""
        ).strip() or None
        primary_url = source_page_url if tool_name == "t2t_search" else (image_url or thumbnail_url or source_page_url)
        if not primary_url:
            continue
        rank_value = raw.get("rank")
        try:
            rank = int(rank_value) if rank_value is not None else None
        except (TypeError, ValueError):
            rank = None
        resource = sft_tools.UrlResource(
            primary_url=primary_url,
            resource_id=str(
                raw.get("resource_id")
                or raw.get("page_id")
                or raw.get("source_page_id")
                or raw.get("image_id")
                or ""
            ).strip(),
            result_id=str(raw.get("result_id") or "").strip() or None,
            kind="text" if tool_name == "t2t_search" else "image",
            title=str(raw.get("title") or "").strip() or None,
            snippet=str(raw.get("snippet") or "").strip() or None,
            image_url=image_url,
            thumbnail_url=thumbnail_url,
            source_page_url=source_page_url,
            search_tool=tool_name,
            search_query=query,
            rank=rank,
        )
        for candidate in (
            primary_url,
            image_url,
            thumbnail_url,
            source_page_url,
            resource.resource_id,
            resource.result_id,
            raw.get("page_id"),
            raw.get("source_page_id"),
            raw.get("image_id"),
        ):
            key = _normalize_resource_url(candidate)
            if key:
                registry[key] = resource


def _resolve_registered_resource(
    registry: dict[str, sft_tools.UrlResource],
    alias: object,
) -> Optional[sft_tools.UrlResource]:
    """Resolve a search result URL or compact page/image ID."""

    key = _normalize_resource_url(alias)
    if not key:
        return None
    resource = registry.get(key)
    if resource is not None:
        return resource
    # Keep this fallback for registries created by older code that only keyed
    # resources by URL and did not register compact IDs explicitly.
    for candidate in registry.values():
        if key in {
            _normalize_resource_url(candidate.primary_url),
            _normalize_resource_url(candidate.resource_id),
            _normalize_resource_url(candidate.result_id),
            _normalize_resource_url(candidate.source_page_url),
            _normalize_resource_url(candidate.image_url),
            _normalize_resource_url(candidate.thumbnail_url),
        }:
            return candidate
    return None


def _resolve_image_for_search(
    image_paths_dict: dict,
    image_ref: str,
    image_url_param: str,
    case_idx: int,
    turn_num: int,
    filename_prefix: str,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(resolved_url, error_message)`` for ``image_search``."""

    target_ref: Optional[str] = None
    image_url: Optional[str] = None

    if image_url_param:
        if image_url_param in image_paths_dict:
            target_ref = image_url_param
        elif isinstance(image_url_param, str) and image_url_param.startswith(
            ("http://", "https://")
        ):
            image_url = image_url_param
        else:
            image_url = image_url_param
    elif image_ref:
        target_ref = image_ref

    if target_ref:
        if target_ref not in image_paths_dict:
            return (
                None,
                f"Image reference '{target_ref}' not found. "
                f"Available: {list(image_paths_dict.keys())}",
            )
        image_data = image_paths_dict[target_ref]
        if isinstance(image_data, str) and image_data.startswith(
            ("http://", "https://")
        ):
            return image_data, None
        if isinstance(image_data, str) and len(image_data) < 500:
            local_path = image_io.local_image_path_from_reference(image_data)
            if local_path and os.path.exists(local_path):
                with open(local_path, "rb") as fh:
                    pil_img = Image.open(io.BytesIO(fh.read()))
                url = cos_upload.upload_pil_image(
                    pil_img, filename_prefix, case_idx, turn_num, "image_search"
                )
                return (url, None) if url else (
                    None,
                    "Failed to upload local image to COS for image_search.",
                )
        if isinstance(image_data, bytes):
            pil_img = Image.open(io.BytesIO(image_data))
            url = cos_upload.upload_pil_image(
                pil_img, filename_prefix, case_idx, turn_num, "image_search"
            )
            return (url, None) if url else (
                None,
                "Failed to upload image bytes to COS for image_search.",
            )
        return None, f"Unsupported image data type for {target_ref!r}"

    if image_url:
        return image_url, None
    return None, "image_search requires either a 'url' or an image reference."


def _apply_image_op(
    operation: Callable[[ImageEnhancementEngine], object],
    image_data,
    return_engine: ImageEnhancementEngine,
) -> Optional[Image.Image]:
    return_engine.load_image(image_data)
    operation(return_engine)
    return return_engine.to_pil()


def _normalize_region_bbox(region: object) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
    """Return ``(x, y, width, height)`` for a region payload."""

    if region in (None, ""):
        return None, None

    if isinstance(region, dict):
        if all(key in region for key in ("x", "y", "width", "height")):
            try:
                x = int(region["x"])
                y = int(region["y"])
                width = int(region["width"])
                height = int(region["height"])
            except (TypeError, ValueError):
                return None, "Region dict values must be numeric."
            if width <= 0 or height <= 0:
                return None, "Region width and height must be positive."
            return (x, y, width, height), None
        return None, "Region dict must contain x, y, width, and height."

    if isinstance(region, (list, tuple)):
        if len(region) != 4:
            return None, "Region list must contain exactly 4 numbers."
        try:
            x1 = int(region[0])
            y1 = int(region[1])
            x2 = int(region[2])
            y2 = int(region[3])
        except (TypeError, ValueError):
            return None, "Region list values must be numeric."
        width = x2 - x1
        height = y2 - y1
        if width <= 0 or height <= 0:
            return None, "Region list must be [x1, y1, x2, y2] with x2 > x1 and y2 > y1."
        return (x1, y1, width, height), None

    return None, "Region must be a 4-number list or a dict with x/y/width/height."


def _crop_region_for_i2i_search(
    *,
    image_ref: str,
    region: object,
    image_paths_dict: dict,
    case_id: str,
    case_idx: int,
    turn_num: int,
    intermediate_dir: str,
    filename_prefix: str,
) -> Tuple[Optional[str], Dict[str, str], Optional[str]]:
    """Crop a region and return ``(search_url, new_images, error)``.

    The cropped image is persisted locally for subsequent model turns.
    Reverse-image search still requires a public URL, so upload is attempted
    only for the search request itself.
    """

    bbox, err = _normalize_region_bbox(region)
    if err:
        return None, {}, err
    if bbox is None:
        return None, {}, None

    if not image_ref:
        return None, {}, "Region-based i2i_search requires an 'image' reference."

    image_data, _ = image_io.ensure_image_local(
        image_ref,
        image_paths_dict,
        intermediate_dir,
        case_idx,
        turn_num,
        tool_name="i2i_search_region",
        case_id=case_id,
        filename_prefix=filename_prefix,
    )
    if image_data is None:
        return None, {}, f"Failed to load image {image_ref!r} for region crop."

    x, y, width, height = bbox
    engine = ImageToolEngine()
    engine.load_image(image_data)
    cropped = engine.crop(x, y, width, height)
    new_id, local_path = _persist_new_image(
        cropped,
        intermediate_dir,
        filename_prefix,
        case_idx,
        turn_num,
        "i2i_search_region",
        image_paths_dict,
    )
    search_url = cos_upload.upload_pil_image(
        cropped,
        filename_prefix,
        case_idx,
        turn_num,
        "i2i_search_region",
    )
    if not search_url:
        return (
            None,
            {new_id: local_path},
            "Failed to upload cropped image to COS for i2i_search.",
        )
    return search_url, {new_id: local_path}, None


def _persist_new_image(
    pil_image: Image.Image,
    intermediate_dir: str,
    filename_prefix: str,
    case_idx: int,
    turn_num: int,
    tool_name: str,
    image_paths_dict: dict,
) -> Tuple[str, str]:
    """Persist a freshly produced image and return ``(image_id, local_path)``."""

    os.makedirs(intermediate_dir, exist_ok=True)
    new_id = f"img_{len(image_paths_dict) + 1}"
    save_path = os.path.join(
        intermediate_dir,
        f"{filename_prefix}_{case_idx}_trajectory_turn{turn_num}_{tool_name}.png",
    )
    pil_image.save(save_path)

    return new_id, save_path


def execute_tool(
    tool_call_json: str | dict,
    image_paths_dict: dict,
    case_id: str,
    case_idx: int,
    turn_num: int,
    intermediate_dir: str,
    filename_prefix: str = "fvqa_train",
    visual_lookup: Optional[Callable[..., object]] = None,
    url_registry: Optional[dict[str, sft_tools.UrlResource]] = None,
) -> ToolResult:
    """Dispatch a parsed tool call. Returns ``(message, new_images)``."""

    try:
        call = (
            json.loads(tool_call_json)
            if isinstance(tool_call_json, str)
            else tool_call_json
        )
    except Exception as exc:
        return f"Tool execution error:\nInvalid tool call payload: {exc}", {}

    name = call.get("name", "")
    params = call.get("parameters", {}) or {}
    resource_registry = url_registry if url_registry is not None else {}

    if name == "t2t_search":
        query = params.get("query") or ""
        if not query:
            return "Tool execution error:\n'query' is required for t2t_search.", {}
        result = sft_tools.t2t_search(
            query=query,
            lang=params.get("lang", "en"),
            top_k=int(params.get("top_k", sft_tools.DEFAULT_SEARCH_TOP_K)),
        )
        if isinstance(result, dict) and result.get("ok"):
            _register_search_resources(resource_registry, tool_name="t2t_search", output=result)
        return json.dumps(result, ensure_ascii=False, indent=2), {}

    if name == "t2i_search":
        query = params.get("query") or ""
        if not query:
            return "Tool execution error:\n'query' is required for t2i_search.", {}
        result = sft_tools.t2i_search(
            query=query,
            lang=params.get("lang", "en"),
            top_k=int(params.get("top_k", sft_tools.DEFAULT_SEARCH_TOP_K)),
        )
        if isinstance(result, dict) and result.get("ok"):
            _register_search_resources(resource_registry, tool_name="t2i_search", output=result)
        return json.dumps(result, ensure_ascii=False, indent=2), {}

    if name == "i2i_search":
        region = params.get("region")
        new_images: Dict[str, str] = {}
        search_image_url: Optional[str] = None
        if region not in (None, ""):
            latest_image_ref = next(reversed(image_paths_dict.keys()), "")
            cropped_location, cropped_new_images, crop_err = _crop_region_for_i2i_search(
                image_ref=latest_image_ref,
                region=region,
                image_paths_dict=image_paths_dict,
                case_id=case_id,
                case_idx=case_idx,
                turn_num=turn_num,
                intermediate_dir=intermediate_dir,
                filename_prefix=filename_prefix,
            )
            if crop_err:
                return f"Tool execution error:\n{crop_err}", {}
            search_image_url = cropped_location or ""
            new_images.update(cropped_new_images)
        else:
            latest_ref = next(reversed(image_paths_dict.keys()), "")
            search_image_url, err = _resolve_image_for_search(
                image_paths_dict,
                image_ref=latest_ref,
                image_url_param="",
                case_idx=case_idx,
                turn_num=turn_num,
                filename_prefix=filename_prefix,
            )
            if err:
                return f"Tool execution error:\n{err}", {}

        result = sft_tools.i2i_search(
            image_url=search_image_url or "",
            visual_lookup=visual_lookup,
            top_k=int(params.get("top_k", sft_tools.DEFAULT_SEARCH_TOP_K)),
        )
        if isinstance(result, dict) and result.get("ok"):
            _register_search_resources(resource_registry, tool_name="i2i_search", output=result)
        return json.dumps(result, ensure_ascii=False, indent=2), new_images

    if name == "read_url":
        url = params.get("url") or params.get("URL") or ""
        resource_id = (
            params.get("resource_id")
            or params.get("page_id")
            or params.get("image_id")
            or ""
        )
        resource = None
        if resource_id:
            resource = _resolve_registered_resource(resource_registry, resource_id)
            if resource is not None:
                url = resource.primary_url
            elif isinstance(resource_id, str) and resource_id.startswith(
                ("http://", "https://")
            ):
                # Some search backends expose a full URL under resource_id.
                url = resource_id
        if not url:
            return (
                "Tool execution error:\nread_url requires either 'url' or a "
                "resolvable 'resource_id'.",
                {},
            )
        result = sft_tools.read_url(
            url=url,
            goal=str(params.get("goal") or "").strip(),
            resource=resource or _resolve_registered_resource(resource_registry, url),
        )
        if not result.get("ok", False):
            return f"Tool execution error:\n{result.get('error', 'Unknown read_url error')}", {}
        if result.get("local_path"):
            local_path = result.get("local_path", "")
            if not local_path:
                return "Tool execution error:\nread_url returned an image without a local path.", {}
            new_id = f"img_{len(image_paths_dict) + 1}"
            msg = "读取图片如下:\n<image>"
            return msg, {new_id: local_path}

        title = result.get("title", "") or "(untitled)"
        content = result.get("content", "") or ""
        report = (
            "Tool execution result:\n"
            f"Title: {title}\n"
            f"URL: {result.get('url', url)}\n"
            f"Content:\n{content}"
        )
        return report, {}

    return f"Tool execution error:\nUnknown tool: {name!r}", {}
