"""Agent tool definitions, parsing helpers and the dispatcher."""

from __future__ import annotations

import io
import json
import logging
import math
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


def get_tools_definition(disabled_tools=()) -> str:
    """Return the public OpenAI-style ``tools`` array as a JSON string."""
    definitions = sft_tools.get_tool_definitions()
    if "t2i_search" in disabled_tools:
        for tool in definitions:
            function = tool.get("function", {})
            if function.get("name") == "read_url":
                function["description"] = (
                    "Read webpage or PDF text and optionally extract evidence relevant to the tool goal. "
                    "Image downloading is disabled; do not request image resources or image URLs. "
                    "Prefer page resource_id from a prior search result; direct webpage/PDF URLs are supported."
                )
                function["parameters"]["properties"]["resource_id"]["description"] = (
                    "Compact page_id returned by a prior search result. Image resources are not allowed."
                )
    return json.dumps(
        [tool for tool in definitions
         if tool.get("function", {}).get("name") not in disabled_tools],
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


def _parse_tool_payload(raw: str) -> tuple[dict, list[str]]:
    """Repair known punctuation mistakes without evaluating model output."""
    repairs: list[str] = []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        # Restrict replacements to the leading name/arguments fields so query
        # strings containing JSON-like text are preserved verbatim.
        fixed = re.sub(r'^(\s*\{\s*"name")\s*,\s*(?=")', r'\1: ', raw)
        if fixed != raw:
            repairs.append("name_comma_to_colon")
        fixed, count = re.subn(
            r'^(\s*\{\s*"name"\s*:\s*"(?:[^"\\]|\\.)*"\s*,\s*)"(arguments|parameters)\s*:',
            r'\1"\2":', fixed,
        )
        if count:
            repairs.append("close_arguments_key_quote")
        payload, end = json.JSONDecoder().raw_decode(fixed.lstrip())
        trailing = fixed.lstrip()[end:].strip()
        if trailing:
            if not re.fullmatch(r'}+', trailing):
                raise ValueError("Unexpected text after tool JSON")
            repairs.append("remove_extra_closing_braces")
    if not isinstance(payload, dict):
        raise ValueError("Tool call must be an object")
    name = payload.get("name")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("Tool name must be a nonempty string")
    params = payload.get("arguments", payload.get("parameters", {}))
    if isinstance(params, str) and name in {
        "t2t_search", "t2i_search", "text_search", "web_search",
    }:
        params = {"query": params}
        repairs.append("search_string_to_query")
    if not isinstance(params, dict):
        raise ValueError("Tool arguments must be an object")
    return {"name": name, "parameters": params}, repairs


def extract_tool_call(text: str, diagnostics: Optional[dict] = None, *, recover_format: bool = False) -> Optional[str]:
    """Pull the next tool invocation out of an agent message.

    Returns a JSON string ``{"name": str, "parameters": {...}}`` on success
    and ``None`` when no recognisable tool call is present.
    """

    match = _TOOL_CALL_RE.search(text)
    if match:
        try:
            payload, repairs = _parse_tool_payload(match.group(1).strip())
            name = payload["name"]
            params = _normalize_search_aliases(name, payload["parameters"])
            if diagnostics is not None and repairs:
                diagnostics["repairs"] = repairs
            return json.dumps({"name": name, "parameters": params}, ensure_ascii=False)
        except (ValueError, TypeError) as exc:
            if diagnostics is not None:
                diagnostics["error"] = str(exc)
            return None

    if "<tool_call>" in text:
        if recover_format:
            raw = text.split("<tool_call>", 1)[1].strip()
            # Only repair the wrapper. Require a complete JSON object and a
            # known closing-tag typo (or EOF); never invent tools/arguments.
            raw = re.sub(r'\s*</(?:tool_call|tool_all|toolcall)\s*>?\s*$', '', raw)
            try:
                payload, repairs = _parse_tool_payload(raw)
                params = _normalize_search_aliases(payload["name"], payload["parameters"])
                if diagnostics is not None:
                    diagnostics["repairs"] = repairs + ["recover_tool_call_closing_tag"]
                return json.dumps({"name": payload["name"], "parameters": params}, ensure_ascii=False)
            except (ValueError, TypeError) as exc:
                if diagnostics is not None:
                    diagnostics["error"] = str(exc)
                return None
        if diagnostics is not None:
            diagnostics["error"] = "Expected a complete <tool_call> JSON object </tool_call>"
        return None

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


def _postprocess_search_result(
    registry: dict[str, sft_tools.UrlResource],
    *,
    tool_name: str,
    output: dict,
) -> dict:
    """Expose the same compact search-result IDs used during training.

    The synthesis tool definitions promise ``source_page_id``/``image_id``
    references, while the low-level Serper functions return raw provider
    results.  Keep full URLs private in ``registry`` and expose only the
    compact, resolvable representation to the model.
    """

    compact, resources = sft_tools.postprocess_search_output(
        tool_name=tool_name,
        output=output,
        extract_url_keywords=False,
    )
    for resource in resources:
        for candidate in (
            resource.primary_url,
            resource.resource_id,
            resource.result_id,
            resource.source_page_url,
            resource.image_url,
            resource.thumbnail_url,
        ):
            if candidate:
                registry[str(candidate)] = resource
        # Backward compatibility for checkpoints that learned to refer to a
        # search result by its displayed rank ("1", "2", ...).  New tool
        # outputs expose stable hash IDs, but resolving rank aliases prevents
        # older checkpoints from entering an avoidable read_url error loop.
        if resource.rank is not None:
            registry[str(resource.rank)] = resource
    return compact


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


def _normalize_region_bbox(
    region: object, *, image_size: Tuple[int, int]
) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[str]]:
    """Convert public 0–1000 xyxy boxes to bounded pixel xywh boxes.

    Legacy x/y/width/height dictionaries retain their pixel-coordinate meaning.
    """

    if region in (None, ""):
        return None, None

    normalized = False
    if isinstance(region, dict):
        if all(key in region for key in ("x", "y", "width", "height")):
            try:
                x = float(region["x"])
                y = float(region["y"])
                width = float(region["width"])
                height = float(region["height"])
            except (TypeError, ValueError, OverflowError):
                return None, "Region dict values must be numeric."
            if width <= 0 or height <= 0:
                return None, "Region width and height must be positive."
            coords = (x, y, x + width, y + height)
        else:
            return None, "Region dict must contain x, y, width, and height."
    elif isinstance(region, (list, tuple)):
        if len(region) != 4:
            return None, "Region list must contain exactly 4 numbers."
        try:
            coords = tuple(float(value) for value in region)
        except (TypeError, ValueError, OverflowError):
            return None, "Region list values must be numeric."
        normalized = True
    else:
        return None, "Region must be a 4-number list or a dict with x/y/width/height."

    if not all(math.isfinite(value) for value in coords):
        return None, "Region coordinates must be finite numbers."
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        return None, "Region must have x2 > x1 and y2 > y1."

    image_width, image_height = image_size
    if normalized:
        x1, x2 = (round(min(max(x, 0), 1000) / 1000 * image_width) for x in (x1, x2))
        y1, y2 = (round(min(max(y, 0), 1000) / 1000 * image_height) for y in (y1, y2))
    x1, x2 = (min(max(int(x), 0), image_width) for x in (x1, x2))
    y1, y2 = (min(max(int(y), 0), image_height) for y in (y1, y2))
    if x2 <= x1 or y2 <= y1:
        return None, "Region is empty after conversion and clipping to image bounds."
    return (x1, y1, x2 - x1, y2 - y1), None


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

    if region in (None, ""):
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

    engine = ImageToolEngine()
    engine.load_image(image_data)
    bbox, err = _normalize_region_bbox(region, image_size=engine.current_image.size)
    if err:
        return None, {}, err
    x, y, width, height = bbox
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
    disabled_tools=(),
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
    if name in disabled_tools:
        return f"Tool execution error:\nTool '{name}' is disabled for this ablation.", {}
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
            result = _postprocess_search_result(
                resource_registry, tool_name="t2t_search", output=result
            )
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
            result = _postprocess_search_result(
                resource_registry, tool_name="t2i_search", output=result
            )
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
            result = _postprocess_search_result(
                resource_registry, tool_name="i2i_search", output=result
            )
        return json.dumps(result, ensure_ascii=False, indent=2), new_images

    if name == "read_url":
        url = params.get("url") or params.get("URL") or ""
        resource_id = (
            params.get("resource_id")
            or params.get("page_id")
            or params.get("image_id")
            or ""
        )
        # Be tolerant of model calls that put a compact search-result ID in
        # the legacy ``url`` field. Otherwise normalize_http_url() would
        # reinterpret it as a hostname such as https://image_.../.
        if not resource_id and sft_tools.is_compact_resource_id(url):
            resource_id = url
            url = ""
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
            **({"allow_images": False} if "t2i_search" in disabled_tools else {}),
        )
        if not result.get("ok", False):
            return f"Tool execution error:\n{result.get('error', 'Unknown read_url error')}", {}
        if result.get("local_path"):
            if "t2i_search" in disabled_tools:
                return "Tool execution error:\nImage downloading is disabled for this ablation.", {}
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
