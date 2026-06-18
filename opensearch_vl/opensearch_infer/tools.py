"""Agent tool definitions, parsing helpers and the dispatcher."""

from __future__ import annotations

import io
import json
import logging
import os
import re
from typing import Callable, Dict, List, Optional, Tuple

from PIL import Image

from . import cos_upload
from . import image_io
from . import search
from .image_engines import ImageEnhancementEngine, ImageToolEngine


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON tool schema
# ---------------------------------------------------------------------------


def get_tools_definition() -> str:
    """Return the public OpenAI-style ``tools`` array as a JSON string."""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "t2t_search",
                "description": (
                    "Search text/web documents from a text query. Uses the "
                    "synthesis Serper backend, then reads pages through Jina "
                    "Reader and summarizes query-relevant content."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "q": {
                            "type": "string",
                            "description": "Alias for 'query'.",
                        },
                        "hl": {"type": "string", "default": "en"},
                        "lang": {
                            "type": "string",
                            "description": "Alias for 'hl'.",
                            "default": "en",
                        },
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "t2i_search",
                "description": (
                    "Search images from a text query via the synthesis "
                    "Serper image backend."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "q": {
                            "type": "string",
                            "description": "Alias for 'query'.",
                        },
                        "hl": {"type": "string", "default": "en"},
                        "lang": {
                            "type": "string",
                            "description": "Alias for 'hl'.",
                            "default": "en",
                        },
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "i2i_search",
                "description": (
                    "Search for visually similar or matching images from an "
                    "input image reference or image URL."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Image reference such as 'img_1'.",
                        },
                        "url": {
                            "type": "string",
                            "description": "Direct remote image URL.",
                        },
                        "region": {
                            "type": "array",
                            "description": (
                                "Optional bounding box to crop before search. "
                                "Preferred format: [x1, y1, x2, y2]."
                            ),
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "read_url",
                "description": (
                    "Read a URL. If it returns text/html, fetch content via "
                    "Jina Reader and optionally summarize the part relevant "
                    "to the provided query. If it returns an image, download "
                    "the image for later use."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "URL": {"type": "string"},
                        "url": {
                            "type": "string",
                            "description": "Alias for 'URL'.",
                        },
                        "query": {
                            "type": "string",
                            "description": "Optional query used for focused summarization.",
                            "default": "",
                        },
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "crop",
                "description": (
                    "Crop a specific region from an image. The target "
                    "(text/object) should cover < 30% of the image, or "
                    "multiple distinct sections need analysis. This "
                    "drastically improves OCR and recognition accuracy by "
                    "removing noise."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": "Image reference (e.g., 'img_1', 'img_2').",
                        },
                        "x": {"type": "integer", "description": "Top-left X."},
                        "y": {"type": "integer", "description": "Top-left Y."},
                        "width": {"type": "integer", "description": "Crop width."},
                        "height": {"type": "integer", "description": "Crop height."},
                    },
                    "required": ["image", "x", "y", "width", "height"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "layout_parsing",
                "description": (
                    "Perform advanced document layout parsing on an image to "
                    "extract structured text. Prefer the 'image' parameter "
                    "with an image reference; the system handles file path "
                    "operations in the background."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {
                            "type": "string",
                            "description": (
                                "Image reference such as 'img_1'. Preferred "
                                "over file_path."
                            ),
                        },
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to a local image file (optional).",
                        },
                        "use_chart_recognition": {"type": "boolean", "default": False},
                        "use_doc_orientation_classify": {"type": "boolean", "default": False},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": (
                    "Perform a web search query to retrieve up-to-date "
                    "information. Implemented as Serper + Jina + Qwen "
                    "summarization (same backbone as text_search)."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "hl": {"type": "string", "default": "en"},
                    },
                    "required": ["q"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "image_search",
                "description": (
                    "Visually identify the contents of an image. Returns "
                    "summarized title/source pairs filtered by Qwen3-32B."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": (
                                "Image reference (e.g., 'img_1') or a direct "
                                "image URL."
                            ),
                        }
                    },
                    "required": ["url"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "text_search",
                "description": (
                    "Search for text documents using Serper + Jina + Qwen "
                    "summarization. Use for entity / fact lookups."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string"},
                        "query": {
                            "type": "string",
                            "description": "Alias for 'q'.",
                        },
                        "hl": {"type": "string", "default": "en"},
                        "lang": {
                            "type": "string",
                            "description": "Alias for 'hl'.",
                            "default": "en",
                        },
                        "top_k": {"type": "integer", "default": 5},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "perspective_correct",
                "description": "Correct perspective distortion in an image.",
                "parameters": {
                    "type": "object",
                    "properties": {"image": {"type": "string"}},
                    "required": ["image"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "super_resolution",
                "description": "Super-resolve a low-resolution image.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string"},
                        "scale": {"type": "integer", "default": 4},
                    },
                    "required": ["image"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "sharpen",
                "description": "Sharpen an image to reduce blur.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "image": {"type": "string"},
                        "amount": {"type": "number", "default": 1.5},
                    },
                    "required": ["image"],
                },
            },
        },
    ]
    return json.dumps(tools, indent=2, ensure_ascii=False)


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
    """Convert ``query``/``lang`` aliases to canonical ``q``/``hl``."""

    if name in {"text_search", "local_search", "web_search", "t2t_search", "t2i_search"}:
        if "query" in params and "q" not in params:
            params["q"] = params.pop("query")
        if "lang" in params and "hl" not in params:
            params["hl"] = params.pop("lang")
    if name == "read_url" and "URL" in params and "url" not in params:
        params["url"] = params.pop("URL")
    return params


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

    return "<response>" in text and ("</response>" in text or "boxed{" in text)


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


ToolResult = Tuple[str, Dict[str, str]]


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
        if (
            isinstance(image_data, str)
            and len(image_data) < 500
            and os.path.exists(image_data)
        ):
            with open(image_data, "rb") as fh:
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
    """Crop a region and return ``(search_url_or_path, new_images, error)``."""

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
    new_id, location = _persist_new_image(
        cropped,
        intermediate_dir,
        filename_prefix,
        case_idx,
        turn_num,
        "i2i_search_region",
        image_paths_dict,
    )
    return location, {new_id: location}, None


def _persist_new_image(
    pil_image: Image.Image,
    intermediate_dir: str,
    filename_prefix: str,
    case_idx: int,
    turn_num: int,
    tool_name: str,
    image_paths_dict: dict,
) -> Tuple[str, str]:
    """Persist a freshly produced image and return ``(image_id, url_or_path)``."""

    os.makedirs(intermediate_dir, exist_ok=True)
    new_id = f"img_{len(image_paths_dict) + 1}"
    save_path = os.path.join(
        intermediate_dir,
        f"{filename_prefix}_{case_idx}_trajectory_turn{turn_num}_{tool_name}.png",
    )
    pil_image.save(save_path)

    cos_url = cos_upload.upload_pil_image(
        pil_image,
        filename_prefix,
        case_idx,
        turn_num,
        tool_name,
    )
    return new_id, cos_url or save_path


def execute_tool(
    tool_call_json: str | dict,
    image_paths_dict: dict,
    case_id: str,
    case_idx: int,
    turn_num: int,
    intermediate_dir: str,
    filename_prefix: str = "fvqa_train",
    visual_lookup: Optional[Callable[..., object]] = None,
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

    if name in {"text_search", "local_search", "web_search", "t2t_search"}:
        query = params.get("q") or params.get("query") or ""
        if not query:
            return "Tool execution error:\n'q' is required for text_search.", {}
        if name == "t2t_search":
            return search.t2t_search(
                query=query,
                lang=params.get("hl", "en") or params.get("lang", "en"),
                top_k=int(params.get("top_k", 5)),
            ), {}
        return search.text_search(
            query=query,
            lang=params.get("hl", "en") or params.get("lang", "en"),
            top_k=int(params.get("top_k", 5)),
        ), {}

    if name == "t2i_search":
        query = params.get("q") or params.get("query") or ""
        if not query:
            return "Tool execution error:\n'q' is required for t2i_search.", {}
        return search.t2i_search(
            query=query,
            lang=params.get("hl", "en") or params.get("lang", "en"),
            top_k=int(params.get("top_k", 5)),
        ), {}

    if name in {"image_search", "local_image_search", "lens_scan", "i2i_search"}:
        if name == "i2i_search":
            region = params.get("region")
            if region not in (None, ""):
                cropped_location, new_images, crop_err = _crop_region_for_i2i_search(
                    image_ref=params.get("image", ""),
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
                result = search.i2i_search(
                    image_url=cropped_location or "",
                    visual_lookup=visual_lookup,
                )
                if new_images:
                    cropped_id = next(iter(new_images.keys()))
                    result = (
                        f"{result}\n\n"
                        f"Cropped region image ID: {cropped_id}\n"
                        f"Cropped region source: {cropped_location}"
                    )
                return result, new_images

        url, err = _resolve_image_for_search(
            image_paths_dict,
            image_ref=params.get("image", ""),
            image_url_param=params.get("url", ""),
            case_idx=case_idx,
            turn_num=turn_num,
            filename_prefix=filename_prefix,
        )
        if err:
            return f"Tool execution error:\n{err}", {}
        if name == "i2i_search":
            return search.i2i_search(image_url=url, visual_lookup=visual_lookup), {}
        return search.image_search(image_url=url, visual_lookup=visual_lookup), {}

    if name == "read_url":
        url = params.get("url") or params.get("URL") or ""
        if not url:
            return "Tool execution error:\n'url' is required for read_url.", {}
        result = search.read_url(url=url, query=params.get("query", "") or "")
        if result.get("error"):
            return f"Tool execution error:\n{result['error']}", {}
        if result.get("kind") == "image":
            local_path = result.get("local_path", "")
            if not local_path:
                return "Tool execution error:\nread_url returned an image without a local path.", {}
            new_id = f"img_{len(image_paths_dict) + 1}"
            msg = (
                f"Tool execution result:\nDownloaded image from {result.get('url', url)}.\n"
                f"New image ID: {new_id}\n"
                f"Local path: {local_path}"
            )
            return msg, {new_id: local_path}

        title = result.get("title", "") or "(untitled)"
        summary = result.get("summary", "") or ""
        content = result.get("content", "") or ""
        body = summary if summary else content
        report = (
            "Tool execution result:\n"
            f"Title: {title}\n"
            f"URL: {result.get('url', url)}\n"
            f"{'Summary' if summary else 'Content'}:\n{body}"
        )
        return report, {}

    if name == "crop":
        image_ref = params.get("image", "")
        x, y = int(params.get("x", 0)), int(params.get("y", 0))
        width = int(params.get("width", 0))
        height = int(params.get("height", 0))
        if not image_ref:
            return "Tool execution error:\n'image' is required for crop.", {}
        if width <= 0 or height <= 0:
            return (
                "Tool execution error:\nwidth and height must be positive integers.",
                {},
            )
        image_data, _ = image_io.ensure_image_local(
            image_ref,
            image_paths_dict,
            intermediate_dir,
            case_idx,
            turn_num,
            tool_name="crop",
            case_id=case_id,
            filename_prefix=filename_prefix,
        )
        if image_data is None:
            return f"Tool execution error:\nFailed to load image {image_ref!r}.", {}
        engine = ImageToolEngine()
        engine.load_image(image_data)
        cropped = engine.crop(x, y, width, height)
        new_id, location = _persist_new_image(
            cropped,
            intermediate_dir,
            filename_prefix,
            case_idx,
            turn_num,
            "crop",
            image_paths_dict,
        )
        msg = (
            f"Image cropped successfully. New image ID: {new_id}. "
            f"Available at: {location}"
        )
        return msg, {new_id: location}

    if name == "layout_parsing":
        file_path = params.get("file_path", "")
        image_ref = params.get("image", "")
        if not file_path and image_ref:
            image_data, local_path = image_io.ensure_image_local(
                image_ref,
                image_paths_dict,
                intermediate_dir,
                case_idx,
                turn_num,
                tool_name="layout_parsing",
                case_id=case_id,
                filename_prefix=filename_prefix,
            )
            if local_path:
                file_path = local_path
            elif isinstance(image_data, bytes):
                os.makedirs(intermediate_dir, exist_ok=True)
                file_path = os.path.join(
                    intermediate_dir,
                    f"{filename_prefix}_{case_idx}_trajectory_turn{turn_num}_layout_source.png",
                )
                with open(file_path, "wb") as fh:
                    fh.write(image_data)
                image_paths_dict[image_ref] = file_path
        if not file_path:
            return (
                "Tool execution error:\nlayout_parsing needs 'file_path' "
                "or a resolvable 'image' reference.",
                {},
            )
        result = search.layout_parsing(
            file_path,
            use_chart_recognition=bool(params.get("use_chart_recognition", False)),
            use_doc_orientation_classify=bool(
                params.get("use_doc_orientation_classify", False)
            ),
            use_doc_unwarping=bool(params.get("use_doc_unwarping", False)),
        )
        if result.get("error"):
            return f"Tool execution error:\n{result['error']}", {}
        body = result.get("formatted_text", "") or "(no text detected)"
        report = (
            "Tool execution result:\n"
            "Layout Parsing SUCCESS.\n\n"
            "ALL RECOGNIZED TEXT:\n"
            f"{body}"
        )
        return report, {}

    if name in {"perspective_correct", "super_resolution", "sharpen"}:
        image_ref = params.get("image", "")
        if not image_ref:
            return f"Tool execution error:\n'image' is required for {name}.", {}
        image_data, _ = image_io.ensure_image_local(
            image_ref,
            image_paths_dict,
            intermediate_dir,
            case_idx,
            turn_num,
            tool_name=name,
            case_id=case_id,
            filename_prefix=filename_prefix,
        )
        if image_data is None:
            return f"Tool execution error:\nFailed to load image {image_ref!r}.", {}
        engine = ImageEnhancementEngine()
        if not ImageEnhancementEngine.available():
            return (
                "Tool execution error:\nOpenCV is required for image "
                "enhancement tools. Install with: pip install opencv-python.",
                {},
            )
        if name == "perspective_correct":
            op = lambda e: e.auto_correct_perspective()
        elif name == "super_resolution":
            scale = int(params.get("scale", 4))
            model_path = params.get("model_path") or os.environ.get(
                "SR_MODEL_PATH", "EDSR_x4.pb"
            )
            op = lambda e: e.apply_super_resolution(model_path=model_path, scale=scale)
        else:
            amount = float(params.get("amount", 1.5))
            op = lambda e: e.enhance_sharpness(amount=amount)
        pil_result = _apply_image_op(op, image_data, engine)
        if pil_result is None:
            return f"Tool execution error:\n{name} produced no image.", {}
        new_id, location = _persist_new_image(
            pil_result,
            intermediate_dir,
            filename_prefix,
            case_idx,
            turn_num,
            name,
            image_paths_dict,
        )
        msg = (
            f"{name} succeeded. New image ID: {new_id}. Available at: {location}"
        )
        return msg, {new_id: location}

    return f"Tool execution error:\nUnknown tool: {name!r}", {}
