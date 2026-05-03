"""
Visual Investigation Tools – ported from qwen_search_fvqa.py.

Provides: crop, layout_parsing, text_search, image_search,
          web_search, perspective_correct, super_resolution, sharpen.

All tools follow the async DeepResearchTool interface.
"""

import asyncio
import base64
import io
import json
import os
import re
import tempfile
import time
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import requests
from PIL import Image

from vision_deepresearch_async_workflow.tools.shared import (
    DeepResearchTool,
    get_cache_async,
    get_cache_key,
    log_tool_event,
    run_with_retries_async,
    set_cache_async,
)

# ---------------------------------------------------------------------------
# Optional dependency: OpenCV (for perspective_correct, super_resolution, sharpen)
# ---------------------------------------------------------------------------
try:
    import cv2
    import numpy as np

    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# ---------------------------------------------------------------------------
# Optional dependency: COS upload
# ---------------------------------------------------------------------------
UPLOAD_AVAILABLE = False
upload_cos = None

# Search for an optional ``upload.py`` module (COS uploader). By default we
# only look at the project root; additional search roots can be injected via
# the ``COS_UPLOAD_PATHS`` env var (colon-separated).
_upload_paths = [
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."),
]
_extra_paths = os.getenv("COS_UPLOAD_PATHS", "")
if _extra_paths:
    _upload_paths.extend(p for p in _extra_paths.split(":") if p)

for _p in _upload_paths:
    _ufile = os.path.join(_p, "upload.py")
    if not os.path.exists(_ufile):
        continue
    try:
        import importlib.util

        _spec = importlib.util.spec_from_file_location("upload", _ufile)
        if _spec and _spec.loader:
            _mod = importlib.util.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            upload_cos = _mod.upload_cos
            if callable(upload_cos):
                UPLOAD_AVAILABLE = True
                print(f"[visual_tools] COS upload loaded from {_ufile}")
                break
    except Exception as _e:
        print(f"[visual_tools] Failed to load upload.py from {_ufile}: {_e}")
        continue

if not UPLOAD_AVAILABLE:
    print("[visual_tools] WARNING: COS upload not available. image_search will not work for local images.")
    def upload_cos(*args, **kwargs):
        return None, None

# ---------------------------------------------------------------------------
# Layout Parsing API config
# ---------------------------------------------------------------------------
LAYOUT_PARSING_API_URL = os.getenv("LAYOUT_PARSING_API_URL", "")
LAYOUT_PARSING_TOKEN = os.getenv("LAYOUT_PARSING_TOKEN", "")

# ---------------------------------------------------------------------------
# Optional proxy gateway / Serper / JINA config (read from env or .env)
# ---------------------------------------------------------------------------
API_HOST = os.getenv("API_GATEWAY_HOST", "")
API_USER = os.getenv("API_GATEWAY_USER", "")
API_KEY = os.getenv("API_GATEWAY_KEY", "")

SERP_SEARCH_URL = os.getenv("TEXT_SEARCH_URL", "https://google.serper.dev/search")

# Judge / Extract model endpoint used for Qwen3-32B summarisation
QWEN_API_BASE = os.getenv("QWEN_API_BASE", "")
QWEN_MODEL = os.getenv("QWEN_MODEL", "")

# ---------------------------------------------------------------------------
# env.py lens_scan / web_search (Polaris)
# ---------------------------------------------------------------------------
_env_lens_scan = None
_env_web_search = None

try:
    import sys
    _env_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", ".."
    )
    _search_paths = [_env_dir]
    _extra_env = os.getenv("VDR_EXTRA_ENV_PATHS", "")
    if _extra_env:
        _search_paths.extend(p for p in _extra_env.split(":") if p)
    for _try_path in _search_paths:
        try:
            if _try_path not in sys.path:
                sys.path.insert(0, _try_path)
            from env import web_search as _ws, lens_scan as _ls  # type: ignore
            _env_web_search = _ws
            _env_lens_scan = _ls
            break
        except Exception:
            continue
except Exception:
    pass


# ===================================================================
# Helper: upload PIL image to COS
# ===================================================================

def _upload_pil_to_cos(
    pil_img: Image.Image,
    tool_name: str,
    userid: str = os.getenv("COS_USERID", "opensearch-vl"),
) -> Optional[str]:
    if not UPLOAD_AVAILABLE:
        print(f"[visual_tools] _upload_pil_to_cos: COS upload not available (UPLOAD_AVAILABLE=False)")
        return None
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil_img.save(tmp.name, format="PNG")
            tmp_path = tmp.name
        date_str = datetime.now().strftime("%Y-%m-%d")
        ts = int(time.time() * 1000)
        filename = f"vdr_{tool_name}_{ts}.png"
        cos_key, cos_url = upload_cos(
            tmp_path, filename, date_str,
            f"vdr_{date_str}", userid, use_direct_url=True,
        )
        if cos_url:
            print(f"[visual_tools] Uploaded to COS: {cos_url}")
        else:
            print(f"[visual_tools] _upload_pil_to_cos: upload_cos returned None for {filename}")
        return cos_url or None
    except Exception as exc:
        print(f"[visual_tools] _upload_pil_to_cos failed: {type(exc).__name__}: {exc}")
        return None
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ===================================================================
# Helper: download image from URL
# ===================================================================

def _download_image_bytes(url: str, timeout: int = 60, retries: int = 3) -> Optional[bytes]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=(timeout, timeout), stream=True)
            resp.raise_for_status()
            return resp.content
        except Exception:
            if attempt < retries - 1:
                time.sleep((attempt + 1) * 2)
    return None


# ===================================================================
# Helper: resolve image reference → local path
# ===================================================================

def _resolve_image_ref(
    image_ref: str,
    image_paths: dict,
    intermediate_dir: str,
    tool_name: str,
) -> Optional[str]:
    """Return a local file path for *image_ref* (downloading if needed).

    Handles: local path str, URL str, bytes, PIL.Image, HuggingFace dict
    ({"bytes": ..., "path": ...}), and base64 data URIs.
    """
    if image_ref not in image_paths:
        return None
    data = image_paths[image_ref]

    # Already a local file path
    if isinstance(data, str) and len(data) < 500 and os.path.exists(data):
        return data

    # URL → download
    if isinstance(data, str) and data.startswith(("http://", "https://")):
        img_bytes = _download_image_bytes(data)
        if img_bytes is None:
            return None
        os.makedirs(intermediate_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
        with open(local, "wb") as f:
            f.write(img_bytes)
        image_paths[image_ref] = local
        return local

    # Base64 data URI → decode
    if isinstance(data, str) and data.startswith("data:image/"):
        try:
            _, encoded = data.split(",", 1)
            raw = base64.b64decode(encoded)
            os.makedirs(intermediate_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
            Image.open(io.BytesIO(raw)).save(local)
            image_paths[image_ref] = local
            return local
        except Exception:
            pass

    # Raw bytes
    if isinstance(data, bytes):
        os.makedirs(intermediate_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
        with open(local, "wb") as f:
            f.write(data)
        image_paths[image_ref] = local
        return local

    # PIL Image
    if isinstance(data, Image.Image):
        os.makedirs(intermediate_dir, exist_ok=True)
        ts = int(time.time() * 1000)
        local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
        data.save(local)
        image_paths[image_ref] = local
        return local

    # HuggingFace dict format: {"bytes": b"...", "path": "..."}
    if isinstance(data, dict):
        pil = None
        if "bytes" in data and data["bytes"] is not None:
            try:
                pil = Image.open(io.BytesIO(data["bytes"])).convert("RGB")
            except Exception:
                pass
        if pil is None and "path" in data and isinstance(data["path"], str):
            try:
                pil = Image.open(data["path"]).convert("RGB")
            except Exception:
                pass
        if pil is None:
            d_str = data.get("data") or data.get("url") or ""
            if isinstance(d_str, str) and d_str.startswith("data:image/"):
                try:
                    _, encoded = d_str.split(",", 1)
                    pil = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
                except Exception:
                    pass
        if pil is not None:
            os.makedirs(intermediate_dir, exist_ok=True)
            ts = int(time.time() * 1000)
            local = os.path.join(intermediate_dir, f"vdr_{tool_name}_{ts}.png")
            pil.save(local)
            image_paths[image_ref] = local
            return local

    print(f"[visual_tools] _resolve_image_ref: unhandled data type {type(data).__name__} for {image_ref}")
    return None


# ===================================================================
# Helper: ensure an image ref becomes a public URL (upload if needed)
# ===================================================================

def _ensure_image_url(
    image_ref: str,
    image_paths: dict,
    intermediate_dir: str,
    tool_name: str,
) -> Optional[str]:
    """
    Resolve *image_ref* to a public URL suitable for APIs that require one.

    Priority:
      1. Already a URL string in image_paths → return directly
      2. Local file / PIL / bytes → upload to COS → return URL
      3. image_ref itself looks like a URL → return as-is

    On success the URL is cached back into image_paths[image_ref] so
    subsequent calls skip the upload.
    """
    if image_ref in image_paths:
        data = image_paths[image_ref]
        # Already a URL
        if isinstance(data, str) and data.startswith(("http://", "https://")):
            return data

        # Resolve to a local file first
        local = _resolve_image_ref(image_ref, image_paths, intermediate_dir, tool_name)
        if local:
            try:
                pil = Image.open(local)
                cos_url = _upload_pil_to_cos(pil, tool_name)
                if cos_url:
                    image_paths[image_ref] = cos_url
                    return cos_url
            except Exception as exc:
                print(f"[visual_tools] _ensure_image_url: failed to upload {image_ref}: {exc}")
        return None

    # image_ref is not a known key — maybe it's already a raw URL
    if isinstance(image_ref, str) and image_ref.startswith(("http://", "https://")):
        return image_ref

    return None


# ===================================================================
# Layout Parsing standalone function
# ===================================================================

def layout_parsing(
    file_path: str,
    use_chart_recognition: bool = False,
    use_doc_orientation_classify: bool = False,
) -> dict:
    """Call the remote Layout Parsing API and return structured result."""
    try:
        if not os.path.exists(file_path):
            return {"error": f"File not found: {file_path}", "rec_texts": [], "formatted_text": "", "blocks": []}
        with open(file_path, "rb") as fh:
            file_data = base64.b64encode(fh.read()).decode("ascii")

        headers = {
            "Authorization": f"token {LAYOUT_PARSING_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "file": file_data,
            "fileType": 1,
            "useDocOrientationClassify": use_doc_orientation_classify,
            "useDocUnwarping": False,
            "useChartRecognition": use_chart_recognition,
        }
        resp = requests.post(LAYOUT_PARSING_API_URL, json=payload, headers=headers, timeout=60)
        if resp.status_code != 200:
            return {"error": f"API HTTP {resp.status_code}: {resp.text[:200]}", "rec_texts": [], "formatted_text": "", "blocks": []}

        result = resp.json().get("result", {})
        TEXT_LABELS = {"paragraph_title", "text", "vision_footnote"}
        blocks = (
            result.get("layoutParsingResults", [{}])[0]
            .get("prunedResult", {})
            .get("parsing_res_list", [])
        )
        texts = [
            blk["block_content"].strip()
            for blk in blocks
            if blk.get("block_label") in TEXT_LABELS and blk.get("block_content", "").strip()
        ]
        return {"rec_texts": texts, "formatted_text": "\n".join(texts), "blocks": blocks, "error": None}
    except Exception as exc:
        return {"error": str(exc), "rec_texts": [], "formatted_text": "", "blocks": []}


# ===================================================================
# ImageEnhancementEngine (perspective_correct, super_resolution, sharpen)
# ===================================================================

class ImageEnhancementEngine:
    """OpenCV-based image enhancement."""

    def __init__(self):
        self.current_image = None

    def load_from_path(self, path: str):
        self.current_image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
        return self

    def load_from_pil(self, pil_img: Image.Image):
        self.current_image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return self

    # -- perspective correct --
    def auto_correct_perspective(self):
        if self.current_image is None:
            return None
        if len(self.current_image.shape) == 3 and self.current_image.shape[2] == 4:
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 75, 200)
        cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]
        screen_cnt = None
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                screen_cnt = approx
                break
        if screen_cnt is None:
            return self.current_image
        pts = screen_cnt.reshape(4, 2).astype("float32")
        s = pts.sum(axis=1)
        rect = np.zeros((4, 2), dtype="float32")
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        (tl, tr, br, bl) = rect
        wA = np.linalg.norm(br - bl)
        wB = np.linalg.norm(tr - tl)
        maxW = max(int(wA), int(wB))
        hA = np.linalg.norm(tr - br)
        hB = np.linalg.norm(tl - bl)
        maxH = max(int(hA), int(hB))
        dst = np.array([[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]], dtype="float32")
        M = cv2.getPerspectiveTransform(rect, dst)
        self.current_image = cv2.warpPerspective(self.current_image, M, (maxW, maxH))
        return self.current_image

    # -- super resolution --
    def apply_super_resolution(self, model_path: str = "EDSR_x4.pb", scale: int = 4):
        if self.current_image is None:
            return None
        if not os.path.exists(model_path):
            return self.current_image
        try:
            sr = cv2.dnn_superres.DnnSuperResImpl_create()
            sr.readModel(model_path)
            sr.setModel("edsr", scale)
            self.current_image = sr.upsample(self.current_image)
        except Exception:
            pass
        return self.current_image

    # -- sharpen --
    def enhance_sharpness(self, amount: float = 1.5):
        if self.current_image is None:
            return None
        blurred = cv2.GaussianBlur(self.current_image, (0, 0), 3)
        self.current_image = cv2.addWeighted(self.current_image, 1.0 + amount, blurred, -amount, 0)
        return self.current_image

    def to_pil(self) -> Optional[Image.Image]:
        if self.current_image is None:
            return None
        return Image.fromarray(cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB))


# ===================================================================
# Qwen summariser helper
# ===================================================================

def _summarize_with_qwen(content: str, query: str, title: str) -> str:
    """Use an external Qwen model to summarise web-page content."""
    if not QWEN_API_BASE or not QWEN_MODEL:
        return content[:500] + ("..." if len(content) > 500 else "")
    try:
        prompt = (
            f'Based on the following webpage content, provide a concise summary '
            f'relevant to the query: "{query}"\n\n'
            f'Webpage Title: {title}\nContent:\n{content[:2000]}\n\n'
            f'Provide a focused summary (2-4 sentences) that directly addresses the query.'
        )
        payload = {
            "model": QWEN_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.3,
        }
        resp = requests.post(
            f"{QWEN_API_BASE}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if resp.status_code == 200:
            choices = resp.json().get("choices", [])
            if choices:
                text = choices[0].get("message", {}).get("content", "")
                if text:
                    return text.strip()
    except Exception:
        pass
    return content[:500] + ("..." if len(content) > 500 else "")


# ===================================================================
# Tool classes
# ===================================================================

class CropTool(DeepResearchTool):
    """Crop a specific region from an image."""

    def __init__(self):
        super().__init__(
            name="crop",
            description="Crop a specific region from an image.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference (e.g., 'img_1')"},
                    "x": {"type": "integer", "description": "Starting X coordinate"},
                    "y": {"type": "integer", "description": "Starting Y coordinate"},
                    "width": {"type": "integer", "description": "Width of the crop region"},
                    "height": {"type": "integer", "description": "Height of the crop region"},
                },
                "required": ["image", "x", "y", "width", "height"],
            },
        )

    async def call(self, image: str = "", x: int = 0, y: int = 0,
                   width: int = 0, height: int = 0, **ctx) -> str:
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not image or image not in image_paths:
            return f"Error: Image reference '{image}' not found. Available: {list(image_paths.keys())}"
        if width <= 0 or height <= 0:
            return "Error: width and height must be positive integers"

        local = _resolve_image_ref(image, image_paths, intermediate_dir, "crop")
        if not local:
            return f"Error: Failed to resolve image for {image}"

        try:
            pil_img = Image.open(local)
            cropped = pil_img.crop((x, y, x + width, y + height))
            new_id = f"img_{len(image_paths) + 1}"

            cos_url = _upload_pil_to_cos(cropped, "crop")
            if cos_url:
                image_paths[new_id] = cos_url
                return f"Image cropped successfully. New image ID: {new_id}. Uploaded to: {cos_url}"

            os.makedirs(intermediate_dir, exist_ok=True)
            save = os.path.join(intermediate_dir, f"vdr_crop_{int(time.time()*1000)}.png")
            cropped.save(save)
            image_paths[new_id] = save
            return f"Image cropped successfully. New image ID: {new_id}. Saved to: {save}"
        except Exception as exc:
            return f"Error executing crop: {exc}"


class LayoutParsingTool(DeepResearchTool):
    """Document layout parsing and text extraction via remote API."""

    def __init__(self):
        super().__init__(
            name="layout_parsing",
            description="Perform document layout parsing on an image to extract structured text.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference (e.g., 'img_1')"},
                    "file_path": {"type": "string", "description": "Absolute path to an image file (optional)."},
                    "use_chart_recognition": {"type": "boolean", "default": False},
                    "use_doc_orientation_classify": {"type": "boolean", "default": False},
                },
                "required": [],
            },
        )

    async def call(self, image: str = "", file_path: str = "",
                   use_chart_recognition: bool = False,
                   use_doc_orientation_classify: bool = False, **ctx) -> str:
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not file_path and image:
            local = _resolve_image_ref(image, image_paths, intermediate_dir, "layout_parsing")
            if local:
                file_path = local

        if not file_path:
            return "Error: 'file_path' or 'image' parameter is required for layout_parsing"

        result = await asyncio.get_event_loop().run_in_executor(
            None, layout_parsing, file_path, use_chart_recognition, use_doc_orientation_classify,
        )

        if result.get("error"):
            return f"Tool execution error:\n{result['error']}"

        rec_texts = result.get("rec_texts", [])
        formatted = result.get("formatted_text", "")
        parts = []
        if rec_texts:
            parts.append(f"Layout Parsing SUCCESS: {len(rec_texts)} text blocks detected.")
            parts.append("")
            for i, t in enumerate(rec_texts, 1):
                parts.append(f"[Text Block {i}] {t}")
            parts.append("")
            parts.append("=" * 50)
            parts.append("ALL RECOGNIZED TEXT:")
            parts.append(formatted)
            parts.append("=" * 50)
        else:
            parts.append("Layout Parsing: No text blocks detected.")
        return "\n".join(parts)


class TextSearchTool(DeepResearchTool):
    """Search web + JINA reader + Qwen summarisation (Serper → JINA → Qwen3-32B)."""

    def __init__(self):
        super().__init__(
            name="text_search",
            description="Search the web and return AI-summarised passages from top-k pages.",
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query"},
                    "query": {"type": "string", "description": "Alternative for q"},
                    "hl": {"type": "string", "default": "en"},
                    "top_k": {"type": "integer", "default": 5},
                },
                "required": [],
            },
        )

    async def call(self, q: str = "", query: str = "", hl: str = "en",
                   top_k: int = 5, lang: str = "", **ctx) -> str:
        q = q or query
        hl = hl or lang or "en"
        if not q:
            return "Error: 'q' or 'query' parameter is required"

        proxies = self._get_requests_proxies()
        use_gateway = bool(API_USER and API_KEY)

        # Step 1 – Serper search
        if use_gateway:
            headers = {
                "Authorization": f"Bearer {API_USER}:{API_KEY}?provider=serper&timeout=60",
                "Content-Type": "application/json",
            }
            search_url = f"{API_HOST}/search"
        else:
            serp_key = os.getenv("SERP_API_KEY", "")
            headers = {"X-API-KEY": serp_key, "Content-Type": "application/json"}
            search_url = SERP_SEARCH_URL

        payload = {"q": q, "location": "United States", "hl": hl, "num": min(top_k, 20)}

        try:
            def _do_search():
                return requests.post(search_url, headers=headers, json=payload, timeout=60, proxies=proxies)

            resp = await run_with_retries_async(_do_search, executor=self.executor)
            if resp.status_code != 200:
                return f"Serper error: HTTP {resp.status_code}: {resp.text[:200]}"
            organic = resp.json().get("organic", [])
            if not organic:
                return f"No results found for '{q}'"
        except Exception as exc:
            return f"Search error: {exc}"

        # Step 2 – JINA reader + Qwen summary
        jina_headers = {}
        if use_gateway:
            jina_headers = {
                "Authorization": f"Bearer {API_USER}:{API_KEY}?provider=jina_ai&timeout=60",
                "Content-Type": "application/json",
            }

        formatted = []
        for i, item in enumerate(organic[:top_k], 1):
            url = item.get("link", "")
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            page_content = snippet

            if url and jina_headers:
                try:
                    def _jina(u=url):
                        return requests.post(
                            f"{API_HOST}/images", headers=jina_headers,
                            json={"url": u}, timeout=30, proxies=proxies,
                        )

                    jr = await run_with_retries_async(_jina, executor=self.executor)
                    if jr.status_code == 200:
                        jd = jr.json()
                        if isinstance(jd, dict):
                            c = jd.get("data", {}).get("content", "") or jd.get("content", "")
                            if c:
                                page_content = c
                except Exception:
                    pass

            summary = await asyncio.get_event_loop().run_in_executor(
                None, _summarize_with_qwen, page_content, q, title,
            )
            formatted.append(f"[Passage {i}]\nTitle: {title}\nURL: {url}\nSummary:\n{summary}")

        return "\n\n".join(formatted)


class WebSearchTool(DeepResearchTool):
    """Simple Serper web search (returns snippets, no JINA/Qwen)."""

    def __init__(self):
        super().__init__(
            name="web_search",
            description="Perform a web search and return snippet results.",
            parameters={
                "type": "object",
                "properties": {
                    "q": {"type": "string", "description": "Search query keywords"},
                    "hl": {"type": "string", "default": "en"},
                },
                "required": ["q"],
            },
        )

    async def call(self, q: str = "", hl: str = "en", **ctx) -> str:
        if not q:
            return "Error: 'q' parameter is required"

        proxies = self._get_requests_proxies()
        use_gateway = bool(API_USER and API_KEY)

        if use_gateway:
            headers = {
                "Authorization": f"Bearer {API_USER}:{API_KEY}?provider=serper&timeout=60",
                "Content-Type": "application/json",
            }
            url = f"{API_HOST}/search"
        else:
            serp_key = os.getenv("SERP_API_KEY", "")
            headers = {"X-API-KEY": serp_key, "Content-Type": "application/json"}
            url = SERP_SEARCH_URL

        payload = {"q": q, "location": "United States", "hl": hl, "num": 10}

        try:
            def _do():
                return requests.post(url, headers=headers, json=payload, timeout=60, proxies=proxies)

            resp = await run_with_retries_async(_do, executor=self.executor)
            if resp.status_code != 200:
                return f"Search error: HTTP {resp.status_code}"
            items = resp.json().get("organic", [])
            if not items:
                return f"No results for '{q}'"
        except Exception as exc:
            return f"Search error: {exc}"

        snippets = []
        for idx, it in enumerate(items[:10], 1):
            t = it.get("title", "Untitled")
            link = it.get("link", "")
            snip = it.get("snippet", "")
            entry = f"{idx}. [{t}]({link})"
            if snip:
                entry += f"\n   {snip}"
            snippets.append(entry)
        return f"Search for '{q}' returned {len(snippets)} results:\n\n" + "\n\n".join(snippets)


class ImageSearchTool(DeepResearchTool):
    """Visual search via Polaris Lens API."""

    def __init__(self):
        super().__init__(
            name="image_search",
            description="Analyze an image using visual search to identify objects.",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Image reference (e.g., 'img_1') or direct URL"},
                },
                "required": ["url"],
            },
        )

    async def call(self, url: str = "", **ctx) -> str:
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not url:
            return "Error: 'url' parameter is required"

        image_url = _ensure_image_url(url, image_paths, intermediate_dir, "image_search")
        if not image_url:
            avail = list(image_paths.keys())
            return (
                f"Error: Cannot resolve '{url}' to a public URL for image_search. "
                f"COS upload available: {UPLOAD_AVAILABLE}. "
                f"Known image refs: {avail}"
            )

        if _env_lens_scan is None:
            return "Error: lens_scan (env module) is not available"

        try:
            def _do_lens():
                return _env_lens_scan(image_url=image_url)

            result = await asyncio.get_event_loop().run_in_executor(None, _do_lens)
            if isinstance(result, dict) and "error" in result:
                return f"image_search error: {result['error']}"
            return f"Tool execution result:\n{json.dumps(result, ensure_ascii=False, indent=2)}"
        except Exception as exc:
            return f"image_search error: {exc}"


class PerspectiveCorrectTool(DeepResearchTool):
    """Auto perspective correction via OpenCV."""

    def __init__(self):
        super().__init__(
            name="perspective_correct",
            description="Correct perspective distortion in an image.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference"},
                },
                "required": ["image"],
            },
        )

    async def call(self, image: str = "", **ctx) -> str:
        if not CV2_AVAILABLE:
            return "Error: OpenCV not available. pip install opencv-python"
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not image or image not in image_paths:
            return f"Error: Image '{image}' not found. Available: {list(image_paths.keys())}"

        local = _resolve_image_ref(image, image_paths, intermediate_dir, "perspective_correct")
        if not local:
            return f"Error: Failed to resolve image for {image}"

        try:
            eng = ImageEnhancementEngine()
            eng.load_from_path(local)
            eng.auto_correct_perspective()
            pil = eng.to_pil()
            if not pil:
                return "Error: perspective correction failed"

            new_id = f"img_{len(image_paths) + 1}"
            cos = _upload_pil_to_cos(pil, "perspective_correct")
            if cos:
                image_paths[new_id] = cos
                return f"Perspective corrected. New image ID: {new_id}. Uploaded to: {cos}"
            save = os.path.join(intermediate_dir, f"vdr_persp_{int(time.time()*1000)}.png")
            os.makedirs(intermediate_dir, exist_ok=True)
            pil.save(save)
            image_paths[new_id] = save
            return f"Perspective corrected. New image ID: {new_id}. Saved to: {save}"
        except Exception as exc:
            return f"Error: {exc}"


class SuperResolutionTool(DeepResearchTool):
    """Super resolution via OpenCV DNN."""

    def __init__(self):
        super().__init__(
            name="super_resolution",
            description="Enhance image resolution using super-resolution.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference"},
                    "scale": {"type": "integer", "default": 4},
                },
                "required": ["image"],
            },
        )

    async def call(self, image: str = "", scale: int = 4, **ctx) -> str:
        if not CV2_AVAILABLE:
            return "Error: OpenCV not available. pip install opencv-python"
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not image or image not in image_paths:
            return f"Error: Image '{image}' not found."

        local = _resolve_image_ref(image, image_paths, intermediate_dir, "super_resolution")
        if not local:
            return f"Error: Failed to resolve image for {image}"

        try:
            eng = ImageEnhancementEngine()
            eng.load_from_path(local)
            eng.apply_super_resolution(scale=scale)
            pil = eng.to_pil()
            if not pil:
                return "Error: super resolution failed"

            new_id = f"img_{len(image_paths) + 1}"
            cos = _upload_pil_to_cos(pil, "super_resolution")
            if cos:
                image_paths[new_id] = cos
                return f"Super resolution done. New image ID: {new_id}. Uploaded to: {cos}"
            save = os.path.join(intermediate_dir, f"vdr_sr_{int(time.time()*1000)}.png")
            os.makedirs(intermediate_dir, exist_ok=True)
            pil.save(save)
            image_paths[new_id] = save
            return f"Super resolution done. New image ID: {new_id}. Saved to: {save}"
        except Exception as exc:
            return f"Error: {exc}"


class SharpenTool(DeepResearchTool):
    """Sharpen / unsharp-mask via OpenCV."""

    def __init__(self):
        super().__init__(
            name="sharpen",
            description="Sharpen an image to reduce blur.",
            parameters={
                "type": "object",
                "properties": {
                    "image": {"type": "string", "description": "Image reference"},
                    "amount": {"type": "number", "default": 1.5},
                },
                "required": ["image"],
            },
        )

    async def call(self, image: str = "", amount: float = 1.5, **ctx) -> str:
        if not CV2_AVAILABLE:
            return "Error: OpenCV not available. pip install opencv-python"
        image_paths: dict = ctx.get("image_paths", {})
        intermediate_dir: str = ctx.get("intermediate_dir", "/tmp/vdr_tools")

        if not image or image not in image_paths:
            return f"Error: Image '{image}' not found."

        local = _resolve_image_ref(image, image_paths, intermediate_dir, "sharpen")
        if not local:
            return f"Error: Failed to resolve image for {image}"

        try:
            eng = ImageEnhancementEngine()
            eng.load_from_path(local)
            eng.enhance_sharpness(amount=amount)
            pil = eng.to_pil()
            if not pil:
                return "Error: sharpening failed"

            new_id = f"img_{len(image_paths) + 1}"
            cos = _upload_pil_to_cos(pil, "sharpen")
            if cos:
                image_paths[new_id] = cos
                return f"Image sharpened. New image ID: {new_id}. Uploaded to: {cos}"
            save = os.path.join(intermediate_dir, f"vdr_sharp_{int(time.time()*1000)}.png")
            os.makedirs(intermediate_dir, exist_ok=True)
            pil.save(save)
            image_paths[new_id] = save
            return f"Image sharpened. New image ID: {new_id}. Saved to: {save}"
        except Exception as exc:
            return f"Error: {exc}"
