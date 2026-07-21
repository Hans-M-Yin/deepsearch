"""Sample image nodes from a graph and test whether each image_url appears in its source page markdown.

Uses the raw markdown reader (default port 8003) so URLs are preserved as much as possible.

Example:
  python debug/check_image_url_in_source_page.py \
    --graph-dir runs/0712_multi_seed_visual_test_8192_6 \
    --sample-size 100 \
    --pretty
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.store import JsonlGraphStore
from synthesis.wiki_text_builder import RawMarkdownReaderClient


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check whether image_url appears in source_page markdown for sampled image nodes.")
    parser.add_argument("--graph-dir", required=True, help="Directory containing nodes.jsonl and edges.jsonl.")
    parser.add_argument("--sample-size", type=int, default=100, help="Maximum number of image nodes to sample.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument("--reader-base-url", type=str, default="http://127.0.0.1:8003/raw", help="Raw markdown Reader base URL.")
    parser.add_argument("--reader-timeout-s", type=float, default=180.0, help="Raw markdown Reader timeout.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _image_nodes(store: JsonlGraphStore) -> list[dict[str, Any]]:
    nodes: list[dict[str, Any]] = []
    for node in store.list_nodes():
        if node.get("node_type") != "image":
            continue
        image_url = str(node.get("image_url") or "").strip()
        source_page_url = str(node.get("source_page_url") or "").strip()
        if not image_url or not source_page_url:
            continue
        nodes.append(node)
    return nodes


def _url_forms(url: str) -> list[str]:
    text = str(url or "").strip()
    if not text:
        return []
    parsed = urlparse(text)
    forms: list[str] = []
    forms.append(text)
    if parsed.scheme and parsed.netloc:
        no_fragment = parsed._replace(fragment="").geturl()
        if no_fragment not in forms:
            forms.append(no_fragment)
        no_query_no_fragment = parsed._replace(query="", fragment="").geturl()
        if no_query_no_fragment not in forms:
            forms.append(no_query_no_fragment)
        path_only = parsed.path or ""
        if path_only and path_only not in forms:
            forms.append(path_only)
        decoded_path = unquote(path_only)
        if decoded_path and decoded_path not in forms:
            forms.append(decoded_path)
        basename = Path(decoded_path).name
        if basename and basename not in forms:
            forms.append(basename)
    return forms


def _match_info(page_markdown: str, image_url: str) -> dict[str, Any]:
    text = page_markdown or ""
    forms = _url_forms(image_url)
    exact = next((item for item in forms if item and item in text), None)
    if exact:
        return {"matched": True, "match_type": "substring", "matched_form": exact}

    basename = Path(unquote(urlparse(image_url).path or "")).name
    if basename:
        escaped = re.escape(basename)
        if re.search(escaped, text):
            return {"matched": True, "match_type": "basename_regex", "matched_form": basename}

    return {"matched": False, "match_type": "none", "matched_form": None}


def main() -> int:
    args = parse_args()
    store = JsonlGraphStore(Path(args.graph_dir))
    reader = RawMarkdownReaderClient(base_url=args.reader_base_url, timeout_s=args.reader_timeout_s)

    nodes = _image_nodes(store)
    random.Random(args.seed).shuffle(nodes)
    sampled = nodes[: max(0, int(args.sample_size))]

    results: list[dict[str, Any]] = []
    match_counter: Counter[str] = Counter()
    error_counter: Counter[str] = Counter()

    for node in sampled:
        node_id = str(node.get("node_id") or "")
        image_url = str(node.get("image_url") or "").strip()
        source_page_url = str(node.get("source_page_url") or "").strip()
        title = _normalize_text(node.get("title") or node.get("caption") or node.get("summary") or node_id)
        try:
            document = reader.read(source_page_url)
            markdown = document.raw_markdown or document.content or ""
            match = _match_info(markdown, image_url)
            record = {
                "node_id": node_id,
                "title": title,
                "image_url": image_url,
                "source_page_url": source_page_url,
                "matched": bool(match["matched"]),
                "match_type": match["match_type"],
                "matched_form": match["matched_form"],
                "markdown_chars": len(markdown),
            }
            match_counter[record["match_type"]] += 1
        except Exception as exc:
            error_key = f"{exc.__class__.__name__}"
            error_counter[error_key] += 1
            record = {
                "node_id": node_id,
                "title": title,
                "image_url": image_url,
                "source_page_url": source_page_url,
                "matched": False,
                "match_type": "reader_error",
                "matched_form": None,
                "error": f"{exc.__class__.__name__}: {exc}",
            }
        results.append(record)

    matched_count = sum(1 for item in results if item.get("matched"))
    payload = {
        "graph_dir": str(Path(args.graph_dir).resolve()),
        "reader_base_url": args.reader_base_url,
        "sample_size_requested": int(args.sample_size),
        "sample_size_actual": len(sampled),
        "matched_count": matched_count,
        "matched_rate": (matched_count / len(sampled)) if sampled else 0.0,
        "match_type_counts": dict(match_counter),
        "reader_error_counts": dict(error_counter),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
