#!/usr/bin/env python3
"""Render one review row per image node in a graph run."""

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


PAGE_WIDTH = 2200
MARGIN_X = 42
MARGIN_Y = 28
HEADER_H = 132
ROW_GAP = 22
ROW_H = 340
SOURCE_W = 330
ARROW_W = 140
IMAGE_W = 300
IMAGE_H = 210
DETAIL_W = 1220
CARD_PAD = 18


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def esc(value):
    return html.escape(str(value or ""))


def short(value, limit):
    text = str(value or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "…"


def wrap_text(value, width, max_lines):
    text = str(value or "").replace("\n", " ").strip()
    if not text:
        return []
    words = text.split()
    lines = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = short(lines[-1], width)
    return lines


def image_origin(node):
    metadata = node.get("metadata") or {}
    source = node.get("source") or {}
    source_type = str(source.get("source_type") or "").strip()
    origin = str(metadata.get("image_origin") or "").strip().lower()
    variant_sources = {
        str(variant.get("source") or "").strip()
        for variant in node.get("image_variants") or []
        if isinstance(variant, dict)
    }
    if source_type == "wikipedia_inline_image" or origin == "wikipedia_inline" or "wikipedia_inline" in variant_sources:
        return "wiki_inline"
    if source_type in {"image_search", "image_search_bundle"}:
        return "visual_plan"
    return f"other:{source_type or 'unknown'}"


def image_href(node):
    candidates = [node.get("image_url"), node.get("thumb_oss_uri"), node.get("oss_uri")]
    for variant in node.get("image_variants") or []:
        if isinstance(variant, dict):
            candidates.extend([variant.get("thumbnail_url"), variant.get("image_url")])
    return next((str(url) for url in candidates if url), None)


def node_title(node):
    return node.get("title") or node.get("caption") or node.get("canonical_id") or node.get("node_id") or "Untitled"


def edge_relation(edge):
    metadata = edge.get("metadata") or {}
    relation_info = metadata.get("relation_info") if isinstance(metadata.get("relation_info"), dict) else {}
    return edge.get("relation") or relation_info.get("predicate") or metadata.get("query") or edge.get("edge_type") or ""


def search_query(node):
    metadata = node.get("metadata") or {}
    return metadata.get("search_query") or metadata.get("query") or ""


def grounded_entities(node):
    metadata = node.get("metadata") or {}
    entities = metadata.get("grounded_entities") or node.get("grounded_entities") or []
    rendered = []
    for entity in entities:
        if isinstance(entity, dict):
            name = entity.get("name") or entity.get("entity") or entity.get("title") or "unnamed entity"
            relation = entity.get("relation_to_image") or entity.get("relation") or ""
            evidence = entity.get("evidence") or ""
            parts = [str(name)]
            if relation:
                parts.append(str(relation))
            if evidence:
                parts.append(str(evidence))
            rendered.append(" — ".join(parts))
        elif entity:
            rendered.append(str(entity))
    return rendered


def image_records(nodes, edges):
    nodes_by_id = {node.get("node_id"): node for node in nodes if node.get("node_id")}
    incoming = defaultdict(list)
    for edge in edges:
        src, dst = edge.get("src_node_id"), edge.get("dst_node_id")
        if src in nodes_by_id and dst in nodes_by_id:
            incoming[dst].append(edge)

    records = []
    for node in nodes:
        if node.get("node_type") != "image" or not node.get("node_id"):
            continue
        origin = image_origin(node)
        source_node = None
        source_edge = None
        # Only a visual-plan image represents an explicit text -> image search.
        # Wiki-inline images intentionally leave the source text card empty.
        if origin == "visual_plan":
            candidates = []
            for edge in incoming.get(node["node_id"], []):
                candidate = nodes_by_id.get(edge.get("src_node_id"))
                if candidate and candidate.get("node_type") == "text":
                    candidates.append((edge, candidate))
            if candidates:
                candidates.sort(key=lambda item: (item[0].get("edge_type") != "search_retrieved", item[1].get("node_id", "")))
                source_edge, source_node = candidates[0]
        records.append(
            {
                "node": node,
                "origin": origin,
                "source_node": source_node,
                "source_edge": source_edge,
            }
        )
    return sorted(records, key=lambda record: (record["origin"], node_title(record["node"]).lower(), record["node"]["node_id"]))


def text(svg, x, y, value, *, size=14, fill="#334155", weight="normal", anchor=None):
    anchor_attr = f' text-anchor="{anchor}"' if anchor else ""
    svg.append(
        f'<text x="{x:.1f}" y="{y:.1f}"{anchor_attr} font-family="Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{esc(value)}</text>'
    )


def render_lines(svg, x, y, value, *, width, max_lines, size=14, fill="#334155", line_gap=18, weight="normal"):
    for index, line in enumerate(wrap_text(value, width, max_lines)):
        text(svg, x, y + index * line_gap, line, size=size, fill=fill, weight=weight)


def render_source_card(svg, x, y, source_node):
    svg.append(f'<rect x="{x}" y="{y}" width="{SOURCE_W}" height="{ROW_H}" rx="18" fill="#ffffff" stroke="#93c5fd" stroke-width="1.5"/>')
    svg.append(f'<rect x="{x}" y="{y}" width="{SOURCE_W}" height="36" rx="18" fill="#dbeafe"/>')
    svg.append(f'<rect x="{x}" y="{y + 18}" width="{SOURCE_W}" height="18" fill="#dbeafe"/>')
    text(svg, x + CARD_PAD, y + 24, "SOURCE TEXT NODE", size=11, fill="#2563eb", weight="bold")
    if source_node is None:
        text(svg, x + SOURCE_W / 2, y + 154, "No source text node", size=16, fill="#94a3b8", weight="bold", anchor="middle")
        text(svg, x + SOURCE_W / 2, y + 180, "Expected for wiki_inline images", size=12, fill="#94a3b8", anchor="middle")
        return
    render_lines(svg, x + CARD_PAD, y + 68, node_title(source_node), width=34, max_lines=3, size=18, fill="#0f172a", weight="bold", line_gap=22)
    canonical_id = source_node.get("canonical_id") or source_node.get("node_id")
    render_lines(svg, x + CARD_PAD, y + 150, canonical_id, width=43, max_lines=2, size=12, fill="#64748b")
    summary = source_node.get("summary") or source_node.get("description") or ""
    if summary:
        render_lines(svg, x + CARD_PAD, y + 210, summary, width=42, max_lines=5, size=13, fill="#475569")


def render_image_card(svg, x, y, record):
    node = record["node"]
    origin = record["origin"]
    svg.append(f'<rect x="{x}" y="{y}" width="{IMAGE_W}" height="{ROW_H}" rx="18" fill="#ffffff" stroke="#f0abfc" stroke-width="1.5"/>')
    svg.append(f'<rect x="{x}" y="{y}" width="{IMAGE_W}" height="36" rx="18" fill="#fce7f3"/>')
    svg.append(f'<rect x="{x}" y="{y + 18}" width="{IMAGE_W}" height="18" fill="#fce7f3"/>')
    text(svg, x + CARD_PAD, y + 24, f"IMAGE NODE · {origin}", size=11, fill="#be185d", weight="bold")
    image_x, image_y = x + CARD_PAD, y + 54
    image_w, image_h = IMAGE_W - 2 * CARD_PAD, IMAGE_H
    svg.append(f'<rect x="{image_x}" y="{image_y}" width="{image_w}" height="{image_h}" rx="12" fill="#fff7ed" stroke="#fed7aa"/>')
    href = image_href(node)
    if href:
        svg.append(f'<image href="{esc(href)}" x="{image_x + 1}" y="{image_y + 1}" width="{image_w - 2}" height="{image_h - 2}" preserveAspectRatio="xMidYMid slice"/>')
    else:
        text(svg, image_x + image_w / 2, image_y + image_h / 2, "No image URL", size=14, fill="#9a6700", anchor="middle")
    render_lines(svg, x + CARD_PAD, y + 292, node_title(node), width=32, max_lines=2, size=13, fill="#334155", weight="bold")


def render_details(svg, x, y, record):
    node = record["node"]
    svg.append(f'<rect x="{x}" y="{y}" width="{DETAIL_W}" height="{ROW_H}" rx="18" fill="#ffffff" stroke="#d7e0ea" stroke-width="1.5"/>')
    text(svg, x + CARD_PAD, y + 28, "IMAGE NODE METADATA", size=12, fill="#475569", weight="bold")
    fields = [
        ("type", record["origin"]),
        ("search_query", search_query(node) or "—"),
        ("title", node.get("title") or "—"),
        ("caption", node.get("caption") or node.get("summary") or "—"),
    ]
    field_y = y + 60
    for label, value in fields:
        text(svg, x + CARD_PAD, field_y, label, size=13, fill="#64748b", weight="bold")
        render_lines(svg, x + 160, field_y, value, width=105, max_lines=2, size=14, fill="#0f172a", line_gap=17)
        field_y += 48

    text(svg, x + CARD_PAD, y + 258, "grounded_entities", size=13, fill="#64748b", weight="bold")
    entities = grounded_entities(node)
    if not entities:
        text(svg, x + 160, y + 258, "—", size=14, fill="#94a3b8")
    else:
        for index, entity in enumerate(entities[:4]):
            render_lines(svg, x + 160, y + 258 + index * 20, f"• {entity}", width=104, max_lines=1, size=13, fill="#334155")
        if len(entities) > 4:
            text(svg, x + 160, y + 338, f"+{len(entities) - 4} more", size=12, fill="#64748b")


def render_svg(run_dir, records, total_images):
    row_count = max(1, len(records))
    height = HEADER_H + MARGIN_Y * 2 + row_count * ROW_H + max(0, row_count - 1) * ROW_GAP
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_WIDTH}" height="{height}" viewBox="0 0 {PAGE_WIDTH} {height}">',
        '<rect width="100%" height="100%" fill="#f6f8fb"/>',
        '<rect x="0" y="0" width="100%" height="100" fill="#0f172a"/>',
        '<rect x="0" y="100" width="100%" height="32" fill="#e6ecf5"/>',
        '<defs><marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto"><path d="M0,0 L0,6 L8,3 z" fill="#64748b"/></marker></defs>',
    ]
    text(svg, 30, 42, "Image Node Review", size=29, fill="#ffffff", weight="bold")
    text(svg, 30, 72, f"run: {run_dir}   image_nodes: {len(records)}/{total_images}", size=15, fill="#cbd5e1")
    text(svg, 30, 121, "Each row shows an image node, its image metadata, and its visual-plan source text node when that source edge exists.", size=13, fill="#475569")

    if not records:
        svg.append(f'<rect x="{MARGIN_X}" y="{HEADER_H + MARGIN_Y}" width="{PAGE_WIDTH - 2 * MARGIN_X}" height="130" rx="18" fill="#ffffff" stroke="#d7e0ea"/>')
        text(svg, MARGIN_X + 24, HEADER_H + MARGIN_Y + 56, "No image nodes found", size=22, fill="#0f172a", weight="bold")
        svg.append("</svg>")
        return "\n".join(svg)

    source_x = MARGIN_X
    arrow_x = source_x + SOURCE_W
    image_x = arrow_x + ARROW_W
    details_x = image_x + IMAGE_W + 22
    for index, record in enumerate(records):
        y = HEADER_H + MARGIN_Y + index * (ROW_H + ROW_GAP)
        render_source_card(svg, source_x, y, record["source_node"])
        svg.append(f'<path d="M {arrow_x + 18} {y + ROW_H / 2} L {image_x - 18} {y + ROW_H / 2}" stroke="#64748b" stroke-width="2" marker-end="url(#arrow)"/>')
        relation = edge_relation(record["source_edge"]) if record["source_edge"] else "wiki_inline: no source edge"
        text(svg, arrow_x + ARROW_W / 2, y + ROW_H / 2 - 14, short(relation, 22), size=11, fill="#64748b", anchor="middle")
        render_image_card(svg, image_x, y, record)
        render_details(svg, details_x, y, record)

    svg.append("</svg>")
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser(description="Render one review row per image node in a graph run.")
    parser.add_argument("run_dir", help="Directory containing nodes.jsonl and edges.jsonl.")
    parser.add_argument("--nodes-file", default="nodes.jsonl")
    parser.add_argument("--edges-file", default="edges.jsonl")
    parser.add_argument("--output", default="image_node_overview.svg")
    parser.add_argument("--max-images", type=int, default=0, help="Optional cap after sorting; 0 means all image nodes.")
    args = parser.parse_args()

    if args.max_images < 0:
        raise SystemExit("--max-images must be >= 0")
    run_dir = Path(args.run_dir)
    nodes = load_jsonl(run_dir / args.nodes_file)
    edges = load_jsonl(run_dir / args.edges_file)
    if not nodes:
        raise SystemExit(f"no nodes found in {run_dir / args.nodes_file}")
    records = image_records(nodes, edges)
    total_images = len(records)
    if args.max_images:
        records = records[: args.max_images]
    output = run_dir / args.output
    output.write_text(render_svg(run_dir, records, total_images), encoding="utf-8")
    print(f"wrote: {output}")


if __name__ == "__main__":
    main()
