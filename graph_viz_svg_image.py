#!/usr/bin/env python3
import argparse
import html
import json
from collections import defaultdict
from pathlib import Path


PAGE_WIDTH = 1680
MARGIN_X = 36
MARGIN_Y = 28
HEADER_H = 122
ROW_GAP = 20
ROW_H = 250
IMAGE_W = 260
IMAGE_H = 180
COLUMN_GAP = 22
TITLE_CHARS = 56
TEXT_CHARS = 44


def load_jsonl(path):
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


def short(text, limit):
    text = str(text or "").replace("\n", " ").strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def wrap_text(text, width):
    text = str(text or "").replace("\n", " ").strip()
    if not text:
        return []
    words = text.split()
    if not words:
        return [text[:width]]
    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if len(candidate) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def fit_lines(lines, max_lines, width):
    if not lines:
        return []
    merged = []
    for line in lines:
        merged.extend(wrap_text(line, width))
    if len(merged) <= max_lines:
        return merged
    trimmed = merged[:max_lines]
    trimmed[-1] = short(trimmed[-1], max(4, width))
    return trimmed


def node_label(node):
    if node.get("node_type") == "image":
        return (
            node.get("title")
            or node.get("caption")
            or ((node.get("metadata") or {}).get("visual_target"))
            or node.get("node_id")
            or "image"
        )
    return node.get("title") or node.get("canonical_id") or node.get("node_id") or "unknown"


def node_href(node):
    if node.get("node_type") != "image":
        return None
    return node.get("image_url") or node.get("thumb_oss_uri") or node.get("oss_uri")


def edge_descriptor(edge, other_node):
    relation = edge.get("relation") or edge.get("edge_type") or "edge"
    kind = edge.get("edge_type") or "edge"
    label = node_label(other_node)
    node_type = other_node.get("node_type") or "node"
    return f"{label} [{node_type}] ({kind}: {relation})"


def image_records(nodes, edges):
    node_by_id = {node.get("node_id"): node for node in nodes if node.get("node_id")}
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    for edge in edges:
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src:
            outgoing[src].append(edge)
        if dst:
            incoming[dst].append(edge)

    images = []
    for node in nodes:
        if node.get("node_type") != "image":
            continue
        node_id = node["node_id"]
        upstream = []
        for edge in incoming.get(node_id, []):
            src_node = node_by_id.get(edge.get("src_node_id"))
            if src_node is None:
                continue
            upstream.append(edge_descriptor(edge, src_node))

        downstream = []
        for edge in outgoing.get(node_id, []):
            dst_node = node_by_id.get(edge.get("dst_node_id"))
            if dst_node is None:
                continue
            downstream.append(edge_descriptor(edge, dst_node))

        images.append(
            {
                "node": node,
                "upstream": sorted(dict.fromkeys(upstream)),
                "downstream": sorted(dict.fromkeys(downstream)),
            }
        )

    images.sort(key=lambda item: node_label(item["node"]).lower())
    return images


def _render_text_block(svg, x, y, title, items, width_chars, max_lines, empty_text):
    svg.append(
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="14" '
        f'font-weight="bold" fill="#0F172A">{esc(title)}</text>'
    )
    lines = []
    if items:
        for item in items:
            lines.append(f"- {item}")
    else:
        lines.append(empty_text)
    fitted = fit_lines(lines, max_lines, width_chars)
    for idx, line in enumerate(fitted, start=1):
        svg.append(
            f'<text x="{x}" y="{y + 20 * idx}" font-family="Arial, sans-serif" font-size="12.5" '
            f'fill="#334155">{esc(line)}</text>'
        )


def render_svg(run_dir, images, total_images=None):
    row_count = max(1, len(images))
    height = HEADER_H + MARGIN_Y * 2 + row_count * ROW_H + max(0, row_count - 1) * ROW_GAP
    total_count = len(images) if total_images is None else total_images
    count_text = f"image_nodes={len(images)}"
    if total_count != len(images):
        count_text = f"image_nodes={len(images)}/{total_count}"

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_WIDTH}" height="{height}" viewBox="0 0 {PAGE_WIDTH} {height}">',
        '<rect width="100%" height="100%" fill="#F6F8FB"/>',
        '<rect x="0" y="0" width="100%" height="96" fill="#0F172A"/>',
        '<rect x="0" y="96" width="100%" height="26" fill="#E6ECF5"/>',
        '<text x="28" y="40" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#FFFFFF">Image Node Overview</text>',
        f'<text x="28" y="68" font-family="Arial, sans-serif" font-size="15" fill="#CBD5E1">run: {esc(run_dir)}   {esc(count_text)}</text>',
        '<text x="28" y="112" font-family="Arial, sans-serif" font-size="13" fill="#475569">One row per image node. This view shows title, primary image, upstream source nodes, and downstream linked nodes.</text>',
    ]

    if not images:
        svg.append(
            f'<rect x="{MARGIN_X}" y="{HEADER_H + MARGIN_Y}" width="{PAGE_WIDTH - 2 * MARGIN_X}" height="140" '
            'rx="18" ry="18" fill="#FFFFFF" stroke="#D7E0EA" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{MARGIN_X + 24}" y="{HEADER_H + MARGIN_Y + 54}" font-family="Arial, sans-serif" font-size="22" '
            'font-weight="bold" fill="#0F172A">No image nodes found</text>'
        )
        svg.append(
            f'<text x="{MARGIN_X + 24}" y="{HEADER_H + MARGIN_Y + 86}" font-family="Arial, sans-serif" font-size="14" '
            'fill="#475569">Check whether the run contains image discovery results in nodes.jsonl.</text>'
        )
        svg.append("</svg>")
        return "\n".join(svg)

    col1_x = MARGIN_X + 22
    col2_x = col1_x + IMAGE_W + COLUMN_GAP
    col3_x = col2_x + 520 + COLUMN_GAP
    text_top_offset = 112

    for index, item in enumerate(images):
        node = item["node"]
        y = HEADER_H + MARGIN_Y + index * (ROW_H + ROW_GAP)
        href = node_href(node)
        row_width = PAGE_WIDTH - 2 * MARGIN_X

        svg.append(
            f'<rect x="{MARGIN_X}" y="{y}" width="{row_width}" height="{ROW_H}" '
            'rx="20" ry="20" fill="#FFFFFF" stroke="#D7E0EA" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{MARGIN_X + 20}" y="{y + 28}" font-family="Arial, sans-serif" font-size="13" '
            f'font-weight="bold" fill="#C77700">image #{index + 1}</text>'
        )

        title_lines = fit_lines([node_label(node)], 2, TITLE_CHARS)
        for line_index, line in enumerate(title_lines):
            svg.append(
                f'<text x="{MARGIN_X + 96}" y="{y + 29 + line_index * 20}" font-family="Arial, sans-serif" '
                f'font-size="18" font-weight="bold" fill="#0F172A">{esc(line)}</text>'
            )

        meta_line = " | ".join(
            part
            for part in [
                node.get("node_id"),
                f'{node.get("width")}x{node.get("height")}' if node.get("width") and node.get("height") else None,
                node.get("content_type"),
            ]
            if part
        )
        if meta_line:
            svg.append(
                f'<text x="{MARGIN_X + 96}" y="{y + 72}" font-family="Arial, sans-serif" font-size="12.5" '
                f'fill="#64748B">{esc(short(meta_line, 120))}</text>'
            )

        svg.append(
            f'<rect x="{col1_x}" y="{y + text_top_offset}" width="{IMAGE_W}" height="{IMAGE_H}" '
            'rx="14" ry="14" fill="#FFF7ED" stroke="#F3C98B" stroke-width="1.2"/>'
        )
        if href:
            svg.append(
                f'<image href="{esc(href)}" x="{col1_x + 1}" y="{y + text_top_offset + 1}" '
                f'width="{IMAGE_W - 2}" height="{IMAGE_H - 2}" preserveAspectRatio="xMidYMid slice"/>'
            )
        else:
            svg.append(
                f'<text x="{col1_x + 28}" y="{y + text_top_offset + 92}" font-family="Arial, sans-serif" '
                'font-size="16" fill="#9A6700">No image URL</text>'
            )

        source_page = short(node.get("source_page_url") or "", 90)
        if source_page:
            svg.append(
                f'<text x="{col1_x}" y="{y + text_top_offset + IMAGE_H + 20}" font-family="Arial, sans-serif" '
                f'font-size="11.5" fill="#64748B">source_page: {esc(source_page)}</text>'
            )

        _render_text_block(
            svg,
            col2_x,
            y + text_top_offset + 4,
            "Expanded From",
            item["upstream"],
            width_chars=TEXT_CHARS,
            max_lines=8,
            empty_text="No upstream source nodes",
        )
        _render_text_block(
            svg,
            col3_x,
            y + text_top_offset + 4,
            "Downstream Links",
            item["downstream"],
            width_chars=TEXT_CHARS,
            max_lines=8,
            empty_text="No downstream linked nodes",
        )

    svg.append("</svg>")
    return "\n".join(svg)


def parse_args():
    parser = argparse.ArgumentParser(description="Render a row-based SVG overview of image nodes in a graph run.")
    parser.add_argument("run_dir", help="Directory containing nodes.jsonl and edges.jsonl.")
    parser.add_argument(
        "--output",
        help="Optional output SVG path. Defaults to <run_dir>/graph_image_overview.svg.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=0,
        help="Optional cap on how many image nodes to render after sorting. 0 means no limit.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    nodes = load_jsonl(run_dir / "nodes.jsonl")
    edges = load_jsonl(run_dir / "edges.jsonl")
    if not nodes:
        raise SystemExit(f"no nodes found in {run_dir}")

    images = image_records(nodes, edges)
    total_images = len(images)
    if args.max_images < 0:
        raise SystemExit("--max-images must be >= 0")
    if args.max_images:
        images = images[: args.max_images]
    svg = render_svg(run_dir, images, total_images=total_images)
    output_path = Path(args.output) if args.output else run_dir / "graph_image_overview.svg"
    output_path.write_text(svg, encoding="utf-8")
    if args.max_images:
        print(f"rendered {len(images)} of {total_images} image nodes")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
