#!/usr/bin/env python3
"""Render text nodes and their graph connections as a portable SVG overview.

Example:
    python debug/visualize_text_node.py runs/my_graph --output /tmp/text_nodes.svg
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PAGE_WIDTH = 1800
MARGIN_X = 36
MARGIN_Y = 28
HEADER_H = 118
CARD_GAP = 18
CARD_PADDING = 22
COLUMN_GAP = 34
LINE_HEIGHT = 18
TITLE_CHARS = 74
EDGE_CHARS = 98
MAX_EDGE_LINES_PER_DIRECTION = 14


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc
    return records


def esc(value: object) -> str:
    return html.escape(str(value or ""))


def short(value: object, limit: int) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else f"{text[: max(1, limit - 3)]}..."


def wrap_text(value: object, width: int) -> list[str]:
    text = " ".join(str(value or "").split())
    if not text:
        return []
    words = text.split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = word if not current else f"{current} {word}"
        if current and len(candidate) > width:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def node_label(node: dict[str, Any] | None, node_id: str | None) -> str:
    if not node:
        return f"missing node: {node_id or 'unknown'}"
    return str(node.get("title") or node.get("canonical_id") or node.get("node_id") or node_id or "unknown")


def edge_lines(edges: list[dict[str, Any]], node_by_id: dict[str, dict[str, Any]], *, direction: str) -> list[str]:
    lines: list[str] = []
    for edge in edges:
        other_id = edge.get("dst_node_id") if direction == "out" else edge.get("src_node_id")
        other = node_by_id.get(str(other_id)) if other_id else None
        relation = short(edge.get("relation") or "(no relation)", EDGE_CHARS)
        edge_type = edge.get("edge_type") or "edge"
        status = edge.get("status", "active")
        target = short(node_label(other, str(other_id) if other_id else None), TITLE_CHARS)
        target_type = (other or {}).get("node_type") or "missing"
        lines.append(f"{relation}  ->  {target} [{target_type}; {edge_type}; {status}]")
    return sorted(lines, key=str.lower)


def text_node_records(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], *, all_edges: bool) -> list[dict[str, Any]]:
    node_by_id = {str(node["node_id"]): node for node in nodes if node.get("node_id")}
    incoming: dict[str, list[dict[str, Any]]] = defaultdict(list)
    outgoing: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for edge in edges:
        if not all_edges and edge.get("status", "active") != "active":
            continue
        source_id = edge.get("src_node_id")
        target_id = edge.get("dst_node_id")
        if source_id:
            outgoing[str(source_id)].append(edge)
        if target_id:
            incoming[str(target_id)].append(edge)

    records: list[dict[str, Any]] = []
    for node in nodes:
        if node.get("node_type") != "text" or not node.get("node_id"):
            continue
        node_id = str(node["node_id"])
        records.append(
            {
                "node": node,
                "outgoing": edge_lines(outgoing[node_id], node_by_id, direction="out"),
                "incoming": edge_lines(incoming[node_id], node_by_id, direction="in"),
            }
        )
    return sorted(records, key=lambda record: node_label(record["node"], None).lower())


def _direction_height(lines: list[str]) -> int:
    rendered = [line for item in lines[:MAX_EDGE_LINES_PER_DIRECTION] for line in wrap_text(item, EDGE_CHARS)]
    if not rendered:
        return LINE_HEIGHT
    return max(LINE_HEIGHT, len(rendered) * LINE_HEIGHT)


def _render_direction(
    svg: list[str],
    *,
    x: int,
    y: int,
    width: int,
    heading: str,
    lines: list[str],
    color: str,
    empty_text: str,
) -> None:
    svg.append(
        f'<text x="{x}" y="{y}" font-family="Arial, sans-serif" font-size="14" font-weight="bold" fill="{color}">{esc(heading)}</text>'
    )
    visible = lines[:MAX_EDGE_LINES_PER_DIRECTION]
    rendered = [line for item in visible for line in wrap_text(item, EDGE_CHARS)]
    if not rendered:
        rendered = [empty_text]
    for index, line in enumerate(rendered, start=1):
        svg.append(
            f'<text x="{x}" y="{y + index * LINE_HEIGHT}" font-family="Arial, sans-serif" font-size="12.5" fill="#334155">{esc(line)}</text>'
        )
    if len(lines) > len(visible):
        extra_y = y + (len(rendered) + 1) * LINE_HEIGHT
        svg.append(
            f'<text x="{x}" y="{extra_y}" font-family="Arial, sans-serif" font-size="12.5" font-style="italic" fill="#64748B">{esc(f'... {len(lines) - len(visible)} more edges')}</text>'
        )
    del width


def render_svg(graph_dir: Path, records: list[dict[str, Any]], *, total_text_nodes: int) -> str:
    cards: list[tuple[dict[str, Any], int]] = []
    for record in records:
        edge_height = max(_direction_height(record["outgoing"]), _direction_height(record["incoming"]))
        cards.append((record, 94 + edge_height + CARD_PADDING))
    body_height = sum(height for _, height in cards) + max(0, len(cards) - 1) * CARD_GAP
    page_height = HEADER_H + MARGIN_Y * 2 + max(body_height, 130)
    count = f"text_nodes={len(records)}" if len(records) == total_text_nodes else f"text_nodes={len(records)}/{total_text_nodes}"
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{PAGE_WIDTH}" height="{page_height}" viewBox="0 0 {PAGE_WIDTH} {page_height}">',
        '<rect width="100%" height="100%" fill="#F6F8FB"/>',
        '<rect x="0" y="0" width="100%" height="92" fill="#0F172A"/>',
        '<rect x="0" y="92" width="100%" height="26" fill="#E6ECF5"/>',
        '<text x="28" y="40" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#FFFFFF">Text Node Connections</text>',
        f'<text x="28" y="68" font-family="Arial, sans-serif" font-size="15" fill="#CBD5E1">graph: {esc(graph_dir)}   {esc(count)}</text>',
        '<text x="28" y="112" font-family="Arial, sans-serif" font-size="13" fill="#475569">Each card shows a text node and its active incoming and outgoing graph edges.</text>',
    ]
    if not records:
        svg.extend(
            [
                f'<rect x="{MARGIN_X}" y="{HEADER_H + MARGIN_Y}" width="{PAGE_WIDTH - 2 * MARGIN_X}" height="120" rx="18" fill="#FFFFFF" stroke="#D7E0EA"/>',
                f'<text x="{MARGIN_X + 22}" y="{HEADER_H + MARGIN_Y + 52}" font-family="Arial, sans-serif" font-size="20" font-weight="bold" fill="#0F172A">No text nodes found</text>',
                '</svg>',
            ]
        )
        return "\n".join(svg)

    left_x = MARGIN_X + CARD_PADDING
    column_width = (PAGE_WIDTH - 2 * MARGIN_X - 2 * CARD_PADDING - COLUMN_GAP) // 2
    right_x = left_x + column_width + COLUMN_GAP
    y = HEADER_H + MARGIN_Y
    for index, (record, card_height) in enumerate(cards, start=1):
        node = record["node"]
        node_id = node.get("node_id") or ""
        title = short(node_label(node, node_id), TITLE_CHARS)
        outgoing = record["outgoing"]
        incoming = record["incoming"]
        svg.append(
            f'<rect x="{MARGIN_X}" y="{y}" width="{PAGE_WIDTH - 2 * MARGIN_X}" height="{card_height}" rx="18" fill="#FFFFFF" stroke="#D7E0EA" stroke-width="1.2"/>'
        )
        svg.append(
            f'<text x="{left_x}" y="{y + 28}" font-family="Arial, sans-serif" font-size="13" font-weight="bold" fill="#2563EB">text #{index}</text>'
        )
        svg.append(
            f'<text x="{left_x + 70}" y="{y + 28}" font-family="Arial, sans-serif" font-size="18" font-weight="bold" fill="#0F172A">{esc(title)}</text>'
        )
        svg.append(
            f'<text x="{left_x}" y="{y + 52}" font-family="monospace" font-size="11.5" fill="#64748B">{esc(node_id)}</text>'
        )
        _render_direction(
            svg,
            x=left_x,
            y=y + 78,
            width=column_width,
            heading=f"Outgoing ({len(outgoing)})",
            lines=outgoing,
            color="#0F766E",
            empty_text="(no outgoing edges)",
        )
        _render_direction(
            svg,
            x=right_x,
            y=y + 78,
            width=column_width,
            heading=f"Incoming ({len(incoming)})",
            lines=incoming,
            color="#9333EA",
            empty_text="(no incoming edges)",
        )
        y += card_height + CARD_GAP
    svg.append("</svg>")
    return "\n".join(svg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("graph_dir", type=Path, help="Directory containing nodes.jsonl and edges.jsonl.")
    parser.add_argument("--output", type=Path, default=None, help="Output SVG path (default: <graph_dir>/text_nodes.svg).")
    parser.add_argument("--all-edges", action="store_true", help="Include inactive edges; active edges are shown by default.")
    parser.add_argument("--max-nodes", type=int, default=0, help="Render at most this many text nodes after title sorting; 0 renders all.")
    args = parser.parse_args()

    graph_dir = args.graph_dir.expanduser().resolve()
    nodes_path = graph_dir / "nodes.jsonl"
    edges_path = graph_dir / "edges.jsonl"
    if not nodes_path.is_file() or not edges_path.is_file():
        parser.error(f"Expected nodes.jsonl and edges.jsonl under {graph_dir}")
    if args.max_nodes < 0:
        parser.error("--max-nodes must be non-negative")

    nodes = load_jsonl(nodes_path)
    edges = load_jsonl(edges_path)
    records = text_node_records(nodes, edges, all_edges=args.all_edges)
    total_text_nodes = len(records)
    if args.max_nodes:
        records = records[: args.max_nodes]

    output_path = (args.output or graph_dir / "text_nodes.svg").expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_svg(graph_dir, records, total_text_nodes=total_text_nodes), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Rendered text_nodes={len(records)}/{total_text_nodes} edges={'all' if args.all_edges else 'active'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
