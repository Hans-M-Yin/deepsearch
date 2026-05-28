#!/usr/bin/env python3
import html
import json
import math
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path


CARD_W = 220
CARD_H = 64
COL_GAP = 170
ROW_GAP = 44
COMPONENT_GAP = 120
MARGIN_X = 48
HEADER_H = 170
FOOTER_H = 40


def load_jsonl(path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def short(text, n=28):
    text = str(text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "..."


def esc(x):
    return html.escape(str(x or ""))


def edge_kind(edge):
    return edge.get("edge_type") or "edge"


def edge_label(edge):
    meta = edge.get("metadata") or {}
    relation = edge.get("relation")
    anchor = meta.get("anchor_text")
    predicate = ((meta.get("relation_info") or {}).get("predicate")) if isinstance(meta.get("relation_info"), dict) else None
    label = relation or anchor or predicate or edge_kind(edge)
    if edge_kind(edge) == "wiki_link" and label and label != "wiki_link":
        return f"mentions: {label}"
    return label


def node_label(node):
    return node.get("title") or node.get("canonical_id") or node.get("node_id") or "unknown"


def node_type_color(node_type):
    palette = {
        "text": ("#DCEEFF", "#2F6DB5"),
        "image": ("#FFE7C2", "#C77700"),
        "region": ("#E8DDFE", "#6E46C6"),
    }
    return palette.get(node_type, ("#E9EDF2", "#536273"))


def edge_color(kind):
    palette = {
        "wiki_link": "#4C78A8",
        "semantic": "#59A14F",
        "reference": "#9C755F",
    }
    return palette.get(kind, "#7A7A7A")


def aggregate_edges(edges):
    grouped = {}
    for edge in edges:
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        kind = edge_kind(edge)
        key = (src, dst, kind)
        bucket = grouped.setdefault(
            key,
            {
                "src": src,
                "dst": dst,
                "kind": kind,
                "edges": [],
                "labels": [],
            },
        )
        bucket["edges"].append(edge)
        label = edge_label(edge)
        if label and label not in bucket["labels"]:
            bucket["labels"].append(label)
    return list(grouped.values())


def build_graph(nodes, edges):
    node_ids = [node["node_id"] for node in nodes]
    children = defaultdict(list)
    parents = defaultdict(list)
    undirected = defaultdict(set)
    indegree = Counter()
    outdegree = Counter()
    for edge in edges:
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src not in node_ids or dst not in node_ids:
            continue
        children[src].append(dst)
        parents[dst].append(src)
        undirected[src].add(dst)
        undirected[dst].add(src)
        indegree[dst] += 1
        outdegree[src] += 1
    return children, parents, undirected, indegree, outdegree


def pick_component_root(component, indegree, outdegree, node_order):
    order_index = {node_id: i for i, node_id in enumerate(node_order)}
    return min(
        component,
        key=lambda node_id: (
            indegree.get(node_id, 0),
            -outdegree.get(node_id, 0),
            order_index.get(node_id, math.inf),
        ),
    )


def connected_components(node_ids, undirected):
    seen = set()
    components = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        queue = deque([node_id])
        comp = []
        seen.add(node_id)
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in undirected.get(cur, set()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        components.append(comp)
    return components


def assign_levels(component, root, children, parents):
    level = {root: 0}
    queue = deque([root])
    while queue:
        cur = queue.popleft()
        neighbors = list(children.get(cur, [])) + list(parents.get(cur, []))
        for nxt in neighbors:
            if nxt not in component or nxt in level:
                continue
            proposed = level[cur] + (1 if nxt in children.get(cur, []) else 1)
            level[nxt] = proposed
            queue.append(nxt)
    for node_id in component:
        level.setdefault(node_id, max(level.values(), default=0) + 1)
    return level


def layout_nodes(nodes, edges):
    node_ids = [node["node_id"] for node in nodes]
    children, parents, undirected, indegree, outdegree = build_graph(nodes, edges)
    components = connected_components(node_ids, undirected)
    components.sort(key=len, reverse=True)

    positions = {}
    component_boxes = []
    current_y = HEADER_H
    max_right = 0
    order_index = {node_id: i for i, node_id in enumerate(node_ids)}

    for component in components:
        root = pick_component_root(component, indegree, outdegree, node_ids)
        level = assign_levels(set(component), root, children, parents)
        buckets = defaultdict(list)
        for node_id in component:
            buckets[level[node_id]].append(node_id)
        levels = sorted(buckets)
        for node_ids_in_level in buckets.values():
            node_ids_in_level.sort(key=lambda nid: (indegree.get(nid, 0), -outdegree.get(nid, 0), order_index[nid]))

        comp_height = 0
        for lv in levels:
            comp_height = max(comp_height, len(buckets[lv]) * CARD_H + max(0, len(buckets[lv]) - 1) * ROW_GAP)
        comp_height = max(comp_height, CARD_H)

        for lv in levels:
            x = MARGIN_X + lv * (CARD_W + COL_GAP)
            col_height = len(buckets[lv]) * CARD_H + max(0, len(buckets[lv]) - 1) * ROW_GAP
            start_y = current_y + (comp_height - col_height) / 2
            for idx, node_id in enumerate(buckets[lv]):
                y = start_y + idx * (CARD_H + ROW_GAP)
                positions[node_id] = (x, y)
                max_right = max(max_right, x + CARD_W)

        component_boxes.append(
            {
                "root": root,
                "top": current_y - 18,
                "bottom": current_y + comp_height + 18,
                "height": comp_height,
                "levels": len(levels),
            }
        )
        current_y += comp_height + COMPONENT_GAP

    width = max(1400, int(max_right + MARGIN_X))
    height = max(900, int(current_y + FOOTER_H))
    return positions, component_boxes, width, height


def path_between(src_xy, dst_xy, bend=0):
    x1 = src_xy[0] + CARD_W
    y1 = src_xy[1] + CARD_H / 2
    x2 = dst_xy[0]
    y2 = dst_xy[1] + CARD_H / 2
    dx = max(50, (x2 - x1) * 0.45)
    mid_lift = bend * 28
    c1x = x1 + dx
    c2x = x2 - dx
    return x1, y1, c1x, y1 + mid_lift, c2x, y2 + mid_lift, x2, y2


def label_position(src_xy, dst_xy, bend=0):
    x1 = src_xy[0] + CARD_W
    y1 = src_xy[1] + CARD_H / 2
    x2 = dst_xy[0]
    y2 = dst_xy[1] + CARD_H / 2
    return x1 + (x2 - x1) * 0.58, y1 + (y2 - y1) * 0.58 - 8 + bend * 24


def aggregated_edge_label(group):
    count = len(group["edges"])
    kind = group["kind"]
    labels = group["labels"]
    if count == 1:
        return short(labels[0] if labels else kind, 34)

    preview = ", ".join(labels[:2]) if labels else kind
    extra = count - min(len(labels), 2)
    if kind == "wiki_link":
        base = f"{count} links"
    else:
        base = f"{count} {kind}"
    if preview:
        suffix = f": {preview}"
        if extra > 0:
            suffix += f" +{extra}"
        return short(base + suffix, 40)
    return base


def render_svg(nodes, edges, positions, component_boxes, width, height):
    node_by_id = {node["node_id"]: node for node in nodes}
    edge_counts = Counter(edge_kind(edge) for edge in edges)
    aggregated_edges = aggregate_edges(edges)
    pair_counts = Counter((group["src"], group["dst"]) for group in aggregated_edges)
    pair_offsets = defaultdict(int)

    svg = []
    svg.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    svg.append(
        '<rect width="100%" height="100%" fill="#F7F8FB"/>'
        '<rect x="0" y="0" width="100%" height="150" fill="#0F172A"/>'
        '<rect x="0" y="150" width="100%" height="24" fill="#E9EEF7"/>'
    )
    svg.append('<text x="28" y="42" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#FFFFFF">Graph Overview</text>')
    svg.append(
        f'<text x="28" y="72" font-family="Arial, sans-serif" font-size="16" fill="#CBD5E1">'
        f'nodes={len(nodes)}   edges={len(edges)}   components={len(component_boxes)}</text>'
    )
    svg.append(
        '<text x="28" y="102" font-family="Arial, sans-serif" font-size="14" fill="#E2E8F0">'
        'Edge meaning: source node points to target node. For wiki edges, the label shows the anchor text used in the source page.</text>'
    )

    legend_x = 28
    legend_y = 120
    legend_items = [
        ("text", "#DCEEFF", "#2F6DB5"),
        ("image", "#FFE7C2", "#C77700"),
        ("region", "#E8DDFE", "#6E46C6"),
    ]
    for idx, (name, fill, stroke) in enumerate(legend_items):
        x = legend_x + idx * 150
        svg.append(f'<rect x="{x}" y="{legend_y}" rx="8" ry="8" width="88" height="22" fill="{fill}" stroke="{stroke}" stroke-width="1"/>')
        svg.append(f'<text x="{x + 44}" y="{legend_y + 15}" text-anchor="middle" font-family="Arial, sans-serif" font-size="12" fill="#111827">{esc(name)}</text>')

    edge_summary = ", ".join(f"{k}={v}" for k, v in sorted(edge_counts.items()))
    svg.append(
        f'<text x="{legend_x + 500}" y="{legend_y + 15}" font-family="Arial, sans-serif" font-size="12" fill="#E2E8F0">edge types: {esc(edge_summary or "none")}</text>'
    )

    svg.append(
        """
    <defs>
      <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="3" stdDeviation="4" flood-color="#94A3B8" flood-opacity="0.25"/>
      </filter>
      <marker id="arrow" markerWidth="11" markerHeight="8" refX="10" refY="4" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,8 L11,4 z" fill="#64748B"/>
      </marker>
    </defs>
    """
    )

    for idx, box in enumerate(component_boxes, start=1):
        svg.append(
            f'<rect x="20" y="{box["top"]:.1f}" width="{width - 40}" height="{box["bottom"] - box["top"]:.1f}" '
            'rx="18" ry="18" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.2"/>'
        )
        root_title = short(node_label(node_by_id[box["root"]]), 52)
        svg.append(
            f'<text x="36" y="{box["top"] + 24:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#64748B">'
            f'component {idx}  root: {esc(root_title)}</text>'
        )

    for group in aggregated_edges:
        src = group["src"]
        dst = group["dst"]
        if src not in positions or dst not in positions:
            continue
        kind = group["kind"]
        color = edge_color(kind)
        pair_key = (src, dst)
        idx = pair_offsets[pair_key]
        pair_offsets[pair_key] += 1
        total = pair_counts[pair_key]
        bend = idx - (total - 1) / 2
        x1, y1, c1x, c1y, c2x, c2y, x2, y2 = path_between(positions[src], positions[dst], bend=bend)
        stroke_w = min(4.8, 1.8 + math.log2(len(group["edges"]) + 1) * 0.85)
        svg.append(
            f'<path d="M{x1:.1f},{y1:.1f} C{c1x:.1f},{c1y:.1f} {c2x:.1f},{c2y:.1f} {x2:.1f},{y2:.1f}" '
            f'fill="none" stroke="{color}" stroke-width="{stroke_w:.1f}" opacity="0.72" marker-end="url(#arrow)"/>'
        )

        label = aggregated_edge_label(group)
        if label:
            lx, ly = label_position(positions[src], positions[dst], bend=bend)
            text_w = max(56, min(180, 8 + len(label) * 6.8))
            svg.append(
                f'<rect x="{lx - text_w / 2:.1f}" y="{ly - 13:.1f}" width="{text_w:.1f}" height="20" '
                'rx="10" ry="10" fill="#FFFFFF" stroke="#CBD5E1" stroke-width="1"/>'
            )
            svg.append(
                f'<text x="{lx:.1f}" y="{ly + 1:.1f}" text-anchor="middle" font-family="Arial, sans-serif" '
                f'font-size="11" fill="#334155">{esc(label)}</text>'
            )

    for node in nodes:
        node_id = node["node_id"]
        if node_id not in positions:
            continue
        x, y = positions[node_id]
        title = node_label(node)
        label = short(title, 30)
        subtype = node.get("subtype") or node.get("node_type") or "node"
        fill, stroke = node_type_color(node.get("node_type", ""))

        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{CARD_W}" height="{CARD_H}" rx="14" ry="14" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#shadow)"/>'
        )
        svg.append(
            f'<text x="{x + 16:.1f}" y="{y + 26:.1f}" font-family="Arial, sans-serif" font-size="14" '
            f'font-weight="bold" fill="#0F172A">{esc(label)}</text>'
        )
        svg.append(
            f'<text x="{x + 16:.1f}" y="{y + 47:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">'
            f'{esc(subtype)}</text>'
        )
        svg.append(
            f'<title>{esc(title)}&#10;node_id: {esc(node_id)}&#10;type: {esc(node.get("node_type", ""))}</title>'
        )

    svg.append("</svg>")
    return "\n".join(svg)


def main(run_dir):
    run_dir = Path(run_dir)
    nodes = load_jsonl(run_dir / "nodes.jsonl")
    edges = load_jsonl(run_dir / "edges.jsonl")

    if not nodes:
        raise SystemExit(f"no nodes found in {run_dir}")

    positions, component_boxes, width, height = layout_nodes(nodes, edges)
    svg = render_svg(nodes, edges, positions, component_boxes, width, height)

    out = run_dir / "graph_overview.svg"
    out.write_text(svg, encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python graph_viz_svg.py /path/to/run_dir")
    main(sys.argv[1])
