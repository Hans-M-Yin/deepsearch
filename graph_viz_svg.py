#!/usr/bin/env python3
import html
import json
import math
import random
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path


CARD_W = 220
CARD_H = 78
COMPONENT_GAP_X = 72
COMPONENT_GAP_Y = 72
MARGIN_X = 48
MARGIN_Y = 188
FOOTER_H = 40
BOX_PAD = 28


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
    return relation or anchor or predicate or edge_kind(edge)


def node_label(node):
    node_type = node.get("node_type")
    if node_type == "image":
        return (
            node.get("caption")
            or node.get("title")
            or ((node.get("metadata") or {}).get("visual_target"))
            or node.get("canonical_id")
            or node.get("node_id")
            or "image"
        )
    return node.get("title") or node.get("canonical_id") or node.get("node_id") or "unknown"


def node_subtitle(node):
    node_type = node.get("node_type") or "node"
    if node_type == "image":
        accepted = len(node.get("accepted_image_ids") or [])
        rejected = len(node.get("rejected_image_ids") or [])
        size = []
        if node.get("width") and node.get("height"):
            size.append(f'{node["width"]}x{node["height"]}')
        counts = f"acc={accepted} rej={rejected}"
        return f"image bundle · {counts}" + (f" · {' '.join(size)}" if size else "")
    return node.get("subtype") or node_type


def node_extra_line(node):
    node_type = node.get("node_type")
    if node_type == "image":
        meta = node.get("metadata") or {}
        query = meta.get("search_query")
        return short(query, 32) if query else short(node.get("source_page_url"), 32)
    summary = node.get("summary")
    return short(summary, 32) if summary else ""


def node_image_href(node):
    if node.get("node_type") != "image":
        return None
    return node.get("image_url") or node.get("thumb_oss_uri") or node.get("oss_uri")


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
        "search_retrieved": "#B56576",
        "image_depicts": "#D17B0F",
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
    node_id_set = {node["node_id"] for node in nodes}
    children = defaultdict(list)
    parents = defaultdict(list)
    undirected = defaultdict(set)
    indegree = Counter()
    outdegree = Counter()
    usable_edges = []
    for edge in edges:
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src not in node_id_set or dst not in node_id_set:
            continue
        usable_edges.append(edge)
        children[src].append(dst)
        parents[dst].append(src)
        undirected[src].add(dst)
        undirected[dst].add(src)
        indegree[dst] += 1
        outdegree[src] += 1
    return children, parents, undirected, indegree, outdegree, usable_edges


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


def _force_layout_component(component, edges_in_component, node_order):
    count = len(component)
    if count == 1:
        return {component[0]: (0.0, 0.0)}

    order_index = {node_id: i for i, node_id in enumerate(node_order)}
    component = sorted(component, key=lambda nid: order_index.get(nid, math.inf))
    index = {node_id: idx for idx, node_id in enumerate(component)}

    radius = max(120.0, 55.0 * math.sqrt(count))
    positions = {}
    for idx, node_id in enumerate(component):
        angle = (2 * math.pi * idx) / max(1, count)
        positions[node_id] = [math.cos(angle) * radius, math.sin(angle) * radius]

    rng = random.Random(42 + count)
    for node_id in component:
        positions[node_id][0] += rng.uniform(-12.0, 12.0)
        positions[node_id][1] += rng.uniform(-12.0, 12.0)

    adjacency = {(edge.get("src_node_id"), edge.get("dst_node_id")) for edge in edges_in_component}
    adjacency |= {(dst, src) for src, dst in list(adjacency)}

    ideal = max(165.0, min(250.0, 190.0 + count * 1.6))
    repulsion = ideal * ideal * 3.2
    spring = 0.055
    gravity = 0.008
    max_step = 18.0

    for _ in range(110):
        disp = {node_id: [0.0, 0.0] for node_id in component}

        for i, src in enumerate(component):
            x1, y1 = positions[src]
            for j in range(i + 1, count):
                dst = component[j]
                x2, y2 = positions[dst]
                dx = x1 - x2
                dy = y1 - y2
                dist2 = dx * dx + dy * dy + 0.01
                dist = math.sqrt(dist2)
                force = repulsion / dist2
                ux = dx / dist
                uy = dy / dist
                disp[src][0] += ux * force
                disp[src][1] += uy * force
                disp[dst][0] -= ux * force
                disp[dst][1] -= uy * force

        for src, dst in adjacency:
            if index[src] >= index[dst]:
                continue
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy) + 0.01
            force = spring * (dist - ideal)
            ux = dx / dist
            uy = dy / dist
            disp[src][0] += ux * force
            disp[src][1] += uy * force
            disp[dst][0] -= ux * force
            disp[dst][1] -= uy * force

        for node_id in component:
            x, y = positions[node_id]
            disp[node_id][0] += -x * gravity
            disp[node_id][1] += -y * gravity

        for node_id in component:
            dx, dy = disp[node_id]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > max_step:
                scale = max_step / dist
                dx *= scale
                dy *= scale
            positions[node_id][0] += dx
            positions[node_id][1] += dy

    return {node_id: tuple(xy) for node_id, xy in positions.items()}


def _component_bounds(positions):
    xs = [xy[0] for xy in positions.values()]
    ys = [xy[1] for xy in positions.values()]
    min_x = min(xs) - CARD_W / 2 - BOX_PAD
    max_x = max(xs) + CARD_W / 2 + BOX_PAD
    min_y = min(ys) - CARD_H / 2 - BOX_PAD
    max_y = max(ys) + CARD_H / 2 + BOX_PAD
    return min_x, min_y, max_x, max_y


def layout_nodes(nodes, edges):
    node_ids = [node["node_id"] for node in nodes]
    children, parents, undirected, indegree, outdegree, usable_edges = build_graph(nodes, edges)
    components = connected_components(node_ids, undirected)
    components.sort(key=len, reverse=True)

    edges_by_component = []
    for component in components:
        comp_set = set(component)
        edges_by_component.append(
            [edge for edge in usable_edges if edge.get("src_node_id") in comp_set and edge.get("dst_node_id") in comp_set]
        )

    local_layouts = []
    total_area = 0.0
    for component, comp_edges in zip(components, edges_by_component):
        local_positions = _force_layout_component(component, comp_edges, node_ids)
        min_x, min_y, max_x, max_y = _component_bounds(local_positions)
        width = max_x - min_x
        height = max_y - min_y
        local_layouts.append(
            {
                "component": component,
                "positions": local_positions,
                "min_x": min_x,
                "min_y": min_y,
                "width": width,
                "height": height,
                "root": pick_component_root(component, indegree, outdegree, node_ids),
            }
        )
        total_area += width * height

    target_row_width = max(1400.0, math.sqrt(max(total_area, 1.0)) * 1.35)

    positions = {}
    component_boxes = []
    cursor_x = MARGIN_X
    cursor_y = MARGIN_Y
    row_height = 0.0
    max_right = 0.0
    max_bottom = 0.0

    for layout in local_layouts:
        comp_width = layout["width"]
        comp_height = layout["height"]
        if cursor_x > MARGIN_X and cursor_x + comp_width > target_row_width:
            cursor_x = MARGIN_X
            cursor_y += row_height + COMPONENT_GAP_Y
            row_height = 0.0

        offset_x = cursor_x - layout["min_x"]
        offset_y = cursor_y - layout["min_y"]
        for node_id, (cx, cy) in layout["positions"].items():
            positions[node_id] = (cx + offset_x - CARD_W / 2, cy + offset_y - CARD_H / 2)

        top = cursor_y
        left = cursor_x
        bottom = cursor_y + comp_height
        right = cursor_x + comp_width
        component_boxes.append(
            {
                "root": layout["root"],
                "left": left,
                "top": top,
                "right": right,
                "bottom": bottom,
                "width": comp_width,
                "height": comp_height,
            }
        )

        cursor_x += comp_width + COMPONENT_GAP_X
        row_height = max(row_height, comp_height)
        max_right = max(max_right, right)
        max_bottom = max(max_bottom, bottom)

    width = max(1400, int(max_right + MARGIN_X))
    height = max(900, int(max_bottom + FOOTER_H))
    return positions, component_boxes, width, height


def _node_center(xy):
    return xy[0] + CARD_W / 2, xy[1] + CARD_H / 2


def _rect_anchor(center, other_center):
    cx, cy = center
    ox, oy = other_center
    dx = ox - cx
    dy = oy - cy
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return cx, cy
    scale = 1.0 / max(abs(dx) / (CARD_W / 2), abs(dy) / (CARD_H / 2))
    return cx + dx * scale, cy + dy * scale


def path_between(src_xy, dst_xy, bend=0):
    src_center = _node_center(src_xy)
    dst_center = _node_center(dst_xy)
    x1, y1 = _rect_anchor(src_center, dst_center)
    x2, y2 = _rect_anchor(dst_center, src_center)
    dx = x2 - x1
    dy = y2 - y1
    dist = math.sqrt(dx * dx + dy * dy) + 0.01
    ux = dx / dist
    uy = dy / dist
    px = -uy
    py = ux
    curve = min(70.0, dist * 0.18) * bend
    c1x = x1 + dx * 0.35 + px * curve
    c1y = y1 + dy * 0.35 + py * curve
    c2x = x1 + dx * 0.65 + px * curve
    c2y = y1 + dy * 0.65 + py * curve
    return x1, y1, c1x, c1y, c2x, c2y, x2, y2


def label_position(src_xy, dst_xy, bend=0):
    src_center = _node_center(src_xy)
    dst_center = _node_center(dst_xy)
    mx = (src_center[0] + dst_center[0]) / 2
    my = (src_center[1] + dst_center[1]) / 2
    dx = dst_center[0] - src_center[0]
    dy = dst_center[1] - src_center[1]
    dist = math.sqrt(dx * dx + dy * dy) + 0.01
    px = -dy / dist
    py = dx / dist
    return mx + px * bend * 24, my + py * bend * 24 - 10


def aggregated_edge_label(group):
    count = len(group["edges"])
    kind = group["kind"]
    labels = group["labels"]
    if count == 1:
        return short(labels[0] if labels else kind, 34)

    preview = ", ".join(labels[:2]) if labels else kind
    extra = count - min(len(labels), 2)
    base = f"{count} links" if kind == "wiki_link" else f"{count} {kind}"
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
        'Layout uses a force-directed plane view. Image nodes are retrieval bundles, not single photos.</text>'
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
            f'<rect x="{box["left"]:.1f}" y="{box["top"]:.1f}" width="{box["right"] - box["left"]:.1f}" height="{box["bottom"] - box["top"]:.1f}" '
            'rx="18" ry="18" fill="#FFFFFF" stroke="#E2E8F0" stroke-width="1.2"/>'
        )
        root_title = short(node_label(node_by_id[box["root"]]), 52)
        svg.append(
            f'<text x="{box["left"] + 16:.1f}" y="{box["top"] + 24:.1f}" font-family="Arial, sans-serif" font-size="13" fill="#64748B">'
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
        subtitle = short(node_subtitle(node), 34)
        extra = node_extra_line(node)
        fill, stroke = node_type_color(node.get("node_type", ""))
        image_href = node_image_href(node)
        title_x = x + 74 if node.get("node_type") == "image" else x + 16

        svg.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{CARD_W}" height="{CARD_H}" rx="14" ry="14" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" filter="url(#shadow)"/>'
        )

        if node.get("node_type") == "image":
            thumb_x = x + 14
            thumb_y = y + 14
            thumb_w = 48
            thumb_h = 36
            svg.append(
                f'<rect x="{thumb_x:.1f}" y="{thumb_y:.1f}" width="{thumb_w}" height="{thumb_h}" rx="6" ry="6" fill="#FFF7ED" stroke="{stroke}" stroke-width="1.2"/>'
            )
            if image_href:
                svg.append(
                    f'<image href="{esc(image_href)}" x="{thumb_x + 1:.1f}" y="{thumb_y + 1:.1f}" '
                    f'width="{thumb_w - 2}" height="{thumb_h - 2}" preserveAspectRatio="xMidYMid slice" />'
                )
            else:
                svg.append(
                    f'<circle cx="{x + 31:.1f}" cy="{y + 28:.1f}" r="4" fill="{stroke}"/>'
                    f'<path d="M{x + 22:.1f},{y + 42:.1f} L{x + 31:.1f},{y + 33:.1f} L{x + 42:.1f},{y + 43:.1f}" fill="none" stroke="{stroke}" stroke-width="1.4"/>'
                )

        svg.append(
            f'<text x="{title_x:.1f}" y="{y + 26:.1f}" font-family="Arial, sans-serif" font-size="14" '
            f'font-weight="bold" fill="#0F172A">{esc(label)}</text>'
        )
        svg.append(
            f'<text x="{x + 16:.1f}" y="{y + 47:.1f}" font-family="Arial, sans-serif" font-size="11" fill="#475569">'
            f'{esc(subtitle)}</text>'
        )
        if extra:
            svg.append(
                f'<text x="{x + 16:.1f}" y="{y + 64:.1f}" font-family="Arial, sans-serif" font-size="10.5" fill="#64748B">'
                f'{esc(short(extra, 34))}</text>'
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
