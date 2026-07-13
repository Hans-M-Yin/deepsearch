#!/usr/bin/env python3
import argparse
import html
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path


SEED = 7
NODE_R = 10
WIDTH = 2200
HEIGHT = 1600
MARGIN = 80
LEGEND_W = 260
PANEL_GAP = 24
INNER_W = WIDTH - 2 * MARGIN - LEGEND_W - PANEL_GAP
INNER_H = HEIGHT - 2 * MARGIN
LABEL_FONT = 12
TITLE_FONT = 13
EDGE_LABEL_FONT = 11

TYPE_COLORS = {
    "text": ("#4f83cc", "#dbeafe"),
    "image": ("#d98c2b", "#ffedd5"),
    "region": ("#7a63cc", "#ede9fe"),
    "default": ("#64748b", "#e2e8f0"),
}


def load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def esc(x):
    return html.escape(str(x or ""))


def short(text, n=36):
    text = str(text or "").replace("\n", " ").strip()
    return text if len(text) <= n else text[: n - 1] + "…"


def wrap_text(text, width=18, max_lines=3):
    text = str(text or "").replace("\n", " ").strip()
    if not text:
        return []
    words = text.split()
    if not words:
        return [short(text, width)]
    lines = []
    cur = words[0]
    for word in words[1:]:
        trial = f"{cur} {word}"
        if len(trial) <= width:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    lines.append(cur)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = short(lines[-1], width)
    return lines


def node_type(node):
    return node.get("node_type") or "default"


def node_title(node):
    return (
        node.get("title")
        or node.get("caption")
        or node.get("canonical_id")
        or node.get("summary")
        or node.get("node_id")
        or "unknown"
    )


def edge_label(edge):
    meta = edge.get("metadata") or {}
    relation_info = meta.get("relation_info") if isinstance(meta.get("relation_info"), dict) else {}
    return (
        edge.get("relation")
        or relation_info.get("predicate")
        or meta.get("anchor_text")
        or edge.get("edge_type")
        or ""
    )


def build_graph(nodes, edges):
    node_by_id = {node["node_id"]: node for node in nodes if node.get("node_id")}
    valid_edges = []
    undirected = defaultdict(set)
    outgoing = defaultdict(list)
    incoming = defaultdict(list)
    degree = Counter()

    for edge in edges:
        src = edge.get("src_node_id")
        dst = edge.get("dst_node_id")
        if src not in node_by_id or dst not in node_by_id or src == dst:
            continue
        valid_edges.append(edge)
        undirected[src].add(dst)
        undirected[dst].add(src)
        outgoing[src].append(dst)
        incoming[dst].append(src)
        degree[src] += 1
        degree[dst] += 1

    return node_by_id, valid_edges, undirected, outgoing, incoming, degree


def connected_components(node_ids, undirected):
    seen = set()
    comps = []
    for node_id in node_ids:
        if node_id in seen:
            continue
        queue = deque([node_id])
        seen.add(node_id)
        comp = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nxt in undirected.get(cur, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def assign_component_centers(components):
    centers = {}
    main_area_w = INNER_W
    main_area_h = INNER_H
    cx = MARGIN + main_area_w / 2
    cy = MARGIN + main_area_h / 2

    if not components:
        return centers

    centers[0] = (cx, cy)
    if len(components) == 1:
        return centers

    ring_r = min(main_area_w, main_area_h) * 0.33
    for idx in range(1, len(components)):
        angle = 2 * math.pi * (idx - 1) / max(1, len(components) - 1)
        centers[idx] = (
            cx + ring_r * math.cos(angle),
            cy + ring_r * math.sin(angle),
        )
    return centers


def initialize_positions(components, centers, node_by_id, degree):
    rng = random.Random(SEED)
    positions = {}
    type_order = {"text": 0, "region": 1, "image": 2, "default": 3}

    for comp_idx, comp in enumerate(components):
        center_x, center_y = centers[comp_idx]
        buckets = defaultdict(list)
        for node_id in comp:
            buckets[node_type(node_by_id[node_id])].append(node_id)

        ordered_types = sorted(buckets.keys(), key=lambda t: type_order.get(t, 99))
        band_count = max(1, len(ordered_types))
        base_radius = 45 + 18 * math.sqrt(len(comp))

        for band_idx, t in enumerate(ordered_types):
            bucket = buckets[t]
            rng.shuffle(bucket)
            radius = base_radius + band_idx * 55
            spread = 2 * math.pi / max(1, len(bucket))
            phase = rng.random() * math.pi * 2
            for i, node_id in enumerate(bucket):
                jitter_r = rng.uniform(-18, 18)
                jitter_a = rng.uniform(-0.35, 0.35)
                angle = phase + i * spread + jitter_a
                x = center_x + (radius + jitter_r) * math.cos(angle)
                y = center_y + (radius + jitter_r) * math.sin(angle)
                x += rng.uniform(-16, 16)
                y += rng.uniform(-16, 16)
                positions[node_id] = [x, y]

        hubs = sorted(comp, key=lambda nid: degree[nid], reverse=True)[: max(3, len(comp) // 18)]
        for j, node_id in enumerate(hubs):
            angle = 2 * math.pi * j / max(1, len(hubs))
            positions[node_id] = [
                center_x + 18 * math.cos(angle),
                center_y + 18 * math.sin(angle),
            ]
    return positions


def force_layout(components, positions, node_by_id, valid_edges, degree):
    edge_pairs = [(edge["src_node_id"], edge["dst_node_id"]) for edge in valid_edges]
    node_to_comp = {}
    for comp_idx, comp in enumerate(components):
        for node_id in comp:
            node_to_comp[node_id] = comp_idx

    comp_sizes = {idx: len(comp) for idx, comp in enumerate(components)}
    k_base = 42.0
    iterations = 260 if len(positions) <= 220 else 180
    temp = 34.0

    for _ in range(iterations):
        disp = {node_id: [0.0, 0.0] for node_id in positions}
        node_ids = list(positions.keys())

        by_comp = defaultdict(list)
        for node_id in node_ids:
            by_comp[node_to_comp[node_id]].append(node_id)

        for comp_idx, comp_nodes in by_comp.items():
            comp_n = len(comp_nodes)
            sample_all = comp_n <= 180
            for i, src in enumerate(comp_nodes):
                x1, y1 = positions[src]
                others = comp_nodes if sample_all else random.sample(comp_nodes, min(80, comp_n))
                for dst in others:
                    if src == dst:
                        continue
                    x2, y2 = positions[dst]
                    dx = x1 - x2
                    dy = y1 - y2
                    dist2 = dx * dx + dy * dy + 0.01
                    dist = math.sqrt(dist2)
                    rep = (k_base * k_base) / dist
                    disp[src][0] += dx / dist * rep
                    disp[src][1] += dy / dist * rep

        for src, dst in edge_pairs:
            x1, y1 = positions[src]
            x2, y2 = positions[dst]
            dx = x2 - x1
            dy = y2 - y1
            dist = math.sqrt(dx * dx + dy * dy) + 0.01
            src_t = node_type(node_by_id[src])
            dst_t = node_type(node_by_id[dst])
            desired = 70.0
            if "image" in (src_t, dst_t):
                desired = 92.0
            if "region" in (src_t, dst_t):
                desired += 12.0
            attr = (dist - desired) * 0.055
            fx = dx / dist * attr
            fy = dy / dist * attr
            disp[src][0] += fx
            disp[src][1] += fy
            disp[dst][0] -= fx
            disp[dst][1] -= fy

        for node_id, (dx, dy) in disp.items():
            comp_idx = node_to_comp[node_id]
            comp_scale = 1.0 + min(1.2, math.sqrt(comp_sizes[comp_idx]) / 18.0)
            x, y = positions[node_id]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                step = min(temp * comp_scale, dist)
                x += dx / dist * step
                y += dy / dist * step
            positions[node_id][0] = min(MARGIN + INNER_W - 20, max(MARGIN + 20, x))
            positions[node_id][1] = min(MARGIN + INNER_H - 20, max(MARGIN + 20, y))

        temp *= 0.985

    return positions


def spread_components(components, positions):
    boxes = []
    for comp in components:
        xs = [positions[nid][0] for nid in comp]
        ys = [positions[nid][1] for nid in comp]
        boxes.append([
            min(xs) - 45,
            min(ys) - 45,
            max(xs) + 45,
            max(ys) + 45,
        ])

    for _ in range(20):
        moved = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                ax1, ay1, ax2, ay2 = boxes[i]
                bx1, by1, bx2, by2 = boxes[j]
                overlap_x = min(ax2, bx2) - max(ax1, bx1)
                overlap_y = min(ay2, by2) - max(ay1, by1)
                if overlap_x > 0 and overlap_y > 0:
                    moved = True
                    shift_x = overlap_x / 2 + 16
                    shift_y = overlap_y / 2 + 16
                    if overlap_x < overlap_y:
                        dir_sign = -1 if (ax1 + ax2) / 2 < (bx1 + bx2) / 2 else 1
                        for nid in components[i]:
                            positions[nid][0] -= dir_sign * shift_x
                        for nid in components[j]:
                            positions[nid][0] += dir_sign * shift_x
                    else:
                        dir_sign = -1 if (ay1 + ay2) / 2 < (by1 + by2) / 2 else 1
                        for nid in components[i]:
                            positions[nid][1] -= dir_sign * shift_y
                        for nid in components[j]:
                            positions[nid][1] += dir_sign * shift_y
        if not moved:
            break
    return positions


def normalize_positions(components, positions):
    if not positions:
        return positions

    min_x = min(x for x, _ in positions.values())
    max_x = max(x for x, _ in positions.values())
    min_y = min(y for _, y in positions.values())
    max_y = max(y for _, y in positions.values())

    graph_w = max(1.0, max_x - min_x)
    graph_h = max(1.0, max_y - min_y)
    avail_w = INNER_W - 60
    avail_h = INNER_H - 60

    scale = min(avail_w / graph_w, avail_h / graph_h, 1.0)
    offset_x = MARGIN + (INNER_W - graph_w * scale) / 2
    offset_y = MARGIN + (INNER_H - graph_h * scale) / 2

    for node_id, (x, y) in list(positions.items()):
        nx = offset_x + (x - min_x) * scale
        ny = offset_y + (y - min_y) * scale
        positions[node_id] = [
            min(MARGIN + INNER_W - 18, max(MARGIN + 18, nx)),
            min(MARGIN + INNER_H - 18, max(MARGIN + 18, ny)),
        ]
    return positions


def compute_label_nodes(nodes, degree):
    scored = []
    for node in nodes:
        nid = node["node_id"]
        title = node_title(node)
        score = degree[nid]
        score += 3 if node_type(node) == "image" else 0
        score += min(len(title) / 18.0, 2.5)
        scored.append((score, nid))
    scored.sort(reverse=True)
    keep = set()
    for _, nid in scored[: max(45, min(140, len(nodes) // 4))]:
        keep.add(nid)
    return keep


def compute_edge_labels(valid_edges, degree):
    buckets = []
    for edge in valid_edges:
        label = edge_label(edge)
        if not label:
            continue
        src = edge["src_node_id"]
        dst = edge["dst_node_id"]
        score = degree[src] + degree[dst]
        if edge.get("relation"):
            score += 2
        buckets.append((score, edge))
    buckets.sort(reverse=True, key=lambda x: x[0])
    return {id(edge) for _, edge in buckets[: max(40, min(120, len(buckets) // 3 or 0))]}


def render_svg(run_dir, nodes, valid_edges, positions, degree):
    label_nodes = compute_label_nodes(nodes, degree)
    edge_label_ids = compute_edge_labels(valid_edges, degree)
    node_by_id = {node["node_id"]: node for node in nodes}

    svg = []
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">')
    svg.append('<rect width="100%" height="100%" fill="#f8fafc"/>')
    svg.append(f'<text x="{MARGIN}" y="44" font-family="Arial, sans-serif" font-size="28" font-weight="bold" fill="#0f172a">Knowledge Graph Overview</text>')
    min_x = min(x for x, _ in positions.values()) if positions else 0
    max_x = max(x for x, _ in positions.values()) if positions else 0
    min_y = min(y for _, y in positions.values()) if positions else 0
    max_y = max(y for _, y in positions.values()) if positions else 0
    svg.append(f'<text x="{MARGIN}" y="72" font-family="Arial, sans-serif" font-size="15" fill="#475569">run={esc(run_dir)} · nodes={len(nodes)} · edges={len(valid_edges)} · layout=component-aware force layout</text>')
    svg.append(f'<rect x="{MARGIN}" y="{MARGIN}" width="{INNER_W}" height="{INNER_H}" rx="18" ry="18" fill="#ffffff" stroke="#dbe4ee"/>')
    legend_x = MARGIN + INNER_W + PANEL_GAP
    svg.append(f'<rect x="{legend_x}" y="{MARGIN}" width="{LEGEND_W}" height="{INNER_H}" rx="18" ry="18" fill="#ffffff" stroke="#dbe4ee"/>')

    svg.append('''
    <defs>
      <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
        <path d="M0,0 L0,6 L8,3 z" fill="#94a3b8"/>
      </marker>
      <filter id="softShadow" x="-20%" y="-20%" width="140%" height="140%">
        <feDropShadow dx="0" dy="1" stdDeviation="1.4" flood-color="#0f172a" flood-opacity="0.10"/>
      </filter>
    </defs>
    ''')

    for edge in valid_edges:
        src = edge["src_node_id"]
        dst = edge["dst_node_id"]
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx * dx + dy * dy) + 0.01
        ux, uy = dx / dist, dy / dist
        start_x = x1 + ux * (NODE_R + 1)
        start_y = y1 + uy * (NODE_R + 1)
        end_x = x2 - ux * (NODE_R + 2)
        end_y = y2 - uy * (NODE_R + 2)
        ctrl_x = (start_x + end_x) / 2 + (-uy) * min(16, dist * 0.06)
        ctrl_y = (start_y + end_y) / 2 + (ux) * min(16, dist * 0.06)
        svg.append(
            f'<path d="M {start_x:.1f} {start_y:.1f} Q {ctrl_x:.1f} {ctrl_y:.1f} {end_x:.1f} {end_y:.1f}" '
            'fill="none" stroke="#94a3b8" stroke-width="1.15" opacity="0.58" marker-end="url(#arrow)"/>'
        )
        if id(edge) in edge_label_ids:
            label = short(edge_label(edge), 24)
            if label:
                lx = (start_x + end_x) / 2
                ly = (start_y + end_y) / 2
                svg.append(f'<rect x="{lx - 28:.1f}" y="{ly - 10:.1f}" width="56" height="16" rx="6" ry="6" fill="#ffffff" opacity="0.82"/>')
                svg.append(f'<text x="{lx:.1f}" y="{ly + 2:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{EDGE_LABEL_FONT}" fill="#475569">{esc(label)}</text>')

    for node in nodes:
        nid = node["node_id"]
        x, y = positions[nid]
        t = node_type(node)
        stroke, fill = TYPE_COLORS.get(t, TYPE_COLORS["default"])
        radius = NODE_R + min(8, math.sqrt(degree[nid]))
        svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius:.1f}" fill="{fill}" stroke="{stroke}" stroke-width="1.7" filter="url(#softShadow)"/>')
        svg.append(f'<title>{esc(node_title(node))}&#10;node_id={esc(nid)}&#10;type={esc(t)}&#10;degree={degree[nid]}</title>')

    occupied = []
    for node in nodes:
        nid = node["node_id"]
        if nid not in label_nodes:
            continue
        x, y = positions[nid]
        lines = wrap_text(node_title(node), width=16 if node_type(node) == "image" else 18, max_lines=2)
        if not lines:
            continue
        box_w = max(72, min(150, max(len(line) for line in lines) * 7 + 16))
        box_h = 18 + len(lines) * 15
        candidates = [
            (x + 16, y - box_h - 8),
            (x + 16, y + 8),
            (x - box_w - 16, y - box_h - 8),
            (x - box_w - 16, y + 8),
        ]
        best = None
        best_penalty = None
        for bx, by in candidates:
            penalty = 0
            if bx < MARGIN or bx + box_w > MARGIN + INNER_W:
                penalty += 1000
            if by < MARGIN or by + box_h > MARGIN + INNER_H:
                penalty += 1000
            for ox1, oy1, ox2, oy2 in occupied:
                overlap_x = min(bx + box_w, ox2) - max(bx, ox1)
                overlap_y = min(by + box_h, oy2) - max(by, oy1)
                if overlap_x > 0 and overlap_y > 0:
                    penalty += overlap_x * overlap_y
            penalty += abs(by - y) * 0.03
            if best_penalty is None or penalty < best_penalty:
                best_penalty = penalty
                best = (bx, by)
        bx, by = best
        occupied.append((bx, by, bx + box_w, by + box_h))
        svg.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{box_w:.1f}" height="{box_h:.1f}" rx="8" ry="8" fill="#ffffff" stroke="#cbd5e1" opacity="0.95"/>')
        for idx, line in enumerate(lines):
            svg.append(f'<text x="{bx + box_w / 2:.1f}" y="{by + 15 + idx * 14:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{LABEL_FONT}" fill="#1e293b">{esc(line)}</text>')

    legend_y = MARGIN + 34
    svg.append(f'<text x="{legend_x + 20}" y="{legend_y}" font-family="Arial, sans-serif" font-size="22" font-weight="bold" fill="#0f172a">Legend</text>')
    legend_y += 28
    svg.append(f'<text x="{legend_x + 20}" y="{legend_y}" font-family="Arial, sans-serif" font-size="13" fill="#64748b">Node color = modality</text>')
    legend_y += 26

    counts = Counter(node_type(node) for node in nodes)
    for key in ("text", "image", "region"):
        stroke, fill = TYPE_COLORS[key]
        svg.append(f'<circle cx="{legend_x + 34}" cy="{legend_y - 4}" r="10" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
        svg.append(f'<text x="{legend_x + 56}" y="{legend_y}" font-family="Arial, sans-serif" font-size="14" fill="#0f172a">{esc(key)} ({counts.get(key, 0)})</text>')
        legend_y += 28

    other_count = sum(v for k, v in counts.items() if k not in {"text", "image", "region"})
    if other_count:
        stroke, fill = TYPE_COLORS["default"]
        svg.append(f'<circle cx="{legend_x + 34}" cy="{legend_y - 4}" r="10" fill="{fill}" stroke="{stroke}" stroke-width="1.7"/>')
        svg.append(f'<text x="{legend_x + 56}" y="{legend_y}" font-family="Arial, sans-serif" font-size="14" fill="#0f172a">other ({other_count})</text>')
        legend_y += 28

    legend_y += 10
    stats = [
        f"labeled nodes: {len(label_nodes)}",
        f"labeled edges: {len(edge_label_ids)}",
        f"x-range: {min_x:.0f}..{max_x:.0f}",
        f"y-range: {min_y:.0f}..{max_y:.0f}",
        "layout: component-aware + force-directed",
        "goal: avoid single-ring clustering",
        "titles: only high-signal nodes shown",
        "detail: hover node for full title",
    ]
    for line in stats:
        svg.append(f'<text x="{legend_x + 20}" y="{legend_y}" font-family="Arial, sans-serif" font-size="13" fill="#475569">{esc(line)}</text>')
        legend_y += 22

    svg.append('</svg>')
    return "\n".join(svg)


def main():
    parser = argparse.ArgumentParser(description="Render a more natural SVG overview for multimodal graph runs.")
    parser.add_argument("run_dir", help="Directory containing nodes.jsonl and edges.jsonl.")
    parser.add_argument("--nodes-file", default="nodes.jsonl")
    parser.add_argument("--edges-file", default="edges.jsonl")
    parser.add_argument("--output", default="graph_overview_better.svg")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    nodes = load_jsonl(run_dir / args.nodes_file)
    edges = load_jsonl(run_dir / args.edges_file)
    if not nodes:
        raise SystemExit(f"no nodes found in {run_dir / args.nodes_file}")

    node_by_id, valid_edges, undirected, _outgoing, _incoming, degree = build_graph(nodes, edges)
    components = connected_components(list(node_by_id.keys()), undirected)
    centers = assign_component_centers(components)
    positions = initialize_positions(components, centers, node_by_id, degree)
    positions = force_layout(components, positions, node_by_id, valid_edges, degree)
    positions = spread_components(components, positions)
    positions = normalize_positions(components, positions)

    svg = render_svg(run_dir, nodes, valid_edges, positions, degree)
    out = run_dir / args.output
    out.write_text(svg, encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
