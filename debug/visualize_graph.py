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
# These are updated for each graph before layout.  A fixed-size canvas made
# card-sized nodes pile up as soon as a run contained more than a few nodes.
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


def image_href(node):
    """Return the best browser-loadable image URL stored on an image node."""
    candidates = [node.get("image_url"), node.get("thumb_oss_uri"), node.get("oss_uri")]
    for variant in node.get("image_variants") or []:
        if isinstance(variant, dict):
            candidates.extend([variant.get("thumbnail_url"), variant.get("image_url")])
    for candidate in candidates:
        if candidate:
            return str(candidate)
    return None


def node_dimensions(node):
    """Dimensions of the visible card, used by both layout and rendering."""
    if node_type(node) == "image":
        return 230.0, 202.0
    if node_type(node) == "region":
        return 190.0, 98.0
    return 184.0, 90.0


def card_edge_distance(node, ux, uy):
    """Distance from a card centre to its border in direction (ux, uy)."""
    width, height = node_dimensions(node)
    denominator = abs(ux) / (width / 2) + abs(uy) / (height / 2)
    return 0.0 if denominator == 0 else 1.0 / denominator


def configure_canvas(nodes):
    """Reserve enough physical room for cards instead of squeezing all runs."""
    global WIDTH, HEIGHT, INNER_W, INNER_H
    card_area = sum(node_dimensions(node)[0] * node_dimensions(node)[1] for node in nodes)
    # The area estimate leaves room for edges and relation labels.  Keep a
    # readable minimum for small graphs and use a landscape page for larger ones.
    graph_area = max(1500.0 * 1050.0, card_area * 6.0)
    aspect = 1.45
    graph_w = math.sqrt(graph_area * aspect)
    graph_h = graph_w / aspect
    WIDTH = max(2200, int(graph_w + 2 * MARGIN + LEGEND_W + PANEL_GAP))
    HEIGHT = max(1600, int(graph_h + 2 * MARGIN))
    INNER_W = WIDTH - 2 * MARGIN - LEGEND_W - PANEL_GAP
    INNER_H = HEIGHT - 2 * MARGIN


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


def initialize_positions(components, centers, node_by_id, degree, undirected):
    rng = random.Random(SEED)
    positions = {}

    for comp_idx, comp in enumerate(components):
        center_x, center_y = centers[comp_idx]
        hub = max(comp, key=lambda nid: degree[nid])
        positions[hub] = [center_x, center_y]
        levels = defaultdict(list)
        distance = {hub: 0}
        queue = deque([hub])
        while queue:
            current = queue.popleft()
            for neighbor in sorted(undirected.get(current, ()), key=lambda nid: (-degree[nid], nid)):
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    levels[distance[neighbor]].append(neighbor)
                    queue.append(neighbor)

        # Nodes are initialized in graph-distance rings rather than modality
        # rings: an image reached from a text node starts beside that text node.
        phase = rng.random() * math.pi * 2
        for depth, level in sorted(levels.items()):
            radius = 285 + (depth - 1) * 250
            for index, node_id in enumerate(level):
                angle = phase + 2 * math.pi * index / len(level) + rng.uniform(-0.10, 0.10)
                positions[node_id] = [
                    center_x + radius * math.cos(angle) + rng.uniform(-14, 14),
                    center_y + radius * math.sin(angle) + rng.uniform(-14, 14),
                ]
    return positions


def force_layout(components, positions, node_by_id, valid_edges, degree):
    edge_pairs = [(edge["src_node_id"], edge["dst_node_id"]) for edge in valid_edges]
    node_to_comp = {}
    for comp_idx, comp in enumerate(components):
        for node_id in comp:
            node_to_comp[node_id] = comp_idx

    comp_sizes = {idx: len(comp) for idx, comp in enumerate(components)}
    k_base = 155.0
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
                    src_w, src_h = node_dimensions(node_by_id[src])
                    dst_w, dst_h = node_dimensions(node_by_id[dst])
                    # Repel by the cards' half-diagonals, not by the tiny dot
                    # that older versions used to draw.
                    safe_dist = math.hypot((src_w + dst_w) / 2, (src_h + dst_h) / 2) + 34
                    rep = (k_base * k_base) / dist
                    if dist < safe_dist:
                        rep += (safe_dist - dist) * 9.0
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
            src_w, src_h = node_dimensions(node_by_id[src])
            dst_w, dst_h = node_dimensions(node_by_id[dst])
            desired = max(src_w, dst_w) * 0.68 + max(src_h, dst_h) * 0.45 + 90.0
            attr = (dist - desired) * 0.042
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
            card_w, card_h = node_dimensions(node_by_id[node_id])
            positions[node_id][0] = min(MARGIN + INNER_W - card_w / 2 - 12, max(MARGIN + card_w / 2 + 12, x))
            positions[node_id][1] = min(MARGIN + INNER_H - card_h / 2 - 12, max(MARGIN + card_h / 2 + 12, y))

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


def resolve_card_collisions(positions, node_by_id, padding=26.0, iterations=90):
    """Separate overlapping visible card rectangles after the force pass."""
    node_ids = list(positions)
    for _ in range(iterations):
        moved = False
        for index, left_id in enumerate(node_ids):
            x1, y1 = positions[left_id]
            w1, h1 = node_dimensions(node_by_id[left_id])
            for right_id in node_ids[index + 1 :]:
                x2, y2 = positions[right_id]
                w2, h2 = node_dimensions(node_by_id[right_id])
                overlap_x = (w1 + w2) / 2 + padding - abs(x2 - x1)
                overlap_y = (h1 + h2) / 2 + padding - abs(y2 - y1)
                if overlap_x <= 0 or overlap_y <= 0:
                    continue
                moved = True
                # Move along the axis requiring the smaller displacement.
                if overlap_x < overlap_y:
                    direction = -1 if x1 < x2 else 1
                    shift = overlap_x / 2
                    positions[left_id][0] += direction * shift
                    positions[right_id][0] -= direction * shift
                else:
                    direction = -1 if y1 < y2 else 1
                    shift = overlap_y / 2
                    positions[left_id][1] += direction * shift
                    positions[right_id][1] -= direction * shift

        for node_id, (x, y) in positions.items():
            card_w, card_h = node_dimensions(node_by_id[node_id])
            positions[node_id] = [
                min(MARGIN + INNER_W - card_w / 2 - 12, max(MARGIN + card_w / 2 + 12, x)),
                min(MARGIN + INNER_H - card_h / 2 - 12, max(MARGIN + card_h / 2 + 12, y)),
            ]
        if not moved:
            break
    return positions


def has_card_collisions(positions, node_by_id, padding=8.0):
    node_ids = list(positions)
    for index, left_id in enumerate(node_ids):
        x1, y1 = positions[left_id]
        w1, h1 = node_dimensions(node_by_id[left_id])
        for right_id in node_ids[index + 1 :]:
            x2, y2 = positions[right_id]
            w2, h2 = node_dimensions(node_by_id[right_id])
            if abs(x2 - x1) < (w1 + w2) / 2 + padding and abs(y2 - y1) < (h1 + h2) / 2 + padding:
                return True
    return False


def grid_fallback_positions(node_by_id, degree, undirected):
    """Non-overlapping fallback that keeps BFS-neighbors next to each other."""
    ordered_ids = []
    seen = set()
    while len(seen) < len(node_by_id):
        root = max((nid for nid in node_by_id if nid not in seen), key=lambda nid: (degree[nid], nid))
        queue = deque([root])
        seen.add(root)
        while queue:
            current = queue.popleft()
            ordered_ids.append(current)
            for neighbor in sorted(undirected.get(current, ()), key=lambda nid: (-degree[nid], nid)):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
    cell_w, cell_h = 280.0, 250.0
    columns = max(1, int((INNER_W - 50) // cell_w))
    positions = {}
    for index, node_id in enumerate(ordered_ids):
        row, column = divmod(index, columns)
        positions[node_id] = [MARGIN + 35 + column * cell_w + cell_w / 2, MARGIN + 35 + row * cell_h + cell_h / 2]
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
    # Do not scale a carefully separated layout back into overlaps.  The
    # dynamic canvas above grows when necessary, so only translate it here.
    avail_w = INNER_W - 80
    avail_h = INNER_H - 80

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

    pair_offsets = defaultdict(int)
    for edge in valid_edges:
        src = edge["src_node_id"]
        dst = edge["dst_node_id"]
        x1, y1 = positions[src]
        x2, y2 = positions[dst]
        dx = x2 - x1
        dy = y2 - y1
        dist = math.sqrt(dx * dx + dy * dy) + 0.01
        ux, uy = dx / dist, dy / dist
        start_dist = card_edge_distance(node_by_id[src], ux, uy) + 3
        end_dist = card_edge_distance(node_by_id[dst], ux, uy) + 8
        start_x = x1 + ux * start_dist
        start_y = y1 + uy * start_dist
        end_x = x2 - ux * end_dist
        end_y = y2 - uy * end_dist
        pair_key = tuple(sorted((src, dst)))
        pair_index = pair_offsets[pair_key]
        pair_offsets[pair_key] += 1
        bend = (pair_index - (pair_offsets[pair_key] - 1) / 2) * 20 + (18 if pair_index % 2 == 0 else -18)
        ctrl_x = (start_x + end_x) / 2 + (-uy) * bend
        ctrl_y = (start_y + end_y) / 2 + (ux) * bend
        svg.append(
            f'<path d="M {start_x:.1f} {start_y:.1f} Q {ctrl_x:.1f} {ctrl_y:.1f} {end_x:.1f} {end_y:.1f}" '
            'fill="none" stroke="#94a3b8" stroke-width="1.15" opacity="0.58" marker-end="url(#arrow)"/>'
        )
        label = short(edge_label(edge), 34)
        if label:
            # The midpoint of a quadratic Bezier is not the midpoint of its end
            # points; use it so the relation sits directly on its curved edge.
            lx = 0.25 * start_x + 0.5 * ctrl_x + 0.25 * end_x
            ly = 0.25 * start_y + 0.5 * ctrl_y + 0.25 * end_y
            label_w = max(52, min(220, len(label) * 6.6 + 16))
            svg.append(f'<rect x="{lx - label_w / 2:.1f}" y="{ly - 11:.1f}" width="{label_w:.1f}" height="20" rx="7" ry="7" fill="#ffffff" stroke="#cbd5e1" opacity="0.96"/>')
            svg.append(f'<text x="{lx:.1f}" y="{ly + 3:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="{EDGE_LABEL_FONT}" font-weight="bold" fill="#334155">{esc(label)}</text>')

    for node in nodes:
        nid = node["node_id"]
        x, y = positions[nid]
        t = node_type(node)
        stroke, fill = TYPE_COLORS.get(t, TYPE_COLORS["default"])
        card_w, card_h = node_dimensions(node)
        left, top = x - card_w / 2, y - card_h / 2
        svg.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{card_w:.1f}" height="{card_h:.1f}" rx="14" ry="14" fill="#ffffff" stroke="{stroke}" stroke-width="2" filter="url(#softShadow)"/>')
        svg.append(f'<rect x="{left:.1f}" y="{top:.1f}" width="{card_w:.1f}" height="30" rx="14" ry="14" fill="{fill}"/>')
        svg.append(f'<rect x="{left:.1f}" y="{top + 16:.1f}" width="{card_w:.1f}" height="14" fill="{fill}"/>')
        svg.append(f'<text x="{left + 12:.1f}" y="{top + 20:.1f}" font-family="Arial, sans-serif" font-size="11" font-weight="bold" fill="{stroke}">{esc(t.upper())} · DEG {degree[nid]}</text>')
        title_lines = wrap_text(node_title(node), width=29 if t == "image" else 23, max_lines=2)
        if t == "image":
            href = image_href(node)
            image_x, image_y = left + 10, top + 40
            image_w, image_h = card_w - 20, 112
            svg.append(f'<rect x="{image_x:.1f}" y="{image_y:.1f}" width="{image_w:.1f}" height="{image_h:.1f}" rx="9" ry="9" fill="#fff7ed" stroke="#fed7aa"/>')
            if href:
                svg.append(f'<image href="{esc(href)}" x="{image_x + 1:.1f}" y="{image_y + 1:.1f}" width="{image_w - 2:.1f}" height="{image_h - 2:.1f}" preserveAspectRatio="xMidYMid slice"/>')
            else:
                svg.append(f'<text x="{x:.1f}" y="{image_y + 62:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#9a6700">No image URL</text>')
            title_y = top + 169
        else:
            title_y = top + 51
        for line_idx, line in enumerate(title_lines):
            svg.append(f'<text x="{left + 11:.1f}" y="{title_y + line_idx * 16:.1f}" font-family="Arial, sans-serif" font-size="{LABEL_FONT}" font-weight="bold" fill="#1e293b">{esc(line)}</text>')
        svg.append(f'<title>{esc(node_title(node))}&#10;node_id={esc(nid)}&#10;type={esc(t)}&#10;degree={degree[nid]}</title>')

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
        f"titled nodes: {len(nodes)}",
        f"relation labels: {sum(bool(edge_label(edge)) for edge in valid_edges)}",
        f"x-range: {min_x:.0f}..{max_x:.0f}",
        f"y-range: {min_y:.0f}..{max_y:.0f}",
        "layout: card-aware + force-directed",
        "nodes: title shown on every card",
        "images: embedded when a URL is available",
        "detail: hover node for full metadata",
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

    configure_canvas(nodes)
    node_by_id, valid_edges, undirected, _outgoing, _incoming, degree = build_graph(nodes, edges)
    components = connected_components(list(node_by_id.keys()), undirected)
    centers = assign_component_centers(components)
    positions = initialize_positions(components, centers, node_by_id, degree, undirected)
    positions = force_layout(components, positions, node_by_id, valid_edges, degree)
    positions = spread_components(components, positions)
    positions = resolve_card_collisions(positions, node_by_id)
    positions = normalize_positions(components, positions)
    positions = resolve_card_collisions(positions, node_by_id, padding=18.0, iterations=40)
    if has_card_collisions(positions, node_by_id):
        # A deterministic grid is preferable to an illegible force layout when
        # a dense run cannot be separated within the page bounds.
        positions = grid_fallback_positions(node_by_id, degree, undirected)

    svg = render_svg(run_dir, nodes, valid_edges, positions, degree)
    out = run_dir / args.output
    out.write_text(svg, encoding="utf-8")
    print(f"wrote: {out}")


if __name__ == "__main__":
    main()
