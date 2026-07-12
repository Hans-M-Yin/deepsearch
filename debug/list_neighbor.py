"""List one graph node's neighbors and show how each connection is formed.

Examples:
  python debug/list_neighbor.py \
    --graph-dir synthesis/runs/mock_graph_review_20260712_env/query_overlap \
    --node-id text_a8fca8f8fd340934

  python debug/list_neighbor.py \
    --graph-dir runs/0712_multi_seed_visual_test4 \
    --node-id image_1234567890abcdef \
    --json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.store import JsonlGraphStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print one graph node's incoming/outgoing edges and its neighboring nodes."
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory containing nodes.jsonl and edges.jsonl.",
    )
    parser.add_argument(
        "--node-id",
        required=True,
        help="Node id to inspect.",
    )
    parser.add_argument(
        "--summary-chars",
        type=int,
        default=140,
        help="Max characters used when printing title/summary snippets.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a single JSON object instead of human-readable text.",
    )
    return parser.parse_args()


def _short(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split())
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: max(0, limit - 3)] + "..."


def _source_url(node: dict[str, Any]) -> str | None:
    source = node.get("source") or {}
    if isinstance(source, dict):
        url = source.get("url")
        if url:
            return str(url)
    for key in ("image_url", "source_page_url", "oss_uri", "thumb_oss_uri"):
        value = node.get(key)
        if value:
            return str(value)
    return None


def _node_title(node: dict[str, Any], *, summary_chars: int) -> str:
    for key in ("title", "caption", "summary", "canonical_id", "node_id"):
        value = node.get(key)
        if value:
            return _short(value, summary_chars)
    return "<untitled>"


def _node_brief(node: dict[str, Any] | None, *, node_id: str | None, summary_chars: int) -> dict[str, Any]:
    if node is None:
        return {
            "node_id": node_id,
            "node_type": None,
            "title": None,
            "status": None,
            "source_url": None,
            "missing": True,
        }
    return {
        "node_id": node.get("node_id") or node_id,
        "node_type": node.get("node_type"),
        "title": _node_title(node, summary_chars=summary_chars),
        "status": node.get("status"),
        "source_url": _source_url(node),
        "missing": False,
    }


def _edge_sort_key(edge: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(edge.get("edge_type") or ""),
        str(edge.get("relation") or ""),
        str(edge.get("dst_node_id") or ""),
        str(edge.get("src_node_id") or ""),
    )


def _edge_record(
    *,
    focus_node: dict[str, Any],
    edge: dict[str, Any],
    neighbor: dict[str, Any] | None,
    direction: str,
    summary_chars: int,
) -> dict[str, Any]:
    if direction == "out":
        src = focus_node
        dst = neighbor
        neighbor_id = edge.get("dst_node_id")
    else:
        src = neighbor
        dst = focus_node
        neighbor_id = edge.get("src_node_id")

    src_label = _node_title(src or {}, summary_chars=summary_chars) if src is not None else str(edge.get("src_node_id"))
    dst_label = _node_title(dst or {}, summary_chars=summary_chars) if dst is not None else str(edge.get("dst_node_id"))
    edge_type = str(edge.get("edge_type") or "")
    relation = str(edge.get("relation") or "")
    relation_suffix = f":{relation}" if relation else ""
    description = f"{src_label} -[{edge_type}{relation_suffix}]-> {dst_label}"

    return {
        "edge_id": edge.get("edge_id"),
        "direction": direction,
        "edge_type": edge.get("edge_type"),
        "relation": edge.get("relation"),
        "status": edge.get("status"),
        "confidence": edge.get("confidence"),
        "src_node_id": edge.get("src_node_id"),
        "dst_node_id": edge.get("dst_node_id"),
        "neighbor_node_id": neighbor_id,
        "neighbor": _node_brief(neighbor, node_id=neighbor_id, summary_chars=summary_chars),
        "description": description,
    }


def _node_report(node: dict[str, Any], *, summary_chars: int) -> dict[str, Any]:
    source = node.get("source") or {}
    return {
        "node_id": node.get("node_id"),
        "node_type": node.get("node_type"),
        "title": _node_title(node, summary_chars=summary_chars),
        "status": node.get("status"),
        "subtype": node.get("subtype"),
        "canonical_id": node.get("canonical_id"),
        "source_type": source.get("source_type") if isinstance(source, dict) else None,
        "source_url": _source_url(node),
        "summary": _short(node.get("summary") or node.get("caption"), summary_chars),
    }


def build_report(*, graph_dir: Path, node_id: str, summary_chars: int) -> dict[str, Any]:
    store = JsonlGraphStore(graph_dir)
    focus_node = store.get_node(node_id)
    if focus_node is None:
        raise KeyError(f"Node not found: {node_id}")

    nodes_by_id = {record["node_id"]: record for record in store.list_nodes()}
    out_edges_raw = sorted(store.edges_from(node_id), key=_edge_sort_key)
    in_edges_raw = sorted(store.edges_to(node_id), key=_edge_sort_key)

    out_edges = [
        _edge_record(
            focus_node=focus_node,
            edge=edge,
            neighbor=nodes_by_id.get(edge.get("dst_node_id")),
            direction="out",
            summary_chars=summary_chars,
        )
        for edge in out_edges_raw
    ]
    in_edges = [
        _edge_record(
            focus_node=focus_node,
            edge=edge,
            neighbor=nodes_by_id.get(edge.get("src_node_id")),
            direction="in",
            summary_chars=summary_chars,
        )
        for edge in in_edges_raw
    ]

    neighbors_by_id: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"neighbor": None, "connections": []}
    )
    for item in out_edges + in_edges:
        neighbor_id = item.get("neighbor_node_id")
        if not neighbor_id:
            continue
        bucket = neighbors_by_id[neighbor_id]
        bucket["neighbor"] = item["neighbor"]
        bucket["connections"].append(
            {
                "direction": item["direction"],
                "edge_id": item["edge_id"],
                "edge_type": item["edge_type"],
                "relation": item["relation"],
                "status": item["status"],
                "description": item["description"],
            }
        )

    neighbors = []
    for neighbor_id, payload in neighbors_by_id.items():
        connections = sorted(
            payload["connections"],
            key=lambda item: (
                str(item.get("direction") or ""),
                str(item.get("edge_type") or ""),
                str(item.get("relation") or ""),
                str(item.get("edge_id") or ""),
            ),
        )
        neighbors.append(
            {
                "neighbor": payload["neighbor"]
                or _node_brief(None, node_id=neighbor_id, summary_chars=summary_chars),
                "connection_count": len(connections),
                "connections": connections,
            }
        )
    neighbors.sort(
        key=lambda item: (
            str((item.get("neighbor") or {}).get("title") or ""),
            str((item.get("neighbor") or {}).get("node_id") or ""),
        )
    )

    return {
        "graph_dir": str(graph_dir),
        "node": _node_report(focus_node, summary_chars=summary_chars),
        "degree": len(in_edges) + len(out_edges),
        "in_degree": len(in_edges),
        "out_degree": len(out_edges),
        "incident_edge_count": len(in_edges) + len(out_edges),
        "neighbor_count": len(neighbors),
        "unique_neighbor_count": len(neighbors),
        "in_edges": in_edges,
        "out_edges": out_edges,
        "neighbors": neighbors,
    }


def _print_edge_section(title: str, edges: list[dict[str, Any]]) -> None:
    print(f"\n{title} ({len(edges)})")
    if not edges:
        print("  <none>")
        return
    for index, edge in enumerate(edges, start=1):
        neighbor = edge.get("neighbor") or {}
        neighbor_title = neighbor.get("title") or "<untitled>"
        neighbor_type = neighbor.get("node_type") or "?"
        print(
            f"  {index:>2}. edge_id={edge.get('edge_id')} direction={edge.get('direction')} "
            f"type={edge.get('edge_type')} relation={edge.get('relation')!r}"
        )
        print(
            f"      neighbor={neighbor.get('node_id')} [{neighbor_type}] {neighbor_title}"
        )
        print(f"      path={edge.get('description')}")


def _print_neighbor_section(neighbors: list[dict[str, Any]]) -> None:
    print(f"\nUnique Neighbors ({len(neighbors)})")
    if not neighbors:
        print("  <none>")
        return
    for index, item in enumerate(neighbors, start=1):
        neighbor = item.get("neighbor") or {}
        print(
            f"  {index:>2}. {neighbor.get('node_id')} [{neighbor.get('node_type') or '?'}] "
            f"{neighbor.get('title') or '<untitled>'}"
        )
        print(f"      connection_count={item.get('connection_count', 0)}")
        for conn_index, conn in enumerate(item.get("connections") or [], start=1):
            print(
                f"      {conn_index:>2}. direction={conn.get('direction')} edge_id={conn.get('edge_id')} "
                f"type={conn.get('edge_type')} relation={conn.get('relation')!r}"
            )
            print(f"          path={conn.get('description')}")


def print_report(report: dict[str, Any]) -> None:
    node = report["node"]
    print("Node")
    print(f"  node_id={node.get('node_id')}")
    print(f"  node_type={node.get('node_type')}")
    print(f"  title={node.get('title')!r}")
    print(f"  status={node.get('status')}")
    if node.get("subtype"):
        print(f"  subtype={node.get('subtype')}")
    if node.get("canonical_id"):
        print(f"  canonical_id={node.get('canonical_id')}")
    if node.get("source_type"):
        print(f"  source_type={node.get('source_type')}")
    if node.get("source_url"):
        print(f"  source_url={node.get('source_url')}")
    if node.get("summary"):
        print(f"  summary={node.get('summary')!r}")
    print(f"  degree={report.get('degree')}")
    print(f"  in_degree={report.get('in_degree')}")
    print(f"  out_degree={report.get('out_degree')}")
    print(f"  incident_edge_count={report.get('incident_edge_count')}")
    print(f"  neighbor_count={report.get('neighbor_count')}")
    print(f"  unique_neighbor_count={report.get('unique_neighbor_count')}")

    _print_edge_section("Outgoing Edges", report.get("out_edges") or [])
    _print_edge_section("Incoming Edges", report.get("in_edges") or [])
    _print_neighbor_section(report.get("neighbors") or [])


def main() -> int:
    args = parse_args()
    graph_dir = Path(args.graph_dir).expanduser().resolve()
    if not graph_dir.exists():
        print(f"Graph directory does not exist: {graph_dir}", file=sys.stderr)
        return 1
    try:
        report = build_report(
            graph_dir=graph_dir,
            node_id=args.node_id,
            summary_chars=max(20, int(args.summary_chars)),
        )
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        try:
            print_report(report)
        except BrokenPipeError:
            return 141
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
