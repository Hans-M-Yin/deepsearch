"""Find hard and likely duplicate text nodes in a persisted graph.

The audit is offline: it never calls Wikipedia/Wikidata.  It reports exact
identity collisions (normalized URL, canonical ID, QID) separately from the
lower-confidence title/alias collisions that need review.

Example:
  python debug/check_duplicate_text_nodes.py --graph-dir runs/example --pretty
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, unquote, urlsplit, urlunsplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit duplicate text-node identities in a graph directory.")
    parser.add_argument("--graph-dir", required=True, help="Directory containing nodes.jsonl and edges.jsonl.")
    parser.add_argument("--limit", type=int, default=50, help="Maximum groups to include per duplicate check; <=0 means all.")
    parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON output.")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                records.append(record)
    return records


def _normalize_label(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _source_url(node: dict[str, Any]) -> str:
    source = node.get("source") or {}
    return str(source.get("url") or "").strip() if isinstance(source, dict) else ""


def _normalize_url(value: Any) -> str:
    url = str(value or "").strip()
    if not url:
        return ""
    parsed = urlsplit(url)
    if not parsed.netloc:
        return url.split("#", 1)[0]
    host = parsed.netloc.casefold()
    path = parsed.path or "/"
    if host.endswith("wikipedia.org") and path.startswith("/wiki/"):
        title = unquote(path.removeprefix("/wiki/")).split("#", 1)[0]
        path = "/wiki/" + quote(title, safe=":/()_,")
    return urlunsplit(("https", host, path, parsed.query, ""))


def _wikidata_qid(node: dict[str, Any]) -> str:
    values: list[Any] = [node.get("canonical_id")]
    source = node.get("source") or {}
    if isinstance(source, dict):
        values.append(source.get("source_id"))
    for value in values:
        match = re.search(r"(?:wikidata:)?\b(Q\d+)\b", str(value or ""), flags=re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return ""


def _node_summary(node: dict[str, Any], degrees: Counter[str]) -> dict[str, Any]:
    node_id = str(node.get("node_id") or "")
    return {
        "node_id": node_id,
        "title": node.get("title"),
        "canonical_id": node.get("canonical_id"),
        "source_url": _source_url(node),
        "normalized_source_url": _normalize_url(_source_url(node)),
        "wikidata_qid": _wikidata_qid(node) or None,
        "aliases": list(node.get("aliases") or []),
        "degree": degrees[node_id],
    }


def _duplicate_groups(
    nodes: list[dict[str, Any]],
    *,
    key_fn: Callable[[dict[str, Any]], str],
    degrees: Counter[str],
    limit: int,
) -> dict[str, Any]:
    by_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for node in nodes:
        key = key_fn(node)
        if key:
            by_key[key].append(node)
    groups = [(key, members) for key, members in by_key.items() if len(members) > 1]
    groups.sort(key=lambda item: (-len(item[1]), item[0]))
    shown = groups if limit <= 0 else groups[:limit]
    return {
        "duplicate_group_count": len(groups),
        "duplicate_node_count": sum(len(members) for _, members in groups),
        "excess_node_count": sum(len(members) - 1 for _, members in groups),
        "groups": [
            {
                "identity_key": key,
                "node_count": len(members),
                "nodes": sorted((_node_summary(node, degrees) for node in members), key=lambda item: item["node_id"]),
            }
            for key, members in shown
        ],
        "omitted_group_count": len(groups) - len(shown),
    }


def _label_collision_groups(nodes: list[dict[str, Any]], degrees: Counter[str], limit: int) -> dict[str, Any]:
    by_label: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    label_sources: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for node in nodes:
        node_id = str(node.get("node_id") or "")
        labels = [(node.get("title"), "title")]
        labels.extend((alias, "alias") for alias in node.get("aliases") or [])
        for raw, source in labels:
            label = _normalize_label(raw)
            if label:
                by_label[label][node_id] = node
                label_sources[label][node_id].add(source)

    groups = [(label, list(members.values())) for label, members in by_label.items() if len(members) > 1]
    groups.sort(key=lambda item: (-len(item[1]), item[0]))
    shown = groups if limit <= 0 else groups[:limit]
    return {
        "confidence": "candidate_only",
        "warning": "Shared labels can be ambiguous; do not merge solely from this check.",
        "duplicate_group_count": len(groups),
        "duplicate_node_count": sum(len(members) for _, members in groups),
        "excess_node_count": sum(len(members) - 1 for _, members in groups),
        "groups": [
            {
                "identity_key": label,
                "node_count": len(members),
                "nodes": [
                    {
                        **_node_summary(node, degrees),
                        "matching_label_sources": sorted(label_sources[label][str(node.get("node_id") or "")]),
                    }
                    for node in sorted(members, key=lambda item: str(item.get("node_id") or ""))
                ],
            }
            for label, members in shown
        ],
        "omitted_group_count": len(groups) - len(shown),
    }


def build_report(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], limit: int) -> dict[str, Any]:
    text_nodes = [node for node in nodes if node.get("node_type") == "text"]
    degrees: Counter[str] = Counter()
    for edge in edges:
        for key in ("src_node_id", "dst_node_id"):
            node_id = str(edge.get(key) or "")
            if node_id:
                degrees[node_id] += 1
    return {
        "input_records": {"text_nodes": len(text_nodes), "all_nodes": len(nodes), "edges": len(edges)},
        "hard_duplicate_checks": {
            "normalized_source_url": _duplicate_groups(
                text_nodes, key_fn=lambda node: _normalize_url(_source_url(node)), degrees=degrees, limit=limit
            ),
            "canonical_id": _duplicate_groups(
                text_nodes, key_fn=lambda node: str(node.get("canonical_id") or "").strip().casefold(), degrees=degrees, limit=limit
            ),
            "wikidata_qid": _duplicate_groups(text_nodes, key_fn=_wikidata_qid, degrees=degrees, limit=limit),
        },
        "likely_duplicate_checks": {
            "shared_title_or_alias": _label_collision_groups(text_nodes, degrees, limit),
        },
        "interpretation": {
            "hard_duplicate": "Same normalized URL, canonical ID, or Wikidata QID; normally safe to investigate as duplicate identity.",
            "likely_duplicate": "Shared title/alias only; inspect context before merging because names can be ambiguous.",
            "degree": "Total in-degree plus out-degree. Duplicate entities with split, low-degree neighborhoods are especially harmful.",
        },
    }


def main() -> int:
    args = parse_args()
    graph_dir = Path(args.graph_dir).expanduser().resolve()
    if not graph_dir.exists():
        raise SystemExit(f"Graph directory does not exist: {graph_dir}")
    report = build_report(
        _load_jsonl(graph_dir / "nodes.jsonl"),
        _load_jsonl(graph_dir / "edges.jsonl"),
        args.limit,
    )
    report["graph_dir"] = str(graph_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2 if args.pretty else None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
