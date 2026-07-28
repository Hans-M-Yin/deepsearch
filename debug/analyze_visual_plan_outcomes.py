"""Audit persisted visual plans against a completed synthesis graph.

The script is deliberately offline: it reads the JSONL artifacts already present in
``--graph-dir`` and does not rerun planning, image search, or grounding.

Examples:
  python debug/analyze_visual_plan_outcomes.py --graph-dir runs/my_graph
  python debug/analyze_visual_plan_outcomes.py --graph-dir runs/my_graph --sample-size 100 --seed 7
  python debug/analyze_visual_plan_outcomes.py --graph-dir runs/my_graph --all

Important limitation:
  The current graph format persists successful image nodes and plan definitions,
  but normally does not persist ImageDiscoveryResult.candidate_decisions for
  rejected / dropped plans. Therefore a no-image plan can often be classified
  only as ``not_materialized`` (or as a task-level execution failure when the
  runner state records one), rather than attributed exactly to uniqueness,
  search, validation, consistency, or grounding filtering.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


DEFAULT_SAMPLE_SIZE = 100
TEXT_NODE_TYPE = "text"
IMAGE_NODE_TYPE = "image"
SEARCH_RETRIEVED_EDGE_TYPE = "search_retrieved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline audit of persisted visual plans: recover source text nodes, "
            "match plans to image nodes, and report the strongest failure explanation "
            "available in an existing graph directory."
        )
    )
    parser.add_argument(
        "--graph-dir",
        required=True,
        help="Directory containing visual_plans.jsonl and graph JSONL files.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help=f"Number of visual plans to sample (default: {DEFAULT_SAMPLE_SIZE}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260728,
        help="Deterministic random seed used for sampling.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Print every persisted visual plan instead of sampling.",
    )
    parser.add_argument(
        "--show-variants",
        action="store_true",
        help="For successful plans, include each persisted image variant's validation result.",
    )
    parser.add_argument(
        "--max-text-chars",
        type=int,
        default=180,
        help="Maximum characters for long fields in terminal output.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                print(f"warning: ignoring malformed JSONL line {path}:{line_no}", file=sys.stderr)
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def shorten(value: Any, limit: int) -> str:
    text = clean_text(value)
    if len(text) <= limit:
        return text or "-"
    return text[: max(0, limit - 3)].rstrip() + "..."


def normalized_text(value: Any) -> str:
    return clean_text(value).casefold()


def node_label(node: dict[str, Any] | None) -> str:
    for key in ("title", "caption", "canonical_id", "summary", "node_id"):
        value = (node or {}).get(key)
        if value:
            return clean_text(value)
    return "<missing>"


def plan_query_ids(plan: dict[str, Any]) -> set[str]:
    return {
        str(query.get("query_id"))
        for query in as_list(plan.get("query_specs"))
        if isinstance(query, dict) and query.get("query_id")
    }


def plan_queries(plan: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for query in as_list(plan.get("query_specs")):
        if isinstance(query, dict) and clean_text(query.get("query")):
            values.append(clean_text(query["query"]))
    if not values:
        values = [clean_text(value) for value in as_list(plan.get("queries")) if clean_text(value)]
    return list(dict.fromkeys(values))


def plan_target_evidence_id(plan: dict[str, Any]) -> str | None:
    value = plan.get("target_evidence_id") or as_dict(plan.get("target")).get("evidence_id")
    return str(value) if value else None


def evidence_ids(edge: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    for ref in as_list(edge.get("evidence_refs")):
        if isinstance(ref, dict) and ref.get("evidence_id"):
            ids.add(str(ref["evidence_id"]))
    return ids


def origin_is_visual_plan(node: dict[str, Any], incoming_edges: list[dict[str, Any]]) -> bool:
    metadata = as_dict(node.get("metadata"))
    source = as_dict(node.get("source"))
    source_type = clean_text(source.get("source_type")).lower()
    variant_sources = {
        clean_text(as_dict(variant).get("source")).lower()
        for variant in as_list(node.get("image_variants"))
        if isinstance(variant, dict)
    }
    if source_type in {"image_search", "image_search_bundle"}:
        return True
    if clean_text(metadata.get("image_origin")).lower() == "visual_plan":
        return True
    if source_type == "wikipedia_inline_image" or "wikipedia_inline" in variant_sources:
        return False
    return any(clean_text(edge.get("edge_type")) == SEARCH_RETRIEVED_EDGE_TYPE for edge in incoming_edges)


@dataclass
class ImageMatch:
    image_node: dict[str, Any]
    edge: dict[str, Any] | None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    match_reasons: list[str] = field(default_factory=list)

    @property
    def node_id(self) -> str:
        return str(self.image_node.get("node_id") or "")


@dataclass
class TaskSignal:
    status: str
    error: str | None
    task_url: str | None
    task_title: str | None
    image_result_count: int | None
    visual_plan_count: int | None


class GraphAudit:
    def __init__(self, graph_dir: Path) -> None:
        self.graph_dir = graph_dir
        self.plans = load_jsonl(graph_dir / "visual_plans.jsonl")
        self.nodes = load_jsonl(graph_dir / "nodes.jsonl")
        self.edges = load_jsonl(graph_dir / "edges.jsonl")
        self.evidence = load_jsonl(graph_dir / "evidence.jsonl")
        self.snapshots = load_jsonl(graph_dir / "search_snapshots.jsonl")
        self.runner_state = load_json(graph_dir / "graph_runner_state.json")

        self.nodes_by_id = {
            str(node["node_id"]): node for node in self.nodes if node.get("node_id")
        }
        self.evidence_by_id = {
            str(item["evidence_id"]): item for item in self.evidence if item.get("evidence_id")
        }
        self.edges_to: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self.edges_from: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for edge in self.edges:
            if edge.get("dst_node_id"):
                self.edges_to[str(edge["dst_node_id"])].append(edge)
            if edge.get("src_node_id"):
                self.edges_from[str(edge["src_node_id"])].append(edge)

        self.image_matches_by_query_id: dict[str, list[ImageMatch]] = defaultdict(list)
        self.image_matches_by_target_evidence_id: dict[str, list[ImageMatch]] = defaultdict(list)
        self.image_matches_by_source_and_query: dict[tuple[str, str], list[ImageMatch]] = defaultdict(list)
        self.task_signals_by_source_node_id: dict[str, list[TaskSignal]] = defaultdict(list)
        self.task_signals_by_url: dict[str, list[TaskSignal]] = defaultdict(list)
        self._index_images()
        self._index_runner_state()

    def _index_images(self) -> None:
        for node in self.nodes:
            if node.get("node_type") != IMAGE_NODE_TYPE:
                continue
            incoming = self.edges_to.get(str(node.get("node_id") or ""), [])
            if not origin_is_visual_plan(node, incoming):
                continue

            matching_edges = [
                edge
                for edge in incoming
                if clean_text(edge.get("edge_type")) == SEARCH_RETRIEVED_EDGE_TYPE
                and self.nodes_by_id.get(str(edge.get("src_node_id") or ""), {}).get("node_type") == TEXT_NODE_TYPE
            ]
            related_evidence = self._image_evidence_for_node(node)
            if not matching_edges:
                matching_edges = [None]

            for edge in matching_edges:
                match = ImageMatch(image_node=node, edge=edge, evidence=related_evidence)
                edge_metadata = as_dict((edge or {}).get("metadata"))
                source_node_id = str((edge or {}).get("src_node_id") or "")
                query_id = edge_metadata.get("query_id")
                if query_id:
                    match.match_reasons.append("edge.query_id")
                    self.image_matches_by_query_id[str(query_id)].append(match)
                for evidence_item in related_evidence:
                    metadata = as_dict(evidence_item.get("metadata"))
                    evidence_query_id = metadata.get("query_id")
                    if evidence_query_id:
                        match.match_reasons.append("evidence.query_id")
                        self.image_matches_by_query_id[str(evidence_query_id)].append(match)
                    target_id = metadata.get("target_evidence_id")
                    if target_id:
                        match.match_reasons.append("image_evidence.target_evidence_id")
                        self.image_matches_by_target_evidence_id[str(target_id)].append(match)
                for target_id in evidence_ids(edge or {}):
                    if target_id in self.evidence_by_id and self.evidence_by_id[target_id].get("evidence_type") == "visual_target":
                        match.match_reasons.append("edge.visual_target_evidence_ref")
                        self.image_matches_by_target_evidence_id[target_id].append(match)
                query_text = clean_text(edge_metadata.get("query") or as_dict(node.get("metadata")).get("search_query"))
                if source_node_id and query_text:
                    self.image_matches_by_source_and_query[(source_node_id, normalized_text(query_text))].append(match)

    def _image_evidence_for_node(self, node: dict[str, Any]) -> list[dict[str, Any]]:
        node_id = str(node.get("node_id") or "")
        return [
            item
            for item in self.evidence
            if item.get("evidence_type") == "image" and node_id in {str(value) for value in as_list(item.get("node_ids"))}
        ]

    def _index_runner_state(self) -> None:
        if not self.runner_state:
            return
        for section, status in (
            ("completed_tasks", "completed"),
            ("failed_tasks", "failed"),
            ("skipped_tasks", "skipped"),
        ):
            for record in as_list(self.runner_state.get(section)):
                if not isinstance(record, dict):
                    continue
                task = as_dict(record.get("task"))
                metadata = as_dict(task.get("metadata"))
                task_type = clean_text(task.get("task_type")).lower()
                origin = clean_text(metadata.get("task_origin")).lower()
                # Older run-state files may omit task_type. In that case task_origin
                # is the reliable discriminator for image-expansion tasks.
                if task_type and task_type != "image_expand" and origin != "image_expand":
                    continue
                if not task_type and origin != "image_expand":
                    continue
                signal = TaskSignal(
                    status=status,
                    error=clean_text(record.get("error") or record.get("attribute_error")) or None,
                    task_url=clean_text(task.get("url")) or None,
                    task_title=clean_text(task.get("title")) or None,
                    image_result_count=_as_int_or_none(record.get("image_result_count")),
                    visual_plan_count=_as_int_or_none(record.get("visual_plan_count")),
                )
                source_node_id = clean_text(metadata.get("source_text_node_id"))
                if source_node_id:
                    self.task_signals_by_source_node_id[source_node_id].append(signal)
                if signal.task_url:
                    self.task_signals_by_url[signal.task_url].append(signal)

    def find_matches(self, plan: dict[str, Any]) -> list[ImageMatch]:
        matches: list[ImageMatch] = []
        for query_id in plan_query_ids(plan):
            matches.extend(self.image_matches_by_query_id.get(query_id, []))
        target_evidence_id = plan_target_evidence_id(plan)
        if target_evidence_id:
            matches.extend(self.image_matches_by_target_evidence_id.get(target_evidence_id, []))

        source_node_id = clean_text(plan.get("node_id"))
        for query in plan_queries(plan):
            matches.extend(self.image_matches_by_source_and_query.get((source_node_id, normalized_text(query)), []))

        unique: dict[tuple[str, str], ImageMatch] = {}
        for match in matches:
            edge_id = str((match.edge or {}).get("edge_id") or "")
            key = (match.node_id, edge_id)
            existing = unique.get(key)
            if existing is None:
                unique[key] = match
            else:
                existing.match_reasons = sorted(set(existing.match_reasons) | set(match.match_reasons))
        return sorted(unique.values(), key=lambda item: (item.node_id, str((item.edge or {}).get("edge_id") or "")))

    def task_signals(self, plan: dict[str, Any], source_node: dict[str, Any] | None) -> list[TaskSignal]:
        signals = list(self.task_signals_by_source_node_id.get(clean_text(plan.get("node_id")), []))
        source_url = clean_text(as_dict((source_node or {}).get("source")).get("url"))
        task_url = clean_text(plan.get("task_url"))
        for url in (source_url, task_url):
            if url:
                signals.extend(self.task_signals_by_url.get(url, []))
        unique: dict[tuple[str, str, str], TaskSignal] = {}
        for signal in signals:
            key = (signal.status, signal.task_url or "", signal.error or "")
            unique[key] = signal
        return sorted(unique.values(), key=lambda item: (item.status, item.task_url or "", item.error or ""))


def _as_int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def classify_plan(audit: GraphAudit, plan: dict[str, Any]) -> tuple[str, str, list[ImageMatch], list[TaskSignal]]:
    matches = audit.find_matches(plan)
    source_node = audit.nodes_by_id.get(clean_text(plan.get("node_id")))
    signals = audit.task_signals(plan, source_node)
    if matches:
        return (
            "SUCCESS",
            "matched persisted image node via " + ", ".join(sorted({reason for match in matches for reason in match.match_reasons})),
            matches,
            signals,
        )
    failed = [signal for signal in signals if signal.status == "failed"]
    if failed:
        details = "; ".join(
            f"{signal.task_url or '<unknown task>'}: {signal.error or 'task marked failed without an error string'}"
            for signal in failed
        )
        return "TASK_ERROR", details, matches, signals
    completed = [signal for signal in signals if signal.status == "completed"]
    if completed:
        image_counts = [signal.image_result_count for signal in completed if signal.image_result_count is not None]
        suffix = f"; runner recorded image_result_count={max(image_counts)}" if image_counts else ""
        return (
            "NOT_MATERIALIZED",
            "image-expansion task completed, but no persisted image node matches this plan" + suffix
            + ". Existing artifacts do not retain its per-candidate filter decision.",
            matches,
            signals,
        )
    return (
        "UNRESOLVED",
        "no matching persisted image node and no usable image-expansion task record; "
        "existing artifacts cannot distinguish no-result from search/validation/grounding filtering",
        matches,
        signals,
    )


def image_summary(match: ImageMatch, max_chars: int, show_variants: bool) -> list[str]:
    node = match.image_node
    metadata = as_dict(node.get("metadata"))
    edge = match.edge or {}
    lines = [
        f"    image: {node.get('node_id') or '-'} | title={shorten(node_label(node), max_chars)}",
        f"      relation={shorten(edge.get('relation'), max_chars)} | edge_query={shorten(as_dict(edge.get('metadata')).get('query'), max_chars)}",
        f"      image_query={shorten(metadata.get('search_query'), max_chars)} | target={shorten(metadata.get('visual_target'), max_chars)}",
    ]
    entities = as_list(metadata.get("grounded_entities"))
    entity_names = [clean_text(as_dict(entity).get("name")) for entity in entities if clean_text(as_dict(entity).get("name"))]
    lines.append(
        f"      grounded_entities={len(entities)}"
        + (f" [{', '.join(entity_names[:5])}]" if entity_names else "")
    )
    if show_variants:
        variants = [variant for variant in as_list(node.get("image_variants")) if isinstance(variant, dict)]
        if not variants:
            lines.append("      variants: <not persisted>")
        for variant in variants:
            lines.append(
                "      variant: "
                f"status={variant.get('validation_status') or '-'} "
                f"primary={bool(variant.get('is_primary'))} "
                f"reason={shorten(variant.get('validation_reason'), max_chars)} "
                f"title={shorten(variant.get('title'), max_chars)}"
            )
    return lines


def print_plan(
    *,
    index: int,
    total_printed: int,
    audit: GraphAudit,
    plan: dict[str, Any],
    max_chars: int,
    show_variants: bool,
) -> str:
    status, outcome, matches, signals = classify_plan(audit, plan)
    node = audit.nodes_by_id.get(clean_text(plan.get("node_id")))
    target = as_dict(plan.get("target"))
    target_metadata = as_dict(target.get("metadata"))
    metadata = as_dict(plan.get("metadata"))
    reason = plan.get("reason") or target_metadata.get("reason") or metadata.get("reason")
    judge_reason = (
        plan.get("plan_judge_reason")
        or target_metadata.get("plan_judge_reason")
        or metadata.get("plan_judge_reason")
    )
    source = as_dict((node or {}).get("source"))

    lines = [
        "=" * 110,
        f"[{index}/{total_printed}] {status}  plan_id={plan.get('plan_id') or '-'}",
        f"  text_node: {plan.get('node_id') or '-'} | {shorten(node_label(node), max_chars)}",
        f"  text_url: {source.get('url') or plan.get('node_source_url') or plan.get('task_url') or '-'}",
        f"  target: {shorten(plan.get('target_description') or target.get('content'), max_chars)}",
        f"  queries: {' | '.join(shorten(query, max_chars) for query in plan_queries(plan)) or '-'}",
        f"  planner: {plan.get('planner') or '-'} | target_type={plan.get('target_type') or target_metadata.get('target_type') or '-'}",
        f"  plan_reason: {shorten(reason, max_chars)}",
        f"  uniqueness_judge: {shorten(judge_reason, max_chars)}",
        f"  outcome: {shorten(outcome, max_chars * 2)}",
    ]
    if matches:
        for match in matches:
            lines.extend(image_summary(match, max_chars, show_variants))
    elif signals:
        for signal in signals:
            lines.append(
                "  task_signal: "
                f"status={signal.status} url={signal.task_url or '-'} "
                f"plans={signal.visual_plan_count if signal.visual_plan_count is not None else '-'} "
                f"image_results={signal.image_result_count if signal.image_result_count is not None else '-'} "
                f"error={shorten(signal.error, max_chars)}"
            )
    return "\n".join(lines)


def sample_plans(plans: list[dict[str, Any]], sample_size: int, seed: int, all_plans: bool) -> list[dict[str, Any]]:
    ordered = sorted(plans, key=lambda item: str(item.get("plan_id") or ""))
    if all_plans or sample_size <= 0 or sample_size >= len(ordered):
        return ordered
    return random.Random(seed).sample(ordered, sample_size)


def status_counts(audit: GraphAudit, plans: Iterable[dict[str, Any]]) -> Counter[str]:
    return Counter(classify_plan(audit, plan)[0] for plan in plans)


def main() -> None:
    args = parse_args()
    graph_dir = Path(args.graph_dir).expanduser()
    if not graph_dir.is_absolute():
        graph_dir = (Path.cwd() / graph_dir).resolve()
    if not graph_dir.is_dir():
        raise SystemExit(f"error: graph directory does not exist: {graph_dir}")

    audit = GraphAudit(graph_dir)
    if not audit.plans:
        raise SystemExit(
            "error: no persisted visual plans found. Expected "
            f"{graph_dir / 'visual_plans.jsonl'}"
        )

    selected = sample_plans(audit.plans, args.sample_size, args.seed, args.all)
    all_counts = status_counts(audit, audit.plans)
    sample_counts = status_counts(audit, selected)
    successful = all_counts["SUCCESS"]
    total = len(audit.plans)

    print("Visual Plan Offline Audit")
    print(f"graph_dir: {graph_dir}")
    print(
        "artifacts: "
        f"plans={len(audit.plans)} nodes={len(audit.nodes)} edges={len(audit.edges)} "
        f"evidence={len(audit.evidence)} snapshots={len(audit.snapshots)} "
        f"runner_state={'yes' if audit.runner_state else 'no'}"
    )
    print(
        f"printed: {len(selected)}/{total} "
        f"({'all' if args.all else f'sample seed={args.seed}'})"
    )
    print(
        "status semantics: SUCCESS=joined to a persisted image node; "
        "TASK_ERROR=runner task error; NOT_MATERIALIZED=completed task but no joined image; "
        "UNRESOLVED=no task-level evidence."
    )

    for index, plan in enumerate(selected, start=1):
        print(
            print_plan(
                index=index,
                total_printed=len(selected),
                audit=audit,
                plan=plan,
                max_chars=max(40, args.max_text_chars),
                show_variants=args.show_variants,
            )
        )

    print("=" * 110)
    print("Summary (all persisted visual plans)")
    print(f"  total_visual_plans: {total}")
    print(f"  successful_visual_plans: {successful}")
    print(f"  success_ratio: {successful}/{total} = {successful / total:.2%}")
    for name in ("SUCCESS", "TASK_ERROR", "NOT_MATERIALIZED", "UNRESOLVED"):
        count = all_counts[name]
        print(f"  {name.lower()}: {count} ({count / total:.2%})")
    print("Summary (printed subset)")
    for name in ("SUCCESS", "TASK_ERROR", "NOT_MATERIALIZED", "UNRESOLVED"):
        count = sample_counts[name]
        print(f"  {name.lower()}: {count}/{len(selected)} ({count / len(selected):.2%})")
    print(
        "Note: uniqueness-rejected planner candidates and detailed search/validation/grounding "
        "filter reasons were not generally persisted by prior runs, so this audit intentionally "
        "does not infer a specific cause when the artifacts only support NOT_MATERIALIZED/UNRESOLVED."
    )


if __name__ == "__main__":
    main()
