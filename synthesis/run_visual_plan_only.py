"""Generate visual-search plans from an existing text graph without image search.

This is a debugging entrypoint for checking whether the visual planner proposes
high-quality visual targets and text-to-image queries before spending search
or MLLM validation cost on image discovery.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import sys
import time
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV_PATH = Path(__file__).with_name(".env")


def load_env_file(path: Path, *, override: bool = False) -> dict[str, str]:
    loaded: dict[str, str] = {}
    if not path.exists():
        return loaded

    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            raise ValueError(f"{path}:{line_no} is not KEY=value syntax: {raw_line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        parsed = shlex.split(value.strip(), comments=False, posix=True)
        env_value = parsed[0] if parsed else ""
        if override or key not in os.environ:
            os.environ[key] = env_value
        loaded[key] = env_value
    return loaded


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"JSONL file does not exist: {path}")
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no} is not valid JSON.") from exc
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def text_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        node
        for node in nodes
        if node.get("node_type") == "text" and (node.get("description") or node.get("summary"))
    ]


def short_text(text: str | None, *, max_chars: int) -> str:
    text = text or ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


def plan_to_record(node: dict[str, Any], plan: Any) -> dict[str, Any]:
    target = plan.target
    return {
        "node_id": node.get("node_id"),
        "node_title": node.get("title") or node.get("canonical_id"),
        "node_source_url": (node.get("source") or {}).get("url"),
        "plan_id": plan.plan_id,
        "target_evidence_id": target.evidence_id,
        "target_description": target.content,
        "target_type": target.metadata.get("target_type"),
        "downstream_use": target.metadata.get("downstream_use"),
        "source_passage": target.metadata.get("source_passage"),
        "source_quote": target.metadata.get("source_quote"),
        "uniqueness": target.metadata.get("uniqueness"),
        "reason": target.metadata.get("reason"),
        "expected_visual": target.metadata.get("expected_visual"),
        "queries": [query.query for query in plan.queries],
        "query_specs": [query.to_dict() for query in plan.queries],
        "target": target.to_dict(),
        "planner": plan.planner,
        "metadata": plan.metadata,
    }


def markdown_escape(text: Any) -> str:
    return str(text or "").replace("|", "\\|").replace("\n", " ")


def write_report(path: Path, records: list[dict[str, Any]], errors: list[dict[str, Any]]) -> None:
    lines = [
        "# Visual Plan Report",
        "",
        f"- planned targets: {len(records)}",
        f"- nodes with errors: {len(errors)}",
        "",
    ]

    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["node_id"], []).append(record)

    for node_id, node_records in grouped.items():
        first = node_records[0]
        lines.extend(
            [
                f"## {first.get('node_title') or node_id}",
                "",
                f"- node_id: `{node_id}`",
                f"- source: {first.get('node_source_url') or ''}",
                "",
            ]
        )
        for index, record in enumerate(node_records, start=1):
            lines.extend(
                [
                    f"### Target {index}: {record.get('target_description')}",
                    "",
                    f"- type: `{record.get('target_type')}`",
                    f"- use: `{record.get('downstream_use')}`",
                    f"- source_passage: `{record.get('source_passage') or ''}`",
                    f"- source_quote: {record.get('source_quote') or ''}",
                    f"- uniqueness: {record.get('uniqueness') or ''}",
                    f"- reason: {record.get('reason') or ''}",
                    f"- expected_visual: {record.get('expected_visual') or ''}",
                    "- queries:",
                ]
            )
            for query in record.get("queries") or []:
                lines.append(f"  - {query}")
            lines.append("")

    if errors:
        lines.extend(["## Errors", ""])
        for error in errors:
            lines.append(
                f"- `{markdown_escape(error.get('node_id'))}` {markdown_escape(error.get('title'))}: "
                f"{markdown_escape(error.get('error'))}"
            )

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", default=str(DEFAULT_ENV_PATH), help="Path to synthesis env file.")
    parser.add_argument("--override-env", action="store_true", help="Let --env-file override existing env vars.")
    parser.add_argument("--run-dir", required=True, help="Existing graph run directory containing nodes.jsonl.")
    parser.add_argument("--nodes-file", default="nodes.jsonl", help="Node JSONL filename inside --run-dir.")
    parser.add_argument("--output-jsonl", default="visual_plans.jsonl", help="Output JSONL filename inside --run-dir.")
    parser.add_argument("--output-report", default="visual_plan_report.md", help="Output Markdown report filename inside --run-dir.")
    parser.add_argument("--max-nodes", type=int, default=20, help="Maximum text nodes to plan from.")
    parser.add_argument("--start", type=int, default=0, help="Start offset after filtering text nodes.")
    parser.add_argument("--max-targets", type=int, default=3, help="Maximum visual targets per node.")
    parser.add_argument("--max-queries-per-target", type=int, default=4, help="Maximum queries per target.")
    parser.add_argument("--max-content-chars", type=int, default=12000, help="Max node description chars sent to planner. <=0 disables truncation.")
    parser.add_argument("--model-alias", default=None, help="Optional model alias overriding VISUAL_PLANNER_MODEL.")
    parser.add_argument("--run-id", default=None, help="Optional run id recorded in generated evidence.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)

    env_path = Path(args.env_file)
    if not env_path.is_absolute():
        env_path = PROJECT_ROOT / env_path
    loaded_env = load_env_file(env_path, override=args.override_env)

    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from synthesis.model_worker import LLM_WORKER
    from synthesis.visual_planner import LLMVisualSearchPlanner

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = PROJECT_ROOT / run_dir
    nodes = text_nodes(load_jsonl(run_dir / args.nodes_file))
    selected_nodes = nodes[max(0, args.start) : max(0, args.start) + max(0, args.max_nodes)]

    planner = LLMVisualSearchPlanner(
        model_client=LLM_WORKER,
        model_alias=args.model_alias,
        max_targets=args.max_targets,
        max_queries_per_target=args.max_queries_per_target,
    )

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    started_at = time.perf_counter()
    for index, node in enumerate(selected_nodes, start=1):
        title = node.get("title") or node.get("canonical_id") or node.get("node_id")
        print(f"[{index}/{len(selected_nodes)}] planning: {title}", flush=True)
        page_text = short_text(node.get("description") or node.get("summary") or "", max_chars=args.max_content_chars)
        try:
            plans = planner.plan(
                node=node,
                page_text=page_text,
                source_evidence_ids=[],
                run_id=args.run_id,
            )
        except Exception as exc:
            errors.append(
                {
                    "node_id": node.get("node_id"),
                    "title": title,
                    "error": f"{exc.__class__.__name__}: {exc}",
                }
            )
            continue
        for plan in plans:
            records.append(plan_to_record(node, plan))

    output_jsonl = run_dir / args.output_jsonl
    output_report = run_dir / args.output_report
    write_jsonl(output_jsonl, records)
    write_report(output_report, records, errors)

    elapsed_s = time.perf_counter() - started_at
    print("=== visual plan only ===")
    print(f"env_file: {env_path} ({len(loaded_env)} vars loaded)")
    print(f"run_dir: {run_dir}")
    print(f"nodes_seen: {len(nodes)}")
    print(f"nodes_selected: {len(selected_nodes)}")
    print(f"plans: {len(records)}")
    print(f"errors: {len(errors)}")
    print(f"elapsed_s: {elapsed_s:.2f}")
    print(f"output_jsonl: {output_jsonl}")
    print(f"output_report: {output_report}")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
