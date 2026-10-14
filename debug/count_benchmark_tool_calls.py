"""Audit saved inference tool invocations, including max-turn failures.

Run from the repository root: python debug/count_benchmark_tool_calls.py
Only final trajectory/failure files are inputs; intermediate snapshots are not.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("opensearch_infer/outputs/Qwen3-VL-8B")
OUT = Path("opensearch_infer/analysis/Qwen3-VL-8B_tool_calls_0921")
BENCHMARKS = ("searchweave", "browsecomp_vl", "mmsearch", "visbrowse")


def read_sample(path, benchmark, completed):
    raw = path.read_bytes()
    data = json.loads(raw)
    original = data.get("original_data") or {}
    case_id = str(original.get("data_id") or data.get("case_id") or "")
    if not case_id:
        raise ValueError(f"Missing sample ID: {path}")
    if data.get("case_id") and str(data["case_id"]) != case_id:
        raise ValueError(f"Conflicting sample IDs: {path}")
    turns = data.get("turns", [])
    kind = data.get("failure_kind", "")
    max_turns = data.get("max_turns")
    failed_turn = data.get("failed_turn")
    max_failure = not completed and (
        kind in ("max_turns", "max_turns_exceeded")
        or (kind == "invalid_response" and isinstance(max_turns, int)
            and isinstance(failed_turn, int) and failed_turn >= max_turns - 1
            and len(turns) >= max_turns)
    )
    requests, executed, skipped = Counter(), Counter(), Counter()
    for turn in turns:
        call = turn.get("tool_call")
        if not call:
            continue
        payload = json.loads(call) if isinstance(call, str) else call
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Malformed structured tool call: {path}, turn={turn.get('turn')}")
        requests[name] += 1
        unexecuted = (turn.get("tool_output_type") == "max_turns_finalization"
                      or (kind == "tool_call_limit" and turn.get("turn") == failed_turn))
        (skipped if unexecuted else executed)[name] += 1
    if sum(requests.values()) != data.get("tool_call_count", sum(requests.values())):
        raise ValueError(f"Stored tool count mismatch: {path}")
    assert requests == executed + skipped
    return dict(benchmark=benchmark, case_id=case_id, case_idx=data.get("case_idx"),
                completed=completed, max_turns_failure=max_failure,
                failure_kind=kind, failed_turn=failed_turn, max_turns=max_turns,
                turns=len(turns), timestamp=data.get("timestamp", ""),
                executed=dict(executed), requested=dict(requests), skipped=dict(skipped),
                path=str(path), sha256=hashlib.sha256(raw).hexdigest())


def aggregate(samples):
    result = {}
    for benchmark in BENCHMARKS:
        rows = [r for r in samples if r["benchmark"] == benchmark]
        scopes = {}
        for scope, selected in (
            ("completed_plus_max_turns", [r for r in rows if r["completed"] or r["max_turns_failure"]]),
            ("all_samples", rows),
        ):
            totals = {field: Counter() for field in ("executed", "requested", "skipped")}
            for row in selected:
                for field in totals:
                    totals[field].update(row[field])
            scopes[scope] = dict(
                unique_samples=len(selected), completed_samples=sum(r["completed"] for r in selected),
                max_turns_failures=sum(r["max_turns_failure"] for r in selected),
                other_failures=sum(not r["completed"] and not r["max_turns_failure"] for r in selected),
                failure_kinds=dict(Counter(r["failure_kind"] for r in selected if not r["completed"])),
                **{field: dict(sorted(value.items())) for field, value in totals.items()},
                total_executed=sum(totals["executed"].values()),
                total_requested=sum(totals["requested"].values()),
                total_skipped=sum(totals["skipped"].values()),
            )
        result[benchmark] = scopes
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--benchmark", action="append", metavar="NAME=DIRECTORY",
                        help="Repeat for each benchmark; defaults to the 0921 directories.")
    args = parser.parse_args()
    directories = dict(item.split("=", 1) for item in args.benchmark) if args.benchmark else {
        name: f"{name}_0921_final" for name in BENCHMARKS}
    if set(directories) != set(BENCHMARKS):
        parser.error("Specify exactly searchweave, browsecomp_vl, mmsearch, and visbrowse.")
    output = args.output
    groups = defaultdict(list)
    for benchmark in BENCHMARKS:
        directory = args.root / directories[benchmark]
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        inputs = [(p, True) for p in sorted(directory.glob("*_trajectory.json"))]
        inputs += [(p, False) for p in sorted((directory / "failures").glob("*_failure.json"))]
        print(f"Reading {directory}: {len(inputs)} final sample files", flush=True)
        for path, completed in inputs:
            row = read_sample(path, benchmark, completed)
            groups[(benchmark, row["case_id"])].append(row)
    # Prefer a completed result; otherwise use the most recent final attempt.
    samples = [max(rows, key=lambda r: (r["completed"], str(r["timestamp"]), r["path"]))
               for rows in groups.values()]
    duplicates = {f"{bench}/{case_id}": [r["path"] for r in rows]
                  for (bench, case_id), rows in groups.items() if len(rows) > 1}
    totals = aggregate(samples)
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(input_root=str(args.root), input_directories=directories, duplicate_sample_files=duplicates, benchmarks=totals,
                   counting_rule="One parsed tool call per turn, excluding synthetic max_turns_finalization and unexecuted tool_call_limit requests. Tool execution errors count as attempted invocations.")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    (output / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples))
    tools = sorted({name for r in samples for name in r["requested"]})
    with (output / "samples.csv").open("w", newline="") as handle:
        fields = ["benchmark", "case_id", "completed", "max_turns_failure", "failure_kind", "turns", "total_executed", "total_requested", "total_skipped"] + tools + ["path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in samples:
            row = {k: r[k] for k in fields if k in r}
            row.update({name: r["executed"].get(name, 0) for name in tools})
            row.update({f"total_{field}": sum(r[field].values()) for field in ("executed", "requested", "skipped")})
            writer.writerow(row)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
