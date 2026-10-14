"""Extract inspectable category diagnostics from saved SearchWeave trajectories."""
import ast
import csv
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path("opensearch_infer/outputs/Qwen3-VL-32B/searchweave_0921_final")
OUT = Path("opensearch_infer/analysis/Qwen3-VL-32B_searchweave_category_audit")


def structured(value):
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                return parser(value)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                pass
    return value


def english(value):
    value = structured(value)
    return str(value.get("en") or value.get("zh") or "") if isinstance(value, dict) else str(value or "")


def normal(text):
    return " ".join(re.sub(r"[^\w]+", " ", str(text).casefold()).split())


def extract(path, judge, completed):
    d = json.loads(path.read_text())
    original = d.get("original_data", {})
    metadata = structured(original.get("source_metadata", {}))
    if not isinstance(metadata, dict):
        metadata = {}
    chain = metadata.get("search_chain", {})
    nodes = chain.get("nodes", [])
    question = english(metadata.get("question") or original.get("question"))
    answer = english(metadata.get("answer") or original.get("answer"))
    calls, counters, output_keys = [], Counter(), Counter()
    image_reads = 0
    first_response = ""
    generated_tokens = 0
    for t in d.get("turns", []):
        first_response = first_response or t.get("response_text", "")
        response = t.get("response", {})
        raw = response.get("raw_response") or {}
        generated_tokens += (raw.get("usage") or {}).get("completion_tokens", 0) or 0
        output_keys.update(t.keys())
        call = structured(t.get("tool_call"))
        if not isinstance(call, dict):
            continue
        name = call.get("name")
        if t.get("tool_output_type") == "max_turns_finalization":
            continue
        parameters = call.get("parameters", call.get("arguments", {}))
        counters[name] += 1
        if name == "read_url" and "image_" in json.dumps(parameters):
            image_reads += 1
        output = str(t.get("tool_output", ""))
        calls.append(dict(turn=t.get("turn"), name=name, parameters=parameters,
                          reasoning=t.get("response_text", ""), output=output))
    images = original.get("images") or []
    node_types = [n.get("node_type") for n in nodes]
    image_positions = [i for i, kind in enumerate(node_types) if kind == "image"]
    inferred = "image-interleave"
    if len(image_positions) == 1:
        if image_positions[0] == 0:
            inferred = "image-start"
        elif image_positions[0] == len(nodes) - 1:
            inferred = "image-end"
    mentions = []
    for i, n in enumerate(nodes):
        name = english(n.get("name"))
        if i > 0 and n.get("node_type") == "text" and len(normal(name)) >= 4 and normal(name) in normal(question):
            mentions.append(dict(index=i, name=name))
    record = dict(case_id=d["case_id"], category=d.get("category"), completed=completed,
                  correct=int(judge[d["case_id"]]["acc"]) if d["case_id"] in judge else None,
                  question=question, reference_answer=answer,
                  model_answer=d.get("final_response_text", ""),
                  judge_reason=judge.get(d["case_id"], {}).get("reasoning", ""),
                  hop_count=chain.get("hop_count"), node_types=node_types,
                  path_nodes=[dict(index=i, node_type=n.get("node_type"), name=english(n.get("name")), wiki_url=n.get("wiki_url")) for i, n in enumerate(nodes)],
                  inferred_category=inferred, input_image_count=len(images),
                  expected_image_nodes=len(image_positions), turns=len(d.get("turns", [])),
                  total_calls=sum(counters.values()), tools=dict(counters), image_reads=image_reads,
                  visual_search_calls=counters["t2i_search"] + counters["i2i_search"],
                  first_tool=calls[0]["name"] if calls else None, generated_tokens=generated_tokens,
                  middle_text_names_in_question=mentions,
                  max_turns_finalization=any(t.get("tool_output_type") == "max_turns_finalization" for t in d.get("turns", [])),
                  failure_kind=d.get("failure_kind"), failure_reason=d.get("failure_reason"),
                  first_response=first_response, calls=calls, file=str(path))
    return record


def main():
    judge_rows = [json.loads(l) for l in (ROOT / "llm_eval_report_details.jsonl").read_text().splitlines() if l.strip()]
    judge = {r["case_id"]: r for r in judge_rows}
    assert len(judge) == len(judge_rows) == 296
    records = []
    for directory, pattern, completed in ((ROOT, "*_trajectory.json", True), (ROOT / "failures", "*_failure.json", False)):
        paths = sorted(directory.glob(pattern))
        print(f"Reading {len(paths)} files: completed={completed}", flush=True)
        for i, path in enumerate(paths, 1):
            records.append(extract(path, judge, completed))
            if i % 50 == 0:
                print(f"Read {i}/{len(paths)}", flush=True)
    assert len(records) == len({r["case_id"] for r in records}) == 300
    assert {r["case_id"] for r in records if r["completed"]} == set(judge)
    assert all(r["category"] == r["inferred_category"] for r in records)
    OUT.mkdir(parents=True, exist_ok=True)
    # Tool outputs can be large; retain a compact analysis table and a separate trace file.
    compact = [{k: v for k, v in r.items() if k != "calls"} for r in records]
    (OUT / "samples.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in compact))
    (OUT / "traces.jsonl").write_text("".join(json.dumps(dict(case_id=r["case_id"], calls=r["calls"]), ensure_ascii=False) + "\n" for r in records))
    with (OUT / "samples.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["case_id", "category", "completed", "correct", "hop_count", "input_image_count", "turns", "total_calls", "visual_search_calls", "image_reads", "first_tool", "generated_tokens", "max_turns_finalization", "question", "reference_answer", "model_answer", "judge_reason", "file"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in records)
    for category in ("image-start", "image-end", "image-interleave"):
        rows = [r for r in records if r["category"] == category]
        print(category, "samples", len(rows), "completed", sum(r["completed"] for r in rows),
              "correct", sum(r["correct"] or 0 for r in rows), "hop_counts", dict(Counter(r["hop_count"] for r in rows)),
              "input_images", dict(Counter(r["input_image_count"] for r in rows)),
              "no_visual_search", sum(r["visual_search_calls"] == 0 for r in rows))


if __name__ == "__main__":
    main()
