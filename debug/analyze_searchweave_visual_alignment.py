"""Audit saved visual tool use against each trajectory's construction snapshot.

Read-only with respect to source traces and benchmark. Semantic judgments are
kept separate from deterministic counts; no model/API calls are made.
"""
import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


def structured(value):
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            pass
    return {}


def tool_json(value):
    text = str(value or "")
    text = re.sub(r"^\s*<tool_response>\s*|\s*</tool_response>\s*$", "", text)
    return structured(text)


def english(value):
    return value.get("en", value.get("zh", "")) if isinstance(value, dict) else str(value or "")


def analyze(trace_dir, output_dir):
    # A successful retry supersedes the old failure for the same question.
    selected = {}
    for file in sorted((trace_dir / "failures").glob("*.json")):
        data = json.loads(file.read_text())
        selected[data["case_id"]] = (file, data)
    failure_files = len(selected)
    for file in sorted(trace_dir.glob("*_trajectory.json")):
        data = json.loads(file.read_text())
        selected[data["case_id"]] = (file, data)
    judges = {}
    judge_file = trace_dir / "llm_eval_report_details.jsonl"
    if judge_file.exists():
        for line in judge_file.open():
            if line.strip():
                record = json.loads(line)
                judges[record["case_id"]] = record
    cases, all_calls = [], []
    for case_id, (file, data) in sorted(selected.items()):
        original = data.get("original_data", {})
        meta = structured(original.get("source_metadata"))
        chain = meta.get("search_chain", {})
        nodes = sorted(chain.get("nodes", []), key=lambda n: n.get("index", 0))
        graph = [{"index": n["index"], "type": n.get("node_type"),
                  "name": english(n.get("name"))} for n in nodes]
        edges = [{"source": e.get("source_index"), "target": e.get("target_index"),
                  "relation": english(e.get("relation"))} for e in chain.get("edges", [])]
        image_indices = [i for i, n in enumerate(nodes) if n.get("node_type") == "image"]
        input_count = len(original.get("images") or [])
        latest = {"kind": "initial_input", "crop_depth": 0} if input_count else None
        resources = data.get("url_resources", {}).get("resources", {})
        image_origins = {}
        calls, downloads = [], []
        turns = data.get("turns", [])
        for idx, turn in enumerate(turns):
            call = structured(turn.get("tool_call"))
            # A parsed proposal without a tool output was never executed.
            if not call or "tool_output" not in turn or turn.get("tool_output_type") == "max_turns_finalization":
                continue
            name = call.get("name")
            args = call.get("parameters", call.get("arguments", {})) or {}
            text = str(turn.get("tool_output") or "")
            result = tool_json(text)
            error = ("Tool execution error:" in text or result.get("ok") is False)
            turn_id = turn.get("turn", idx)
            record = {"case_id": case_id, "category": data.get("category"),
                      "turn": turn_id, "tool": name, "arguments": args,
                      "error": error, "output_excerpt": text[:3000],
                      "reasoning": turn.get("response_text", ""),
                      "next_reasoning": turns[idx + 1].get("response_text", "") if idx + 1 < len(turns) else ""}
            if name in ("t2i_search", "i2i_search"):
                record["input_image_provenance"] = dict(latest) if name == "i2i_search" and latest else None
                record["no_input_image_error"] = name == "i2i_search" and not latest and error
                matches = result.get("results", result.get("matches", [])) or []
                record["result_titles"] = [r.get("title", "") for r in matches if isinstance(r, dict)]
                calls.append(record)
                for item in matches:
                    if isinstance(item, dict) and item.get("image_id"):
                        image_origins[item["image_id"]] = {
                            "kind": name, "search_turn": turn_id,
                            "query": args.get("query", ""), "title": item.get("title", ""),
                            "resource_id": item["image_id"], "crop_depth": 0}
                # Current implementation appends a crop to image history when
                # cropping succeeds. This is inferred, not explicit trace state.
                if name == "i2i_search" and args.get("region") not in (None, "") and latest and not error:
                    latest = {**latest, "crop_depth": latest.get("crop_depth", 0) + 1,
                              "last_crop_turn": turn_id}
            if name == "read_url" and "读取图片如下" in text:
                ref = args.get("resource_id", args.get("url", ""))
                origin = image_origins.get(ref, {})
                registered = resources.get(ref, {})
                latest = {**origin, "kind": origin.get("kind", "read_url_other"),
                          "read_turn": turn_id, "resource_id": ref, "crop_depth": 0,
                          "goal": args.get("goal", ""), "url": registered.get("primary_url", args.get("url", ""))}
                downloads.append(dict(latest))
            all_calls.append(record)
        used_searches = {(d.get("kind"), d.get("search_turn")) for d in downloads}
        for call in calls:
            call["result_image_downloaded"] = (call["tool"], call["turn"]) in used_searches
        counts = Counter(c["tool"] for c in calls)
        judge = judges.get(case_id, {})
        cases.append({"case_id": case_id, "category": data.get("category"), "file": str(file),
                      "case_idx": data.get("case_idx"),
                      "evaluation_eligible": not meta.get("is_placeholder", False) and meta.get("evaluation_eligible", True) and not meta.get("exclude_from_metrics", False),
                      "status": data.get("status"), "failure_kind": data.get("failure_kind"),
                      "question": original.get("question", english(meta.get("question"))),
                      "reference_answer": original.get("answer", english(meta.get("answer"))),
                      "judge_acc": judge.get("acc"), "model_answer": judge.get("model_answer", ""),
                      "graph": graph, "edges": edges, "input_image_count": input_count,
                      "has_interior_image": any(0 < i < len(nodes)-1 for i in image_indices),
                      "visual_calls": calls, "downloads": downloads,
                      "t2i_calls": counts["t2i_search"], "i2i_calls": counts["i2i_search"],
                      "turn_count": len(turns), "timestamp": data.get("timestamp"),
                      "served_models": sorted({t.get("response", {}).get("modelVersion", "") for t in turns} - {""})})
    summary = {"trace_directory": str(trace_dir), "cases": len(cases),
               "eligible_cases": sum(c["evaluation_eligible"] for c in cases),
               "excluded_case_ids": [c["case_id"] for c in cases if not c["evaluation_eligible"]],
               "raw_failure_files": failure_files,
               "status": dict(Counter(c["status"] for c in cases)),
               "served_model_case_counts": dict(Counter(m for c in cases for m in c["served_models"])),
               "by_category": {}, "scope": "Saved original_data paths. Category tool statistics exclude explicitly ineligible placeholders; raw status/model totals include them.",
               "provenance_caveat": "Last-image provenance and crop depth inferred from saved calls and current tool implementation; image-state IDs are not recorded per turn.",
               "semantic_alignment": "Not inferred from token overlap; requires manual evidence review."}
    for category in sorted({c["category"] for c in cases}):
        group = [c for c in cases if c["category"] == category and c["evaluation_eligible"]]
        vcalls = [v for c in group for v in c["visual_calls"]]
        summary["by_category"][category] = {
            "cases": len(group), "completed": sum(c["status"] == "completed" for c in group),
            "with_t2i": sum(c["t2i_calls"] > 0 for c in group),
            "with_i2i": sum(c["i2i_calls"] > 0 for c in group),
            "with_either": sum(c["t2i_calls"] + c["i2i_calls"] > 0 for c in group),
            "with_interior_image": sum(c["has_interior_image"] for c in group),
            "downloads": sum(len(c["downloads"]) for c in group),
            "visual_calls": dict(Counter(v["tool"] for v in vcalls)),
            "visual_errors": dict(Counter(v["tool"] for v in vcalls if v["error"])),
            "i2i_missing_input_calls": sum(v["no_input_image_error"] for v in vcalls),
            "i2i_missing_input_cases": sum(any(v["no_input_image_error"] for v in c["visual_calls"]) for c in group),
            "search_calls_with_download": dict(Counter(v["tool"] for v in vcalls if v["result_image_downloaded"])),
            "i2i_inferred_input_origin": dict(Counter((v["input_image_provenance"] or {}).get("kind", "none") for v in vcalls if v["tool"] == "i2i_search")),
            "i2i_inferred_input_previously_cropped": sum(v["tool"] == "i2i_search" and (v["input_image_provenance"] or {}).get("crop_depth", 0) > 0 for v in vcalls)}
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, records in [("cases", cases), ("calls", all_calls)]:
        (output_dir / f"{name}.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in records))
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    fields = ["case_id", "case_idx", "category", "evaluation_eligible", "status", "judge_acc", "input_image_count", "has_interior_image", "t2i_calls", "i2i_calls", "turn_count", "question", "reference_answer", "file"]
    with (output_dir / "cases.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(cases)
    assert len(cases) == len({c["case_id"] for c in cases})
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return cases


def audit_crop_cache(cases, output_dir):
    """Inspect recovered crop files; never edit or regenerate source images."""
    from datetime import datetime, timezone
    import numpy as np
    from PIL import Image

    records = []
    for case in cases:
        if not case["evaluation_eligible"]:
            continue
        cache = Path(case["file"]).parent
        if cache.name == "failures":
            cache = cache.parent
        cache /= "intermediate"
        for call in case["visual_calls"]:
            region = call["arguments"].get("region")
            if call["tool"] != "i2i_search" or not isinstance(region, list) or len(region) != 4:
                continue
            file = cache / f"fvqa_test_{case['case_idx']}_trajectory_turn{call['turn']}_i2i_search_region.png"
            if not file.exists():
                continue
            modified = datetime.fromtimestamp(file.stat().st_mtime, timezone.utc).isoformat()
            if modified[:10] != case["timestamp"][:10]:
                continue
            with Image.open(file) as image:
                black = np.all(np.asarray(image.convert("RGB")) == 0, axis=2)
                size = image.size
            records.append({"case_id": case["case_id"], "category": case["category"],
                            "turn": call["turn"], "region": region, "crop_size": list(size),
                            "size_matches_pixel_interpretation": size == (int(region[2]) - int(region[0]), int(region[3]) - int(region[1])),
                            "black_fraction": float(black.mean()), "all_black": bool(black.all()),
                            "tool_error": call["error"], "file": str(file),
                            "cache_mtime": modified, "trace_timestamp": case["timestamp"]})
    summary = {
        "cache_images_checked": len(records),
        "pixel_sized_crops": sum(r["size_matches_pixel_interpretation"] for r in records),
        "all_black_crops": sum(r["all_black"] for r in records),
        "all_black_cases": len({r["case_id"] for r in records if r["all_black"]}),
        "at_least_half_black": sum(r["black_fraction"] >= 0.5 for r in records),
        "category_counts": dict(Counter(r["category"] for r in records)),
        "all_black_by_category": dict(Counter(r["category"] for r in records if r["all_black"])),
        "limits": "Eligible cases only. Existing cached files matched by case_idx, turn and UTC date. Shared caches can retain/overwrite earlier files, so this is recovered-cache evidence, not full runtime prevalence. Partly black images need visual interpretation."
    }
    (output_dir / "crop_cache_audit.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (output_dir / "crop_cache_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, default=Path("opensearch_infer/outputs/Qwen3-VL-8B-SFT13_5-RL5-step110/searchweave_1005"))
    parser.add_argument("--output-dir", type=Path, default=Path("opensearch_infer/analysis/searchweave_visual_alignment_20261015"))
    parser.add_argument("--audit-crops", action="store_true")
    args = parser.parse_args()
    records = analyze(args.trace_dir, args.output_dir)
    if args.audit_crops:
        audit_crop_cache(records, args.output_dir)
