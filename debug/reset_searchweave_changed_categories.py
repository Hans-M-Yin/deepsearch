"""Archive changed SearchWeave categories while preserving image-start for resume."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "opensearch_infer/outputs"
CHANGED = {"image-end", "image-interleave", "image-interleaving", "mix-interleaving"}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def plan(directory, known):
    files = sorted(set(directory.glob("*_trajectory.json")) |
                   set(directory.glob("*_failure.json")) |
                   set((directory / "failures").glob("*_failure.json")))
    remove, preserve, categories = [], [], dict(known)
    for path in files:
        # These fields precede prompt/turn data in our saved trajectories. Avoid
        # reading megabytes of embedded image bytes just to classify a file.
        with path.open() as handle:
            header = handle.read(1024)
        case_match = re.search(r'^\s*"case_id"\s*:\s*("(?:[^"\\]|\\.)*")', header, re.MULTILINE)
        category_match = re.search(r'^\s*"category"\s*:\s*("(?:[^"\\]|\\.)*")', header, re.MULTILINE)
        if case_match and category_match:
            case_id = json.loads(case_match.group(1))
            category = json.loads(category_match.group(1))
        else:
            data = json.loads(path.read_text())
            case_id = data.get("case_id")
            category = data.get("category") or data.get("original_data", {}).get("category") or known.get(case_id)
        if category not in CHANGED | {"image-start"}:
            raise ValueError(f"Unknown category: {path}: {category!r}")
        if case_id:
            categories[case_id] = category
        (remove if category in CHANGED else preserve).append(path)
    details = directory / "llm_eval_report_details.jsonl"
    rows = read_rows(details) if details.exists() else []
    kept = []
    for row in rows:
        category = row.get("category") or categories.get(row.get("case_id"))
        if category not in CHANGED | {"image-start"}:
            raise ValueError(f"Unknown evaluation category in {directory}: {row.get('case_id')}")
        if category == "image-start":
            kept.append(row)
    return remove, preserve, rows, kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", action="append", required=True)
    parser.add_argument("--apply", action="store_true", help="Archive changed trajectories and filter judge cache; default is dry run")
    args = parser.parse_args()
    known = {}
    reference = OUTPUTS / "GPT5/searchweave_no_tools/llm_eval_report_details.jsonl"
    for row in read_rows(reference):
        known[row["case_id"]] = row["category"]
    plans = []
    for value in args.directory:
        directory = Path(value).resolve()
        root_ablation = ROOT / "Qwen3-VL-8B-ablation"
        if (not (directory.is_relative_to(OUTPUTS) or directory.is_relative_to(root_ablation))
                or not directory.is_dir()):
            raise ValueError(f"Not a supported evaluation directory: {directory}")
        plans.append((directory, *plan(directory, known)))
    archive = ROOT / "opensearch_infer/archive" / ("searchweave_category_reset_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    manifest = {"applied": args.apply, "archive": str(archive) if args.apply else None, "directories": []}
    for directory, remove, preserve, rows, kept in plans:
        item = {"directory": str(directory), "remove_case_files": len(remove), "preserve_case_files": len(preserve), "judge_rows_before": len(rows), "judge_rows_after": len(kept)}
        manifest["directories"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        if not args.apply:
            continue
        relative = directory.relative_to(OUTPUTS) if directory.is_relative_to(OUTPUTS) else directory.relative_to(ROOT)
        backup = archive / relative
        backup.mkdir(parents=True, exist_ok=False)
        hashes = {str(path.relative_to(directory)): sha(path) for path in preserve}
        for name in ["llm_eval_report_details.jsonl", "llm_eval_report.json"]:
            path = directory / name
            if path.exists():
                shutil.copy2(path, backup / name)
        for path in remove:
            destination = backup / path.relative_to(directory)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), destination)
        details = directory / "llm_eval_report_details.jsonl"
        if details.exists():
            temporary = directory / "llm_eval_report_details.reset.tmp"
            temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
            temporary.replace(details)
        report_path = directory / "llm_eval_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text())
            total = len(kept)
            correct = sum(int(row.get("acc", 0)) for row in kept)
            accuracy = f"{100 * correct / total:.2f}%" if total else "0.00%"
            report.update(total=total, correct=correct, accuracy=accuracy,
                          error_count=sum(bool(row.get("error")) for row in kept),
                          label_distribution=dict(Counter(str(row.get("acc", 0)) for row in kept)),
                          category_accuracy={"image-start": {"total": total, "correct": correct, "accuracy": accuracy}} if total else {})
            temporary = directory / "llm_eval_report.reset.tmp"
            temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            temporary.replace(report_path)
        for relative, digest in hashes.items():
            if sha(directory / relative) != digest:
                raise RuntimeError(f"Preserved file changed: {directory / relative}")
        remaining = plan(directory, known)
        if remaining[0] or len(remaining[3]) != len(kept):
            raise RuntimeError(f"Post-reset verification failed: {directory}")
        item["preserved_file_hashes"] = hashes
        item["verified"] = True
        (archive / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    if args.apply:
        (archive / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
