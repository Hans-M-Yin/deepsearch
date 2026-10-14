"""Evaluate the same uniqueness reference dataset across configured LLMs."""
import argparse
import csv
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


MODELS = {
    "gpt54": "gpt54_internal_azure",
    "claude46": "claude_4_6_sonnet",
    "gemini38flash": "gemini38flash_internal_azure",
    "gemini35flash": "gemini35flash_internal_azure",
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--discussion", type=Path, default=Path(__file__).resolve().parents[1] / "synthesis/ignore/DiscussioN_about_uniqueness.txt")
    parser.add_argument("--workers", type=int, default=4, help="Concurrent requests per model")
    parser.add_argument("--model", action="append", metavar="DIRECTORY=ALIAS", help="Override the models to evaluate; repeat for multiple models")
    args = parser.parse_args()
    if args.workers <= 0:
        parser.error("--workers must be positive")
    models = dict(item.split("=", 1) for item in args.model) if args.model else MODELS
    if not models or any(not name or Path(name).name != name or name in (".", "..") or not alias for name, alias in models.items()):
        parser.error("Model directories must be simple names and aliases must be nonempty")
    from eval_image_uniqueness import load_gold_examples
    discussion = args.discussion.resolve()
    expected = [asdict(example) for example in load_gold_examples(discussion)]
    expected_by_id = {example["image_id"]: example for example in expected}
    expected_count = len(expected)
    assert expected_count and len(expected_by_id) == expected_count
    repo = Path(__file__).resolve().parents[1]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    def run_model(name, alias):
        directory = output / name
        directory.mkdir(exist_ok=True)
        # A separate directory prevents resuming predictions from another model.
        config = directory / "run_config.json"
        if config.exists() and json.loads(config.read_text())["model_alias"] != alias:
            raise ValueError(f"Existing results belong to another model: {directory}")
        gold = directory / "gold_examples.jsonl"
        if gold.exists() and [json.loads(line) for line in gold.read_text().splitlines() if line.strip()] != expected:
            raise ValueError(f"Existing results use a different reference dataset: {directory}")
        command = [sys.executable, "-u", str(repo / "debug/eval_image_uniqueness.py"),
                   "--model", alias, "--workers", str(args.workers), "--retries", "1",
                   "--rerun-failures", "--discussion", str(discussion), "--output-dir", str(directory)]
        print(f"[start] {alias}: {directory}", flush=True)
        with (directory / "run.log").open("a") as log:
            process = subprocess.run(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
        print(f"[done] {alias}: exit={process.returncode}", flush=True)
        return name, process.returncode

    with ThreadPoolExecutor(max_workers=len(models)) as pool:
        futures = [pool.submit(run_model, name, alias) for name, alias in models.items()]
        outcomes = dict(f.result() for f in as_completed(futures))

    rows, audits = [], {}
    gold_hashes, prompt_hashes = set(), set()
    for name, alias in models.items():
        directory = output / name
        metrics = json.loads((directory / "metrics.json").read_text())
        records = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line.strip()]
        assert len(records) == len({r["image_id"] for r in records}) == metrics["total"] == expected_count
        assert {r["image_id"] for r in records} == set(expected_by_id)
        assert all(r["gold_label"] == expected_by_id[r["image_id"]]["gold_label"] for r in records)
        assert all(r["model_alias"] == alias for r in records)
        assert metrics["correct"] == sum(r["predicted_label"] == r["gold_label"] for r in records)
        gold_hash = hashlib.sha256((directory / "gold_examples.jsonl").read_bytes()).hexdigest()
        prompt_hash = hashlib.sha256((directory / "system_prompt.txt").read_bytes()).hexdigest()
        gold_hashes.add(gold_hash)
        prompt_hashes.add(prompt_hash)
        row = {"model_alias": alias, "served_models": ", ".join(sorted({r["served_model"] or "unknown" for r in records})),
               **{key: metrics[key] for key in ("total", "parsed", "correct", "accuracy_all", "macro_f1", "unique_image_precision", "unique_image_recall", "request_errors")},
               "unique_image_f1": metrics["unique_image_metrics"]["f1"]}
        rows.append(row)
        audits[alias] = {"exit_code": outcomes[name], "directory": str(directory),
                        "gold_sha256": gold_hash, "prompt_sha256": prompt_hash, "metrics": metrics}
    assert len(gold_hashes) == len(prompt_hashes) == 1, "Models used different gold data or prompts"
    with (output / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(rows, ensure_ascii=False, indent=2), flush=True)
    return int(any(outcomes.values()) or any(row["request_errors"] or row["parsed"] != expected_count for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
