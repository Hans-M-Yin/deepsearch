"""Merge Unique/Semi-unique predictions and audit a multi-model evaluation."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path


POSITIVE = {"唯一性", "半唯一性"}
NEGATIVE = "不唯一性"


def class_metrics(tp, fp, fn):
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "support": tp + fn,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True)
    args = parser.parse_args()
    output = args.results_dir
    source = json.loads((output / "comparison.json").read_text())
    rows, details, cases = [], {}, {}
    reference = None
    for alias, info in source.items():
        records = [json.loads(line) for line in (Path(info["directory"]) / "results.jsonl").read_text().splitlines() if line.strip()]
        by_id = {r["image_id"]: r for r in records}
        gold = {r["image_id"]: r["gold_label"] for r in records}
        assert len(records) == len(by_id) == info["metrics"]["total"]
        assert all(not r["error"] and r["predicted_label"] in POSITIVE | {NEGATIVE} for r in records), "Rerun failed or unparsed samples first"
        if reference is None:
            reference = gold
        assert gold == reference, "Models used different samples or gold labels"
        cm = Counter((r["gold_label"], r["predicted_label"]) for r in records)
        # Reconcile against the independently generated three-class report.
        for actual, values in info["metrics"]["confusion_matrix"].items():
            for prediction, count in values.items():
                assert cm[actual, prediction] == count
        tp = sum(cm[a, b] for a in POSITIVE for b in POSITIVE)
        fp = sum(cm[NEGATIVE, b] for b in POSITIVE)
        fn = sum(cm[a, NEGATIVE] for a in POSITIVE)
        tn = cm[NEGATIVE, NEGATIVE]
        pos, neg = class_metrics(tp, fp, fn), class_metrics(tn, fn, fp)
        total = tp + fp + fn + tn
        row = {"model_alias": alias, "total": total, "correct": tp + tn,
               "accuracy": (tp + tn) / total,
               **{key: pos[key] for key in ("precision", "recall", "f1")},
               **{f"non_unique_{key}": neg[key] for key in ("precision", "recall", "f1")},
               **{f"macro_{key}": (pos[key] + neg[key]) / 2 for key in ("precision", "recall", "f1")},
               "TP": tp, "FP": fp, "FN": fn, "TN": tn}
        rows.append(row)
        details[alias] = {**row, "per_class": {"unique_or_semi_unique": pos, "non_unique": neg},
                          "source_directory": info["directory"]}
        for r in records:
            case = cases.setdefault(r["image_id"], {key: r[key] for key in ("image_id", "description", "gold_label", "expert_reason")})
            case[alias] = r["predicted_label"]
    with (output / "comparison_binary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "comparison_binary.json").write_text(json.dumps({"positive_labels": sorted(POSITIVE), "negative_label": NEGATIVE, "zero_division": 0, "models": details}, ensure_ascii=False, indent=2) + "\n")
    with (output / "case_comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fields = ["image_id", "description", "gold_label", "expert_reason"] + list(source) + ["binary_correct_model_count"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case in cases.values():
            case["binary_correct_model_count"] = sum((case[alias] in POSITIVE) == (case["gold_label"] in POSITIVE) for alias in source)
            writer.writerow(case)
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
