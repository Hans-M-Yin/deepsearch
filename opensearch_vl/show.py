import argparse
import json
import random
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a few evaluated samples from a details jsonl file.")
    parser.add_argument("path", help="Path to llm_eval_report_details.jsonl")
    parser.add_argument("--acc", type=int, choices=[0, 1], default=0, help="Which acc label to display.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling matched items.")
    parser.add_argument("--num", type=int, default=5, help="How many matched items to show.")
    args = parser.parse_args()

    path = Path(args.path)
    matched = []
    for line in path.open("r", encoding="utf-8"):
        item = json.loads(line)
        if item.get("acc") == args.acc:
            matched.append(item)

    rng = random.Random(args.seed)
    if len(matched) > args.num:
        matched = rng.sample(matched, args.num)

    for item in matched:
        print("=" * 80)
        print("case_id:", item.get("case_id"))
        print("question:", item.get("question"))
        print("correct_answer:", item.get("correct_answer"))
        print("model_answer:", item.get("model_answer"))
        print("reasoning:", item.get("reasoning"))


if __name__ == "__main__":
    main()
