import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Show a few evaluated samples from a details jsonl file.")
    parser.add_argument("path", help="Path to llm_eval_report_details.jsonl")
    args = parser.parse_args()

    path = Path(args.path)
    count = 0
    for line in path.open("r", encoding="utf-8"):
        item = json.loads(line)
        if item.get("acc") == 0:
            print("=" * 80)
            print("case_id:", item.get("case_id"))
            print("question:", item.get("question"))
            print("correct_answer:", item.get("correct_answer"))
            print("model_answer:", item.get("model_answer"))
            print("reasoning:", item.get("reasoning"))
            count += 1
            if count >= 5:
                break


if __name__ == "__main__":
    main()
