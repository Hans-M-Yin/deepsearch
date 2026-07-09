import argparse
import json
import random
import re
from pathlib import Path


def _short(text, limit=2000):
    text = str(text or "").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _extract_block(text: str, tag: str) -> str:
    match = re.search(rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.DOTALL | re.IGNORECASE)
    if not match:
        return ""
    return match.group(1).strip()


def _load_trajectory(traj_dir: Path, case_id: str) -> dict | None:
    path = traj_dir / f"{case_id}_trajectory.json"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _print_trajectory(trajectory: dict, *, show_raw_response: bool) -> None:
    turns = trajectory.get("turns") or []
    if not turns:
        print("trajectory: <empty>")
        return

    print("trajectory:")
    for turn in turns:
        turn_id = turn.get("turn")
        print(f"  [turn {turn_id}]")
        if turn.get("error"):
            print(f"    error: {turn.get('error')}")
            continue

        response_text = str(turn.get("response_text") or "")
        thinking = _extract_block(response_text, "thinking") or _extract_block(response_text, "think")
        tool_call = _extract_block(response_text, "tool_call")
        answer = _extract_block(response_text, "answer")
        tool_output = str(turn.get("tool_output") or "").strip()

        if thinking:
            print("    thinking:")
            print(_indent_block(_short(thinking)))
        if tool_call:
            print("    tool_call:")
            print(_indent_block(_short(tool_call)))
        if tool_output:
            print("    tool_output:")
            print(_indent_block(_short(tool_output)))
        if answer:
            print("    answer:")
            print(_indent_block(_short(answer)))
        if show_raw_response:
            print("    raw_response_text:")
            print(_indent_block(_short(response_text, limit=6000)))


def _indent_block(text: str, prefix: str = "      ") -> str:
    lines = str(text or "").splitlines() or [""]
    return "\n".join(f"{prefix}{line}" for line in lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Show evaluated samples and optional agentic trajectories.")
    parser.add_argument("path", help="Path to llm_eval_report_details.jsonl")
    parser.add_argument("--acc", type=int, choices=[0, 1], default=0, help="Which acc label to display.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling matched items.")
    parser.add_argument("--num", type=int, default=5, help="How many matched items to show.")
    parser.add_argument(
        "--traj-dir",
        default="",
        help="Optional trajectory directory containing *_trajectory.json files. Defaults to the details file parent.",
    )
    parser.add_argument(
        "--hide-trajectory",
        action="store_true",
        help="Only show final judged fields, without printing trajectory turns.",
    )
    parser.add_argument(
        "--show-raw-response",
        action="store_true",
        help="Also print each turn's raw response_text in addition to parsed thinking/tool_call/answer blocks.",
    )
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

    traj_dir = Path(args.traj_dir) if args.traj_dir else path.parent

    for item in matched:
        print("=" * 80)
        print("case_id:", item.get("case_id"))
        print("question:", item.get("question"))
        print("correct_answer:", item.get("correct_answer"))
        print("model_answer:", item.get("model_answer"))
        print("reasoning:", item.get("reasoning"))

        if args.hide_trajectory:
            continue

        trajectory = _load_trajectory(traj_dir, str(item.get("case_id") or ""))
        if trajectory is None:
            print(f"trajectory: <missing> searched in {traj_dir}")
            continue
        _print_trajectory(trajectory, show_raw_response=args.show_raw_response)


if __name__ == "__main__":
    main()
