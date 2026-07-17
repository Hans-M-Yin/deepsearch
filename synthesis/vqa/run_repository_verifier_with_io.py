"""Run the repository verifier in verbose input/output mode.

This is the explicit merged runner for the two repository-verifier branches:
- repository-grounded solve
- question-only shortcut solve

It delegates to ``synthesis.vqa.debug.debug_repository_verifier`` with
``--run-verification`` enabled, so it prints the model requests and raw outputs
while executing verification.

Example:
    python -m synthesis.vqa.run_repository_verifier_with_io       --vqa-dir /path/to/vqa_dir       --graph-dir /path/to/graph_dir       --answer-model-alias <answer_model>       --judge-model-alias <judge_model>
"""

from __future__ import annotations

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "synthesis.vqa"

from synthesis.vqa.debug.debug_repository_verifier import main as _debug_main


def main(argv: list[str] | None = None) -> int:
    forwarded = list(argv if argv is not None else sys.argv[1:])
    if "--run-verification" not in forwarded:
        forwarded.append("--run-verification")
    return _debug_main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())
