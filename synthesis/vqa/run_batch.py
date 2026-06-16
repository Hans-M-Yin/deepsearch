"""Generate VQA samples from an existing graph with parallel LLM workers."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from synthesis.model_worker import LLM_WORKER
from synthesis.store import JsonlGraphStore

from .batch_runner import VqaBatchRunner
from .graph_view import GraphView
from .path_sampler import RandomPathSampler, SamplerConfiguration
from .pipeline import VqaGenerationPipeline
from .question_writer import QuestionWriter


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-inflight", type=int, default=None)
    parser.add_argument("--min-hops", type=int, default=3)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-penalty-alpha", type=float, default=1.0)
    parser.add_argument(
        "--hop-sampling-strategy",
        choices=("uniform", "middle_biased"),
        default="middle_biased",
    )
    parser.add_argument(
        "--model-alias",
        default=None,
        help="Model alias registered in synthesis/models.json. Defaults to VQA_WRITER_MODEL.",
    )
    parser.add_argument(
        "--sampler-model-alias",
        default=None,
        help="Optional model alias for LLM-guided next-hop selection. Defaults to VQA_SAMPLER_MODEL.",
    )
    parser.add_argument(
        "--neighbor-selection-strategy",
        choices=("random", "llm_guided"),
        default="random",
    )
    parser.add_argument("--llm-candidate-count", type=int, default=6)
    parser.add_argument("--llm-score-temperature", type=float, default=0.35)
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    graph_dir = args.graph_dir.resolve()
    output_dir = (args.output_dir or graph_dir / "vqa").resolve()
    model_alias = args.model_alias or os.environ.get("VQA_WRITER_MODEL")
    sampler_model_alias = args.sampler_model_alias or os.environ.get("VQA_SAMPLER_MODEL")
    config = SamplerConfiguration(
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        max_samples=args.samples,
        random_seed=args.seed,
        edge_penalty_alpha=args.edge_penalty_alpha,
        hop_sampling_strategy=args.hop_sampling_strategy,
        neighbor_selection_strategy=args.neighbor_selection_strategy,
        llm_candidate_count=args.llm_candidate_count,
        llm_score_temperature=args.llm_score_temperature,
    )
    store = JsonlGraphStore(graph_dir)
    graph = GraphView(store, allowed_edge_types=set(config.allowed_edge_types))
    sampler = RandomPathSampler(
        graph=graph,
        config=config,
        model_client=LLM_WORKER if sampler_model_alias and args.neighbor_selection_strategy == "llm_guided" else None,
        model=sampler_model_alias,
    )
    writer = QuestionWriter(
        model_client=LLM_WORKER if model_alias else None,
        model=model_alias,
    )
    pipeline = VqaGenerationPipeline(
        store=store,
        config=config,
        sampler=sampler,
        writer=writer,
    )
    runner = VqaBatchRunner(
        pipeline=pipeline,
        output_dir=output_dir,
        workers=args.workers,
        resume=not args.no_resume,
        max_inflight=args.max_inflight,
    )
    summary = runner.run(limit=args.samples)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    print(f"samples: {runner.samples_path}")
    print(f"questions: {runner.questions_path}")
    print(f"errors: {runner.errors_path}")
    print(f"warnings: {runner.warnings_path}")
    print(f"summary: {runner.summary_path}")
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
