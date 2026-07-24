"""Generate VQA samples from an existing graph with parallel LLM workers."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shlex
import sys

from synthesis.model_worker import LLM_WORKER
from synthesis.store import JsonlGraphStore

from .batch_runner import VqaBatchRunner
from .graph_view import GraphView
from .path_sampler import DEFAULT_HISTORY_EXPOSURE_MODEL, RandomPathSampler, SamplerConfiguration
from .pipeline import VqaGenerationPipeline
from .question_writer import QuestionWriter


def _default_output_dir(graph_dir: Path) -> Path:
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    return (graph_dir / "vqa" / timestamp).resolve()


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
        "--history-exposure-model-alias",
        default=os.environ.get("VQA_HISTORY_EXPOSURE_MODEL") or DEFAULT_HISTORY_EXPOSURE_MODEL,
        help="Model alias for sampler history-exposure filtering. Defaults to VQA_HISTORY_EXPOSURE_MODEL or multimodal_process.",
    )
    parser.add_argument(
        "--compress-hop-model-alias",
        default=None,
        help="Optional model alias for compress_hop. Defaults to VQA_COMPRESS_HOP_MODEL.",
    )
    parser.add_argument(
        "--image-bridge-model-alias",
        default=None,
        help="Optional model alias for hidden image-bridge normalization. Defaults to VQA_IMAGE_BRIDGE_MODEL.",
    )
    parser.add_argument(
        "--image-target-ask-model-alias",
        default=None,
        help="Optional model alias for hidden final-image target-ask normalization. Defaults to VQA_IMAGE_TARGET_ASK_MODEL.",
    )
    parser.add_argument(
        "--neighbor-selection-strategy",
        choices=("random", "llm_guided"),
        default="random",
    )
    parser.add_argument("--llm-candidate-count", type=int, default=6)
    parser.add_argument("--llm-score-temperature", type=float, default=0.35)
    parser.add_argument(
        "--llm-generic-category-score-cap",
        type=float,
        default=0.15,
        help="Hard maximum LLM score for targets classified as generic category/concept nodes.",
    )
    parser.add_argument(
        "--sampler-state",
        type=Path,
        default=None,
        help="Optional sampler state JSON file to import before sampling.",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    argv_to_parse = list(argv) if argv is not None else sys.argv[1:]
    args = build_arg_parser().parse_args(argv_to_parse)
    graph_dir = args.graph_dir.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else _default_output_dir(graph_dir)
    sampler_state_input_path = args.sampler_state.resolve() if args.sampler_state else None
    model_alias = args.model_alias or os.environ.get("VQA_WRITER_MODEL")
    sampler_model_alias = args.sampler_model_alias or os.environ.get("VQA_SAMPLER_MODEL")
    history_exposure_model_alias = args.history_exposure_model_alias
    compress_hop_model_alias = args.compress_hop_model_alias or os.environ.get("VQA_COMPRESS_HOP_MODEL")
    image_bridge_model_alias = args.image_bridge_model_alias or os.environ.get("VQA_IMAGE_BRIDGE_MODEL")
    image_target_ask_model_alias = args.image_target_ask_model_alias or os.environ.get("VQA_IMAGE_TARGET_ASK_MODEL")
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
        llm_generic_category_score_cap=args.llm_generic_category_score_cap,
    )
    store = JsonlGraphStore(graph_dir)
    graph = GraphView(store, allowed_edge_types=set(config.allowed_edge_types))
    sampler = RandomPathSampler(
        graph=graph,
        config=config,
        model_client=LLM_WORKER if sampler_model_alias and args.neighbor_selection_strategy == "llm_guided" else None,
        model=sampler_model_alias,
        history_exposure_model_client=LLM_WORKER,
        history_exposure_model=history_exposure_model_alias,
    )
    writer = QuestionWriter(
        model_client=LLM_WORKER if model_alias else None,
        model=model_alias,
        compress_hop_model_client=LLM_WORKER if compress_hop_model_alias else None,
        compress_hop_model=compress_hop_model_alias,
        image_bridge_model_client=LLM_WORKER if image_bridge_model_alias else None,
        image_bridge_model=image_bridge_model_alias,
        image_target_ask_model_client=LLM_WORKER if image_target_ask_model_alias else None,
        image_target_ask_model=image_target_ask_model_alias,
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
        sampler_state_input_path=sampler_state_input_path,
        question_metadata={
            "entrypoint": "synthesis.vqa.run_batch",
            "invocation": {
                "argv": argv_to_parse,
                "replay_command": shlex.join(
                    [sys.executable, "-m", "synthesis.vqa.run_batch", *argv_to_parse]
                ),
                "cwd": str(Path.cwd().resolve()),
            },
            "paths": {
                "graph_dir": str(graph_dir),
                "output_dir": str(output_dir),
            },
            "sampling_parameters": config.to_dict(),
            "batch_parameters": {
                "samples": args.samples,
                "workers": args.workers,
                "max_inflight": args.max_inflight,
                "resume": not args.no_resume,
            },
            "models": {
                "writer_model_alias": model_alias,
                "sampler_model_alias": sampler_model_alias,
                "history_exposure_model_alias": history_exposure_model_alias,
                "compress_hop_model_alias": compress_hop_model_alias,
                "image_bridge_model_alias": image_bridge_model_alias,
                "image_target_ask_model_alias": image_target_ask_model_alias,
            },
            "sampler_state_request": {
                "input_path": str(sampler_state_input_path) if sampler_state_input_path else None,
            },
        },
    )
    summary = runner.run(limit=args.samples)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    print(f"samples: {runner.samples_path}")
    print(f"questions: {runner.questions_path}")
    print(f"question_metadata: {runner.question_metadata_path}")
    print(f"errors: {runner.errors_path}")
    print(f"warnings: {runner.warnings_path}")
    print(f"summary: {runner.summary_path}")
    print(f"sampler_state: {runner.sampler_state_path}")
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
