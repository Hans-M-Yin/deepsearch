#!/usr/bin/env python3
"""Locate a Ray worker's resolved final-model output directory.

Run this from the same SFT environment / Ray head node that launched training.
It is read-only: it schedules a tiny probe on a chosen node and lists the
relative output directory as that worker actually sees it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


DEFAULT_NODE_IP = "10.124.139.223"
DEFAULT_OUTPUT_DIR = (
    "saves/qwen3_vl_8b/full/"
    "data_refined_v2_rewritten_15k_2node16gpu_lr_2e_5_slow_cosine_epoch3"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print where a Ray worker resolves a relative training output_dir."
    )
    parser.add_argument("--node-ip", default=DEFAULT_NODE_IP)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ray.init(address="auto")

    node = next(
        (
            item
            for item in ray.nodes()
            if item.get("Alive") and item.get("NodeManagerAddress") == args.node_ip
        ),
        None,
    )
    if node is None:
        alive = [
            item.get("NodeManagerAddress")
            for item in ray.nodes()
            if item.get("Alive")
        ]
        raise SystemExit(
            f"No alive Ray node with IP {args.node_ip}. Alive nodes: {alive}"
        )

    output_dir = args.output_dir

    @ray.remote(num_cpus=0)
    def probe() -> dict[str, object]:
        cwd = Path.cwd()
        resolved = (cwd / output_dir).resolve()
        return {
            "node_ip": args.node_ip,
            "worker_cwd": str(cwd),
            "configured_output_dir": output_dir,
            "resolved_output_dir": str(resolved),
            "exists": resolved.exists(),
            "files": sorted(path.name for path in resolved.iterdir())
            if resolved.is_dir()
            else [],
        }

    result = ray.get(
        probe.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node["NodeID"], soft=False
            )
        ).remote()
    )
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
