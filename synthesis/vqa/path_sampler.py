"""Path sampling interfaces and first-pass random sampler."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from hashlib import sha256
import random
from typing import Any

from .graph_view import GraphView
from .schemas import PathCandidate, TrajectoryStats


def _stable_hash(*parts: object, length: int = 16) -> str:
    payload = "||".join("" if part is None else str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()[:length]


@dataclass(slots=True)
class SamplerConfiguration:
    """Configuration for first-pass random trajectory sampling."""

    min_hops: int = 3
    max_hops: int = 5
    max_samples: int = 100
    random_seed: int = 0
    max_attempts_multiplier: int = 20
    min_attempts: int = 100
    require_simple_path: bool = True
    min_modality_switches: int = 0
    dedup_by_exact_signature: bool = True
    allowed_start_node_types: tuple[str, ...] = ()
    allowed_end_node_types: tuple[str, ...] = ()
    edge_penalty_alpha: float = 1.0
    allowed_edge_types: tuple[str, ...] = (
        "wiki_link",
        "wiki_attribute",
        "web_link",
        "search_retrieved",
        "image_source_page",
        "image_depicts",
    )

    def __post_init__(self) -> None:
        if self.min_hops <= 0:
            raise ValueError("min_hops must be positive")
        if self.max_hops < self.min_hops:
            raise ValueError("max_hops must be >= min_hops")
        if self.max_samples <= 0:
            raise ValueError("max_samples must be positive")
        if self.max_attempts_multiplier <= 0:
            raise ValueError("max_attempts_multiplier must be positive")
        if self.min_attempts <= 0:
            raise ValueError("min_attempts must be positive")
        if self.min_modality_switches < 0:
            raise ValueError("min_modality_switches must be >= 0")
        if self.edge_penalty_alpha < 0:
            raise ValueError("edge_penalty_alpha must be >= 0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SamplerGenerationStats:
    """Basic run-level stats for one sampler invocation."""

    requested: int
    accepted: int = 0
    attempts: int = 0
    rejected_too_short: int = 0
    rejected_dead_end: int = 0
    rejected_cycle: int = 0
    rejected_duplicate_exact: int = 0
    rejected_modality_switch: int = 0
    rejected_start_type: int = 0
    rejected_end_type: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PathSampler:
    """Abstract-ish path sampler interface."""

    graph: GraphView
    config: SamplerConfiguration
    last_generation_stats: SamplerGenerationStats | None = None

    def generate_one(self) -> PathCandidate | None:
        raise NotImplementedError

    def generate(self, limit: int | None = None) -> list[PathCandidate]:
        raise NotImplementedError


@dataclass(slots=True)
class RandomPathSampler(PathSampler):
    """First-pass random path sampler.

    This class intentionally keeps policy light. We use natural random sampling
    plus minimal path validity constraints, while recording trajectory labels
    for later analysis.
    """

    used_exact_signatures: set[str] = field(default_factory=set)
    edge_usage_counts: dict[str, int] = field(default_factory=dict)

    def generate_one(self) -> PathCandidate | None:
        rng = random.Random(self.config.random_seed)
        node_ids = self._candidate_start_nodes()
        if not node_ids:
            self.last_generation_stats = SamplerGenerationStats(requested=1, attempts=0, accepted=0)
            return None

        stats = SamplerGenerationStats(requested=1)
        max_attempts = max(self.config.max_attempts_multiplier, self.config.min_attempts)
        attempts = 0
        while attempts < max_attempts:
            attempts += 1
            stats.attempts = attempts
            start_node_id = rng.choice(node_ids)
            candidate, reject_reason = self._sample_one(start_node_id=start_node_id, rng=rng)
            if candidate is None:
                self._count_rejection(stats, reject_reason)
                continue
            if self.config.dedup_by_exact_signature and candidate.exact_signature in self.used_exact_signatures:
                stats.rejected_duplicate_exact += 1
                continue
            self.used_exact_signatures.add(candidate.exact_signature)
            self._register_edge_usage(candidate.edge_ids)
            stats.accepted = 1
            self.last_generation_stats = stats
            return candidate
        self.last_generation_stats = stats
        return None

    def generate(self, limit: int | None = None) -> list[PathCandidate]:
        target_count = self.config.max_samples if limit is None else limit
        candidates: list[PathCandidate] = []
        aggregate = SamplerGenerationStats(requested=target_count)
        for _ in range(target_count):
            candidate = self.generate_one()
            one_stats = self.last_generation_stats
            if one_stats is not None:
                aggregate.attempts += one_stats.attempts
                aggregate.accepted += one_stats.accepted
                aggregate.rejected_too_short += one_stats.rejected_too_short
                aggregate.rejected_dead_end += one_stats.rejected_dead_end
                aggregate.rejected_cycle += one_stats.rejected_cycle
                aggregate.rejected_duplicate_exact += one_stats.rejected_duplicate_exact
                aggregate.rejected_modality_switch += one_stats.rejected_modality_switch
                aggregate.rejected_start_type += one_stats.rejected_start_type
                aggregate.rejected_end_type += one_stats.rejected_end_type
            if candidate is None:
                continue
            candidates.append(candidate)
        self.last_generation_stats = aggregate
        return candidates

    def sample_candidates(self, limit: int) -> list[PathCandidate]:
        """Backward-compatible alias for early callers."""
        return self.generate(limit=limit)

    def _sample_one(self, *, start_node_id: str, rng: random.Random) -> tuple[PathCandidate | None, str | None]:
        node_ids = [start_node_id]
        edge_ids: list[str] = []
        edge_types: list[str] = []
        relations: list[str] = []
        used_edge_ids: set[str] = set()
        current = start_node_id
        hop_count = rng.randint(self.config.min_hops, self.config.max_hops)

        for _ in range(hop_count):
            neighbors = self.graph.neighbors(current)
            if self.config.require_simple_path:
                neighbors = [edge for edge in neighbors if edge.get("dst_node_id") not in node_ids]
            neighbors = [edge for edge in neighbors if edge.get("edge_id") not in used_edge_ids]
            if not neighbors:
                if len(edge_ids) < self.config.min_hops:
                    return None, "too_short"
                return None, "dead_end"
            edge = self._weighted_edge_choice(neighbors, rng=rng)
            current = edge["dst_node_id"]
            if self.config.require_simple_path and current in node_ids:
                return None, "cycle"
            node_ids.append(current)
            edge_ids.append(edge["edge_id"])
            edge_types.append(edge.get("edge_type") or "")
            relations.append(edge.get("relation") or "")
            used_edge_ids.add(edge["edge_id"])

        if len(edge_ids) < self.config.min_hops:
            return None, "too_short"
        node_types = [self.graph.node_type(node_id) or "unknown" for node_id in node_ids]
        trajectory = self._trajectory_stats(node_types)
        if trajectory.modality_switch_count < self.config.min_modality_switches:
            return None, "modality_switch"
        if not self._valid_end_type(node_types[-1]):
            return None, "end_type"
        exact_signature = "|".join(node_ids)
        skeleton_signature = "|".join(node_types + edge_types)
        core_signature = "|".join(node_ids[-3:]) if len(node_ids) >= 3 else exact_signature

        candidate = PathCandidate(
            path_id=f"path_{_stable_hash(exact_signature)}",
            node_ids=node_ids,
            edge_ids=edge_ids,
            node_types=node_types,
            edge_types=edge_types,
            relations=relations,
            target_node_id=node_ids[-1],
            start_node_id=node_ids[0],
            trajectory=trajectory,
            exact_signature=exact_signature,
            skeleton_signature=skeleton_signature,
            core_signature=core_signature,
            metadata={
                "sampling_policy": "random",
                "sampler_config": self.config.to_dict(),
            },
        )
        return candidate, None

    def _candidate_start_nodes(self) -> list[str]:
        node_ids = self.graph.list_node_ids()
        if not self.config.allowed_start_node_types:
            return node_ids
        allowed = set(self.config.allowed_start_node_types)
        return [
            node_id
            for node_id in node_ids
            if self.graph.node_type(node_id) in allowed
        ]

    def _valid_end_type(self, node_type: str) -> bool:
        if not self.config.allowed_end_node_types:
            return True
        return node_type in set(self.config.allowed_end_node_types)

    def _weighted_edge_choice(self, neighbors: list[dict[str, Any]], *, rng: random.Random) -> dict[str, Any]:
        if len(neighbors) == 1 or self.config.edge_penalty_alpha == 0:
            return rng.choice(neighbors)
        weights = [self._edge_weight(edge.get("edge_id")) for edge in neighbors]
        return rng.choices(neighbors, weights=weights, k=1)[0]

    def _edge_weight(self, edge_id: str | None) -> float:
        if not edge_id:
            return 1.0
        count = self.edge_usage_counts.get(edge_id, 0)
        return 1.0 / (1.0 + self.config.edge_penalty_alpha * count)

    def _register_edge_usage(self, edge_ids: list[str]) -> None:
        for edge_id in edge_ids:
            self.edge_usage_counts[edge_id] = self.edge_usage_counts.get(edge_id, 0) + 1

    @staticmethod
    def _count_rejection(stats: SamplerGenerationStats, reject_reason: str | None) -> None:
        if reject_reason == "too_short":
            stats.rejected_too_short += 1
        elif reject_reason == "dead_end":
            stats.rejected_dead_end += 1
        elif reject_reason == "cycle":
            stats.rejected_cycle += 1
        elif reject_reason == "modality_switch":
            stats.rejected_modality_switch += 1
        elif reject_reason == "end_type":
            stats.rejected_end_type += 1

    @staticmethod
    def _trajectory_stats(node_types: list[str]) -> TrajectoryStats:
        modality_sequence = ["image" if node_type == "image" else "text" for node_type in node_types]
        image_positions = [idx for idx, modality in enumerate(modality_sequence) if modality == "image"]
        switch_count = sum(
            1 for idx in range(1, len(modality_sequence)) if modality_sequence[idx] != modality_sequence[idx - 1]
        )
        return TrajectoryStats(
            start_modality=modality_sequence[0],
            end_modality=modality_sequence[-1],
            modality_sequence=modality_sequence,
            hop_count=max(0, len(node_types) - 1),
            image_node_count=modality_sequence.count("image"),
            text_node_count=modality_sequence.count("text"),
            modality_switch_count=switch_count,
            starts_with_image=modality_sequence[0] == "image",
            ends_with_image=modality_sequence[-1] == "image",
            image_only_at_start=image_positions == [0],
            image_only_at_end=image_positions == [len(modality_sequence) - 1],
            has_mid_image=any(0 < pos < len(modality_sequence) - 1 for pos in image_positions),
        )
