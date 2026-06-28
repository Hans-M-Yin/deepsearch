"""Path sampling interfaces and first-pass random sampler."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
from hashlib import sha256
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    __package__ = "synthesis.vqa"

from synthesis.model_worker import ModelMessage, ModelRequest, ModelWorkerClient
from synthesis.model_worker import LLM_WORKER
from synthesis.store import JsonlGraphStore

from .graph_view import GraphView
from .schemas import PathCandidate, TrajectoryStats


PROMPT_LLM_NEXT_HOP_SELECTION = """You are reviewing candidate next hops for graph trajectory sampling in a multi-hop question-generation pipeline.

Your job is NOT to choose the most generally related next node.
Your job is to judge which candidate next hop is most promising for extending the CURRENT trajectory into a high-quality multi-hop question chain.

A good next hop should help produce a future question that is:
- genuinely multi-hop: later targets should depend on earlier ones
- coherent: the chain should stay on a clear topic rather than drift
- specific: the relation should constrain the next target instead of being broad or generic
- askable: the extended chain should still plausibly lead to a clear, non-trivial final question
- resistant to shortcuts: the next hop should not make later answers too obvious without following the chain

Important evaluation principles:
1. Evaluate each candidate as an extension of the existing trajectory, not in isolation.
2. Prefer candidates whose target is meaningfully constrained by the current source.
3. Prefer candidates that preserve room for 1-2 additional useful hops later.
4. Penalize candidates that look like broad encyclopedia links, weak topic drift, or dead-end facts.
5. If an image hop is involved, prefer it only when the image is likely to provide necessary evidence rather than decorative context.
6. Avoid near-duplicate entities that are too close to entities already present in the trajectory. For example, if the trajectory already contains "iPhone 4S", then "iPhone 5" is usually too similar and should be penalized unless the relation creates a genuinely necessary contrast.

Common bad candidates:
- generic links that could connect to many entities
- candidates whose target can be guessed without knowing the current source
- candidates that reveal a likely future answer too directly
- candidates that lead to thin targets with little downstream askability
- candidates that make the chain read like a loose summary instead of a reasoning path
- candidates whose target is just a near-duplicate, sibling variant, adjacent model/version, or minimally changed entity relative to something already in the trajectory

You will receive:
- a trajectory summary in hop format
- the current node
- a list of candidate next hops

Return valid JSON with exactly this shape:
{
  "ranked_candidates": [
    {
      "edge_id": "candidate edge id",
      "score": 0.0,
      "specificity": 0.0,
      "dependency": 0.0,
      "coherence": 0.0,
      "future_potential": 0.0,
      "askability": 0.0,
      "shortcut_risk": 0.0,
      "reason": "short explanation grounded in the current trajectory"
    }
  ]
}

Rank higher candidates first. Use higher score for better candidates. Use higher shortcut_risk when a candidate is more dangerous.
Give priority to less well-known edges and their corresponding neighbor nodes that involve more niche knowledge.
"""


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
    hop_sampling_strategy: str = "middle_biased"
    require_simple_path: bool = True
    min_modality_switches: int = 0
    dedup_by_exact_signature: bool = True
    allowed_start_node_types: tuple[str, ...] = ()
    allowed_end_node_types: tuple[str, ...] = ()
    edge_penalty_alpha: float = 1.0
    image_spacing_enabled: bool = True
    neighbor_selection_strategy: str = "random"
    llm_candidate_count: int = 6
    llm_score_temperature: float = 0.35
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
        if self.hop_sampling_strategy not in {"uniform", "middle_biased"}:
            raise ValueError("hop_sampling_strategy must be 'uniform' or 'middle_biased'")
        if self.min_modality_switches < 0:
            raise ValueError("min_modality_switches must be >= 0")
        if self.edge_penalty_alpha < 0:
            raise ValueError("edge_penalty_alpha must be >= 0")
        if self.neighbor_selection_strategy not in {"random", "llm_guided"}:
            raise ValueError("neighbor_selection_strategy must be 'random' or 'llm_guided'")
        if self.llm_candidate_count <= 0:
            raise ValueError("llm_candidate_count must be positive")
        if self.llm_score_temperature <= 0:
            raise ValueError("llm_score_temperature must be positive")

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
    accepted_start_node_id: str | None = None
    recent_start_node_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PathSampler:
    """Abstract-ish path sampler interface."""

    graph: GraphView
    config: SamplerConfiguration
    last_generation_stats: SamplerGenerationStats | None = None

    def generate_one(self, start_node_id: str | None = None) -> PathCandidate | None:
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
    model_client: ModelWorkerClient | None = None
    model: str | None = None
    llm_temperature: float = 0.0
    llm_max_tokens: int = 800
    _rng: random.Random = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.random_seed)

    def generate_one(self, start_node_id: str | None = None) -> PathCandidate | None:
        node_ids = self._candidate_start_nodes()
        # ##### DEBUG #####
        node_ids = [
            node_id for node_id in node_ids
            if (self.graph.get_node(node_id) or {}).get("node_type") == "image"
        ]
        # ##### END #####
        if not node_ids:
            self.last_generation_stats = SamplerGenerationStats(requested=1, attempts=0, accepted=0)
            return None

        forced_start_node_id = start_node_id
        if forced_start_node_id is not None and forced_start_node_id not in node_ids:
            self.last_generation_stats = SamplerGenerationStats(
                requested=1,
                attempts=0,
                accepted=0,
                rejected_start_type=1,
                recent_start_node_ids=[forced_start_node_id],
            )
            return None

        stats = SamplerGenerationStats(requested=1)
        max_attempts = max(self.config.max_attempts_multiplier, self.config.min_attempts)
        attempts = 0
        target_hop_count = self._sample_hop_count(self._rng)
        while attempts < max_attempts:
            attempts += 1
            stats.attempts = attempts
            sampled_start_node_id = forced_start_node_id or self._rng.choice(node_ids)
            stats.recent_start_node_ids.append(sampled_start_node_id)
            stats.recent_start_node_ids = stats.recent_start_node_ids[-10:]
            candidate, reject_reason = self._sample_one(
                start_node_id=sampled_start_node_id,
                rng=self._rng,
                hop_count=target_hop_count,
            )
            if candidate is None:
                self._count_rejection(stats, reject_reason)
                continue
            if self.config.dedup_by_exact_signature and candidate.exact_signature in self.used_exact_signatures:
                stats.rejected_duplicate_exact += 1
                continue
            self.used_exact_signatures.add(candidate.exact_signature)
            self._register_edge_usage(candidate.edge_ids)
            stats.accepted = 1
            stats.accepted_start_node_id = sampled_start_node_id
            candidate.metadata["sampled_hop_count"] = target_hop_count
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

    def _sample_one(
        self,
        *,
        start_node_id: str,
        rng: random.Random,
        hop_count: int | None = None,
    ) -> tuple[PathCandidate | None, str | None]:
        node_ids = [start_node_id]
        edge_ids: list[str] = []
        edge_types: list[str] = []
        relations: list[str] = []
        selection_trace: list[dict[str, Any]] = []
        used_edge_ids: set[str] = set()
        current = start_node_id
        hop_count = hop_count if hop_count is not None else self._sample_hop_count(rng)

        for _ in range(hop_count):
            neighbors = self._traversable_neighbors(current, node_ids=node_ids)
            if self.config.require_simple_path:
                neighbors = [edge for edge in neighbors if edge.get("dst_node_id") not in node_ids]
            neighbors = [edge for edge in neighbors if edge.get("edge_id") not in used_edge_ids]
            if not neighbors:
                if len(edge_ids) < self.config.min_hops:
                    return None, "too_short"
                return None, "dead_end"
            edge = self._weighted_edge_choice(
                neighbors,
                node_ids=node_ids,
                rng=rng,
                selection_trace=selection_trace,
            )
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
                "sampling_policy": self.config.neighbor_selection_strategy,
                "sampled_hop_count": hop_count,
                "sampler_config": self.config.to_dict(),
                "selection_trace": selection_trace,
            },
        )
        return candidate, None

    def _traversable_neighbors(self, node_id: str, *, node_ids: list[str]) -> list[dict[str, Any]]:
        node_type = self.graph.node_type(node_id)
        neighbors = self.graph.neighbors(node_id)
        if node_type == "text":
            return [edge for edge in neighbors if edge.get("edge_type") != "image_depicts"]
        if node_type == "image" and len(node_ids) >= 2 and self.graph.node_type(node_ids[-2]) == "text":
            return [
                edge
                for edge in neighbors
                if not (
                    edge.get("edge_type") == "image_depicts"
                    and bool((edge.get("metadata") or {}).get("query_overlap_entity"))
                )
            ]
        return neighbors

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

    def _weighted_edge_choice(
        self,
        neighbors: list[dict[str, Any]],
        *,
        node_ids: list[str],
        rng: random.Random,
        selection_trace: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if len(neighbors) == 1:
            return neighbors[0]
        if self._use_llm_guidance():
            llm_choice = self._llm_guided_edge_choice(
                neighbors,
                node_ids=node_ids,
                rng=rng,
                selection_trace=selection_trace,
            )
            if llm_choice is not None:
                return llm_choice
        if self.config.edge_penalty_alpha == 0:
            return rng.choice(neighbors)
        weights = [self._candidate_weight(edge, node_ids=node_ids) for edge in neighbors]
        return rng.choices(neighbors, weights=weights, k=1)[0]

    def _use_llm_guidance(self) -> bool:
        return self.config.neighbor_selection_strategy == "llm_guided" and self.model_client is not None

    def _llm_guided_edge_choice(
        self,
        neighbors: list[dict[str, Any]],
        *,
        node_ids: list[str],
        rng: random.Random,
        selection_trace: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        candidate_edges = self._select_llm_candidates(neighbors, node_ids=node_ids)
        if len(candidate_edges) <= 1:
            return candidate_edges[0] if candidate_edges else None

        payload = self._llm_next_hop_payload(node_ids=node_ids, candidates=candidate_edges)
        try:
            parsed = self._generate_llm_json(
                system=PROMPT_LLM_NEXT_HOP_SELECTION,
                user_payload=payload,
                trace_label="sampler_next_hop",
            )
        except Exception as exc:
            if selection_trace is not None:
                selection_trace.append(
                    {
                        "mode": "llm_guided",
                        "status": "fallback_random",
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                        "candidate_edge_ids": [str(edge.get("edge_id") or "") for edge in candidate_edges],
                    }
                )
            return None

        ranked_records = parsed.get("ranked_candidates")
        if not isinstance(ranked_records, list):
            return None
        candidate_by_id = {
            str(edge.get("edge_id") or ""): edge
            for edge in candidate_edges
            if str(edge.get("edge_id") or "")
        }
        ranked_weights: list[tuple[dict[str, Any], float, dict[str, Any]]] = []
        trace_candidates: list[dict[str, Any]] = []
        for item in ranked_records:
            if not isinstance(item, dict):
                continue
            edge_id = str(item.get("edge_id") or "").strip()
            if not edge_id or edge_id not in candidate_by_id:
                continue
            score = self._coerce_score(item.get("score"), default=0.0)
            if score <= 0:
                score = 0.01
            weight = self._score_to_weight(score)
            ranked_weights.append((candidate_by_id[edge_id], weight, item))
            trace_candidates.append(
                {
                    "edge_id": edge_id,
                    "score": score,
                    "reason": str(item.get("reason") or "").strip(),
                }
            )
        if not ranked_weights:
            return None
        chosen_edge, _, chosen_item = rng.choices(
            [entry[0] for entry in ranked_weights],
            weights=[entry[1] for entry in ranked_weights],
            k=1,
        )[0], None, None
        chosen_item = next(
            (entry[2] for entry in ranked_weights if entry[0].get("edge_id") == chosen_edge.get("edge_id")),
            None,
        )
        if selection_trace is not None:
            selection_trace.append(
                {
                    "mode": "llm_guided",
                    "status": "selected",
                    "current_node_id": node_ids[-1] if node_ids else None,
                    "candidate_edge_ids": [str(edge.get("edge_id") or "") for edge in candidate_edges],
                    "ranked_candidates": trace_candidates,
                    "selected_edge_id": chosen_edge.get("edge_id"),
                    "selected_reason": str((chosen_item or {}).get("reason") or "").strip(),
                }
            )
        return chosen_edge

    def _select_llm_candidates(
        self,
        neighbors: list[dict[str, Any]],
        *,
        node_ids: list[str],
    ) -> list[dict[str, Any]]:
        ranked = sorted(
            neighbors,
            key=lambda edge: self._candidate_weight(edge, node_ids=node_ids),
            reverse=True,
        )
        return ranked[: self.config.llm_candidate_count]

    def _llm_next_hop_payload(
        self,
        *,
        node_ids: list[str],
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        current_node_id = node_ids[-1]
        current_node = self.graph.get_node(current_node_id) or {}
        history_hops = self._history_hop_strings(node_ids)
        return {
            "trajectory_goal": "choose the next hop that best supports future multi-hop question generation",
            "trajectory_history": history_hops,
            "current_node": self._node_summary(current_node, include_details=True),
            "candidate_next_hops": [
                {
                    "edge_id": edge.get("edge_id"),
                    "hop": self._format_candidate_hop(edge),
                    "edge_type": edge.get("edge_type"),
                    "relation": edge.get("relation") or edge.get("edge_type") or "",
                    "base_weight": round(self._candidate_weight(edge, node_ids=node_ids), 4),
                    "target_node_type": self.graph.node_type(str(edge.get("dst_node_id") or "")) or "unknown",
                    "target_node": self._node_summary(
                        self.graph.get_node(str(edge.get("dst_node_id") or "")) or {},
                        include_details=False,
                    ),
                }
                for edge in candidates
            ],
        }

    def _generate_llm_json(
        self,
        *,
        system: str,
        user_payload: dict[str, Any],
        trace_label: str,
    ) -> dict[str, Any]:
        if self.model_client is None:
            raise RuntimeError("model_client is required for LLM-guided sampling")
        response = self.model_client.generate(
            ModelRequest(
                model=self.model,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    ModelMessage(role="system", content=system),
                    ModelMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False, indent=2)),
                ],
                metadata={"trace_label": trace_label},
            )
        )
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            parsed = self._extract_json_object(response.content)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object from {trace_label}, got {type(parsed)!r}")
        return parsed

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model response does not contain a JSON object: {text[:500]}")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not an object.")
        return parsed

    @staticmethod
    def _coerce_score(value: Any, *, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _score_to_weight(self, score: float) -> float:
        return max(0.001, score) ** (1.0 / self.config.llm_score_temperature)

    def _node_summary(self, node: dict[str, Any], *, include_details: bool) -> dict[str, Any]:
        node_type = node.get("node_type") or "unknown"
        summary = {
            "node_id": node.get("node_id"),
            "node_type": node_type,
            "label": self._node_label(node),
        }
        if node_type == "image":
            metadata = node.get("metadata") or {}
            summary["caption"] = self._short_text(node.get("caption") or node.get("summary"), limit=180)
            if isinstance(metadata, dict):
                summary["search_query"] = self._short_text(metadata.get("search_query"), limit=160)
                summary["visual_facts"] = [
                    self._short_text(item, limit=120)
                    for item in (metadata.get("visual_facts") or [])[: (4 if include_details else 2)]
                    if self._short_text(item, limit=120)
                ]
        else:
            summary["summary"] = self._short_text(node.get("summary"), limit=220)
            if include_details:
                summary["description"] = self._short_text(node.get("description"), limit=260)
                attributes = node.get("attributes") or {}
                if isinstance(attributes, dict):
                    summary["attributes"] = {
                        str(key): self._short_text(value.get("value") if isinstance(value, dict) else value, limit=80)
                        for key, value in list(attributes.items())[:5]
                        if self._short_text(value.get("value") if isinstance(value, dict) else value, limit=80)
                    }
        return summary

    def _history_hop_strings(self, node_ids: list[str]) -> list[str]:
        if len(node_ids) <= 1:
            node = self.graph.get_node(node_ids[0]) or {} if node_ids else {}
            return [f"1. {self._node_label(node)}"] if node_ids else []
        history: list[str] = []
        for index in range(len(node_ids) - 1):
            src_node = self.graph.get_node(node_ids[index]) or {}
            dst_node = self.graph.get_node(node_ids[index + 1]) or {}
            edge = self.graph.get_edge_id_between(node_ids[index], node_ids[index + 1]) or {}
            relation = str(edge.get("relation") or edge.get("edge_type") or "related to").strip()
            history.append(
                f"{index + 1}. {self._node_label(src_node)} -- {relation} --> {self._node_label(dst_node)}"
            )
        return history

    def _format_candidate_hop(self, edge: dict[str, Any]) -> str:
        src_node = self.graph.get_node(str(edge.get("src_node_id") or "")) or {}
        dst_node = self.graph.get_node(str(edge.get("dst_node_id") or "")) or {}
        relation = str(edge.get("relation") or edge.get("edge_type") or "related to").strip()
        return f"{self._node_label(src_node)} -- {relation} --> {self._node_label(dst_node)}"

    @staticmethod
    def _short_text(value: Any, *, limit: int) -> str | None:
        if value is None:
            return None
        text = re.sub(r"\s+", " ", str(value)).strip()
        if not text:
            return None
        if len(text) <= limit:
            return text
        return text[: limit - 3].rstrip() + "..."

    @staticmethod
    def _node_label(node: dict[str, Any]) -> str:
        if not node:
            return "unknown"
        title = str(node.get("title") or "").strip()
        if title:
            return title
        caption = str(node.get("caption") or node.get("summary") or "").strip()
        if caption:
            return caption[:80]
        return str(node.get("node_id") or "unknown")

    def _sample_hop_count(self, rng: random.Random) -> int:
        if self.config.min_hops == self.config.max_hops:
            return self.config.min_hops
        if self.config.hop_sampling_strategy == "uniform":
            return rng.randint(self.config.min_hops, self.config.max_hops)

        mid = (self.config.min_hops + self.config.max_hops) / 2.0
        sampled = rng.triangular(self.config.min_hops, self.config.max_hops, mid)
        hop_count = int(round(sampled))
        return max(self.config.min_hops, min(self.config.max_hops, hop_count))

    def _edge_weight(self, edge_id: str | None) -> float:
        if not edge_id:
            return 1.0
        count = self.edge_usage_counts.get(edge_id, 0)
        return 1.0 / (1.0 + self.config.edge_penalty_alpha * count)

    def _candidate_weight(self, edge: dict[str, Any], *, node_ids: list[str]) -> float:
        weight = self._edge_weight(edge.get("edge_id"))
        if not self.config.image_spacing_enabled:
            return weight
        next_node_id = edge.get("dst_node_id")
        if not isinstance(next_node_id, str):
            return weight
        if self.graph.node_type(next_node_id) != "image":
            return weight
        return weight * self._image_spacing_factor(node_ids)

    def _image_spacing_factor(self, node_ids: list[str]) -> float:
        last_image_index: int | None = None
        for index in range(len(node_ids) - 1, -1, -1):
            if self.graph.node_type(node_ids[index]) == "image":
                last_image_index = index
                break
        if last_image_index is None:
            return 1.0
        distance = len(node_ids) - 1 - last_image_index
        if distance <= 0:
            return 1.0
        return distance / (distance + 1.0)

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


def _debug_main() -> None:
    parser = argparse.ArgumentParser(description="Debug one sampled trajectory from an existing graph store.")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("runs/kobe_text_only"),
        help="Directory containing nodes.jsonl/edges.jsonl graph tables.",
    )
    parser.add_argument("--min-hops", type=int, default=3)
    parser.add_argument("--max-hops", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--start-node-id",
        default=None,
        help="Optional fixed start node id for debugging one specific trajectory root.",
    )
    parser.add_argument("--edge-penalty-alpha", type=float, default=1.0)
    parser.add_argument(
        "--disable-image-spacing",
        action="store_true",
        help="Disable the extra image-spacing penalty when choosing image neighbors.",
    )
    parser.add_argument("--min-modality-switches", type=int, default=0)
    parser.add_argument(
        "--hop-sampling-strategy",
        choices=("uniform", "middle_biased"),
        default="middle_biased",
    )
    parser.add_argument(
        "--neighbor-selection-strategy",
        choices=("random", "llm_guided"),
        default="random",
    )
    parser.add_argument("--llm-candidate-count", type=int, default=6)
    parser.add_argument("--llm-score-temperature", type=float, default=0.35)
    parser.add_argument(
        "--sampler-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for next-hop ranking.",
    )
    args = parser.parse_args()

    store = JsonlGraphStore(args.graph_dir)
    graph = GraphView(store, allowed_edge_types=set(SamplerConfiguration().allowed_edge_types))
    sampler = RandomPathSampler(
        graph=graph,
        config=SamplerConfiguration(
            min_hops=args.min_hops,
            max_hops=args.max_hops,
            random_seed=args.seed,
            edge_penalty_alpha=args.edge_penalty_alpha,
            image_spacing_enabled=not args.disable_image_spacing,
            min_modality_switches=args.min_modality_switches,
            hop_sampling_strategy=args.hop_sampling_strategy,
            neighbor_selection_strategy=args.neighbor_selection_strategy,
            llm_candidate_count=args.llm_candidate_count,
            llm_score_temperature=args.llm_score_temperature,
            max_samples=1,
        ),
        model_client=LLM_WORKER if args.sampler_model_alias and args.neighbor_selection_strategy == "llm_guided" else None,
        model=args.sampler_model_alias,
    )
    candidate = sampler.generate_one(start_node_id=args.start_node_id)
    print(f"graph_dir: {args.graph_dir}")
    print(f"store_stats: {json.dumps(store.stats(), ensure_ascii=False)}")
    print(f"sampler_stats: {json.dumps(sampler.last_generation_stats.to_dict() if sampler.last_generation_stats else {}, ensure_ascii=False)}")
    if candidate is None:
        print("trajectory: null")
        return
    print("trajectory:")
    print(json.dumps(candidate.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _debug_main()
