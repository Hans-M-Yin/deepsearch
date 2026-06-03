"""High-level graph-to-question pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass

from synthesis.store import JsonlGraphStore

from .evidence_builder import EvidenceBuilder
from .graph_view import GraphView
from .obfuscation import ObfuscationProcessor
from .path_sampler import RandomPathSampler, SamplerConfiguration
from .question_writer import QuestionWriter
from .schemas import SampleStatus, VqaSample
from .verifier import SampleVerifier


@dataclass(slots=True)
class VqaGenerationPipeline:
    """Orchestrate sampling, evidence construction, writing, and verification."""

    store: JsonlGraphStore
    config: SamplerConfiguration
    sampler: RandomPathSampler | None = None
    evidence_builder: EvidenceBuilder | None = None
    obfuscator: ObfuscationProcessor | None = None
    writer: QuestionWriter | None = None
    verifier: SampleVerifier | None = None

    def __post_init__(self) -> None:
        graph = GraphView(self.store, allowed_edge_types=set(self.config.allowed_edge_types))
        self.graph = graph
        self.sampler = self.sampler or RandomPathSampler(graph=graph, config=self.config)
        self.evidence_builder = self.evidence_builder or EvidenceBuilder(graph=graph)
        self.obfuscator = self.obfuscator or ObfuscationProcessor()
        self.writer = self.writer or QuestionWriter()
        self.verifier = self.verifier or SampleVerifier()

    def generate(self, *, limit: int | None = None) -> list[VqaSample]:
        sample_limit = self.config.max_samples if limit is None else limit
        samples: list[VqaSample] = []
        for path in self.sampler.generate(limit=sample_limit):
            target_node = self.graph.get_node(path.target_node_id) or {}
            target_title = target_node.get("title")
            evidence = self.evidence_builder.build(path)
            evidence = self.obfuscator.pre_obfuscate(evidence, target_title=target_title)
            draft = self.writer.draft(path=path, evidence=evidence)
            polished = self.writer.polish(draft=draft, path=path, evidence=evidence)
            polished = self.obfuscator.post_obfuscate(polished, target_title=target_title)
            verification = self.verifier.verify(question=polished)
            status = SampleStatus.VERIFIED if verification.final_keep else SampleStatus.REJECTED
            samples.append(
                VqaSample(
                    sample_id=f"sample_{path.path_id}",
                    status=status,
                    path=path,
                    evidence=evidence,
                    draft=draft,
                    polished=polished,
                    verification=verification,
                    metadata={},
                )
            )
        return samples
