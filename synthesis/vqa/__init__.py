"""VQA/data-generation pipeline built on top of the synthesis graph.

This package turns an existing multimodal graph into question candidates via:

1. graph sampling
2. evidence construction
3. obfuscation / anti-shortcut processing
4. question drafting / polishing
5. verification and filtering

The first implementation pass intentionally focuses on clear module boundaries
and stable schemas so that we can iterate on sampling and verification policy
without rewriting the whole pipeline.
"""

from .schemas import (
    EvidenceBundle,
    EvidenceItem,
    PathCandidate,
    QuestionDraft,
    SampleProgress,
    SampleStatus,
    TrajectoryStats,
    VerificationCheck,
    VerificationResult,
    VqaGenerationConfig,
    VqaSample,
)
from .graph_view import GraphView
from .path_sampler import PathSampler, RandomPathSampler, SamplerConfiguration, SamplerGenerationStats
from .evidence_builder import EvidenceBuilder
from .question_writer import QuestionWriter
from .verifier import SampleVerifier
from .repository_verifier import (
    OfflineGraphRepositoryVerifier,
    RepositoryAssembler,
    RepositoryBundle,
    RepositoryItem,
    RepositoryVerificationConfig,
)
from .pipeline import VqaGenerationError, VqaGenerationPipeline
from .batch_runner import VqaBatchRunner, VqaBatchSummary

__all__ = [
    "EvidenceBundle",
    "EvidenceItem",
    "PathCandidate",
    "QuestionDraft",
    "SampleProgress",
    "SampleStatus",
    "TrajectoryStats",
    "VerificationCheck",
    "VerificationResult",
    "VqaGenerationConfig",
    "VqaSample",
    "GraphView",
    "PathSampler",
    "RandomPathSampler",
    "SamplerConfiguration",
    "SamplerGenerationStats",
    "EvidenceBuilder",
    "QuestionWriter",
    "SampleVerifier",
    "RepositoryVerificationConfig",
    "RepositoryItem",
    "RepositoryBundle",
    "RepositoryAssembler",
    "OfflineGraphRepositoryVerifier",
    "VqaGenerationError",
    "VqaGenerationPipeline",
    "VqaBatchRunner",
    "VqaBatchSummary",
]
