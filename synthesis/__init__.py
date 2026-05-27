"""Data synthesis graph objects and utilities."""

from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .nodes import (
    AssetRef,
    ImageNode,
    Node,
    NodeSource,
    NodeStatus,
    NodeType,
    RegionNode,
    TextNode,
)
from .edges import (
    Edge,
    EdgeSource,
    EdgeStatus,
    EdgeType,
    EvidenceRef,
    allowed_edge_types,
)
from .evidence import (
    Asset,
    AssetType,
    Evidence,
    EvidenceType,
    RecordStatus,
    SearchEngine,
    SearchSnapshot,
)
from .store import JsonlGraphStore
from .model_worker import (
    LLM_WORKER,
    ModelMessage,
    ModelRouterWorkerClient,
    OpenAIModelWorkerClient,
    ModelRequest,
    ModelResponse,
    ModelWorkerClient,
)
from .search_client import (
    CommonsImageSearchClient,
    FallbackImageSearchClient,
    ImageSearchResult,
    MockSearchClient,
    OpenSerpSearchClient,
    SearchClient,
    SearchResponse,
    SerperAdapterSearchClient,
    SerpApiSearchClient,
    TextSearchResult,
)
from .image_discovery import (
    DiscoveredImage,
    ImageCandidateStatus,
    ImageDiscoveryBuilder,
    ImageDiscoveryConfig,
    ImageDiscoveryResult,
    ImageValidationResult,
    PROMPT_IMAGE_CHECK,
    PROMPT_IMAGE_GROUND,
)
from .wiki_text_builder import (
    EnhancedReaderClient,
    PROMPT_EXTRACT_ATTRIBUTE,
    ReaderClient,
    ReaderDocument,
    WikiLinkCandidate,
    WikiTextBuilder,
    WikiTextBuildResult,
)
from .graph_expansion import (
    ExpansionTask,
    ExpansionTaskStatus,
    GraphExpansionConfig,
    GraphExpansionStrategy,
    NodeExpansionResult,
)
from .graph_runner import (
    GraphRunner,
    GraphRunnerConfig,
    GraphRunnerResult,
    GraphRunnerState,
)
from .visual_planner import (
    DownstreamUse,
    LLMVisualSearchPlanner,
    PROMPT_VISUAL_SEARCH_PLANNER,
    SearchQuerySpec,
    VisualSearchPlan,
    VisualSearchPlanner,
    VisualTargetType,
)

__all__ = [
    "AssetRef",
    "ImageNode",
    "Node",
    "NodeSource",
    "NodeStatus",
    "NodeType",
    "RegionNode",
    "TextNode",
    "Edge",
    "EdgeSource",
    "EdgeStatus",
    "EdgeType",
    "EvidenceRef",
    "allowed_edge_types",
    "Asset",
    "AssetType",
    "Evidence",
    "EvidenceType",
    "RecordStatus",
    "SearchEngine",
    "SearchSnapshot",
    "JsonlGraphStore",
    "LLM_WORKER",
    "ModelMessage",
    "ModelRouterWorkerClient",
    "OpenAIModelWorkerClient",
    "ModelRequest",
    "ModelResponse",
    "ModelWorkerClient",
    "CommonsImageSearchClient",
    "FallbackImageSearchClient",
    "ImageSearchResult",
    "MockSearchClient",
    "OpenSerpSearchClient",
    "SearchClient",
    "SearchResponse",
    "SerperAdapterSearchClient",
    "SerpApiSearchClient",
    "TextSearchResult",
    "DiscoveredImage",
    "ImageCandidateStatus",
    "ImageDiscoveryBuilder",
    "ImageDiscoveryConfig",
    "ImageDiscoveryResult",
    "ImageValidationResult",
    "PROMPT_IMAGE_CHECK",
    "PROMPT_IMAGE_GROUND",
    "EnhancedReaderClient",
    "PROMPT_EXTRACT_ATTRIBUTE",
    "ReaderClient",
    "ReaderDocument",
    "WikiLinkCandidate",
    "WikiTextBuilder",
    "WikiTextBuildResult",
    "ExpansionTask",
    "ExpansionTaskStatus",
    "GraphExpansionConfig",
    "GraphExpansionStrategy",
    "NodeExpansionResult",
    "GraphRunner",
    "GraphRunnerConfig",
    "GraphRunnerResult",
    "GraphRunnerState",
    "DownstreamUse",
    "LLMVisualSearchPlanner",
    "PROMPT_VISUAL_SEARCH_PLANNER",
    "SearchQuerySpec",
    "VisualSearchPlan",
    "VisualSearchPlanner",
    "VisualTargetType",
]


def _smoke_test() -> None:
    assert TextNode.from_webpage("https://example.com").to_dict()["node_type"] == "text"
    assert EdgeType.DERIVED.value == "derived"
    assert "GraphRunner" in __all__
    print("__init__ smoke test passed")


if __name__ == "__main__":
    _smoke_test()
