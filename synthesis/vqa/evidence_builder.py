"""Build writer/verifier evidence bundles from sampled paths."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256

from .graph_view import GraphView
from .schemas import EvidenceBundle, EvidenceItem, PathCandidate


def _stable_hash(*parts: object, length: int = 16) -> str:
    payload = "||".join("" if part is None else str(part) for part in parts)
    return sha256(payload.encode("utf-8")).hexdigest()[:length]


@dataclass(slots=True)
class EvidenceBuilder:
    """Turn path nodes and edges into structured evidence items."""

    graph: GraphView

    def build(self, path: PathCandidate) -> EvidenceBundle:
        items: list[EvidenceItem] = []
        for index, node_id in enumerate(path.node_ids):
            node = self.graph.get_node(node_id) or {}
            title = node.get("title")
            modality = "image" if node.get("node_type") == "image" else "text"
            raw_content = self._node_content(node)
            relation_hint = path.relations[index - 1] if index > 0 and index - 1 < len(path.relations) else None
            items.append(
                EvidenceItem(
                    evidence_id=f"ev_{_stable_hash(path.path_id, node_id, index)}",
                    source_kind="node",
                    source_node_id=node_id,
                    modality=modality,
                    title=title,
                    raw_content=raw_content,
                    transformed_content=raw_content,
                    relation_hint=relation_hint,
                    metadata={"node_type": node.get("node_type")},
                )
            )

        return EvidenceBundle(
            bundle_id=f"bundle_{_stable_hash(path.path_id)}",
            path_id=path.path_id,
            oracle_evidence=items,
            writer_evidence=list(items),
            verifier_evidence=list(items),
            metadata={"path_id": path.path_id},
        )

    @staticmethod
    def _node_content(node: dict[str, object]) -> str:
        node_type = node.get("node_type")
        if node_type == "image":
            parts = [
                str(node.get("caption") or ""),
            ]
            metadata = node.get("metadata") or {}
            if isinstance(metadata, dict):
                visual_facts = metadata.get("visual_facts") or []
                ocr_texts = metadata.get("ocr_texts") or []
                if visual_facts:
                    parts.append(" ".join(str(item) for item in visual_facts))
                if ocr_texts:
                    parts.append(" ".join(str(item) for item in ocr_texts))
            return " ".join(part for part in parts if part).strip()
        return " ".join(
            part for part in [
                str(node.get("summary") or ""),
                str(node.get("description") or ""),
            ] if part
        ).strip()
