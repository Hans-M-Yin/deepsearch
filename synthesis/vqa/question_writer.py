"""LLM-backed question writer for graph trajectories.

The writer now works directly from ``PathCandidate + GraphView`` instead of a
separate evidence-builder stage. Internally it follows a three-step process:

1. compress each hop into a short statement
2. select an askable target from the final node
3. compose the final multi-hop question
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
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
from .path_sampler import RandomPathSampler, SamplerConfiguration
from .schemas import PathCandidate, QuestionDraft


PROMPT_COMPRESS_HOP = """You are compressing one hop from a multimodal reasoning trajectory.

You are given:
- a source node
- the edge relation connecting to the next node
- a destination node

Write one short declarative statement that captures the essential meaning of
this hop for downstream question composition.

Important semantics by hop type:
- text -> text:
  treat this as a normal entity-to-entity relation
- text -> image:
  treat the target as a key photo / visual scene, not as an image file
  the edge relation is the retrieval query for finding that image
  preserve the query's distinctive details
  do not generalize it into a vague scene description
  you may lightly rewrite it into natural language, but you must keep the key
  entity, event, action, and distinguishing scene details
- image -> text:
  treat the source as a visual clue inside the image, and treat the target as
  the entity identified by that clue

The statement should:
- preserve the relation needed for downstream reasoning
- be concise
- avoid unnecessary details
- avoid asking a question
- produce source/target that are semantically useful for composing a final question

Return valid JSON with exactly these fields:
{
  "statement": "...",
  "source": "...",
  "target": "...",
  "relation": "...",
  "retrieval_query": "..."  // required for text -> image, otherwise empty string
}
"""


PROMPT_SELECT_TARGET = """You are selecting a good final ask from the target node of a multi-hop search problem.

Choose one answerable target from the final node. Prefer a question whose answer is clearly supported by reliable evidence associated with the node, but is not merely a piece of broad common knowledge. When possible, avoid asking for the node name itself or for overly obvious attributes. Instead, select a specific, verifiable detail that requires consulting the relevant evidence while remaining unambiguous and well-supported.

Return valid JSON with exactly these fields:
{
  "answer_type": "entity|attribute|image_content|ocr|other",
  "ask_target": "what the final question should ask about",
  "answer": "the gold answer",
  "support": "short explanation of why this is a good final ask"
}
"""


PROMPT_COMPOSE_QUESTION = """You are writing a multi-hop SEARCH QUESTION from compressed trajectory evidence.

You are given:
- a directed list of hop facts
- the selected final ask target and answer
- the question opening mode

The hop facts are LATENT reasoning support. They are not meant to be narrated
step by step to the user.

Some text -> image hops may include a retrieval_query field. When present,
that retrieval query is a high-precision visual anchor for the image and is
often more specific than the natural-language statement. Preserve its key
distinguishing details, but do not copy it verbatim in a search-engine style.

Your job is to write one natural search or deep-research question that:
- sounds like a realistic user request
- hides most of the intermediate reasoning chain
- preserves the true hop direction internally
- keeps only the clues necessary for the question to remain coherent and solvable
- does not directly reveal the answer
- is easy to read as a SINGLE clear question
- has a clearly understandable final ask

Do NOT do the following:
- do not narrate the chain step by step
- do not explicitly say phrases like:
  - "starting with"
  - "then"
  - "after that"
  - "following that clue"
  - "using that clue"
  - "first ... then ..."
- do not expose every intermediate entity if the question can stay solvable without it
- do not turn the output into a summary of the reasoning path

Important direction rule:
- each hop has an explicit source and target
- keep the semantic direction of each hop
- do not reverse a hop when forming the hidden sub-questions
- for example, if a hop is:
  source = Kobe Bryant
  relation = played_for
  target = Los Angeles Lakers
  then a direction-preserving hidden step is:
  "Kobe Bryant played for which team?"
  and a direction-reversing wrong version is:
  "Who played for the Los Angeles Lakers?"

Question-writing goal:
- produce one final user-facing question, not a narrated derivation
- the user should need to search or reason through the hops, but should not see
  the full chain spelled out
- intermediate facts may be paraphrased, partially hidden, or omitted
- preserve the first-hop-to-last-hop logic as latent support

Clarity rules:
- write one main question, not a loose pile of clues
- the reader should be able to understand what is being asked after one careful read
- keep referents clear: avoid too many vague "this", "that", or "the one"
- prefer layered constraints that narrow the target step by step
- end with a very explicit final ask, such as:
  - "which player ... ?"
  - "what is the name of ... ?"
  - "which team ... ?"
  - "how many ... ?"
- do not let the final ask get buried inside a long sentence

Good style example:
- "In the championship celebration image of a certain NBA team, one player later had an award named after him after winning it four times. Other than him, which player also won that award four times?"

Bad style example:
- "From the image ..., then ..., and following that clue ..., what ... ?"
- a long sentence where the reader cannot tell what the final question is asking

Opening mode rules:
- if opening_mode is "text_start", begin naturally from the first hop's source
  entity/event/concept
- if opening_mode is "image_start", assume the starting image is already given
  to the user as part of the input; begin from a visual clue in that image and
  do not rewrite the opening as if the first source were already known text

Return valid JSON with exactly these fields:
{
  "question": "...",
  "answer": "...",
  "answer_type": "..."
}
"""


@dataclass(slots=True)
class HopContext:
    """Compact readable representation of one trajectory hop."""

    hop_index: int
    src_node_id: str
    dst_node_id: str
    src_modality: str
    dst_modality: str
    edge_id: str
    edge_type: str
    relation: str
    src_content: dict[str, Any] = field(default_factory=dict)
    dst_content: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "hop_index": self.hop_index,
            "src_node_id": self.src_node_id,
            "dst_node_id": self.dst_node_id,
            "src_modality": self.src_modality,
            "dst_modality": self.dst_modality,
            "edge_id": self.edge_id,
            "edge_type": self.edge_type,
            "relation": self.relation,
            "src_content": dict(self.src_content),
            "dst_content": dict(self.dst_content),
        }


@dataclass(slots=True)
class WriterContext:
    """Structured writer input derived directly from a trajectory."""

    path_id: str
    trajectory: dict[str, Any]
    hops: list[HopContext]
    target_node: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path_id": self.path_id,
            "trajectory": dict(self.trajectory),
            "hops": [hop.to_dict() for hop in self.hops],
            "target_node": dict(self.target_node),
        }


@dataclass(slots=True)
class QuestionWriter:
    """Question writer that reads graph content directly from a trajectory."""

    model_client: ModelWorkerClient | None = None
    model: str | None = None
    temperature: float = 0.2
    max_tokens: int = 800

    def build_writer_context(self, *, path: PathCandidate, graph: GraphView) -> WriterContext:
        hops: list[HopContext] = []
        for hop_index, edge_id in enumerate(path.edge_ids):
            src_node_id = path.node_ids[hop_index]
            dst_node_id = path.node_ids[hop_index + 1]
            src_node = graph.get_node(src_node_id) or {}
            dst_node = graph.get_node(dst_node_id) or {}
            hops.append(
                HopContext(
                    hop_index=hop_index,
                    src_node_id=src_node_id,
                    dst_node_id=dst_node_id,
                    src_modality=self._node_modality(src_node),
                    dst_modality=self._node_modality(dst_node),
                    edge_id=edge_id,
                    edge_type=path.edge_types[hop_index] if hop_index < len(path.edge_types) else "",
                    relation=path.relations[hop_index] if hop_index < len(path.relations) else "",
                    src_content=self._node_payload(
                        src_node,
                        full=hop_index == 0,
                    ),
                    dst_content=self._node_payload(
                        dst_node,
                        full=hop_index + 1 == len(path.node_ids) - 1,
                    ),
                )
            )
        target_node = graph.get_node(path.target_node_id) or {}
        return WriterContext(
            path_id=path.path_id,
            trajectory=path.trajectory.to_dict(),
            hops=hops,
            target_node=self._node_payload(target_node, full=True),
        )

    def compress_hop(self, *, hop: HopContext) -> dict[str, Any]:
        if self.model_client is None:
            return self._fallback_compress_hop(hop)
        prompt = {
            "hop_type": f"{hop.src_modality}->{hop.dst_modality}",
            "source_node": hop.src_content,
            "edge": {
                "edge_type": hop.edge_type,
                "relation": hop.relation,
            },
            "destination_node": hop.dst_content,
        }
        parsed = self._generate_json(
            system=PROMPT_COMPRESS_HOP,
            user_payload=prompt,
            trace_label=f"compress_hop_{hop.hop_index}",
        )
        statement = str(parsed.get("statement") or "").strip()
        source = str(parsed.get("source") or "").strip()
        target = str(parsed.get("target") or "").strip()
        relation = str(parsed.get("relation") or hop.relation or hop.edge_type or "").strip()
        retrieval_query = str(parsed.get("retrieval_query") or "").strip()
        if hop.src_modality == "text" and hop.dst_modality == "image" and not retrieval_query:
            retrieval_query = str(hop.relation or "").strip()
        if not statement or not source or not target:
            return self._fallback_compress_hop(hop)
        return {
            "hop_index": hop.hop_index,
            "statement": statement,
            "source": source,
            "target": target,
            "relation": relation,
            "retrieval_query": retrieval_query,
            "edge_id": hop.edge_id,
            "src_node_id": hop.src_node_id,
            "dst_node_id": hop.dst_node_id,
        }

    def select_target_ask(self, *, context: WriterContext) -> dict[str, Any]:
        if self.model_client is None:
            return self._fallback_select_target(context.target_node)
        parsed = self._generate_json(
            system=PROMPT_SELECT_TARGET,
            user_payload={"target_node": context.target_node},
            trace_label="select_target_ask",
        )
        ask_target = str(parsed.get("ask_target") or "").strip()
        answer = str(parsed.get("answer") or "").strip()
        answer_type = str(parsed.get("answer_type") or "other").strip()
        support = str(parsed.get("support") or "").strip()
        if not ask_target or not answer:
            return self._fallback_select_target(context.target_node)
        return {
            "answer_type": answer_type,
            "ask_target": ask_target,
            "answer": answer,
            "support": support,
        }

    def compose_question(
        self,
        *,
        path: PathCandidate,
        graph: GraphView,
        context: WriterContext | None = None,
    ) -> QuestionDraft:
        context = context or self.build_writer_context(path=path, graph=graph)
        hop_summaries = [self.compress_hop(hop=hop) for hop in context.hops]
        target_ask = self.select_target_ask(context=context)
        opening_mode = "image_start" if path.trajectory.starts_with_image else "text_start"
        if self.model_client is None:
            return self._fallback_compose_question(
                path=path,
                hop_summaries=hop_summaries,
                target_ask=target_ask,
                opening_mode=opening_mode,
            )
        parsed = self._generate_json(
            system=PROMPT_COMPOSE_QUESTION,
            user_payload={
                "opening_mode": opening_mode,
                "hop_facts": [
                    {
                        "hop_index": item.get("hop_index"),
                        "source": item.get("source"),
                        "relation": item.get("relation"),
                        "target": item.get("target"),
                        "statement": item.get("statement"),
                        "retrieval_query": item.get("retrieval_query"),
                    }
                    for item in hop_summaries
                ],
                "target_ask": target_ask,
            },
            trace_label="compose_question",
        )
        question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        answer = str(parsed.get("answer") or target_ask.get("answer") or "").strip()
        answer_type = str(parsed.get("answer_type") or target_ask.get("answer_type") or "other").strip()
        if (
            not question
            or not answer
            or self._looks_like_chain_narration(question)
        ):
            rewritten = self._rewrite_chain_narration(
                opening_mode=opening_mode,
                hop_summaries=hop_summaries,
                target_ask=target_ask,
            )
            if rewritten is not None:
                question = rewritten
        if not question or not answer:
            return self._fallback_compose_question(
                path=path,
                hop_summaries=hop_summaries,
                target_ask=target_ask,
                opening_mode=opening_mode,
            )
        return QuestionDraft(
            question=question,
            answer=answer,
            answer_type=answer_type,
            reasoning_steps=hop_summaries,
            used_evidence_ids=[hop.edge_id for hop in context.hops],
            metadata={
                "path_id": path.path_id,
                "target_ask": target_ask,
                "writer_context": context.to_dict(),
            },
        )

    def draft(self, *, path: PathCandidate, graph: GraphView) -> QuestionDraft:
        return self.compose_question(path=path, graph=graph)

    def polish(self, *, draft: QuestionDraft, path: PathCandidate, graph: GraphView) -> QuestionDraft:
        del path, graph
        return draft

    def _generate_json(self, *, system: str, user_payload: dict[str, Any], trace_label: str) -> dict[str, Any]:
        if self.model_client is None:
            raise RuntimeError("model_client is required for _generate_json")
        request = ModelRequest(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=json.dumps(user_payload, ensure_ascii=False, indent=2)),
            ],
            metadata={"trace_label": trace_label},
        )
        response = self.model_client.generate(request)
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            parsed = self._extract_json_object(response.content)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object for {trace_label}, got: {type(parsed)!r}")
        return parsed

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any]:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise ValueError(f"Model response does not contain JSON object: {text[:500]}")
        parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not an object.")
        return parsed

    @staticmethod
    def _node_modality(node: dict[str, Any]) -> str:
        return "image" if node.get("node_type") == "image" else "text"

    def _node_payload(self, node: dict[str, Any], *, full: bool) -> dict[str, Any]:
        node_type = node.get("node_type")
        payload: dict[str, Any] = {
            "node_id": node.get("node_id"),
            "node_type": node_type,
            "title": node.get("title"),
        }
        if node_type == "image":
            metadata = node.get("metadata") or {}
            payload["caption"] = node.get("caption") or node.get("summary")
            payload["visual_facts"] = list((metadata.get("visual_facts") or [])[: (999 if full else 2)]) if isinstance(metadata, dict) else []
            payload["ocr_texts"] = list((metadata.get("ocr_texts") or [])[: (999 if full else 2)]) if isinstance(metadata, dict) else []
            payload["grounded_entities"] = list((metadata.get("grounded_entities") or [])[: (999 if full else 3)]) if isinstance(metadata, dict) else []
            if full:
                payload["source_page_url"] = node.get("source_page_url")
        else:
            payload["summary"] = node.get("summary")
            description = node.get("description")
            payload["description"] = description if full else self._shorten_text(description, limit=240)
            aliases = node.get("aliases") or []
            payload["aliases"] = list(aliases[: (999 if full else 3)])
            if full:
                payload["attributes"] = dict(node.get("attributes") or {})
                payload["canonical_id"] = node.get("canonical_id")
        return payload

    @staticmethod
    def _shorten_text(text: Any, *, limit: int) -> str | None:
        if not text:
            return None
        normalized = re.sub(r"\s+", " ", str(text)).strip()
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."

    @staticmethod
    def _fallback_compress_hop(hop: HopContext) -> dict[str, Any]:
        relation = hop.relation or hop.edge_type or "is connected to"
        retrieval_query = ""
        if hop.src_modality == "text" and hop.dst_modality == "text":
            src_label = hop.src_content.get("title") or hop.src_node_id
            dst_label = hop.dst_content.get("title") or hop.dst_node_id
            statement = f"{src_label} {relation} {dst_label}".strip()
        elif hop.src_modality == "text" and hop.dst_modality == "image":
            src_label = hop.src_content.get("title") or hop.src_node_id
            retrieval_query = str(hop.relation or "").strip()
            dst_label = (
                hop.dst_content.get("title")
                or hop.dst_content.get("caption")
                or retrieval_query
                or "a key visual scene"
            )
            if retrieval_query:
                statement = (
                    f"{src_label} is associated with a key photo or visual scene described by the query: "
                    f"{retrieval_query}."
                )
            else:
                statement = f"{src_label} is associated with a key photo or visual scene: {dst_label}."
        elif hop.src_modality == "image" and hop.dst_modality == "text":
            src_label = QuestionWriter._image_clue_label(hop)
            dst_label = hop.dst_content.get("title") or hop.dst_node_id
            statement = f"{src_label} refers to {dst_label}."
        else:
            src_label = hop.src_content.get("title") or hop.src_content.get("caption") or hop.src_node_id
            dst_label = hop.dst_content.get("title") or hop.dst_content.get("caption") or hop.dst_node_id
            statement = f"{src_label} {relation} {dst_label}".strip()
        return {
            "hop_index": hop.hop_index,
            "statement": statement,
            "source": src_label,
            "target": dst_label,
            "relation": relation,
            "retrieval_query": retrieval_query,
            "edge_id": hop.edge_id,
            "src_node_id": hop.src_node_id,
            "dst_node_id": hop.dst_node_id,
        }

    @staticmethod
    def _image_clue_label(hop: HopContext) -> str:
        relation = (hop.relation or "").strip()
        if relation:
            lowered = relation.lower()
            if any(token in lowered for token in ("left", "right", "logo", "brand", "person", "player", "wearing")):
                return relation
        visual_facts = hop.src_content.get("visual_facts") or []
        if visual_facts:
            return str(visual_facts[0])
        caption = hop.src_content.get("caption")
        if caption:
            return f"a clue in the image: {caption}"
        return "a visual clue in the image"

    def _fallback_select_target(self, target_node: dict[str, Any]) -> dict[str, Any]:
        node_type = target_node.get("node_type")
        if node_type == "image":
            visual_facts = target_node.get("visual_facts") or []
            answer = ""
            if visual_facts:
                answer = str(visual_facts[0])
            elif target_node.get("caption"):
                answer = str(target_node["caption"])
            return {
                "answer_type": "image_content",
                "ask_target": "the key visual content in the final image",
                "answer": answer or "unknown visual content",
                "support": "Selected from the final image caption/visual facts.",
            }

        attributes = target_node.get("attributes") or {}
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                if isinstance(value, (str, int, float)) and str(value).strip():
                    return {
                        "answer_type": "attribute",
                        "ask_target": f"the {key} of the final entity",
                        "answer": str(value),
                        "support": f"Selected attribute {key}.",
                    }
        description = target_node.get("description") or target_node.get("summary")
        if description:
            return {
                "answer_type": "attribute",
                "ask_target": "a key detail described for the final entity",
                "answer": self._shorten_text(description, limit=120) or "",
                "support": "Fell back to description/summary.",
            }
        return {
            "answer_type": "entity",
            "ask_target": "the identity of the final entity",
            "answer": str(target_node.get("title") or "unknown"),
            "support": "Fell back to target title.",
        }

    @staticmethod
    def _fallback_compose_question(
        *,
        path: PathCandidate,
        hop_summaries: list[dict[str, Any]],
        target_ask: dict[str, Any],
        opening_mode: str,
    ) -> QuestionDraft:
        hop_text = " Then ".join(item.get("statement", "") for item in hop_summaries if item.get("statement"))
        ask_target = str(target_ask.get("ask_target") or "the final answer")
        answer = str(target_ask.get("answer") or "unknown")
        answer_type = str(target_ask.get("answer_type") or "other")
        if opening_mode == "image_start":
            question = (
                f"In the given image, follow the relevant visual and factual clues needed to identify {ask_target}. "
                f"{hop_text} What is it?"
            )
        else:
            first_source = str(hop_summaries[0].get("source") or "this subject") if hop_summaries else "this subject"
            question = (
                f"Using {first_source} as the starting clue, search through the relevant facts needed to determine "
                f"{ask_target}. What is it?"
            )
        question = QuestionWriter._clean_composed_question(question)
        return QuestionDraft(
            question=question,
            answer=answer,
            answer_type=answer_type,
            reasoning_steps=hop_summaries,
            used_evidence_ids=[item.get("edge_id", "") for item in hop_summaries if item.get("edge_id")],
            metadata={"target_ask": target_ask},
        )

    @staticmethod
    def _looks_like_chain_narration(question: str) -> bool:
        lowered = question.lower()
        bad_patterns = (
            "starting with",
            "follow this chain",
            "following that clue",
            "using that clue",
            "then looking at",
            "then finding",
            "then following",
            "first find",
            "first identify",
        )
        return any(pattern in lowered for pattern in bad_patterns)

    def _rewrite_chain_narration(
        self,
        *,
        opening_mode: str,
        hop_summaries: list[dict[str, Any]],
        target_ask: dict[str, Any],
    ) -> str | None:
        if self.model_client is None:
            return None
        parsed = self._generate_json(
            system=(
                "You are rewriting a bad multi-hop question draft.\n\n"
                "The previous draft explicitly narrated the reasoning chain or was too hard to read.\n"
                "Rewrite it into ONE natural search question.\n"
                "Do not narrate the chain. Do not use phrases like 'starting with', "
                "'then', 'following that clue', or 'using that clue'.\n"
                "Keep the latent reasoning structure, but hide the step-by-step derivation.\n"
                "Make the question easy to read, with a clear final ask.\n"
                "Prefer one main question with layered constraints, not a loose string of facts.\n\n"
                "Return valid JSON with exactly these fields:\n"
                "{\n"
                '  "question": "..."\n'
                "}\n"
            ),
            user_payload={
                "opening_mode": opening_mode,
                "hop_facts": [
                    {
                        "source": item.get("source"),
                        "relation": item.get("relation"),
                        "target": item.get("target"),
                        "statement": item.get("statement"),
                        "retrieval_query": item.get("retrieval_query"),
                    }
                    for item in hop_summaries
                ],
                "target_ask": target_ask,
            },
            trace_label="rewrite_chain_narration",
        )
        question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        if not question or self._looks_like_chain_narration(question):
            return None
        return question

    @staticmethod
    def _clean_composed_question(question: str) -> str:
        question = re.sub(r"\s+", " ", question).strip()
        for prefix in (
            "Starting with ",
            "Starting from ",
            "Follow this chain of clues: ",
            "Using the following clues, ",
        ):
            if question.startswith(prefix):
                question = question[len(prefix):].strip()
        return question


def _debug_main() -> None:
    parser = argparse.ArgumentParser(description="Debug question writing for one sampled trajectory.")
    parser.add_argument(
        "--graph-dir",
        type=Path,
        default=Path("runs/kobe_text_only"),
        help="Directory containing nodes.jsonl/edges.jsonl graph tables.",
    )
    parser.add_argument("--min-hops", type=int, default=2)
    parser.add_argument("--max-hops", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--edge-penalty-alpha", type=float, default=1.0)
    parser.add_argument(
        "--model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json. If omitted, fallback logic is used.",
    )
    parser.add_argument(
        "--hop-sampling-strategy",
        choices=("uniform", "middle_biased"),
        default="middle_biased",
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
            hop_sampling_strategy=args.hop_sampling_strategy,
            max_samples=1,
        ),
    )
    path = sampler.generate_one()
    print(f"graph_dir: {args.graph_dir}")
    print(f"store_stats: {json.dumps(store.stats(), ensure_ascii=False)}")
    print(f"sampler_stats: {json.dumps(sampler.last_generation_stats.to_dict() if sampler.last_generation_stats else {}, ensure_ascii=False)}")
    if path is None:
        print("path: null")
        return

    writer = QuestionWriter(
        model_client=LLM_WORKER if args.model_alias else None,
        model=args.model_alias,
    )
    context = writer.build_writer_context(path=path, graph=graph)
    hop_summaries = [writer.compress_hop(hop=hop) for hop in context.hops]
    draft = writer.compose_question(path=path, graph=graph, context=context)

    print("path:")
    print(json.dumps(path.to_dict(), ensure_ascii=False, indent=2))
    print("hop_summaries:")
    print(json.dumps(hop_summaries, ensure_ascii=False, indent=2))
    print("question:")
    print(json.dumps(
        {
            "question": draft.question,
            "answer": draft.answer,
            "answer_type": draft.answer_type,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    _debug_main()
