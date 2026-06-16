"""LLM-backed question writer for graph trajectories.

The writer now works directly from ``PathCandidate + GraphView`` instead of a
separate evidence-builder stage. Internally it follows a five-step process:

1. compress each hop into a short statement
2. derive an opening package for the first source + first hop
3. select an askable target from the final node
4. compose and polish the final multi-hop question
5. obfuscate shortcut clues while preserving the reasoning path
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
- stay faithful to THIS hop only
- not introduce entities that are not the current source node or current destination node

Anchor rules:
- source must refer to the current source node only
- target must refer to the current destination node only
- do not replace source or target with entities from earlier or later hops
- do not turn this into a cross-hop summary

Return valid JSON with exactly these fields:
{
  "statement": "...",
  "source": "...",
  "target": "...",
  "relation": "...",
  "retrieval_query": "..."  // required for text -> image, otherwise empty string
}
"""


PROMPT_SELECT_TEXT_TARGET = """Now you are an expert question designer for knowledge-based Q&A. You will create a knowledge competition question for students, designed to challenge their knowledge of a person’s life and background.

You will design the question based on the complete profile of a given object. From the full material, you need to select one or more detailed, non-common-sense factual pieces of information, then organize the information you extract into a complete question. The information you extract will serve as the corresponding answer. The question may cover any aspect, including the object’s identity, life experience, major events, detailed knowledge, relevant numbers, or dates. However, you must ensure that the question is non-trivial and requires reasoning and knowledge retrieval to answer correctly.

Requirements:

1. Use only the information provided to you about the object.
2. The question must have a clear, unambiguous answer supported by evidence.
3. The factual information you select must not be revealed in the question, and the question must not tell students that any material exists.

Output format: JSON, containing the following fields:

{
  "ask_target": "A complete question about the target node",
  "answer": "A standard answer that fully answers every part of ask_target",
  "supporting_facts": ["The exact original facts needed to answer the question"],
  "reasoning": "A concise derivation from supporting_facts to the answer",
  "support": "A brief explanation of why the question is evidence-supported and unambiguous"
}
"""


PROMPT_SELECT_IMAGE_TARGET = """You are an expert question writer designing a high-quality image-grounded web-search question.

Task setting:
- A solver will be given the same image you get.
- The solver must use the image and answer the question you propose.
- The image itself has already been provided to you, and the textual description has already been decided.
- Your job is to write one strong question about the image, together with its gold answer.

Core objective:
Write a question that can only be answered reliably by finding and inspecting the details of the image. The question must be grounded in the visual content of the image, not merely in background knowledge about the subject.

Important constraints:
1. The question must require looking at the image.
   - Do not ask for facts that can be answered from general knowledge, the image description alone, or a Wikipedia page about the subject.
   - Prefer questions about visible objects, spatial relations, counts, relative positions, gestures, clothing, text appearing in the image, composition, or other stable visual properties.

2. Handle image ambiguity carefully.
   - Some image descriptions may retrieve multiple different images referring to the same entity, scene, object, or landmark.
   - If the description is not highly unique, ask only about visual properties that are stable across the plausible retrieved images.
   - Example: for aerial views of a famous building, different retrieved images may differ in angle, orientation, rendering style, or lighting, but core architectural features may remain stable. In such cases, ask about those shared stable features.
   - Avoid questions whose answer may change across plausible search results.

3. Use wider freedom only when the image is highly specific.
   - If the image description points very strongly to one exact image or to one nearly unique photo, artwork, cover, poster, or historically specific shot, you may ask a more specific question about fine-grained visual details.
   - This is appropriate for cases such as a famous artwork, an album cover, a historically iconic photograph, or a well-known image from a specific event and moment.

4. The question must be high quality.
   - It should be clear, natural, and unambiguous.
   - It should have one answer fully supported by the image.
   - It should not be trivial, but it also should not require subjective interpretation.
   - Prefer concrete visual reasoning over vague aesthetic judgment.

5. Use only the provided image evidence.
   - Base the question and answer only on the supplied image material.
   - Do not invent details that are not visibly supported.

Return valid JSON with exactly these fields:
{
  "ask_target": "one complete question about the target image",
  "answer": "the gold answer",
  "supporting_facts": ["the exact visual facts from the image that support the answer"],
  "reasoning": "a concise explanation of how the answer follows from the image",
  "support": "why this question is image-dependent and why its answer is stable across plausible search results"
}
"""


PROMPT_SELECT_OPENING_PACKAGE = """Next, the user will provide you with a declarative sentence describing the relationship between different objects; specifically, it describes the relationship between the source entity and the target entity. The user will then provide you with the Wikipedia page of the source entity. Please extract information about the source entity from the Wikipedia page and organize it into a phrase or short descriptive sentence to replace the source entity in the statement, so that the relationship between the source entity and the corresponding target entity can still be inferred from the revised sentence without causing ambiguity.

Rules:
You may extract any type of relevant information from the Wikipedia page, as long as it ensures that the final description can be used to identify the correct source entity through web search.
The description must not be overly simple, overly generic, or based only on the most obvious and universally known characteristic of the source entity. Prefer a more detailed, less salient, and somewhat niche characteristic when it still ensures uniqueness. In particular, based on this description and the statement in which it is used, it should be possible to recover the source entity through web search without ambiguity.
Use only information that is explicitly present in the provided Wikipedia page material.
Do not output the source title, aliases, abbreviations, initials, canonical ids, or near-copy surface forms of the source name.
The description should be a phrase or a short descriptive sentence, not a full biography.

Input format:
You will receive one JSON object with exactly these fields:
{
  "statement": "one declarative sentence describing the relationship between source and target",
  "source": "source entity title",
  "target": "target entity title or target description",
  "forbidden_labels": ["source title and aliases that must not appear in the output"],
  "wikipedia_page": {
    "title": "source page title",
    "aliases": ["source aliases"],
    "summary": "short Wikipedia summary",
    "description": "longer Wikipedia page content or introduction",
    "attributes": {"optional": "structured attributes"}
  }
}

Output format:
Return valid JSON with exactly these fields:
{
  "source_clue": "a phrase or short descriptive sentence replacing the source entity",
  "rewritten_statement": "the original statement rewritten with source_clue replacing the source entity",
  "source_supporting_facts": ["the exact source-page facts used to form source_clue"],
  "why_relevant": "why the description is specific enough and still keeps the relationship inferable"
}

Example:
Input:
{
  "statement": "Lionel Messi’s first professional football club was Newell’s Old Boys of Argentina.",
  "source": "Lionel Messi",
  "target": "Newell’s Old Boys",
  "forbidden_labels": ["Lionel Messi", "Messi"],
  "wikipedia_page": {
    "title": "Lionel Messi",
    "aliases": ["Messi"],
    "summary": "Wikipedia page content omitted.",
    "description": "Wikipedia page content omitted.",
    "attributes": {}
  }
}

A suitable obfuscation would be:
"the first-team captain of FC Barcelona in the 2018–19 season"

Rewritten statement:
"The first-team captain of FC Barcelona in the 2018–19 season first played for Argentina’s Newell’s Old Boys."
"""


PROMPT_REWRITE_IMAGE_FIRST_HOP = """You will receive one declarative sentence for the first hop of an image-start trajectory.

Rewrite the sentence so that any specific description or title-like reference to the source image is replaced with the deictic phrase "this image".

Rules:
- Preserve the original meaning of the hop.
- Keep the sentence declarative and natural.
- Do not add facts that are not already present.
- Use "this image" when referring to the source image, even if the original sentence says "the image", "the photo", "the picture", or uses a longer image description.
- If the sentence refers to content inside the image, phrases like "in this image" are allowed.
- Return only valid JSON.

Input format:
{
  "statement": "first-hop declarative sentence"
}

Output format:
{
  "rewritten_statement": "sentence rewritten to refer to the source image as 'this image'"
}
"""

PROMPT_COMPOSE_QUESTION = """
You are an expert at composing multi-hop search questions. Below, you will be given the specific structure of each hop in the data, and your task is to assemble these separated pieces into a continuous reasoning question that hides the intermediate steps and is meant for a user to answer.

Each hop contains at least three parts:
- source: the starting point of this hop
- target: the endpoint of this hop, which must be identified through search and reasoning from the known source based on the given relational statement
- statement: a statement describing the relationship between the target and the source

The source of each hop is the target of the previous hop, so you need to integrate all hops into one complete multi-hop question whose reasoning chain can be described as A -> B -> C ..., where A -> B is the first hop and the user must infer B from A, B -> C is the second hop, and so on.

The entities in the intermediate process, including every hop's source and target, must not appear directly in the question. They must instead be recoverable only through clues, so that the user is forced to reason forward step by step.  NOTICE: if there is an image, that means the image will serve as a part of the question, which will be provided along with the question.

For each hop's target, your question design must ensure that the user can infer it only on the premise that they have already inferred that hop's source from the previous step. There must be no shortcut clues or extra clues.

For the first hop's source, since there is no preceding entity, we additionally provide `first_clue` to describe that source. You must use this description in the question to refer to the first source, ensuring that the user must search and reason from this first clue to identify it.

For the final hop's target, we additionally provide `target_ask`. Once the user has inferred the final target entity step by step, they must answer a knowledge question about that entity.

Rules:
1. After composing the question, check that none of the source or target names appear in the question. However, if a source or target is an image, its textual description may appear, as long as it does not introduce a shortcut.
2. The wording must be natural and references must be clear. The question may be very long, so make sure it is unambiguous, especially with pronouns, and avoid referential confusion.
3. Do not list explicit reasoning steps, and do not use expressions such as "starting from...", "then...", "next...", or "based on this clue...".
4. If the first source is an image, it will be shown to the user, so refer to it in the question as "this image".
5. Check for redundant clues yourself and remove them to ensure there are no shortcuts.
6. If any hop is ambiguous, you should flexibly add a restricting modifier. For example, if the true reasoning result is a certain player who once played for a certain club, and you ask who the first captain of that club was in 2023, the club might be ambiguous. In that case, the restriction should be derived from the true target in a way that still depends on the source. For instance, if the true target is FC Barcelona, a better phrasing would be: "the first international club this player played for." Do not use a restriction like "the club that won the 2011 UEFA Champions League," because that would allow the club to be inferred without depending on the previous source, which creates a shortcut.

Example:
Input:
{
  "opening_mode": "text_start",
  "first_clue": "A 20th-century Romanian pioneering sculptor created a war memorial ensemble in Targu Jiu.",
  "forbidden_labels": ["Constantin Brancusi", "Brancusi"],
  "hop_facts": [
    {
      "hop_index": 0,
      "source": "Constantin Brâncuși",
      "target": "image of Constantin Brâncuși's studio in Paris in 1920",
      "statement": "There is a famous photo of Constantin Brâncuși in his Paris studio photographed by Edward Steichen in 1920"
    },
    {
      "hop_index": 1,
      "source": "image of Constantin Brâncuși's studio in Paris in 1920",
      "target": "Bird in Space",
      "statement": "The slender sculpture positioned at the center of the room in the background is Bird in Space."
    },
    {
      "hop_index": 2,
      "source": "Bird in Space",
      "target": "National Gallery of Art",
      "statement": "National Gallery of Art houses the 1925 marble version and the 1927 bronze version of Bird in Space"
    },
    {
      "hop_index": 3,
      "source": "National Gallery of Art",
      "target": "David E. Finley, Jr.",
      "statement": "The director of the museum from 1938 to 1956 was David E. Finley, Jr."
    }
  ],
  "target_ask": {
    "ask_target": "Where did David E. Finley, Jr. earn his professional degree, and in what field was that degree?"
  }
}

Output question:
"A 20th-century Romanian avant-garde sculptor was photographed by Edward Steichen in his Paris studio in 1920. The slender sculpture standing at the center of the studio background existed in multiple versions. Where did the director who served from 1938 to 1956 at the museum that holds its 1925 marble version and its 1927 bronze version receive their professional degree, and in what field was that degree?"

Explanation: This question removes potentially redundant clues from the input, reveals none of the intermediate entities, and remains grammatically natural.

Return valid JSON with exactly these fields:
{
  "question": "..."
}
"""

PROMPT_POLISH_QUESTION = """
You need to check whether a question should be revised from the following aspects:

0. Entity Obfuscation: If any intermediate result is mentioned in the question (an intermediate result means any source or target in the hops), you need to replace it with a vague description. This description must not be sufficient to identify that entity on its own, but it should still allow the entity to be determined within the context of the question. Example:
"After her club, Stabæk, won the Toppserien title in 2010, it qualified for the UEFA Women's Champions League," where the UEFA Women's Champions League is the source of the next hop.
Reason: The UEFA Women's Champions League has already been explicitly mentioned, but it is an intermediate reasoning result.
Revision method: Replace "qualified for the UEFA Women's Champions League" with "qualified for a European women's club competition."

1. Potential shortcuts: If, when inferring the target of any hop, it is actually unnecessary to first infer that hop’s source, and the target can instead be obtained directly from clues in the question, then a shortcut exists. For example:
“This player (Gemma Font, source) joined the women’s team of a major European club (FC Barcelona Femení, target) as a goalkeeper. In what year did this team move into the Johan Cruyff Stadium?”
Reason: Mentioning the Johan Cruyff Stadium reveals that the team is FC Barcelona Femení, so the source does not need to be inferred.
Revision method: Replace “Johan Cruyff Stadium” with “the team’s current main home stadium.” This removes the shortcut without changing the reasoning path.

2. Ambiguity: If, starting from the source of a hop, multiple targets fit the description in the question, then the question is not actually answerable. In that case, you need to disambiguate by adding a restriction. For example:
“This player (Lionel Messi, source) joined a club (FC Barcelona, target), and that club later won the UEFA Champions League. Who was the club’s first captain in the 2018–19 season?”
Reason: Both FC Barcelona and Paris Saint-Germain fit the description that they are clubs Messi played for and that later won the UEFA Champions League, so the question is ambiguous and requires an added restriction.
Revision method: Add a relatively vague restriction before “club” that does not directly introduce a shortcut, such as: “the youth academy of a club that this player joined.” Note that you must not add a restriction like “the club that later won the 2011 UEFA Champions League,” because that would introduce a shortcut, since only FC Barcelona won the 2011 UEFA Champions League.

3. Remove redundancy: If the target can already be inferred from some clues, then any additional clues about that target can be deleted to make the question shorter and more natural. For example:
“That nonprofit railroad museum in Lenox, reporting mark BRMX, moved its excursion train operations to the Hoosac Valley and began service to North Adams in 2016; the museum displays Budd RDC diesel multiple units...”
Reason: The first sentence is already sufficient to identify the museum, so the part from “moved its excursion train operations...” through “began service to North Adams in 2016” is redundant.
Revision method: Delete that entire redundant portion directly.

4. Polish the wording: Make the question sound more natural to human readers, and check whether there is any referential ambiguity and revise it if needed. For example:
“A collector who donated about 400 German Expressionist works to that museum in 1953 is depicted in a photo of a Richard Neutra-designed house in the Hollywood Hills of Los Angeles. According to the image caption for that photo, what setting is the house shown in?”
Reason: This is an artifact of dataset construction. The image found online may not actually have a caption, and the final question is about the setting of the house, so it is unnecessary to explicitly tell the user which image to look at; the user should locate the relevant image based on the description.
Revision method: “A collector who donated about 400 German Expressionist works to that museum in 1953 is depicted in a photo of a Richard Neutra-designed house in the Hollywood Hills of Los Angeles. What setting is the corresponding house shown in?”

NOTICE: if there is an image, that means the image will serve as a part of the question, which will be provided along with the question.

Now, based on the above requirements and examples, revise the upcoming question and output it in the following format:
Reason: xxx
Revision method: xxx
JSON:
{
  "question": "..."
}
"""

PROMPT_OBFUSCATE_QUESTION = """You are revising a multi-hop question to prevent reasoning shortcuts. You will be given: a multi-hop knowledge reasoning question, which may also include an image associated with the question; the ordered source -> target hop chain supporting the question; and the final ask. For each hop, including the direction of the final ask, inspect how the target is described in the question.

For each hop, if the user can identify the target directly from the clues in the question without first identifying that hop’s source, then a shortcut exists. This usually happens because the relationship between the source and target is described too explicitly, or because highly distinctive events, organizations, objects, or places make the target directly identifiable.

For example: “This player once used the ‘Hand of God’ in a World Cup he played in, and in the semifinal of that World Cup, ...” In this example, even without identifying the player (the source), the phrase “Hand of God” already makes it possible to infer that the target is the 1986 World Cup. An appropriate revision would be: “This player once won a crucial match in a World Cup he played in with a goal that should not have counted, and in the semifinal of that World Cup, ...” This ensures that only after inferring Diego Maradona (the source) can one further infer the 1986 World Cup.

Another example: “The attacking midfielder (Fran Kirby) became Chelsea Women’s all-time leading goalscorer in December 2020 and was part of England’s UEFA Women’s Euro 2022-winning squad. She is linked to a photo of England’s women celebrating with the trophy after the Euro 2022 final at Wembley.” Here, even without first inferring the midfielder (the source, Fran Kirby), one can directly search for the target, namely a championship celebration photo, based on England Women winning the tournament. An appropriate revision would be: “...and was part of her national women’s team’s title-winning squad in a continental tournament in 2022. She is linked to a photo of that team celebrating with the trophy after the final of the tournament just mentioned.”

Requirements:
1. Preserve the order and direction of every hop exactly.
2. Once the source of a hop is known, each target must still be uniquely identifiable. Your obfuscation must not introduce ambiguity such that multiple targets would satisfy the description.
3. Do not mechanically remove all explicit proper nouns. Ensure that the revised question remains answerable without ambiguity while eliminating unnecessary shortcuts.
4. Apply the same obfuscation principle to the final question in the sentence, but do not change what is being asked and do not change the answer.
5. Keep pronoun references clear.
6. If there is no safe and necessary room for improvement, return the original question unchanged.

Before rewriting, first conduct careful analysis, understand the intent of the question, and think through the reasoning path. Make sure that for every hop: when the source is unknown, the target cannot be inferred from the clues in the question alone; and when the source is known, there is exactly one target consistent with both the source and the question.

Output format:
Reason: your detailed reasoning process
Question: the revised question
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
        source_label = self._hop_anchor_label(hop.src_content, fallback=hop.src_node_id)
        target_label = self._hop_anchor_label(hop.dst_content, fallback=hop.dst_node_id)
        if self.model_client is None:
            return self._fallback_compress_hop(hop, source_label=source_label, target_label=target_label)
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
        source = source_label
        target = target_label
        relation = str(parsed.get("relation") or hop.relation or hop.edge_type or "").strip()
        retrieval_query = str(parsed.get("retrieval_query") or "").strip()
        if hop.src_modality == "text" and hop.dst_modality == "image" and not retrieval_query:
            retrieval_query = str(hop.relation or "").strip()
        if not statement or not source or not target:
            return self._fallback_compress_hop(hop, source_label=source_label, target_label=target_label)
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
        target_node_type = str(context.target_node.get("node_type") or "")
        system_prompt = PROMPT_SELECT_IMAGE_TARGET if target_node_type == "image" else PROMPT_SELECT_TEXT_TARGET
        target_image_url = self._target_image_url(context.target_node)
        parsed = self._generate_json(
            system=system_prompt,
            user_payload={"target_node": context.target_node},
            trace_label=f"select_target_ask_{target_node_type or 'unknown'}",
            image_url=target_image_url,
        )
        ask_target = self._ensure_question(str(parsed.get("ask_target") or "").strip())
        answer = str(parsed.get("answer") or "").strip()
        supporting_facts = parsed.get("supporting_facts") or []
        if not isinstance(supporting_facts, list):
            supporting_facts = []
        supporting_facts = [str(item).strip() for item in supporting_facts if str(item).strip()]
        reasoning = str(parsed.get("reasoning") or "").strip()
        support = str(parsed.get("support") or "").strip()
        if not ask_target or not answer:
            return self._fallback_select_target(context.target_node)
        return {
            "ask_target": ask_target,
            "answer": answer,
            "supporting_facts": supporting_facts,
            "reasoning": reasoning,
            "support": support,
        }

    def select_opening_package(self, *, context: WriterContext, hop_summaries: list[dict[str, Any]]) -> dict[str, Any]:
        if not context.hops:
            return {
                "source_clue": "this subject",
                "source_supporting_facts": [],
                "packaged_first_hop": "",
                "first_hop_support": "",
                "why_relevant": "No hops available.",
                "forbidden_labels": [],
            }

        first_hop = context.hops[0]
        source_node = first_hop.src_content
        forbidden_labels = self._forbidden_source_labels(source_node)
        first_hop_summary = hop_summaries[0] if hop_summaries else self._fallback_compress_hop(first_hop)
        if first_hop.src_modality != "text":
            packaged_first_hop = self._select_image_opening_first_hop(first_hop_summary)
            return {
                "source_clue": "",
                "source_supporting_facts": [],
                "packaged_first_hop": packaged_first_hop,
                "first_hop_support": "Image-start trajectory; source-image references are normalized to deictic expressions such as 'this image'.",
                "why_relevant": "Image-start trajectory; no text source anchor required.",
                "forbidden_labels": forbidden_labels,
            }

        if self.model_client is None:
            clue = self._fallback_select_source(source_node, forbidden_labels=forbidden_labels)
            packaged_first_hop = self._fallback_package_first_hop(
                source_clue=clue,
                first_hop_summary=first_hop_summary,
                forbidden_labels=forbidden_labels,
            )
            return {
                "source_clue": clue,
                "source_supporting_facts": [clue] if clue else [],
                "packaged_first_hop": packaged_first_hop,
                "first_hop_support": "Fallback first-hop packaging generated from the first-hop summary.",
                "why_relevant": "Fallback source clue generated from the source node summary/attributes.",
                "forbidden_labels": forbidden_labels,
            }

        parsed = self._generate_json(
            system=PROMPT_SELECT_OPENING_PACKAGE,
            user_payload={
                "statement": first_hop_summary.get("statement") or "",
                "source": source_node.get("title") or first_hop_summary.get("source") or "",
                "target": first_hop_summary.get("target") or "",
                "forbidden_labels": forbidden_labels,
                "wikipedia_page": {
                    "title": source_node.get("title"),
                    "aliases": list(source_node.get("aliases") or []),
                    "summary": source_node.get("summary"),
                    "description": source_node.get("description"),
                    "attributes": dict(source_node.get("attributes") or {}),
                },
            },
            trace_label="select_opening_package",
        )
        clue = str(parsed.get("source_clue") or "").strip()
        supporting_facts = parsed.get("source_supporting_facts") or []
        if not isinstance(supporting_facts, list):
            supporting_facts = []
        supporting_facts = [str(item).strip() for item in supporting_facts if str(item).strip()]
        packaged_first_hop = str(parsed.get("rewritten_statement") or "").strip()
        first_hop_support = str(parsed.get("why_relevant") or "").strip()
        why_relevant = str(parsed.get("why_relevant") or "").strip()
        if not clue or self._contains_forbidden_label(clue, forbidden_labels):
            clue = self._fallback_select_source(source_node, forbidden_labels=forbidden_labels)
            if not supporting_facts and clue:
                supporting_facts = [clue]
            if not why_relevant:
                why_relevant = "Fallback source clue generated because the model clue leaked the source name or was empty."
        if not packaged_first_hop or self._contains_forbidden_label(packaged_first_hop, forbidden_labels):
            packaged_first_hop = self._fallback_package_first_hop(
                source_clue=clue,
                first_hop_summary=first_hop_summary,
                forbidden_labels=forbidden_labels,
            )
            if not first_hop_support:
                first_hop_support = "Fallback first-hop packaging generated because the model output leaked the source name or was empty."
        return {
            "source_clue": clue,
            "source_supporting_facts": supporting_facts,
            "packaged_first_hop": packaged_first_hop,
            "first_hop_support": first_hop_support,
            "why_relevant": why_relevant,
            "forbidden_labels": forbidden_labels,
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
        opening_package = self.select_opening_package(context=context, hop_summaries=hop_summaries)
        target_ask = self.select_target_ask(context=context)
        opening_mode = "image_start" if path.trajectory.starts_with_image else "text_start"
        answer_type = self._default_answer_type(context.target_node)
        if self.model_client is None:
            return self._fallback_compose_question(
                path=path,
                hop_summaries=hop_summaries,
                opening_package=opening_package,
                target_ask=target_ask,
                opening_mode=opening_mode,
                answer_type=answer_type,
            )
        compose_hops = hop_summaries
        compose_payload = self._compose_question_payload(
            opening_mode=opening_mode,
            opening_package=opening_package,
            hop_summaries=compose_hops,
            target_ask=target_ask,
        )
        starting_image_url = self._starting_image_url(path=path, graph=graph)
        parsed = self._generate_json(
            system=PROMPT_COMPOSE_QUESTION,
            user_payload=compose_payload,
            trace_label="compose_question",
            image_url=starting_image_url,
        )
        question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        answer = str(target_ask.get("answer") or "").strip()
        if (
            not question
            or not answer
            or self._looks_like_chain_narration(question)
            or self._contains_forbidden_label(question, opening_package.get("forbidden_labels") or [])
        ):
            rewritten = self._rewrite_chain_narration(
                opening_mode=opening_mode,
                hop_summaries=compose_hops,
                opening_package=opening_package,
                target_ask=target_ask,
            )
            if rewritten is not None:
                question = rewritten
        if (
            not question
            or not answer
            or self._contains_forbidden_label(question, opening_package.get("forbidden_labels") or [])
        ):
            return self._fallback_compose_question(
                path=path,
                hop_summaries=hop_summaries,
                opening_package=opening_package,
                target_ask=target_ask,
                opening_mode=opening_mode,
                answer_type=answer_type,
            )
        return QuestionDraft(
            question=question,
            answer=answer,
            answer_type=answer_type,
            reasoning_steps=hop_summaries,
            used_evidence_ids=[hop.edge_id for hop in context.hops],
            metadata={
                "path_id": path.path_id,
                "opening_package": opening_package,
                "compose_payload": compose_payload,
                "starting_image_url": starting_image_url,
                "target_ask": target_ask,
                "writer_context": context.to_dict(),
            },
        )

    def draft(self, *, path: PathCandidate, graph: GraphView) -> QuestionDraft:
        return self.compose_question(path=path, graph=graph)

    def polish(self, *, draft: QuestionDraft, path: PathCandidate, graph: GraphView) -> QuestionDraft:
        if self.model_client is None:
            return draft
        polish_payload = self._polish_question_payload(
            question=draft.question,
            hops=draft.reasoning_steps,
        )
        starting_image_url = self._starting_image_url(path=path, graph=graph)
        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_POLISH_QUESTION),
                        ModelMessage(
                            role="user",
                            content=self._user_message_content(
                                polish_payload,
                                image_url=starting_image_url,
                            ),
                        ),
                    ],
                    metadata={"trace_label": "polish_question"},
                )
            )
        except Exception as exc:
            return self._record_writer_warning(draft, stage="polish_request", error=exc)
        try:
            parsed = self._extract_json_object(response.content)
        except Exception as exc:
            return self._record_writer_warning(draft, stage="polish_parse", error=exc)
        polished_question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        if not polished_question:
            return self._record_writer_warning(
                draft,
                stage="polish_parse",
                error=ValueError("Model returned an empty question."),
            )
        metadata = dict(draft.metadata)
        metadata["polish_payload"] = polish_payload
        metadata["polish_starting_image_url"] = starting_image_url
        metadata["polish_result"] = {
            "raw_response": response.content,
            "question": polished_question,
        }
        return QuestionDraft(
            question=polished_question,
            answer=draft.answer,
            answer_type=draft.answer_type,
            reasoning_steps=list(draft.reasoning_steps),
            used_evidence_ids=list(draft.used_evidence_ids),
            metadata=metadata,
        )

    def obfuscate(self, *, draft: QuestionDraft, path: PathCandidate, graph: GraphView) -> QuestionDraft:
        if self.model_client is None:
            return draft
        target_ask = draft.metadata.get("target_ask") or {}
        obfuscation_payload = self._obfuscation_question_payload(
            question=draft.question,
            hops=draft.reasoning_steps,
            final_ask=str(target_ask.get("ask_target") or "").strip(),
        )
        starting_image_url = self._starting_image_url(path=path, graph=graph)
        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_OBFUSCATE_QUESTION),
                        ModelMessage(
                            role="user",
                            content=self._user_message_content(
                                obfuscation_payload,
                                image_url=starting_image_url,
                            ),
                        ),
                    ],
                    metadata={"trace_label": "obfuscate_question"},
                )
            )
        except Exception as exc:
            return self._record_writer_warning(draft, stage="obfuscation_request", error=exc)

        obfuscated_question = self._extract_labeled_section(response.content, label="Question")
        if not obfuscated_question:
            return self._record_writer_warning(
                draft,
                stage="obfuscation_parse",
                error=ValueError("Model response did not contain a Question section."),
            )
        reason = self._extract_labeled_section(response.content, label="Reason", next_label="Question")
        obfuscated_question = self._clean_composed_question(obfuscated_question)
        if not obfuscated_question:
            return self._record_writer_warning(
                draft,
                stage="obfuscation_parse",
                error=ValueError("Model returned an empty obfuscated question."),
            )

        metadata = dict(draft.metadata)
        metadata["obfuscation_payload"] = obfuscation_payload
        metadata["obfuscation_starting_image_url"] = starting_image_url
        metadata["obfuscation_result"] = {
            "raw_response": response.content,
            "reason": reason,
            "question": obfuscated_question,
        }
        return QuestionDraft(
            question=obfuscated_question,
            answer=draft.answer,
            answer_type=draft.answer_type,
            reasoning_steps=list(draft.reasoning_steps),
            used_evidence_ids=list(draft.used_evidence_ids),
            metadata=metadata,
        )

    @staticmethod
    def _record_writer_warning(
        draft: QuestionDraft,
        *,
        stage: str,
        error: Exception,
    ) -> QuestionDraft:
        metadata = dict(draft.metadata)
        warnings = list(metadata.get("writer_warnings") or [])
        warnings.append(
            {
                "stage": stage,
                "error_type": error.__class__.__name__,
                "error": str(error),
            }
        )
        metadata["writer_warnings"] = warnings
        return QuestionDraft(
            question=draft.question,
            answer=draft.answer,
            answer_type=draft.answer_type,
            reasoning_steps=list(draft.reasoning_steps),
            used_evidence_ids=list(draft.used_evidence_ids),
            metadata=metadata,
        )

    def _generate_json(
        self,
        *,
        system: str,
        user_payload: dict[str, Any],
        trace_label: str,
        image_url: str | None = None,
    ) -> dict[str, Any]:
        if self.model_client is None:
            raise RuntimeError("model_client is required for _generate_json")
        request = ModelRequest(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_object"},
            messages=[
                ModelMessage(role="system", content=system),
                ModelMessage(
                    role="user",
                    content=self._user_message_content(user_payload, image_url=image_url),
                ),
            ],
            metadata={"trace_label": trace_label},
        )
        response = self.model_client.generate(request)
        print('####',response.content,"####")
        try:
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            parsed = self._extract_json_object(response.content)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object for {trace_label}, got: {type(parsed)!r}")
        return parsed

    @staticmethod
    def _starting_image_url(*, path: PathCandidate, graph: GraphView) -> str | None:
        if not path.node_ids or not path.trajectory.starts_with_image:
            return None
        source_node = graph.get_node(path.node_ids[0]) or {}
        if source_node.get("node_type") != "image":
            return None
        for candidate in (
            source_node.get("image_url"),
            source_node.get("oss_uri"),
            source_node.get("thumb_oss_uri"),
        ):
            image_url = str(candidate or "").strip()
            if image_url:
                return image_url
        return None

    @staticmethod
    def _target_image_url(target_node: dict[str, Any]) -> str | None:
        if target_node.get("node_type") != "image":
            return None
        for candidate in (
            target_node.get("image_url"),
            target_node.get("oss_uri"),
            target_node.get("thumb_oss_uri"),
        ):
            image_url = str(candidate or "").strip()
            if image_url:
                return image_url
        return None

    @staticmethod
    def _default_answer_type(target_node: dict[str, Any]) -> str:
        return "image_content" if target_node.get("node_type") == "image" else "other"

    @staticmethod
    def _user_message_content(
        user_payload: dict[str, Any],
        *,
        image_url: str | None = None,
    ) -> str | list[dict[str, Any]]:
        prompt_text = json.dumps(user_payload, ensure_ascii=False, indent=2)
        if not image_url:
            return prompt_text
        return [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]

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
    def _extract_labeled_section(
        text: str,
        *,
        label: str,
        next_label: str | None = None,
    ) -> str:
        if not text:
            return ""
        if next_label:
            pattern = rf"(?:^|\n)\s*{re.escape(label)}\s*:\s*(.*?)(?=\n\s*{re.escape(next_label)}\s*:)"
        else:
            pattern = rf"(?:^|\n)\s*{re.escape(label)}\s*:\s*(.*)\Z"
        match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        value = match.group(1).strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1].strip()
        return value

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
            payload["image_url"] = node.get("image_url")
            payload["oss_uri"] = node.get("oss_uri")
            payload["thumb_oss_uri"] = node.get("thumb_oss_uri")
            payload["search_query"] = metadata.get("search_query") if isinstance(metadata, dict) else None
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
    def _ensure_question(text: str) -> str:
        if not text or text.endswith("?"):
            return text
        return text.rstrip(".!") + "?"

    @classmethod
    def _hop_anchor_label(cls, content: dict[str, Any], *, fallback: str) -> str:
        if content.get("node_type") == "image":
            search_query = cls._shorten_text(content.get("search_query"), limit=180)
            if search_query:
                return f"the image that {search_query}"
            caption = cls._shorten_text(content.get("caption"), limit=180)
            if caption:
                return f"the image showing {caption}"
        return str(content.get("title") or content.get("caption") or fallback)

    @staticmethod
    def _forbidden_source_labels(source_node: dict[str, Any]) -> list[str]:
        labels: list[str] = []
        for candidate in [source_node.get("title"), *(source_node.get("aliases") or [])]:
            text = str(candidate or "").strip()
            if text and text.lower() not in {item.lower() for item in labels}:
                labels.append(text)
        return labels

    @staticmethod
    def _contains_forbidden_label(text: str, labels: list[str]) -> bool:
        normalized_text = QuestionWriter._normalize_label(text)
        if not normalized_text:
            return False
        for label in labels:
            normalized_label = QuestionWriter._normalize_label(label)
            if not normalized_label:
                continue
            if re.search(rf"(?<!\w){re.escape(normalized_label)}(?!\w)", normalized_text):
                return True
        return False

    @staticmethod
    def _normalize_label(text: Any) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z\u00C0-\u024F\u4e00-\u9fff]+", " ", str(text or "").lower())).strip()

    @classmethod
    def _remove_forbidden_labels(cls, text: str, labels: list[str], *, replacement: str = "") -> str:
        rewritten = str(text or "")
        for label in sorted(labels, key=len, reverse=True):
            if not label:
                continue
            rewritten = re.sub(re.escape(label), replacement, rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\s+", " ", rewritten)
        rewritten = re.sub(r"\s+([,.;:!?])", r"\1", rewritten)
        return rewritten.strip(" ,.;:-")

    @staticmethod
    def _compose_question_payload(
        *,
        opening_mode: str,
        opening_package: dict[str, Any],
        hop_summaries: list[dict[str, Any]],
        target_ask: dict[str, Any],
    ) -> dict[str, Any]:
        first_clue = str(
            opening_package.get("packaged_first_hop")
            or opening_package.get("source_clue")
            or ("this image" if opening_mode == "image_start" else "this subject")
        ).strip()
        return {
            "opening_mode": opening_mode,
            "first_clue": first_clue,
            "forbidden_labels": list(opening_package.get("forbidden_labels") or []),
            "hop_facts": [
                {
                    "hop_index": item.get("hop_index"),
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "statement": item.get("statement"),
                    "retrieval_query": item.get("retrieval_query"),
                }
                for item in hop_summaries
            ],
            "target_ask": {
                "ask_target": target_ask.get("ask_target"),
            },
        }

    @staticmethod
    def _polish_question_payload(*, question: str, hops: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "question": question,
            "hops": [
                {
                    "hop_index": item.get("hop_index"),
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "statement": item.get("statement"),
                }
                for item in hops
            ],
        }

    @staticmethod
    def _obfuscation_question_payload(
        *,
        question: str,
        hops: list[dict[str, Any]],
        final_ask: str,
    ) -> dict[str, Any]:
        return {
            "question": question,
            "hops": [
                {
                    "hop_index": item.get("hop_index"),
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "statement": item.get("statement"),
                }
                for item in hops
            ],
            "final_ask": final_ask,
        }




    def _fallback_select_source(self, source_node: dict[str, Any], *, forbidden_labels: list[str]) -> str:
        candidate_texts: list[str] = []
        for candidate in [source_node.get("summary"), source_node.get("description")]:
            if candidate:
                candidate_texts.append(str(candidate))
        attributes = source_node.get("attributes") or {}
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                if isinstance(value, dict):
                    value = value.get("value")
                if isinstance(value, (str, int, float)) and str(value).strip():
                    normalized_key = self._normalize_label(key)
                    if normalized_key in {"occupation", "job", "profession", "role"}:
                        candidate_texts.append(f"the {value}")
                    else:
                        candidate_texts.append(f"the one whose {key} is {value}")
        for candidate in candidate_texts:
            cleaned = self._remove_forbidden_labels(candidate, forbidden_labels)
            cleaned = self._shorten_text(cleaned, limit=180) or ""
            if cleaned and not self._contains_forbidden_label(cleaned, forbidden_labels):
                return cleaned.rstrip(".")
        return "the starting subject"

    @classmethod
    def _fallback_package_first_hop(
        cls,
        *,
        source_clue: str,
        first_hop_summary: dict[str, Any],
        forbidden_labels: list[str],
    ) -> str:
        retrieval_query = str(first_hop_summary.get("retrieval_query") or "").strip()
        statement = str(first_hop_summary.get("statement") or "").strip()
        if retrieval_query:
            cleaned_query = cls._remove_forbidden_labels(retrieval_query, forbidden_labels)
            if cleaned_query:
                return f"A well-known image related to {source_clue} provides the next clue."
        if not statement:
            return f"{source_clue}."
        packaged = cls._remove_forbidden_labels(statement, forbidden_labels, replacement=source_clue)
        packaged = re.sub(r"\s+", " ", packaged).strip()
        if not packaged:
            packaged = source_clue
        if not re.search(r"[.?!]$", packaged):
            packaged = packaged.rstrip(" ,;:") + "."
        return packaged

    @classmethod
    def _fallback_package_image_first_hop(cls, first_hop_summary: dict[str, Any]) -> str:
        statement = str(first_hop_summary.get("statement") or "").strip()
        if not statement:
            return "This image provides the next clue."
        packaged = cls._normalize_image_reference(statement)
        if not re.search(r"[.?!]$", packaged):
            packaged = packaged.rstrip(" ,;:") + "."
        return packaged

    @staticmethod
    def _normalize_image_reference(text: str) -> str:
        rewritten = str(text or "")
        rewritten = re.sub(r"\b(image|photo|picture) of [^.?!,;:]+", "this image", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bin (?:the )?(image|photo|picture)\b", "in this image", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\bthe (image|photo|picture)\b", "this image", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\s+", " ", rewritten).strip()
        if rewritten:
            rewritten = rewritten[0].upper() + rewritten[1:]
        return rewritten or "This image provides the next clue."

    def _select_image_opening_first_hop(self, first_hop_summary: dict[str, Any]) -> str:
        fallback = self._fallback_package_image_first_hop(first_hop_summary)
        statement = str(first_hop_summary.get("statement") or "").strip()
        if self.model_client is None or not statement:
            return fallback

        try:
            parsed = self._generate_json(
                system=PROMPT_REWRITE_IMAGE_FIRST_HOP,
                user_payload={"statement": statement},
                trace_label="rewrite_image_first_hop",
            )
        except Exception:
            return fallback

        rewritten = str(parsed.get("rewritten_statement") or "").strip()
        if not rewritten:
            return fallback
        rewritten = self._normalize_image_reference(rewritten)
        return rewritten or fallback

    @staticmethod
    def _fallback_compress_hop(
        hop: HopContext,
        *,
        source_label: str | None = None,
        target_label: str | None = None,
    ) -> dict[str, Any]:
        relation = hop.relation or hop.edge_type or "is connected to"
        retrieval_query = ""
        source_label = source_label or QuestionWriter._hop_anchor_label(hop.src_content, fallback=hop.src_node_id)
        target_label = target_label or QuestionWriter._hop_anchor_label(hop.dst_content, fallback=hop.dst_node_id)
        if hop.src_modality == "text" and hop.dst_modality == "text":
            statement = f"{source_label} {relation} {target_label}".strip()
        elif hop.src_modality == "text" and hop.dst_modality == "image":
            retrieval_query = str(hop.relation or "").strip()
            if retrieval_query:
                statement = (
                    f"{source_label} is associated with a key photo or visual scene described by the query: "
                    f"{retrieval_query}."
                )
            else:
                statement = f"{source_label} is associated with a key photo or visual scene: {target_label}."
        elif hop.src_modality == "image" and hop.dst_modality == "text":
            statement = f"{QuestionWriter._image_clue_label(hop)} refers to {target_label}."
        else:
            statement = f"{source_label} {relation} {target_label}".strip()
        return {
            "hop_index": hop.hop_index,
            "statement": statement,
            "source": source_label,
            "target": target_label,
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
                "ask_target": "What is the key visual content in the final image?",
                "answer": answer or "unknown visual content",
                "supporting_facts": [answer] if answer else [],
                "reasoning": "The answer is directly extracted from the final image evidence.",
                "support": "Selected from the final image caption/visual facts.",
            }

        attributes = target_node.get("attributes") or {}
        if isinstance(attributes, dict):
            for key, value in attributes.items():
                if isinstance(value, (str, int, float)) and str(value).strip():
                    return {
                        "ask_target": f"What is the {key} of the final entity?",
                        "answer": str(value),
                        "supporting_facts": [f"{key}: {value}"],
                        "reasoning": "The answer is directly extracted from the selected attribute.",
                        "support": f"Selected attribute {key}.",
                    }
        description = target_node.get("description") or target_node.get("summary")
        if description:
            return {
                "ask_target": "What key detail is described for the final entity?",
                "answer": self._shorten_text(description, limit=120) or "",
                "supporting_facts": [str(description)],
                "reasoning": "The answer is directly extracted from the description or summary.",
                "support": "Fell back to description/summary.",
            }
        return {
            "ask_target": "What is the identity of the final entity?",
            "answer": str(target_node.get("title") or "unknown"),
            "supporting_facts": [str(target_node.get("title"))] if target_node.get("title") else [],
            "reasoning": "The answer is the title of the final entity.",
            "support": "Fell back to target title.",
        }

    @staticmethod
    def _fallback_compose_question(
        *,
        path: PathCandidate,
        hop_summaries: list[dict[str, Any]],
        opening_package: dict[str, Any],
        target_ask: dict[str, Any],
        opening_mode: str,
        answer_type: str,
    ) -> QuestionDraft:
        remaining_hops = hop_summaries[1:] if opening_package.get("packaged_first_hop") else hop_summaries
        hop_text = " ".join(item.get("statement", "") for item in remaining_hops if item.get("statement"))
        ask_target = str(target_ask.get("ask_target") or "What is the final answer?")
        answer = str(target_ask.get("answer") or "unknown")
        opening_bridge = str(opening_package.get("packaged_first_hop") or "").strip()
        if opening_mode == "image_start":
            question = (
                f"{opening_bridge} {hop_text} {ask_target}"
                if opening_bridge
                else "Use the relevant visual and factual clues in the given image to identify the final subject. "
                f"{hop_text} {ask_target}"
            )
        else:
            forbidden = list(opening_package.get("forbidden_labels") or [])
            if not opening_bridge:
                source_clue = str(opening_package.get("source_clue") or "this subject").strip()
                opening_bridge = source_clue.rstrip(".") + "."
            masked_hop_text = QuestionWriter._remove_forbidden_labels(hop_text, forbidden)
            question = f"{opening_bridge} {masked_hop_text} {ask_target}"
        question = QuestionWriter._clean_composed_question(question)
        return QuestionDraft(
            question=question,
            answer=answer,
            answer_type=answer_type,
            reasoning_steps=hop_summaries,
            used_evidence_ids=[item.get("edge_id", "") for item in hop_summaries if item.get("edge_id")],
            metadata={"opening_package": opening_package, "target_ask": target_ask},
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
        opening_package: dict[str, Any],
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
                "Use first_clue to refer to the first source instead of naming it directly.\n"
                "If forbidden_labels is provided, never output those labels.\n\n"
                "Return valid JSON with exactly these fields:\n"
                "{\n"
                '  "question": "..."\n'
                "}\n"
            ),
            user_payload=self._compose_question_payload(
                opening_mode=opening_mode,
                opening_package=opening_package,
                hop_summaries=hop_summaries,
                target_ask=target_ask,
            ),
            trace_label="rewrite_chain_narration",
        )
        question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        if (
            not question
            or self._looks_like_chain_narration(question)
            or self._contains_forbidden_label(question, opening_package.get("forbidden_labels") or [])
        ):
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
    debug_hop_summaries = [
        {
            key: value
            for key, value in item.items()
            if key not in {"edge_id", "src_node_id", "dst_node_id"}
        }
        for item in hop_summaries
    ]
    opening_package = writer.select_opening_package(context=context, hop_summaries=hop_summaries)
    draft = writer.compose_question(path=path, graph=graph, context=context)
    polished = writer.polish(draft=draft, path=path, graph=graph)
    obfuscated = writer.obfuscate(draft=polished, path=path, graph=graph)
    polish_result = polished.metadata.get("polish_result") if isinstance(polished.metadata, dict) else None
    obfuscation_result = (
        obfuscated.metadata.get("obfuscation_result")
        if isinstance(obfuscated.metadata, dict)
        else None
    )

    print("path:")
    print(json.dumps(path.to_dict(), ensure_ascii=False, indent=2))
    print(f"writer_model: {args.model_alias or 'fallback(no llm)'}")
    print("hop_summaries:")
    print(json.dumps(debug_hop_summaries, ensure_ascii=False, indent=2))
    print("opening_package:")
    print(json.dumps(opening_package, ensure_ascii=False, indent=2))
    print("draft_question:")
    print(json.dumps(
        {
            "question": draft.question,
            "answer": draft.answer,
            "answer_type": draft.answer_type,
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("polished_question:")
    print(json.dumps(
        {
            "question": polished.question,
            "answer": polished.answer,
            "answer_type": polished.answer_type,
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("polish_raw_output:")
    print((polish_result or {}).get("raw_response") if isinstance(polish_result, dict) else None)
    print("obfuscated_question:")
    print(json.dumps(
        {
            "question": obfuscated.question,
            "answer": obfuscated.answer,
            "answer_type": obfuscated.answer_type,
        },
        ensure_ascii=False,
        indent=2,
    ))
    print("obfuscation_raw_output:")
    print(
        (obfuscation_result or {}).get("raw_response")
        if isinstance(obfuscation_result, dict)
        else None
    )


if __name__ == "__main__":
    _debug_main()
