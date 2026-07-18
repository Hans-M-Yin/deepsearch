"""LLM-backed question writer for graph trajectories.

The writer now works directly from ``PathCandidate + GraphView`` instead of a
separate evidence-builder stage. Internally it follows a seven-step process:

1. compress each hop into a short statement
2. normalize hidden image bridges when they are only evidence carriers
3. derive an opening package for the first source + first hop
4. select a raw askable target from the final node
5. normalize a hidden final image terminal step when needed
6. compose and polish the final multi-hop question
7. obfuscate shortcut clues while preserving the reasoning path
"""

from __future__ import annotations

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import mimetypes
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


_VQA_FIXED_REQUEST_ID = "3200636808"


PROMPT_COMPRESS_HOP_GENERIC = """You are compressing one hop from a multimodal reasoning trajectory.

You are given:
- hop_type: the modality pair for this hop
- source: the current source node title
- relation: the edge relation string
- target: the current target node title

Write one short declarative statement that captures the essential meaning of
this hop for downstream question composition.

The statement should:
- preserve the relation needed for downstream reasoning
- be concise
- avoid unnecessary details
- avoid asking a question
- stay faithful to THIS hop only
- not introduce entities that are not the current source node or current target node

Anchor rules:
- source must refer to the current source node only
- target must refer to the current target node only
- do not replace source or target with entities from earlier or later hops
- do not turn this into a cross-hop summary

Return valid JSON with exactly these fields:
{
  "statement": "...",
  "source": "...",
  "target": "...",
  "relation": "...",
  "retrieval_query": "..."
}
"""


PROMPT_COMPRESS_HOP_TEXT_TO_TEXT = """You are compressing one text-to-text hop from a reasoning trajectory.

You are given:
- source: the current source text node name
- relation: the edge relation string
- target: the current target text node name

Write one short declarative statement that captures the essential meaning of
this hop for downstream question composition.

Treat this as a directed entity-to-entity relation. The statement should make
the transition from the source to the target clear, rather than replacing
either one with a broad summary.

The statement should:
- preserve the relation needed for downstream reasoning
- keep the direction from source to target clear
- preserve dates, roles, numbers, and other constraints when they matter
- be concise
- avoid unnecessary details
- avoid asking a question
- stay faithful to THIS hop only
- not introduce entities that are not the current source node or current target node

Anchor rules:
- source must refer to the current source node only
- target must refer to the current target node only
- do not replace source or target with entities from earlier or later hops
- do not turn this into a cross-hop summary

Example 1:
Input:
{
  "source": "David E. Finley, Jr.",
  "relation": "the school where he earned his professional degree",
  "target": "Harvard Law School"
}
Output:
{
  "statement": "David E. Finley, Jr. earned his professional degree from Harvard Law School.",
  "source": "David E. Finley, Jr.",
  "target": "Harvard Law School",
  "relation": "earned his professional degree from",
  "retrieval_query": ""
}

Example 2:
Input:
{
  "source": "Bird in Space",
  "relation": "the museum that houses its 1925 marble and 1927 bronze versions",
  "target": "National Gallery of Art"
}
Output:
{
  "statement": "The museum that houses the 1925 marble and 1927 bronze versions of Bird in Space is the National Gallery of Art.",
  "source": "Bird in Space",
  "target": "National Gallery of Art",
  "relation": "the museum that houses its 1925 marble and 1927 bronze versions is",
  "retrieval_query": ""
}

Return valid JSON with exactly these fields:
{
  "statement": "...",
  "source": "...",
  "target": "...",
  "relation": "...",
  "retrieval_query": ""
}
"""


PROMPT_COMPRESS_HOP_TEXT_TO_IMAGE = """You are writing one natural-language declarative sentence for a structured text-to-image hop.

You will be given:
- source: the name of the current source text node
- relation: the edge relation string from the source to the image
- target: the name of the current target image

Task:
Write one complete declarative sentence in English that converts the structured
information above into a natural-language statement.

The sentence must preserve the transition from the source text node to the
target image. It should be a relation sentence, not a standalone caption of the
target image.

Requirements:
- include the source explicitly
- include the target explicitly as the image target
- preserve the relation and all uniqueness-bearing details, especially event, time, place, action, and scene details
- make the sentence natural, coherent, and unambiguous
- stay faithful to THIS hop only
- do not introduce any information not provided
- do not introduce literal type prefixes such as "Image:" unless they are already present in the input
- do not replace the target with a free-floating scene description detached from the source
- avoid vague scaffolds like "For X, the relevant image is ..." or "About X, there is ..."
- when possible, phrase the sentence as a direct transition from the source to the target image

Field guidance:
- `statement` should be the natural-language sentence
- `source` should stay aligned with the given source
- `target` should stay aligned with the given target
- `relation` should stay close to the given relation string; only smooth grammar lightly if needed
- `retrieval_query` should be a concise image lookup clue, usually the target itself or a slightly cleaner version

Anchor rules:
- source must refer to the current source node only
- target must refer to the current target node only
- do not replace source or target with entities from earlier or later hops
- do not turn this into a cross-hop summary

Example 1:
Input:
{
  "source": "Port Jackson",
  "relation": "Japanese midget submarine recovered from Sydney Harbour after the 31 May 1942 raid",
  "target": "photo of the recovered Japanese midget submarine"
}
Output:
{
  "statement": "Port Jackson is related to a photo that shows the Japanese midget submarine recovered from Sydney Harbour after the 31 May 1942 raid.",
  "source": "Port Jackson",
  "target": "photo of the recovered Japanese midget submarine",
  "relation": "Japanese midget submarine recovered from Sydney Harbour after the 31 May 1942 raid",
  "retrieval_query": "photo of the recovered Japanese midget submarine"
}

Example 2:
Input:
{
  "source": "Constantin Brancusi",
  "relation": "photo of him in his Paris studio taken by Edward Steichen in 1920",
  "target": "Steichen photo of Brancusi in his Paris studio"
}
Output:
{
  "statement": "Constantin Brancusi is related to a photo that shows him in his Paris studio, taken by Edward Steichen in 1920.",
  "source": "Constantin Brancusi",
  "target": "Steichen photo of Brancusi in his Paris studio",
  "relation": "photo of him in his Paris studio taken by Edward Steichen in 1920",
  "retrieval_query": "Steichen photo of Brancusi in his Paris studio"
}

Return valid JSON with exactly these fields:
{
  "statement": "...",
  "source": "...",
  "target": "...",
  "relation": "...",
  "retrieval_query": "..."
}
"""


PROMPT_COMPRESS_HOP_IMAGE_TO_TEXT = """You are writing one natural-language declarative sentence for a structured image-to-text hop.

You will be given:
- source: the name of the current image node; this is a brief description of the image, but the image itself is not provided
- relation: a description of how the target appears in the image
- target: the name of the current target text node

Task:
Write one short declarative sentence in English that converts the structured
information above into a natural-language statement.

The sentence must preserve the transition from the image to the target text
node. It should make clear that the target is identified from the image,
rather than turning into a generic fact about the target alone.

Requirements:
- include both source and target explicitly
- preserve the descriptive image name, the target name, and the relation
- do not omit key identifying details from the relation
- keep the image as the anchor of the sentence
- make the sentence concise, natural, coherent, and unambiguous
- stay faithful to THIS hop only
- do not introduce any information not provided
- do not introduce literal type prefixes such as "Image:" unless they are already present in the input
- do not add new absolute directional descriptions such as top, bottom, left, right, upper-left, or lower-right
- if the provided relation already contains a directional detail, preserve it rather than inventing a different spatial claim

Field guidance:
- `statement` should be the natural-language sentence
- `source` should stay aligned with the given source
- `target` should stay aligned with the given target
- `relation` should stay close to the given relation string; only smooth grammar lightly if needed
- `retrieval_query` must be an empty string for this hop type

Anchor rules:
- source must refer to the current source node only
- target must refer to the current target node only
- do not replace source or target with entities from earlier or later hops
- do not turn this into a cross-hop summary

Example 1:
Input:
{
  "source": "photo of a player taking the decisive penalty in a World Cup final",
  "relation": "the player taking the penalty in the image is",
  "target": "Gonzalo Montiel"
}
Output:
{
  "statement": "In this photo of a player taking the decisive penalty in 2022 World Cup final, the player taking the penalty in the image is Gonzalo Montiel.",
  "source": "photo of a player taking the decisive penalty in a World Cup final",
  "target": "Gonzalo Montiel",
  "relation": "the player taking the penalty in the image is",
  "retrieval_query": ""
}

Example 2:
Input:
{
  "source": "first page of the original handwritten United States Constitution on parchment with the words We the People",
  "relation": "the chamber of Congress named at the end of Article I, Section 1 is",
  "target": "United States House of Representatives"
}
Output:
{
  "statement": "In the first page of the original handwritten United States Constitution on parchment with the words We the People, the chamber of Congress named at the end of Article I, Section 1 is the United States House of Representatives.",
  "source": "first page of the original handwritten United States Constitution on parchment with the words We the People",
  "target": "United States House of Representatives",
  "relation": "the chamber of Congress named at the end of Article I, Section 1 is",
  "retrieval_query": ""
}

Return valid JSON with exactly these fields:
{
  "statement": "...",
  "source": "...",
  "target": "...",
  "relation": "...",
  "retrieval_query": ""
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
   - Do not ask for facts that can be answered from general knowledge, the image description alone, or a Wikipedia page about the subject. Bad Example: ask what accessory was in the breast pocket of the suit jacket as the answer should be a white pocket square based on common sense.
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

**Example 1:
A photo of Argentina defender Montiel scoring the penalty in the World Cup final.

Reasonable question examples:
1. Which side of the goal did the player kick the ball toward when taking the penalty in the World Cup final?
2. Which side of the goal did the goalkeeper dive toward?
3. What number jersey was the goalkeeper wearing?

Unreasonable question:
What number was the penalty taker wearing?
(Reason: this can be answered directly by searching for Montiel based on the image description, so the image itself is unnecessary.)

**Example 2:
An aerial photo of the University of Chicago.

Reasonable question examples:
1. What is the main color of the roofs of the campus buildings?
2. What is the nearest building next to the tallest building on campus?

Unreasonable question:
1. What two colors are on the roof of the building in the lower-left corner of the image?
(Reason: this is ambiguous because it depends on the orientation of the photo.)

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


PROMPT_NORMALIZE_IMAGE_BRIDGE = """You will be given two declarative statements involving three objects: a source, an intermediate image, and a target. The actual intermediate image will also be provided with the request.

Input format:
statement1: ...
statement2: ...
source: ...
mid-image: ...
target: ...

Task:
Merge the two statements into a single source-to-target step.

This process has two steps.

Step 1: Determine whether the intermediate image can be skipped.
- Prefer `hide_image` when the image mainly records a real-world event, activity, occasion, or scene that exists independently of the photograph, and the target is related to that real-world situation rather than to the photograph as an artifact.
- A clear time reference is a strong signal in favor of `hide_image`, but it is not the only criterion.
- Do not decide based only on the target type. Even a logo, clothing item, object, or number can still support `hide_image` if it belongs to the real-world event or scene rather than to the photograph as an artifact.
- Prefer `keep_image` when the intermediate image is itself the essential object or reference frame, such as a manuscript page, document page, poster, artwork, portrait, cover, screenshot, interface, map, diagram, chart, or similar artifact.
- If removing the explicit mention of the intermediate image would change the meaning or lose the needed reference frame, choose `keep_image`.

Step 2: Merge the statements accordingly.
- If you choose `hide_image`, use the event / activity / scene corresponding to the image instead of explicit media wording.
- If you choose `keep_image`, keep the image / page / artwork / artifact framing explicit and merge the two statements into one natural sentence.

Output requirements:
- Return both a `rewritten_statement` and a `rewritten_relation`.
- `rewritten_relation` should be a short source-to-target relation phrase aligned with the merged sentence. The target itself must not appear directly in the relation.
- `rewritten_statement` should be one complete declarative sentence from the source to the target, and the target should appear explicitly in that sentence.
- If you choose `hide_image`, the final statement and relation must not contain words such as `image`, `photo`, `picture`, or similar media terms.
- Preserve all necessary information. Do not delete important details or add new facts.
- Make the final statement and relation fluent and natural.
- Return only valid JSON.

Return valid JSON with exactly these fields:
{
  "decision": "hide_image" or "keep_image",
  "reason": "brief explanation",
  "rewritten_relation": "short source-to-target relation phrase",
  "rewritten_statement": "one declarative sentence from source to target"
}

Example 1:
Input:
statement1: Joe Biden is related to an image of him delivering the State of the Union address at a joint session of the U.S. Congress in 2023
statement2: In this image of a 2023 joint session of the U.S. Congress listening to President Joe Biden deliver the State of the Union address, the person sitting behind Joe Biden on the right is Kevin McCarthy
source: Joe Biden
mid-image: Image: In 2023, a joint session of the U.S. Congress is listening to President Joe Biden deliver the State of the Union address
target: Kevin McCarthy
Output:
{
  "decision": "hide_image",
  "reason": "The intermediate image records a specific real-world event with a concrete time reference, so the event can replace the photo framing.",
  "rewritten_relation": "the person sitting behind him on the right when he delivered the State of the Union address at a 2023 joint session of the U.S. Congress was",
  "rewritten_statement": "When Joe Biden delivered the State of the Union address at a 2023 joint session of the U.S. Congress, the person sitting behind him on the right was Kevin McCarthy."
}

Example 2:
Input:
statement1: Katherine Johnson is related to an image of her receiving the Silver Snoopy Award from astronaut Leland Melvin in 2017
statement2: In this image, the logo on the chest of the blue flight suit is NASA
source: Katherine Johnson
mid-image: Image: Astronaut Leland Melvin presenting the Silver Snoopy Award to Katherine Johnson in 2017
target: NASA
Output:
{
  "decision": "hide_image",
  "reason": "The intermediate image records a specific award presentation in 2017, so the event can replace the image while preserving the target description.",
  "rewritten_relation": "the logo on the chest of the presenter's blue flight suit when she received the Silver Snoopy Award in 2017 was",
  "rewritten_statement": "When Katherine Johnson received the Silver Snoopy Award in 2017, the logo on the chest of the presenter's blue flight suit was NASA."
}

Example 3:
Input:
statement1: The United States Constitution is related to an image of its first handwritten page beginning with We the People
statement2: In this image, the chamber of Congress named at the end of Article I, Section 1 is the United States House of Representatives
source: United States Constitution
mid-image: Image: The first handwritten page of the original United States Constitution beginning with We the People
target: United States House of Representatives
Output:
{
  "decision": "keep_image",
  "reason": "The intermediate image is a document page whose textual content and page identity are essential, so the image cannot be skipped.",
  "rewritten_relation": "in its first handwritten page beginning with We the People, the chamber of Congress named at the end of Article I, Section 1 is",
  "rewritten_statement": "In the first handwritten page of the original United States Constitution beginning with We the People, the chamber of Congress named at the end of Article I, Section 1 is the United States House of Representatives."
}
"""

PROMPT_NORMALIZE_FINAL_IMAGE_TARGET_ASK = """You will be given a declarative statement and a question involving three elements: a source, an intermediate image, and a final answer. The actual intermediate image will also be provided with the request.

Input format:
statement1: ...
question: ...
answer: ...
source: ...
mid-image: ...

Task:
The statement links the source to an intermediate image. The question asks about a visual detail in that image, and the provided `answer` is the final target.

You are rewriting the question-facing terminal step.
This rewrite will replace the original final `text -> image` hop plus the raw final question.
Do not output a declarative bridge statement. The only question-facing output should be the rewritten final question.

Step 1: Determine whether the intermediate image can be skipped.
- Prefer `hide_image` when the image mainly records a real-world event, activity, occasion, or scene that exists independently of the image, and the asked visual detail belongs to that real-world situation rather than to the image as an artifact.
- A clear time reference is a strong signal in favor of `hide_image`, but it is not the only criterion.
- Do not decide based only on the target type. Even a logo, clothing item, object, number, or attribute can still support `hide_image` if it belongs to the real-world event or scene rather than to the image as an artifact.
- Prefer `keep_image` when the intermediate image is itself the essential object or reference frame, such as a manuscript page, document page, poster, artwork, portrait, cover, screenshot, interface, map, diagram, chart, or similar artifact.
- If removing the explicit mention of the intermediate image would change the meaning, lose the needed reference frame, or make the question depend on layout, cropping, or textual arrangement that only exists inside the image, choose `keep_image`.

Step 2: Rewrite the final question accordingly.
- If you choose `hide_image`, rewrite the question through the underlying event / activity / scene instead of explicit media wording.
- If you choose `keep_image`, keep the image / page / artwork / artifact framing explicit.
- The rewritten question must carry enough information on its own, because the original final `text -> image` hop will not be provided separately during downstream question generation.
- Use the actual image to resolve ambiguity when necessary, but do not add any fact that is not supported by the input or the image.

Output requirements:
- Return only `rewritten_ask_target`.
- `rewritten_ask_target` should be one natural final question whose answer remains exactly the provided `answer`.
- It should replace the original last hop plus the raw question in downstream question generation.
- Do not reveal the answer inside the question.
- If you choose `hide_image`, the rewritten question must not contain words such as `image`, `photo`, `picture`, or similar media terms.
- Preserve all necessary information. Do not delete important details or add new facts.
- Make the rewritten question fluent and natural.
- Return only valid JSON.

Return valid JSON with exactly these fields:
{
  "decision": "hide_image" or "keep_image",
  "reason": "brief explanation",
  "rewritten_ask_target": "one final question whose answer remains the provided answer"
}

Example 1:
Input:
statement1: Lionel Messi is related to an image of him during the 2022 FIFA World Cup final
question: What logo appears on the front of the jersey in the image?
answer: Adidas
source: Lionel Messi
mid-image: Image: Lionel Messi during the 2022 FIFA World Cup final
Output:
{
  "decision": "hide_image",
  "reason": "The intermediate image records a specific real-world event, and the asked jersey logo belongs to that event scene rather than to the image as an artifact.",
  "rewritten_ask_target": "During the 2022 FIFA World Cup final, what logo was on the front of Lionel Messi's jersey?"
}

Example 2:
Input:
statement1: Southern Methodist University is related to an image of five U.S. presidents at the dedication ceremony for the George W. Bush Presidential Center on April 25, 2013
question: How many of the attending presidents in the image are wearing red ties?
answer: 3
source: Southern Methodist University
mid-image: Image: Five U.S. presidents Obama, George W. Bush, Clinton, George H.W. Bush, and Carter together on stage at the George W. Bush Presidential Center dedication ceremony at Southern Methodist University on April 25, 2013
Output:
{
  "decision": "hide_image",
  "reason": "The intermediate image records a specific ceremony with a concrete time reference, so the event can replace the image framing while preserving the restricted group shown in the scene.",
  "rewritten_ask_target": "At the George W. Bush Presidential Center dedication ceremony at Southern Methodist University on April 25, 2013, how many of the attending presidents were wearing red ties?"
}

Example 3:
Input:
statement1: Marc Kinchen is related to the cover art for the Storm Queen single 'Look Right Through (MK Remix)'
question: In the cover art, what letters appear on the second-to-last line, and what color is each of them?
answer: GH and MK; GH is black and MK is red
source: Marc Kinchen
mid-image: Image: Cover art for the Storm Queen single 'Look Right Through (MK Remix)'
Output:
{
  "decision": "keep_image",
  "reason": "The intermediate image is cover art, which is itself the essential artifact and reference frame for the asked visual text layout.",
  "rewritten_ask_target": "In Marc Kinchen's cover art for the Storm Queen single 'Look Right Through (MK Remix)', what letters appear on the second-to-last line, and what color is each of them?"
}

Example 4:
Input:
statement1: The United States Constitution is related to an image of its first handwritten page beginning with We the People
question: What chamber of Congress is named at the end of Article I, Section 1 in the image?
answer: the United States House of Representatives
source: United States Constitution
mid-image: Image: The first handwritten page of the original United States Constitution beginning with We the People
Output:
{
  "decision": "keep_image",
  "reason": "The intermediate image is a document page whose textual content and page identity are essential, so the image cannot be skipped.",
  "rewritten_ask_target": "In the first handwritten page of the original United States Constitution beginning with We the People, what chamber of Congress is named at the end of Article I, Section 1?"
}
"""


# Legacy compose prompt retained for comparison and rollback. Runtime question
# composition uses the revised PROMPT_COMPOSE_QUESTION defined after this block.
PROMPT_COMPOSE_QUESTION_LEGACY = """
You are an expert at composing multi-hop search questions. Below, you will be given the specific structure of each hop in the data, and your task is to assemble these separated pieces into a continuous reasoning question that hides the intermediate steps and is meant for a user to answer.

Each hop contains at least three parts:
- source: the starting point of this hop
- target: the endpoint of this hop, which must be identified through search and reasoning from the known source based on the given relational statement
- statement: a statement describing the relationship between the target and the source

The source of each hop is the target of the previous hop, so you need to integrate all hops into one complete multi-hop question whose reasoning chain can be described as A -> B -> C ..., where A -> B is the first hop and the user must infer B from A, B -> C is the second hop, and so on.

The entities in the intermediate process, including every hop's source and target, must not appear directly in the question. They must instead be recoverable only through clues, so that the user is forced to reason forward step by step.  NOTICE: if there is an image at the beginning, that means the image will serve as a part of the question, which will be provided along with the question.

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


PROMPT_COMPOSE_QUESTION = """
You are an expert at composing multi-hop search questions. Below, you will be
given the structure of each hop in a reasoning path. Transform these separated
pieces into one coherent question that hides intermediate answers while still
requiring the solver to follow the intended search path.

Each hop contains:
- source: the starting point of the hop;
- target: the endpoint that must be recovered from the source;
- statement: a declarative sentence describing their relationship;
- retrieval_query: optional retrieval context, especially for image hops.

The source of each hop is the target of the previous hop. The path can therefore
be represented as A -> B -> C ..., but this notation describes semantic
dependency, not a sentence structure to copy.

The input also provides:
- `first_clue`: the question-facing description of the first source and,
  sometimes, an already rewritten version of the first hop;
- `forbidden_labels`: source names or aliases that must not be exposed;
- `target_ask`: the final question to ask after the last entity or image has
  been reached.

Important: `first_clue` may already express the first item in `hop_facts`. If it
does, use that information once and do not restate the same first hop.

Composition principles:

1. Compose globally instead of restating the path hop by hop.

Read the complete path before writing. Determine the minimum role that each hop
plays in identifying the next entity, then combine the necessary relationships
using whatever structure is most natural: relative clauses, temporal clauses,
appositives, participial phrases, or multiple sentences when they improve
readability.

Do not use a fixed template. Do not mechanically turn every hop statement into
a separate sentence. Sentence boundaries should follow natural semantic units,
not hop boundaries.

2. Preserve semantic dependency and reasoning direction.

The wording must preserve A -> B -> C: A provides the basis for identifying B,
and B provides the basis for identifying C. Local reordering is allowed only
when it improves grammar without breaking this dependency or making a later
clue independently understandable without the preceding result.

Every later entity must remain meaningfully connected to the entity recovered
in the preceding hop.

3. Apply minimum sufficient obfuscation, not maximum vagueness.

Intermediate source and target labels are internal annotations and should not
normally be exposed directly. However, do not blindly obscure every proper noun
or concrete detail. For each reference, implicitly choose the least aggressive
action that prevents leakage while preserving recoverability:

- KEEP: retain a necessary entry anchor, disambiguating detail, or part of the
  final ask when it does not independently reveal an intermediate answer.
- RELATIONALIZE: replace an entity name with a description based on its
  relationship to the entity recovered in the preceding hop.
- GENERALIZE: replace an explicit name with a stable broader type, role, class,
  or category when the exact name is unnecessarily revealing.
- DELETE: remove a detail that does not identify the next entity, resolve a
  genuine ambiguity, preserve an essential visual reference frame, or express
  the final ask.

A good replacement must pass both tests:
- Without the preceding-hop result, it should not independently identify the
  intermediate entity.
- After the preceding-hop result is known, it should provide enough relational
  or contextual information to recover the intended next entity.

If a replacement identifies the entity without the previous hop, it is too
revealing. If it remains unclear even after the previous hop is known, it is too
vague.

Do not obscure a clue merely because it is specific. Dates, visible clothing,
event types, co-attendance, locations, and other concrete details should be
retained when removing them would make the next event, image, or entity unstable
or ambiguous. The goal is the minimum sufficient description, not the shortest
possible question.

4. Allow evidence-preserving abstraction, but do not expand the story.

You may paraphrase, relationalize, or generalize information explicitly stated
or directly entailed by `first_clue`, `hop_facts`, and `target_ask`.

Examples of valid abstraction:
- `Wellesley College` -> `a women's college`;
- `the Presidential Medal of Freedom` -> `a major civilian honor`.

Such abstractions preserve the supplied fact at a broader level. They are not
permission to introduce outside knowledge.

Do not add an unrelated event, achievement, office, famous anecdote,
biographical fact, or identity clue merely because you know it. Do not replace
a supplied relation with a different famous fact about the same entity. Avoid
new highly searchable details that allow the solver to bypass the preceding
hop.

5. Retain functionally necessary information and remove only true redundancy.

Every retained detail must serve at least one purpose:
- identify the next entity through the preceding entity;
- distinguish the intended event, image, or entity from plausible alternatives;
- preserve an essential image or artifact reference frame;
- express the final ask.

When deciding whether to delete a detail, ask: after deletion, can the next hop
still be recovered reliably and uniquely from the preceding result? If not,
retain the detail.

When several statements repeat the same person, event, image, or relationship,
merge the repeated information instead of narrating it again.

6. Handle images naturally.

If the first source is an image, it will be shown with the question. Refer to it
as `this image`; do not repeat a long title-like description unless a visible
detail is necessary for the first transition.

If a later image mainly records a real-world event or scene and the supplied hop
has already been normalized into event-based wording, express the event
naturally rather than instructing the solver to locate another image. If the
image is itself the essential artifact or reference frame, such as a manuscript
page, poster, artwork, cover, screenshot, map, diagram, or chart, preserve that
framing.

Avoid repeatedly saying `in this image`, `in that photo`, or `there is another
image` unless the image itself is required as the reference frame.

7. Keep the final question compact, connected, and natural.

Do not add procedural introductions such as `Please look at`, `First identify`,
`Starting from`, `Then`, `Next`, or `Based on this clue`. Do not explain the
reasoning procedure.

Avoid repeatedly reintroducing the same person, event, institution, or image.
Maintain a continuous reference chain. Use pronouns only when they have one
unambiguous referent. Multiple sentences are allowed, but use another sentence
only when it improves comprehension rather than merely marking a new hop.

8. Avoid shortcuts and unnecessary constraints.

Do not identify a later entity through a newly added famous title, iconic event,
unique achievement, unrelated date, nationality, or recognizable biographical
fact. If a hop is genuinely ambiguous, add only a minimal restriction that
still depends on the preceding entity.

For example, `the first international club this player represented` still
depends on identifying the player. By contrast, `the club that won the 2011
UEFA Champions League` may identify the club independently and creates a
shortcut.

9. Preserve the final ask.

The final question must retain the meaning of `target_ask`. Do not reveal its
answer, replace it with a different question, or add an independent question
that is not part of `target_ask`. The transition into the final ask should
follow naturally from the entity or event established by the preceding clauses.

10. Silently check the result before returning it.

Verify that:
- the path's semantic dependency and direction are preserved;
- `first_clue` and the first hop are not repeated;
- the output is not a one-sentence-per-hop narration;
- intermediate names are hidden only where necessary;
- necessary anchors and disambiguating details have not been over-obscured;
- each later entity is recoverable after, but not independently of, the
  preceding entity;
- no outside background facts were introduced;
- every retained detail performs a transition, disambiguation,
  visual-reference, or final-ask function;
- all references are clear;
- the question is concise without sacrificing unique recoverability;
- the final ask has the same meaning as the supplied `target_ask`.

Few-shot example 1: image-start path with multiple visual transitions

Input:
{
  "opening_mode": "image_start",
  "first_clue": "In this image showing President George H. W. Bush reviewing documents with Dick Cheney and Brent Scowcroft in April 1989, the man on the left is Brent Scowcroft.",
  "forbidden_labels": ["Brent Scowcroft", "Scowcroft"],
  "hop_facts": [
    {
      "hop_index": 0,
      "source": "image of George H. W. Bush reviewing documents with Dick Cheney and Brent Scowcroft in April 1989",
      "target": "Brent Scowcroft",
      "statement": "In this image showing President George H. W. Bush reviewing documents with Dick Cheney and Brent Scowcroft in April 1989, the man on the left is Brent Scowcroft.",
      "retrieval_query": ""
    },
    {
      "hop_index": 1,
      "source": "Brent Scowcroft",
      "target": "image of Brent Scowcroft receiving the Presidential Medal of Freedom in 1991",
      "statement": "Brent Scowcroft is associated with a photograph of President George H. W. Bush placing the Presidential Medal of Freedom around his neck at a White House ceremony in 1991.",
      "retrieval_query": "Brent Scowcroft Presidential Medal of Freedom White House 1991"
    },
    {
      "hop_index": 2,
      "source": "image of Brent Scowcroft receiving the Presidential Medal of Freedom in 1991",
      "target": "Barbara Bush",
      "statement": "In the photograph of the 1991 White House ceremony, the woman standing beside the recipient in a red-and-white polka-dot dress is Barbara Bush.",
      "retrieval_query": ""
    },
    {
      "hop_index": 3,
      "source": "Barbara Bush",
      "target": "image of Barbara Bush and another country's first lady at the Wellesley College commencement",
      "statement": "Barbara Bush is associated with a photograph of her sitting beside another country's first lady at the Wellesley College commencement on June 1, 1990.",
      "retrieval_query": "Barbara Bush Wellesley commencement June 1 1990"
    }
  ],
  "target_ask": {
    "ask_target": "What type of necklace was Barbara Bush wearing at the commencement?"
  }
}

Bad output:
"Please look at the man on the left in this image. In 1991, he received the
Presidential Medal of Freedom from President George H. W. Bush at the White
House. A woman in a red-and-white polka-dot dress was standing beside him. That
woman later attended a famous commencement ceremony at Wellesley College. What
type of necklace was she wearing?"

Why it is bad:
- It mechanically restates the hops as separate sentences.
- It repeats the starting instruction.
- It exposes exact names that can be safely abstracted.
- It adds `famous`, which is unsupported and unnecessary.
- Its references are fragmented instead of forming one dependency chain.

Good output:
"What type of necklace did the woman in the red-and-white polka-dot dress
standing beside the man on the left of this image as he received a major
civilian honor at a 1991 White House ceremony wear when, the previous year, she
attended a women's-college commencement alongside another country's first
lady?"

Why it is good:
- It preserves 1991 and the co-attendance clue because they distinguish the
  intended ceremonies and image.
- `another country's first lady` preserves the supplied relation without adding
  a highly searchable nationality that could bypass the earlier transition.
- It relationalizes the man and woman and generalizes the honor and college.
- It uses the starting image once and preserves the complete dependency chain.
- It does not add background facts or narrate one sentence per hop.

Few-shot example 2: text-start path with a photographed artwork bridge

Input:
{
  "opening_mode": "text_start",
  "first_clue": "A 20th-century Romanian sculptor created a war memorial ensemble in Targu Jiu.",
  "forbidden_labels": ["Constantin Brancusi", "Constantin Brâncuși", "Brancusi", "Brâncuși"],
  "hop_facts": [
    {
      "hop_index": 0,
      "source": "Constantin Brâncuși",
      "target": "image of Constantin Brâncuși's studio in Paris in 1920",
      "statement": "The sculptor was photographed in his Paris studio in 1920.",
      "retrieval_query": "Constantin Brancusi Paris studio 1920"
    },
    {
      "hop_index": 1,
      "source": "image of Constantin Brâncuși's studio in Paris in 1920",
      "target": "Bird in Space",
      "statement": "The slender sculpture standing near the center of the studio is Bird in Space.",
      "retrieval_query": ""
    },
    {
      "hop_index": 2,
      "source": "Bird in Space",
      "target": "National Gallery of Art",
      "statement": "A museum holds both a marble version and a bronze version of the sculpture.",
      "retrieval_query": ""
    },
    {
      "hop_index": 3,
      "source": "National Gallery of Art",
      "target": "David E. Finley, Jr.",
      "statement": "The museum's director from 1938 to 1956 was David E. Finley, Jr.",
      "retrieval_query": ""
    }
  ],
  "target_ask": {
    "ask_target": "Where did David E. Finley, Jr. earn his professional degree, and in what field was that degree?"
  }
}

Bad output:
"A 20th-century Romanian sculptor created a war memorial ensemble in Targu Jiu.
He was photographed in his Paris studio in 1920. In the image, the sculpture in
the center is Bird in Space. The National Gallery of Art holds two versions of
the sculpture. Its director from 1938 to 1956 was David E. Finley, Jr. Where did
he earn his professional degree, and in what field?"

Why it is bad:
- It assigns a separate sentence to nearly every hop.
- It directly reveals the sculpture, museum, and director.
- It reads like an explanation of the path rather than a natural question.

Good output:
"A 20th-century Romanian sculptor who created a war memorial ensemble in Targu
Jiu was photographed in his Paris studio in 1920. Where did the 1938-56 director
of the museum that holds both marble and bronze versions of the slender
sculpture at the center of that studio photograph earn his professional degree,
and in what field?"

Why it is good:
- It uses two sentences because they form natural semantic units, not because
  the path contains multiple hops.
- It preserves the path order while compressing the middle transitions.
- It hides the sculpture, museum, and director names without losing the facts
  needed to recover them.
- It retains dates and material types because they perform useful identifying
  functions.

Now compose the question for the provided input.

Return valid JSON with exactly these fields:
{
  "question": "..."
}
"""


PROMPT_POLISH_ENTITY_OBFUSCATION = """
You are auditing a multi-hop reasoning question for which the intended reasoning chain has already been provided. In this chain, the source of each hop statement is an intermediate answer. An ideal multi-hop reasoning question must require the solver to infer each intermediate result step by step from the beginning, so intermediate answers must not be exposed in advance. It must also ensure that an intermediate answer cannot be inferred without first obtaining the result of the previous step; in other words, shortcuts must be avoided, so that a solver cannot infer an intermediate answer without relying on the earlier part of the question.

You need to compare the question against the reasoning chain sentence by sentence and check whether either of the following problems appears:
1. The question explicitly reveals an intermediate answer.
2. The question does not explicitly mention an intermediate answer, but based on common knowledge, the description in the question allows the real referent of an entity to be directly inferred.

For issue 1, the solution is to replace the intermediate answer with a vague description or a referring expression, while making sure not to introduce ambiguity, that is, not to create multiple possible matches because the reference becomes too vague. For issue 2, the solution is to revise the wording so that the description can only lead to that intermediate result if the previous entity is already known.

If there are multiple issues, list them one by one, without duplication or omission. Please output the result in JSON format:

{
  "issues": "1. ... (2. ...)",
  "advice": "1. ... (2. ...)"
}
If no issue is found, output:
{
  "issues": "None.",
  "advice": "No change needed."
}

Example 1:
(reasoning chain omitted)
"question": This image shows a celebration scene, and the logo on the green banner on the left indicates that it is related to the German national football association. This association belongs to the continental governing body of European football.

Output:
{
    "issues": "According to the provided reasoning trajectory, the German national football association is the reasoning result of the first hop, that is, an intermediate result, but the question mentions it directly and therefore exposes it.",
    "advice": "Replace that intermediate result and remove the specific description of it, for example: '...the football organization indicated by the logo on the banner belongs to the continental governing body of European football.' "
}
Example 2:
"question": ...This player once used the ‘Hand of God’ in a World Cup he played in, and in the semifinal of that World Cup, ...

Output:
{
    "issues": "Although the intermediate result is not explicitly exposed, the phrase 'Hand of God' is mentioned, which refers to one of the most iconic goals in football history. This description allows 'This player' and 'that World Cup' to be directly inferred, creating a shortcut relative to the earlier part of the question.",
    "advice": "Obscure 'Hand of God', for example by changing it to 'a goal that should not have counted', so that only after identifying This player as Diego Maradona can one further infer the year of that World Cup from the description."
}
"""


PROMPT_POLISH_SHORTCUT = """
You are reviewing a multi-hop search reasoning question for which the ideal search chain has already been provided. In this chain, the source of each hop’s statement is the intermediate answer from the previous hop. A well-constructed multi-hop reasoning question must require step-by-step inference from the beginning, so you need to ensure that no intermediate answer can be derived without the result of the previous step. In other words, avoid shortcuts: the next intermediate answer should not be inferable without considering the earlier part of the question.

Shortcuts usually arise when the question describes an entity too explicitly, making it possible to infer the next intermediate answer directly from common knowledge or from the description itself, without needing the previous intermediate result. You should compare the question against the search chain sentence by sentence and analyze whether each intermediate result can be directly inferred from the question description, paying special attention to overly specific wording or to related entities that are explicitly exposed in the question.

If there is a shortcut, provide your analysis and a suggestion for revision. If there are multiple problems, list them all without omission.

If there are multiple issues, list them one by one, without duplication or omission. Please output the result in JSON format:

{
  "issues": "1. ... (2. ...)",
  "advice": "1. ... (2. ...)"
}
If no issue is found, output:
{
  "issues": "None.",
  "advice": "No change needed."
}

Example:
(the search chain is omitted)
"question": "...this player later used the ‘Hand of God’ in a World Cup he played in, and in the semifinal of that World Cup, ..."

Output:
{
    "issues": "Although the intermediate result is not explicitly exposed, the phrase 'Hand of God' is mentioned, which refers to one of the most iconic goals in football history. This description allows 'This player' and 'that World Cup' to be directly inferred, creating a shortcut relative to the earlier part of the question.",
    "advice": "Obscure 'Hand of God', for example by changing it to 'a goal that should not have counted', so that only after identifying This player as Diego Maradona can one further infer the year of that World Cup from the description."
}
"""

PROMPT_POLISH_AMBIGUITY = """
Here is the English translation of the prompt:

You are auditing a multi-hop search-and-reasoning question. The intended search chain for the question has already been provided, and in each hop, the source of the statement is the intermediate answer from the previous hop. An ideal multi-hop reasoning question must not contain ambiguity — that is, cases where a description is too broad, so that multiple intermediate results could plausibly satisfy the question.

If ambiguity exists, you should explain where the ambiguity appears and propose an additional restricting modifier so that only the given intermediate answer fits the question. However, this modifier must remain vague enough that it does not become an independent shortcut. In other words, the solver must still derive the previous intermediate answer first before using the new description to obtain the next one.

If there are multiple issues, list them one by one, without duplication or omission. Please output the result in JSON format:

{
  "issues": "1. ... (2. ...)",
  "advice": "1. ... (2. ...)"
}
If no issue is found, output:
{
  "issues": "None.",
  "advice": "No change needed."
}

Example 1:
(search chain omitted)

question: This university association includes an Australian university among its members. Where is the main campus of that university?

Output:
{
  "issues": "After identifying the university association, there may still be multiple Australian universities among its members, so the question does not uniquely determine which university is being referred to.",
  "advice": "According to the search chain, the university is actually Charles Sturt University. You can add a vague but restricting description, such as 'a university whose coat of arms contains river-like wavy lines.'"
}

Bad revision example:
question: In the team of this player (Lionel Messi) (FC Barcelona), who was the first captain in the 2017–18 season?
Reason it is bad: this player may have played for multiple teams before.

Low-quality advice: change “the team of this player” to “the team that won the 2011 UEFA Champions League.”
Reason: the 2011 UEFA Champions League winner directly points to FCB, so it no longer requires deriving Lionel Messi first, which creates a shortcut.

High-quality advice: “the player’s first club team in Europe.”
Reason: this is restrictive, but it still requires first deriving Lionel Messi.
"""

PROMPT_POLISH_REDUNDANCY = """
You are auditing a multi-hop search-and-reasoning question. The intended search chain for the question has already been provided, and in each hop, the source of each statement is the intermediate answer from the previous hop. In this kind of reasoning question, the more intermediate descriptions are included, the more likely they are to directly expose the intermediate answer, so you need to examine the question and identify redundant descriptions.

A redundant description means an overly detailed description of an entity in the question such that removing it would make the question harder, but would not create ambiguity in the reasoning process. (Note that some descriptions are necessary, because removing them would make multiple entities fit the description and thus introduce ambiguity.)

If there are multiple issues, list them one by one, without duplication or omission. Please output the result in JSON format:

{
  "issues": "1. ... (2. ...)",
  "advice": "1. ... (2. ...)"
}
If no issue is found, output:
{
  "issues": "None.",
  "advice": "No change needed."
}
Example:
question: The winner of that tournament came from Germany’s highest women’s league, which began in 1990 with two regional divisions before later becoming a single nationwide competition. For the 1997–98 season, what structural change was introduced in that league?

Output:
{
  "issues": "Under the intended reasoning chain, the logic is: tournament winner -> the German league they came from. So the added description of the league is unnecessary and may create a shortcut.",
  "advice": "Delete 'which began in 1990 with two regional divisions before later becoming a single nationwide competition.'"
}
"""

PROMPT_POLISH_WORDING = """
You are auditing a generated multi-hop question and focusing on only one issue: wording and fluency.

Judge only whether the question sounds unnatural, awkward, overly like a stitched-together search chain, or contains unclear references. Focus especially on the following aspects:

1. Unclear pronoun references. Based on the search-and-reasoning chain, verify whether every pronoun in the question clearly refers to a unique target.
2. Rigid and overly fragmented phrasing. Sometimes a question is written in an unnecessarily long and scattered way, and we want a more compact structure. For example:
   "In this continental organization, the member association representing France is that country's football governing body. That French member association was created in 1919 by transforming an earlier organization. What was the name of that earlier organization? On what exact date did this transformation occur? Who became the first president after the change?"
   can be improved to:
   "The French representative member of this continental organization was reorganized in 1919. What was the organization’s name before the reorganization, on what exact date did it occur, and who became the first president afterward?"
3. Rigid image descriptions. If the question includes an image, it should refer to it as "this image" rather than giving a stiff literal description of the picture. Also, if solving the question requires searching for another image, the question can hide that operation somewhat to make it sound more natural. For example:
   "...and there is also a draft-night photo of the player that team had selected first overall the previous year alongside the league commissioner. In that draft photo, what is the player wearing on his head?"
   The second sentence explicitly instructs the solver to locate a particular image. This can be made more natural by hiding that operation, such as:
   "What was worn on the head of the player whom that team had selected first overall the previous year when he appeared alongside the league commissioner on draft night?"
   This revised version sounds more natural.

Note: if there is an image, it means the image will be shown to the respondent together with the question. Check the coherence and correctness of the question against the image.

Please identify all such issues in the question, completely and without omission or duplication. Return valid JSON containing only the following fields:
{
  "issues": "1. ... 2. ...",
  "advice": "1. ... 2. ..."
}
If no issue is found, output:
{
  "issues": "None.",
  "advice": "No change needed."
}
"""

PROMPT_POLISH_REWRITE = """
You are revising an existing multi-hop question based on a set of diagnostic reports. This multi-hop search question may have issues related to object obfuscation, shortcuts, ambiguity, grammatical correctness, or fluency. You will be given a question, the hop chain that supports its search and reasoning process, and a set of diagnostic issue descriptions and revision suggestions. Your task is to address all valid diagnostic feedback together and produce one revised question.

Requirements:
1. Preserve the order and reasoning direction of the underlying hops.
2. Preserve the final answer exactly.
3. Ensure that the revised question is natural and fluent, with no referential errors and no ambiguity.

Note: if there is an image, it means the image will be shown to the respondent together with the question. Check the coherence and correctness of the question against the image.

Please return valid JSON with exactly the following field:
{
  "question": "..."
}
"""

PROMPT_DIFFICULTY_ENHANCEMENT = """
You are a difficulty enhancement editor for multi-hop search questions. Rewrite the question so that it becomes harder for strong models to immediately identify the intermediate entities, while keeping the original answer, factual relations, and core reasoning chain unchanged. The rewritten question must remain uniquely solvable and verifiable.

Your goal is NOT to make the question longer, more literary, or more confusing. Instead, you should:
1. reduce direct exposure of intermediate entities;
2. remove strong clues that allow common-sense shortcutting;
3. replace those strong clues with weaker, vaguer expressions that still preserve contextual identification function;
4. reduce the obviously step-by-step sequential reasoning structure, so that the order of clues does not mechanically mirror the hop order;
5. keep the entire question fluent, natural, and benchmark-like.

Strict requirements:
1. Do not change the core question or the final answer.
2. Do not merely perform synonym substitution; you must genuinely reduce the salience of intermediate entities.
3. A replacement expression must satisfy this condition: by itself it should not directly identify the target entity, but within the full question context it should still help uniquely constrain the correct path.
4. While reducing salience, you must preserve or introduce enough non-shortcut constraints to keep the question uniquely solvable. Recommendation: retain the name of one or two objects mentioned at the beginning of the original question to provide an entry point for identification and reasoning if there's no images attached to the questions. Apply stronger obfuscation to widely known entities, and weaker obfuscation to less familiar entities—or leave some unobfuscated in certain questions—to ensure the questions remain solvable and the difficulty stays balanced.
5. If the original question is artificially tied to a specific source framing (“according to profile X,” “in source Y’s description,” etc.) but the answer is really a real-world fact rather than a document-specific wording question, remove or naturalize that framing instead of keeping it mechanically.
6. Note that there might be an image attached to the question, keep the connection between the question and the image content
7. Do not fabricate any information, and do not make any changes unless you are explicitly certain about them.

Preferred rewriting strategies:
1. Use relational, structural, or contextual constraints instead of highly distinctive signals including famous titles, people names, signature works, unique achievements, strong year markers, or iconic paper titles.
2. Remove features that directly expose intermediate entities, but replace them with weaker contextual descriptions rather than simply deleting them.
3. If obfuscation introduces ambiguity, add non-shortcut constraints to eliminate wrong candidates.
4. Avoid an overly explicit hop-by-hop reasoning structure; The order of the clues may be adjusted appropriately according to semantics and logic.
5. Keep the question natural, concise, and benchmark-like rather than turning it into a pile of hints.

Below are examples. Learn the rewriting style from them.

Example 1:
question: The man shown in this image later became nationally prominent for his handling of a devastating 1927 flood centered on the lower Mississippi Valley, a development that helped lead to his 1928 presidential nomination by the political party whose modern encyclopedic profile says Donald Trump narrowly carried three long-Democratic “blue wall” states in 2016. Which three states were they, and what broader effect did that victory have on that party according to the profile?
Output:
{
  "analysis": "The original question was too easy in two important ways. First, the phrase “Donald Trump” plus “three long-Democratic blue wall states” created an overly direct shortcut to the Republican Party, which in turn made the earlier flood-and-nomination clue almost unnecessary for identifying the party connection. The revision removes the explicit mention of “Democratic,” because once that label is stated outright, the party in the earlier clause becomes mechanically recoverable without requiring the intended historical bridge through the flood response and the 1928 nomination. This weakens the partisan cue while still preserving a uniquely recoverable political contrast through “the opposing party.” Second, the original phrasing anchored the answer to a specific “modern encyclopedic profile” and repeatedly referred to what “the profile says” or “the overview says.” The revision removes that framing, because the question is really about the underlying factual content rather than about a particular document as an object. Keeping that document frame made the question sound artificially source-bound and less natural, even though the intended answer is a real-world fact rather than a source-specific wording task. Beyond those two user-motivated improvements, the revised version also softens other high-salience expressions: “centered on the lower Mississippi Valley” becomes the less directly identifying “a catastrophic flood in 1927,” and “blue wall states” becomes the more contextual “states that had long been regarded as reliably belonging to the opposing party in presidential elections.” These replacements reduce direct entity exposure while preserving enough relational structure to keep the question uniquely solvable. The result is more natural, less template-like, and less vulnerable to partisan shortcutting, while keeping the original answer, factual relations, and reasoning chain intact.",
  "question": "The man shown in this image later gained national prominence through his response to a catastrophic flood in 1927, a development that helped him secure his party’s presidential nomination the following year. That same party’s 2016 nominee narrowly flipped three states that had long been regarded as reliably belonging to the opposing party in presidential elections. Which three states were they, and what broader effect were those victories said to have had on that party?"
}

Example 2:
question: This image shows a work on the Moon’s motion. That subject also appears in a geometrical diagram in Newton’s landmark 1687 mathematical treatise on natural philosophy. Before the final three-book structure of the Principia was settled, what was the planned title of the surviving fair-copy draft of its second volume, about when was that draft completed, and which later part of the finished work did it largely correspond to in purpose?
Output:
{
  "analysis": "The original question is too easy because it exposes the bridge entities too directly and presents the clues in almost the same order as the hop chain, allowing a strong model to identify the target work and its composition history during question reading. First, the phrase “Moon’s motion” is not a proper name, but when combined with the later Newton cue it sharply narrows the search space, so it should be replaced with the weaker expression “The topic treated in the work shown here,” which preserves the semantic link to the pictured text without repeating an overly revealing subject label. Second, “Newton’s landmark 1687 mathematical treatise on natural philosophy” is an especially strong identifying bundle: it gives the author, the year, the genre, and a canonical-status cue, which together almost amount to naming the Principia outright. This should therefore be weakened to “a late seventeenth-century mathematical work on natural philosophy,” which keeps the period, disciplinary context, and textual type while removing the most direct author- and year-based shortcuts. Third, the phrase “Before the final three-book structure of the Principia was settled” not only names the work again but also points the solver straight to the specific composition-history page about its evolving structure. The revision replaces this with “was not yet fixed in its eventual three-book arrangement when one surviving fair-copy draft was prepared,” which preserves the same manuscript-history constraint while no longer directly naming the Principia or making the search path so explicit. Fourth, “the surviving fair-copy draft of its second volume” is actually a valuable non-shortcut constraint and should be retained rather than removed, because it is part of what keeps the question uniquely solvable. In the revision, it becomes “the draft corresponding to what was then planned as its second volume,” which preserves the same functional role while making the transition slightly less template-like. Fifth, the original question follows a highly mechanical sequence: pictured topic -> Newton’s treatise -> Principia composition history -> second-volume fair-copy -> title/date/correspondence. The revision improves this by folding the pictured topic and the geometric treatment into a single sentence and by centering the question more on the unstable early structural state of the work, so the solver must reconstruct the composition history instead of simply following a rigid scaffold. Overall, these changes are justified because they remove or weaken highly distinctive cues such as the author name, exact year, work title, and canonical-status framing, while preserving enough non-shortcut constraints—late seventeenth-century natural-philosophical mathematics, eventual three-book structure, surviving fair-copy, planned second volume, and title/date/later-book correspondence—to keep the question uniquely solvable and faithful to the original answer chain.",
  "question": "The topic treated in the work shown in this image also appears in geometrical form in a late seventeenth-century mathematical work on natural philosophy that was not yet fixed in its eventual three-book arrangement when one surviving fair-copy draft was prepared. For the draft corresponding to what was then planned as its second volume, what title was intended, roughly when was it completed, and which later portion of the finished work did it largely match in purpose?"
}

Example 3:
question: The 2015 Copa Libertadores champion was an Argentine giant whose home ground is the Estadio Monumental. In River Plate’s 2024–25 squad list, the player wearing number 29 took the final penalty in the 2022 World Cup final. The provided photo shows Montiel taking that last penalty. Which side of the goal did he send the ball to?
Output:
{
  "analysis": "The question contains multiple overly revealing clues, as well as some redundancy. First, to ensure that the question remains solvable, we keep the Copa Libertadores as the entry point. Since the 2015 Copa Libertadores champion is already sufficient to identify the team as Argentina’s River Plate, the clue \"Estadio Monumental\" is redundant and makes the question easier. In addition, the team’s name is explicitly exposed in the second hop and should be removed. The description of the World Cup year can also be made vaguer; even after doing so, the relevant World Cup can still be identified through Montiel and the penalty he took. In the final sentence, Montiel is named directly, and because the image does not appear at the beginning of the question—that is, it is not actually provided to the user and must instead be located online by the user—this image cue should be hidden. The question should not explicitly mention the image, while still requiring the visual information from that specific image in order to answer correctly. In addition, the order of the clues can be adjusted and compressed to some extent."
  "question": "The number 29 player who was with the 2015 Copa Libertadores-winning club in 2024–25 once took the final penalty in a World Cup final. Which side of the goal did he aim at for that kick?"
}
Now perform the same style of difficulty enhancement on the following question.

Return valid JSON with exactly these fields:
{
  "analysis": "brief analysis of why the original question is too easy or too linear, and what was changed",
  "question": "the revised harder question"
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
    compress_hop_model_client: ModelWorkerClient | None = None
    compress_hop_model: str | None = None
    image_bridge_model_client: ModelWorkerClient | None = None
    image_bridge_model: str | None = None
    image_target_ask_model_client: ModelWorkerClient | None = None
    image_target_ask_model: str | None = None
    temperature: float | None = None
    max_tokens: int = 800
    json_retry_attempts: int = 2

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
        target_label = self._compress_hop_prompt_label(hop.dst_content, fallback=hop.dst_node_id)
        model_client = self.compress_hop_model_client or self.model_client
        model = self.compress_hop_model or self.model
        if model_client is None:
            return self._fallback_compress_hop(hop, source_label=source_label, target_label=target_label)
        prompt = self._compress_hop_prompt_payload(hop=hop)
        try:
            parsed = self._generate_json(
                system=self._compress_hop_prompt(hop=hop),
                user_payload=prompt,
                trace_label=f"compress_hop_{hop.hop_index}",
                model_client=model_client,
                model=model,
            )
        except Exception as exc:
            fallback = self._fallback_compress_hop(hop, source_label=source_label, target_label=target_label)
            fallback["writer_warning"] = self._writer_warning_entry(
                stage=f"compress_hop_{hop.hop_index}",
                error=exc,
            )
            return fallback
        statement = str(parsed.get("statement") or "").strip()
        source = source_label
        target = target_label
        relation = str(parsed.get("relation") or hop.relation or hop.edge_type or "").strip()
        retrieval_query = str(parsed.get("retrieval_query") or "").strip()
        if hop.src_modality == "text" and hop.dst_modality == "image":
            raw_retrieval_query = self._hop_image_retrieval_query(hop)
            if (
                not retrieval_query
                or (
                    raw_retrieval_query
                    and self._normalized_compact_text(retrieval_query) == self._normalized_compact_text(relation)
                )
            ):
                retrieval_query = raw_retrieval_query or retrieval_query
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

    @staticmethod
    def _compress_hop_prompt(*, hop: HopContext) -> str:
        hop_type = (hop.src_modality, hop.dst_modality)
        if hop_type == ("text", "text"):
            return PROMPT_COMPRESS_HOP_TEXT_TO_TEXT
        if hop_type == ("text", "image"):
            return PROMPT_COMPRESS_HOP_TEXT_TO_IMAGE
        if hop_type == ("image", "text"):
            return PROMPT_COMPRESS_HOP_IMAGE_TO_TEXT
        return PROMPT_COMPRESS_HOP_GENERIC

    @classmethod
    def _compress_hop_prompt_payload(cls, *, hop: HopContext) -> dict[str, Any]:
        payload = {
            "source": cls._compress_hop_prompt_label(hop.src_content, fallback=hop.src_node_id),
            "relation": str(hop.relation or hop.edge_type or "").strip(),
            "target": cls._compress_hop_prompt_label(hop.dst_content, fallback=hop.dst_node_id),
        }
        if (hop.src_modality, hop.dst_modality) not in {("text", "text"), ("text", "image"), ("image", "text")}:
            payload["hop_type"] = f"{hop.src_modality}->{hop.dst_modality}"
        return payload

    def select_target_ask(self, *, context: WriterContext) -> dict[str, Any]:
        if self.model_client is None:
            return self._fallback_select_target(context.target_node)
        target_node_type = str(context.target_node.get("node_type") or "")
        system_prompt = PROMPT_SELECT_IMAGE_TARGET if target_node_type == "image" else PROMPT_SELECT_TEXT_TARGET
        target_image_url = self._target_image_url(context.target_node)
        try:
            parsed = self._generate_json(
                system=system_prompt,
                user_payload={"target_node": context.target_node},
                trace_label=f"select_target_ask_{target_node_type or 'unknown'}",
                image_url=target_image_url,
            )
        except Exception as exc:
            fallback = self._fallback_select_target(context.target_node)
            fallback["writer_warning"] = self._writer_warning_entry(
                stage=f"select_target_ask_{target_node_type or 'unknown'}",
                error=exc,
            )
            return fallback
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
            return self._fallback_opening_package(
                source_node=source_node,
                first_hop_summary=first_hop_summary,
                forbidden_labels=forbidden_labels,
            )

        try:
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
        except Exception as exc:
            fallback = self._fallback_opening_package(
                source_node=source_node,
                first_hop_summary=first_hop_summary,
                forbidden_labels=forbidden_labels,
            )
            fallback["writer_warning"] = self._writer_warning_entry(stage="select_opening_package", error=exc)
            return fallback
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
        raw_hop_summaries = self._compress_hops(context.hops)
        opening_mode = "image_start" if path.trajectory.starts_with_image else "text_start"
        question_hop_summaries, image_bridge_normalization = self._normalize_question_hops(
            path=path,
            context=context,
            hop_summaries=raw_hop_summaries,
        )
        opening_package = self.select_opening_package(
            context=context,
            hop_summaries=question_hop_summaries,
        )
        raw_target_ask = self.select_target_ask(context=context)
        question_hop_summaries, question_target_ask, question_terminal_bridge, image_target_terminal_normalization = self._normalize_question_terminal_step(
            path=path,
            context=context,
            hop_summaries=question_hop_summaries,
            raw_target_ask=raw_target_ask,
        )
        draft_warnings = self._collect_writer_warnings(
            raw_hop_summaries,
            question_hop_summaries,
            image_bridge_normalization,
            opening_package,
            raw_target_ask,
            image_target_terminal_normalization,
        )
        answer_type = self._default_answer_type(context.target_node)
        starting_image_url = self._starting_image_url(path=path, graph=graph)
        if self.model_client is None:
            return self._draft_with_writer_warnings(self._fallback_compose_question(
                path=path,
                hop_summaries=question_hop_summaries,
                opening_package=opening_package,
                target_ask=question_target_ask,
                opening_mode=opening_mode,
                answer_type=answer_type,
                raw_target_ask=raw_target_ask,
                raw_hop_summaries=raw_hop_summaries,
                image_bridge_normalization=image_bridge_normalization,
                image_target_terminal_normalization=image_target_terminal_normalization,
                question_terminal_bridge=question_terminal_bridge,
                starting_image_url=starting_image_url,
                writer_context=context.to_dict(),
            ), warnings=draft_warnings)
        compose_hops = question_hop_summaries
        compose_payload = self._compose_question_payload(
            opening_mode=opening_mode,
            opening_package=opening_package,
            hop_summaries=compose_hops,
            target_ask=question_target_ask,
        )
        try:
            parsed = self._generate_json(
                system=PROMPT_COMPOSE_QUESTION,
                user_payload=compose_payload,
                trace_label="compose_question",
                image_url=starting_image_url,
            )
        except Exception as exc:
            draft_warnings.append(self._writer_warning_entry(stage="compose_question", error=exc))
            return self._draft_with_writer_warnings(
                self._fallback_compose_question(
                    path=path,
                    hop_summaries=question_hop_summaries,
                    opening_package=opening_package,
                    target_ask=question_target_ask,
                    opening_mode=opening_mode,
                    answer_type=answer_type,
                    raw_target_ask=raw_target_ask,
                    raw_hop_summaries=raw_hop_summaries,
                    image_bridge_normalization=image_bridge_normalization,
                    image_target_terminal_normalization=image_target_terminal_normalization,
                    question_terminal_bridge=question_terminal_bridge,
                    starting_image_url=starting_image_url,
                    writer_context=context.to_dict(),
                ),
                warnings=draft_warnings,
            )
        question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        answer = str(raw_target_ask.get("answer") or "").strip()
        if (
            not question
            or not answer
            or self._looks_like_chain_narration(question)
            or self._contains_forbidden_label(question, opening_package.get("forbidden_labels") or [])
        ):
            try:
                rewritten = self._rewrite_chain_narration(
                    opening_mode=opening_mode,
                    hop_summaries=compose_hops,
                    opening_package=opening_package,
                    target_ask=question_target_ask,
                )
            except Exception as exc:
                draft_warnings.append(self._writer_warning_entry(stage="rewrite_chain_narration", error=exc))
                rewritten = None
            if rewritten is not None:
                question = rewritten
        if (
            not question
            or not answer
            or self._contains_forbidden_label(question, opening_package.get("forbidden_labels") or [])
        ):
            return self._draft_with_writer_warnings(self._fallback_compose_question(
                path=path,
                hop_summaries=question_hop_summaries,
                opening_package=opening_package,
                target_ask=question_target_ask,
                opening_mode=opening_mode,
                answer_type=answer_type,
                raw_target_ask=raw_target_ask,
                raw_hop_summaries=raw_hop_summaries,
                image_bridge_normalization=image_bridge_normalization,
                image_target_terminal_normalization=image_target_terminal_normalization,
                question_terminal_bridge=question_terminal_bridge,
                starting_image_url=starting_image_url,
                writer_context=context.to_dict(),
            ), warnings=draft_warnings)
        return self._draft_with_writer_warnings(QuestionDraft(
            question=question,
            answer=answer,
            answer_type=answer_type,
            reasoning_steps=question_hop_summaries,
            used_evidence_ids=[hop.edge_id for hop in context.hops],
            metadata={
                "path_id": path.path_id,
                "opening_package": opening_package,
                "compose_payload": compose_payload,
                "raw_hop_summaries": raw_hop_summaries,
                "image_bridge_normalization": image_bridge_normalization,
                "starting_image_url": starting_image_url,
                "target_ask": raw_target_ask,
                "question_target_ask": question_target_ask,
                "question_terminal_bridge": question_terminal_bridge,
                "image_target_terminal_normalization": image_target_terminal_normalization,
                "writer_context": context.to_dict(),
            },
        ), warnings=draft_warnings)

    def draft(self, *, path: PathCandidate, graph: GraphView) -> QuestionDraft:
        return self.compose_question(path=path, graph=graph)

    def polish(self, *, draft: QuestionDraft, path: PathCandidate, graph: GraphView) -> QuestionDraft:
        if self.model_client is None:
            return draft
        polish_payload = self._polish_question_payload(
            question=draft.question,
            hops=draft.reasoning_steps,
        )
        question_target_ask = draft.metadata.get("question_target_ask") or draft.metadata.get("target_ask") or {}
        obfuscation_payload = self._obfuscation_question_payload(
            question=draft.question,
            hops=draft.reasoning_steps,
            final_ask=str(question_target_ask.get("ask_target") or "").strip(),
        )
        starting_image_url = self._starting_image_url(path=path, graph=graph)
        diagnostics: dict[str, dict[str, Any]] = {}
        warnings: list[dict[str, Any]] = []
        subtask_specs = [
            ("entity_obfuscation", PROMPT_POLISH_ENTITY_OBFUSCATION, obfuscation_payload, False),
            # ("shortcut", PROMPT_POLISH_SHORTCUT, polish_payload, False),
            # ("ambiguity", PROMPT_POLISH_AMBIGUITY, polish_payload, False),
            ("redundancy", PROMPT_POLISH_REDUNDANCY, polish_payload, False),
            ("wording", PROMPT_POLISH_WORDING, polish_payload, True),
        ]
        if len(subtask_specs) <= 1:
            subtask_results = [
                self._run_polish_subtask(
                    task_name=task_name,
                    system_prompt=system_prompt,
                    payload=payload,
                    image_url=starting_image_url if use_starting_image else None,
                )
                for task_name, system_prompt, payload, use_starting_image in subtask_specs
            ]
        else:
            with ThreadPoolExecutor(max_workers=min(len(subtask_specs), 4)) as executor:
                subtask_results = list(
                    executor.map(
                        lambda spec: self._run_polish_subtask(
                            task_name=spec[0],
                            system_prompt=spec[1],
                            payload=spec[2],
                            image_url=starting_image_url if spec[3] else None,
                        ),
                        subtask_specs,
                    )
                )
        for result in subtask_results:
            task_name = str(result.get("task_name") or "")
            error = result.get("error")
            if error is not None:
                warnings.append(
                    {
                        "stage": f"polish_{task_name}",
                        "error_type": error.__class__.__name__,
                        "error": str(error),
                    }
                )
                continue
            parsed = result.get("parsed") or {}
            issues = str(parsed.get("issues") or "").strip()
            advice = str(parsed.get("advice") or parsed.get("adjust") or "").strip()
            has_feedback = self._has_effective_feedback(issues=issues, advice=advice)
            diagnostics[task_name] = {
                "issues": issues,
                "advice": advice,
                "has_feedback": has_feedback,
                "input_payload": dict(result.get("input_payload") or {}),
                "image_attached": bool(result.get("image_attached")),
                "raw": parsed,
            }
        effective_diagnostics = {
            name: item
            for name, item in diagnostics.items()
            if bool(item.get("has_feedback"))
        }
        metadata = dict(draft.metadata)
        existing_warnings = list(metadata.get("writer_warnings") or [])
        existing_warnings.extend(warnings)
        if existing_warnings:
            metadata["writer_warnings"] = existing_warnings
        metadata["polish_payload"] = polish_payload
        metadata["polish_starting_image_url"] = starting_image_url
        metadata["polish_subtasks"] = diagnostics
        if not effective_diagnostics:
            metadata["polish_rewrite_skipped"] = True
            metadata["polish_rewrite_skip_reason"] = "no_subtask_feedback"
            metadata["polish_result"] = {
                "raw_response": None,
                "question": draft.question,
                "rewrite_skipped": True,
            }
            return QuestionDraft(
                question=draft.question,
                answer=draft.answer,
                answer_type=draft.answer_type,
                reasoning_steps=list(draft.reasoning_steps),
                used_evidence_ids=list(draft.used_evidence_ids),
                metadata=metadata,
            )
        rewrite_payload = {
            "question": draft.question,
            "hops": polish_payload.get("hops") or [],
            "starting_image_attached": bool(starting_image_url),
            "diagnostics": {
                name: {
                    "issues": item.get("issues") or "",
                    "advice": item.get("advice") or "",
                }
                for name, item in effective_diagnostics.items()
            },
        }
        try:
            parsed = self._generate_json(
                system=PROMPT_POLISH_REWRITE,
                user_payload=rewrite_payload,
                trace_label="polish_rewrite",
                image_url=starting_image_url,
            )
        except Exception as exc:
            warning_draft = self._record_writer_warning(draft, stage="polish_rewrite_request", error=exc)
            if warnings:
                warning_metadata = dict(warning_draft.metadata)
                combined_warnings = list(warning_metadata.get("writer_warnings") or [])
                combined_warnings.extend(warnings)
                warning_metadata["writer_warnings"] = combined_warnings
                warning_draft = QuestionDraft(
                    question=warning_draft.question,
                    answer=warning_draft.answer,
                    answer_type=warning_draft.answer_type,
                    reasoning_steps=list(warning_draft.reasoning_steps),
                    used_evidence_ids=list(warning_draft.used_evidence_ids),
                    metadata=warning_metadata,
                )
            return warning_draft
        polished_question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        if not polished_question:
            warning_draft = self._record_writer_warning(
                draft,
                stage="polish_rewrite_parse",
                error=ValueError("Model returned an empty rewritten question."),
            )
            if warnings:
                warning_metadata = dict(warning_draft.metadata)
                combined_warnings = list(warning_metadata.get("writer_warnings") or [])
                combined_warnings.extend(warnings)
                warning_metadata["writer_warnings"] = combined_warnings
                warning_draft = QuestionDraft(
                    question=warning_draft.question,
                    answer=warning_draft.answer,
                    answer_type=warning_draft.answer_type,
                    reasoning_steps=list(warning_draft.reasoning_steps),
                    used_evidence_ids=list(warning_draft.used_evidence_ids),
                    metadata=warning_metadata,
                )
            return warning_draft
        metadata["polish_rewrite_payload"] = rewrite_payload
        metadata["polish_rewrite_skipped"] = False
        metadata["polish_result"] = {
            "raw_response": parsed,
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
        metadata = dict(draft.metadata)
        metadata["obfuscation_result"] = {
            "raw_response": None,
            "reason": "Integrated into polish subtasks and aggregate rewrite.",
            "question": draft.question,
            "integrated_in_polish": True,
        }
        return QuestionDraft(
            question=draft.question,
            answer=draft.answer,
            answer_type=draft.answer_type,
            reasoning_steps=list(draft.reasoning_steps),
            used_evidence_ids=list(draft.used_evidence_ids),
            metadata=metadata,
        )

    def enhance_difficulty(self, *, draft: QuestionDraft, path: PathCandidate, graph: GraphView) -> QuestionDraft:
        if self.model_client is None:
            return draft
        starting_image_url = self._starting_image_url(path=path, graph=graph)
        return self._enhance_difficulty_with_image(draft=draft, starting_image_url=starting_image_url)

    def enhance_difficulty_direct(
        self,
        *,
        draft: QuestionDraft,
        starting_image_url: str | None = None,
    ) -> QuestionDraft:
        if self.model_client is None:
            return draft
        return self._enhance_difficulty_with_image(draft=draft, starting_image_url=starting_image_url)

    def _enhance_difficulty_with_image(
        self,
        *,
        draft: QuestionDraft,
        starting_image_url: str | None,
    ) -> QuestionDraft:
        payload = self._difficulty_enhancement_payload(
            question=draft.question,
            hops=draft.reasoning_steps,
        )
        try:
            parsed = self._generate_json(
                system=PROMPT_DIFFICULTY_ENHANCEMENT,
                user_payload=payload,
                trace_label="difficulty_enhancement",
                image_url=starting_image_url,
            )
        except Exception as exc:
            return self._record_writer_warning(draft, stage="difficulty_enhancement_request", error=exc)
        enhanced_question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        if not enhanced_question:
            return self._record_writer_warning(
                draft,
                stage="difficulty_enhancement_parse",
                error=ValueError("Model returned an empty difficulty-enhanced question."),
            )
        metadata = dict(draft.metadata)
        metadata["difficulty_enhancement_payload"] = payload
        metadata["difficulty_enhancement_result"] = {
            "raw_response": parsed,
            "analysis": str(parsed.get("analysis") or "").strip(),
            "question": enhanced_question,
            "starting_image_url": starting_image_url,
        }
        return QuestionDraft(
            question=enhanced_question,
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
        warnings.append(QuestionWriter._writer_warning_entry(stage=stage, error=error))
        metadata["writer_warnings"] = warnings
        return QuestionDraft(
            question=draft.question,
            answer=draft.answer,
            answer_type=draft.answer_type,
            reasoning_steps=list(draft.reasoning_steps),
            used_evidence_ids=list(draft.used_evidence_ids),
            metadata=metadata,
        )

    @staticmethod
    def _writer_warning_entry(*, stage: str, error: Exception) -> dict[str, str]:
        return {
            "stage": stage,
            "error_type": error.__class__.__name__,
            "error": str(error),
        }

    @staticmethod
    def _collect_writer_warnings(*items: Any) -> list[dict[str, Any]]:
        warnings: list[dict[str, Any]] = []
        for item in items:
            if isinstance(item, list):
                for sub_item in item:
                    if isinstance(sub_item, dict):
                        warning = sub_item.get("writer_warning")
                        if isinstance(warning, dict):
                            warnings.append(dict(warning))
                continue
            if isinstance(item, dict):
                warning = item.get("writer_warning")
                if isinstance(warning, dict):
                    warnings.append(dict(warning))
        return warnings

    @staticmethod
    def _draft_with_writer_warnings(draft: QuestionDraft, *, warnings: list[dict[str, Any]]) -> QuestionDraft:
        if not warnings:
            return draft
        metadata = dict(draft.metadata)
        existing_warnings = list(metadata.get("writer_warnings") or [])
        existing_warnings.extend(dict(warning) for warning in warnings if isinstance(warning, dict))
        metadata["writer_warnings"] = existing_warnings
        return QuestionDraft(
            question=draft.question,
            answer=draft.answer,
            answer_type=draft.answer_type,
            reasoning_steps=list(draft.reasoning_steps),
            used_evidence_ids=list(draft.used_evidence_ids),
            metadata=metadata,
        )

    @staticmethod
    def _json_retry_system_prompt(system: str, *, attempt_index: int) -> str:
        if attempt_index <= 0:
            return system
        return (
            f"{system}\n\n"
            "CRITICAL OUTPUT FORMAT REMINDER:\n"
            "Return exactly one valid JSON object.\n"
            "Do not output markdown, YAML, bullet lists, tables, explanations, or any text before or after the JSON.\n"
            "All property names must use double quotes, and the JSON must be parseable by Python json.loads()."
        )

    @staticmethod
    def _has_effective_feedback(*, issues: str, advice: str) -> bool:
        normalized_issues = str(issues or "").strip().lower()
        normalized_advice = str(advice or "").strip().lower()
        no_issue_tokens = {"", "none.", "none"}
        no_advice_tokens = {"", "no change needed.", "no change needed"}
        return not (
            normalized_issues in no_issue_tokens
            and normalized_advice in no_advice_tokens
        )

    def _generate_json(
        self,
        *,
        system: str,
        user_payload: Any,
        trace_label: str,
        image_url: str | None = None,
        model_client: ModelWorkerClient | None = None,
        model: str | None = None,
    ) -> dict[str, Any]:
        active_model_client = model_client or self.model_client
        active_model = model or self.model
        if active_model_client is None:
            raise RuntimeError("model_client is required for _generate_json")
        attempts = max(1, int(self.json_retry_attempts) + 1)
        last_error: Exception | None = None
        for attempt_index in range(attempts):
            request = ModelRequest(
                model=active_model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"},
                messages=[
                    ModelMessage(
                        role="system",
                        content=self._json_retry_system_prompt(system, attempt_index=attempt_index),
                    ),
                    ModelMessage(
                        role="user",
                        content=self._user_message_content(user_payload, image_url=image_url),
                    ),
                ],
                metadata={
                    "trace_label": trace_label,
                    "json_attempt": attempt_index + 1,
                    "session_id": _VQA_FIXED_REQUEST_ID,
                    "prompt_cache_key": _VQA_FIXED_REQUEST_ID,
                    "user_id": _VQA_FIXED_REQUEST_ID,
                    "x_tt_logid": _VQA_FIXED_REQUEST_ID,
                },
            )
            try:
                response = active_model_client.generate(request)
                try:
                    parsed = json.loads(response.content)
                except json.JSONDecodeError:
                    parsed = self._extract_json_object(response.content)
                if not isinstance(parsed, dict):
                    raise ValueError(f"Expected JSON object for {trace_label}, got: {type(parsed)!r}")
                return parsed
            except Exception as exc:
                last_error = exc
        if last_error is None:
            raise RuntimeError(f"{trace_label} failed without a captured exception")
        raise last_error

    def _compress_hops(self, hops: list[HopContext]) -> list[dict[str, Any]]:
        if len(hops) <= 1:
            return [self.compress_hop(hop=hop) for hop in hops]
        with ThreadPoolExecutor(max_workers=min(len(hops), 8)) as executor:
            return list(executor.map(lambda hop: self.compress_hop(hop=hop), hops))

    def _normalize_question_hops(
        self,
        *,
        path: PathCandidate,
        context: WriterContext,
        hop_summaries: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if not context.hops or not hop_summaries:
            return list(hop_summaries), []

        normalized: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        hop_index = 0
        while hop_index < len(context.hops):
            incoming_hop = context.hops[hop_index]
            incoming_summary = hop_summaries[hop_index]

            if hop_index + 1 < len(context.hops):
                outgoing_hop = context.hops[hop_index + 1]
                outgoing_summary = hop_summaries[hop_index + 1]
                if self._can_normalize_hidden_image_bridge(
                    path=path,
                    incoming_hop=incoming_hop,
                    outgoing_hop=outgoing_hop,
                ):
                    synthetic_summary, diagnostic = self._normalize_image_bridge(
                        incoming_hop=incoming_hop,
                        incoming_summary=incoming_summary,
                        outgoing_hop=outgoing_hop,
                        outgoing_summary=outgoing_summary,
                    )
                    diagnostics.append(diagnostic)
                    if synthetic_summary is not None:
                        normalized.append(synthetic_summary)
                        hop_index += 2
                        continue

            normalized.append(incoming_summary)
            hop_index += 1

        return normalized, diagnostics

    @classmethod
    def _can_normalize_hidden_image_bridge(
        cls,
        *,
        path: PathCandidate,
        incoming_hop: HopContext,
        outgoing_hop: HopContext,
    ) -> bool:
        if (incoming_hop.src_modality, incoming_hop.dst_modality) != ("text", "image"):
            return False
        if (outgoing_hop.src_modality, outgoing_hop.dst_modality) != ("image", "text"):
            return False
        if incoming_hop.dst_node_id != outgoing_hop.src_node_id:
            return False
        if cls._is_image_node_visible_in_question(path=path, image_node_id=incoming_hop.dst_node_id):
            return False
        return True

    @staticmethod
    def _is_image_node_visible_in_question(*, path: PathCandidate, image_node_id: str) -> bool:
        return bool(
            path.trajectory.starts_with_image
            and path.node_ids
            and path.node_ids[0] == image_node_id
        )

    def _normalize_image_bridge(
        self,
        *,
        incoming_hop: HopContext,
        incoming_summary: dict[str, Any],
        outgoing_hop: HopContext,
        outgoing_summary: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        image_content = incoming_hop.dst_content or outgoing_hop.src_content or {}
        image_label = self._compress_hop_prompt_label(image_content, fallback=incoming_hop.dst_node_id)
        model_client = self.image_bridge_model_client or self.model_client
        model = self.image_bridge_model or self.model
        diagnostic: dict[str, Any] = {
            "incoming_hop_index": incoming_hop.hop_index,
            "outgoing_hop_index": outgoing_hop.hop_index,
            "image_node_id": incoming_hop.dst_node_id,
            "image_label": image_label,
            "model_alias": model,
            "decision": "keep_image",
            "reason": "no_image_bridge_model_available",
            "applied": False,
        }
        if model_client is None or not model:
            return None, diagnostic

        image_url = self._target_image_url(image_content)
        trace_label = f"normalize_image_bridge_{incoming_hop.hop_index}_{outgoing_hop.hop_index}"
        try:
            parsed = self._generate_json(
                system=PROMPT_NORMALIZE_IMAGE_BRIDGE,
                user_payload=self._image_bridge_prompt_text(
                    incoming_hop=incoming_hop,
                    incoming_summary=incoming_summary,
                    outgoing_hop=outgoing_hop,
                    outgoing_summary=outgoing_summary,
                ),
                trace_label=trace_label,
                image_url=image_url,
                model_client=model_client,
                model=model,
            )
        except Exception as exc:
            diagnostic["reason"] = "image_bridge_model_error"
            diagnostic["writer_warning"] = self._writer_warning_entry(stage=trace_label, error=exc)
            return None, diagnostic

        decision = str(parsed.get("decision") or "").strip().lower()
        reason = str(parsed.get("reason") or "").strip()
        rewritten_statement = self._ensure_declarative_statement(
            str(parsed.get("rewritten_statement") or "").strip()
        )
        rewritten_relation = str(parsed.get("rewritten_relation") or "").strip()
        if decision not in {"hide_image", "keep_image"}:
            decision = "keep_image"
            if not reason:
                reason = "unexpected_model_decision"

        diagnostic["decision"] = decision
        diagnostic["reason"] = reason or ("hide_image" if decision == "hide_image" else "keep_image")
        if not rewritten_statement:
            fallback_statement, fallback_relation = self._fallback_merge_image_bridge(
                incoming_summary=incoming_summary,
                outgoing_summary=outgoing_summary,
                hide_image=(decision == "hide_image"),
            )
            rewritten_statement = self._ensure_declarative_statement(fallback_statement)
            rewritten_relation = rewritten_relation or fallback_relation
            if not rewritten_statement:
                diagnostic["decision"] = "keep_image"
                diagnostic["reason"] = reason or "empty_rewritten_statement"
                return None, diagnostic
            diagnostic["fallback_merged_statement_used"] = True

        synthetic_summary = {
            "hop_index": incoming_summary.get("hop_index"),
            "statement": rewritten_statement,
            "source": incoming_summary.get("source"),
            "target": outgoing_summary.get("target"),
            "relation": rewritten_relation or str(outgoing_summary.get("relation") or "").strip(),
            "retrieval_query": (
                str(incoming_summary.get("retrieval_query") or "").strip()
                if decision == "keep_image"
                else ""
            ),
            "edge_id": "|".join(
                item
                for item in (
                    str(incoming_summary.get("edge_id") or "").strip(),
                    str(outgoing_summary.get("edge_id") or "").strip(),
                )
                if item
            ),
            "src_node_id": incoming_summary.get("src_node_id"),
            "dst_node_id": outgoing_summary.get("dst_node_id"),
            "image_bridge_hidden": decision == "hide_image",
            "image_bridge_decision": decision,
            "bridge_image_node_id": incoming_hop.dst_node_id,
        }
        if decision == "hide_image":
            synthetic_summary["hidden_image_node_id"] = incoming_hop.dst_node_id
        diagnostic["applied"] = True
        diagnostic["rewritten_statement"] = rewritten_statement
        return synthetic_summary, diagnostic

    @staticmethod
    def _fallback_merge_image_bridge(
        *,
        incoming_summary: dict[str, Any],
        outgoing_summary: dict[str, Any],
        hide_image: bool,
    ) -> tuple[str, str]:
        incoming_statement = str(incoming_summary.get("statement") or "").strip()
        outgoing_statement = str(outgoing_summary.get("statement") or "").strip()
        if hide_image:
            merged_statement = outgoing_statement or incoming_statement
            merged_relation = str(outgoing_summary.get("relation") or "").strip()
        else:
            merged_statement = " ".join(item for item in (incoming_statement, outgoing_statement) if item)
            merged_relation = str(outgoing_summary.get("relation") or incoming_summary.get("relation") or "").strip()
        return merged_statement, merged_relation

    @classmethod
    def _image_bridge_prompt_text(
        cls,
        *,
        incoming_hop: HopContext,
        incoming_summary: dict[str, Any],
        outgoing_hop: HopContext,
        outgoing_summary: dict[str, Any],
    ) -> str:
        image_content = incoming_hop.dst_content or outgoing_hop.src_content or {}
        source = incoming_summary.get("source") or cls._compress_hop_prompt_label(
            incoming_hop.src_content,
            fallback=incoming_hop.src_node_id,
        )
        target = outgoing_summary.get("target") or cls._compress_hop_prompt_label(
            outgoing_hop.dst_content,
            fallback=outgoing_hop.dst_node_id,
        )
        lines = [
            f"statement1: {cls._prompt_text_value(incoming_summary.get('statement') or '')}",
            f"statement2: {cls._prompt_text_value(outgoing_summary.get('statement') or '')}",
            f"source: {cls._prompt_text_value(source)}",
            f"mid-image: {cls._mid_image_prompt_value(image_content, fallback=incoming_hop.dst_node_id)}",
            f"target: {cls._prompt_text_value(target)}",
        ]
        return "\n".join(lines)

    def _normalize_question_terminal_step(
        self,
        *,
        path: PathCandidate,
        context: WriterContext,
        hop_summaries: list[dict[str, Any]],
        raw_target_ask: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
        question_hop_summaries = list(hop_summaries)
        question_target_ask = dict(raw_target_ask)
        if context.target_node.get("node_type") != "image":
            return question_hop_summaries, question_target_ask, None, None

        final_hop = context.hops[-1] if context.hops else None
        if final_hop is None or not question_hop_summaries:
            diagnostic = {
                "hop_index": final_hop.hop_index if final_hop is not None else None,
                "image_node_id": str(context.target_node.get("node_id") or ""),
                "decision": "keep_image",
                "reason": "missing_final_hop",
                "applied": False,
                "raw_ask_target": str(raw_target_ask.get("ask_target") or "").strip(),
            }
            return question_hop_summaries, question_target_ask, None, diagnostic
        if (final_hop.src_modality, final_hop.dst_modality) != ("text", "image"):
            diagnostic = {
                "hop_index": final_hop.hop_index,
                "image_node_id": final_hop.dst_node_id,
                "decision": "keep_image",
                "reason": f"unsupported_final_hop_type:{final_hop.src_modality}->{final_hop.dst_modality}",
                "applied": False,
                "raw_ask_target": str(raw_target_ask.get("ask_target") or "").strip(),
            }
            return question_hop_summaries, question_target_ask, None, diagnostic
        if self._is_image_node_visible_in_question(path=path, image_node_id=final_hop.dst_node_id):
            diagnostic = {
                "hop_index": final_hop.hop_index,
                "image_node_id": final_hop.dst_node_id,
                "decision": "keep_image",
                "reason": "target_image_visible_in_question",
                "applied": False,
                "raw_ask_target": str(raw_target_ask.get("ask_target") or "").strip(),
            }
            return question_hop_summaries, question_target_ask, None, diagnostic

        final_hop_summary = hop_summaries[-1]
        question_target_ask, question_terminal_bridge, diagnostic = self._normalize_final_image_target_terminal(
            final_hop=final_hop,
            final_hop_summary=final_hop_summary,
            raw_target_ask=raw_target_ask,
        )
        diagnostic["question_target_ask"] = dict(question_target_ask)
        if not diagnostic.get("applied"):
            return question_hop_summaries, question_target_ask, question_terminal_bridge, diagnostic

        updated_hops = list(question_hop_summaries[:-1])
        diagnostic["question_hop_count_before"] = len(question_hop_summaries)
        diagnostic["question_hop_count_after"] = len(updated_hops)
        diagnostic["removed_question_hop"] = {
            key: final_hop_summary.get(key)
            for key in (
                "hop_index",
                "source",
                "target",
                "statement",
                "relation",
                "retrieval_query",
                "edge_id",
                "src_node_id",
                "dst_node_id",
            )
        }
        return updated_hops, question_target_ask, question_terminal_bridge, diagnostic

    def _normalize_final_image_target_terminal(
        self,
        *,
        final_hop: HopContext,
        final_hop_summary: dict[str, Any],
        raw_target_ask: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any]]:
        image_content = final_hop.dst_content or {}
        image_label = self._compress_hop_prompt_label(image_content, fallback=final_hop.dst_node_id)
        model_client = self.image_target_ask_model_client or self.image_bridge_model_client or self.model_client
        model = self.image_target_ask_model or self.image_bridge_model or self.model
        question_target_ask = dict(raw_target_ask)
        target_value = str(raw_target_ask.get("answer") or final_hop_summary.get("target") or "").strip()
        diagnostic: dict[str, Any] = {
            "hop_index": final_hop.hop_index,
            "image_node_id": final_hop.dst_node_id,
            "image_label": image_label,
            "model_alias": model,
            "decision": "keep_image",
            "reason": "no_image_target_ask_model_available",
            "applied": False,
            "raw_ask_target": str(raw_target_ask.get("ask_target") or "").strip(),
            "raw_answer": target_value,
        }
        if model_client is None or not model:
            return question_target_ask, None, diagnostic

        image_url = self._target_image_url(image_content)
        trace_label = f"normalize_image_target_terminal_{final_hop.hop_index}"
        try:
            parsed = self._generate_json(
                system=PROMPT_NORMALIZE_FINAL_IMAGE_TARGET_ASK,
                user_payload=self._image_target_terminal_prompt_text(
                    final_hop=final_hop,
                    final_hop_summary=final_hop_summary,
                    raw_target_ask=raw_target_ask,
                ),
                trace_label=trace_label,
                image_url=image_url,
                model_client=model_client,
                model=model,
            )
        except Exception as exc:
            diagnostic["reason"] = "image_target_terminal_model_error"
            diagnostic["writer_warning"] = self._writer_warning_entry(stage=trace_label, error=exc)
            return question_target_ask, None, diagnostic

        decision = str(parsed.get("decision") or "").strip().lower()
        reason = str(parsed.get("reason") or "").strip()
        rewritten_ask_target = self._ensure_question(str(parsed.get("rewritten_ask_target") or "").strip())
        if decision not in {"hide_image", "keep_image"}:
            decision = "keep_image"
            if not reason:
                reason = "unexpected_model_decision"

        diagnostic["decision"] = decision
        diagnostic["reason"] = reason or ("hide_image" if decision == "hide_image" else "keep_image")
        if not rewritten_ask_target:
            if not reason:
                diagnostic["reason"] = "empty_rewritten_ask_target"
            return question_target_ask, None, diagnostic

        question_target_ask["ask_target"] = rewritten_ask_target
        question_terminal_bridge = self._build_final_image_target_terminal_bridge(
            final_hop=final_hop,
            final_hop_summary=final_hop_summary,
            raw_target_ask=raw_target_ask,
            target_value=target_value,
            rewritten_ask_target=rewritten_ask_target,
            decision=decision,
        )
        diagnostic["applied"] = True
        diagnostic["rewritten_ask_target"] = rewritten_ask_target
        diagnostic["question_terminal_bridge"] = dict(question_terminal_bridge)
        return question_target_ask, question_terminal_bridge, diagnostic

    @staticmethod
    def _build_final_image_target_terminal_bridge(
        *,
        final_hop: HopContext,
        final_hop_summary: dict[str, Any],
        raw_target_ask: dict[str, Any],
        target_value: str,
        rewritten_ask_target: str,
        decision: str,
    ) -> dict[str, Any]:
        bridge = {
            "terminal_question_bridge": True,
            "terminal_bridge_decision": decision,
            "terminal_image_node_id": final_hop.dst_node_id,
            "replaces_terminal_text_to_image_hop": True,
            "source": final_hop_summary.get("source"),
            "target_image": final_hop_summary.get("target"),
            "answer": target_value or str(raw_target_ask.get("answer") or "").strip(),
            "raw_ask_target": str(raw_target_ask.get("ask_target") or "").strip(),
            "rewritten_ask_target": rewritten_ask_target,
            "removed_question_hop": {
                key: final_hop_summary.get(key)
                for key in (
                    "hop_index",
                    "source",
                    "target",
                    "statement",
                    "relation",
                    "retrieval_query",
                    "edge_id",
                    "src_node_id",
                    "dst_node_id",
                )
            },
        }
        if decision == "hide_image":
            bridge["hidden_image_node_id"] = final_hop.dst_node_id
        return bridge

    @classmethod
    def _image_target_terminal_prompt_text(
        cls,
        *,
        final_hop: HopContext,
        final_hop_summary: dict[str, Any],
        raw_target_ask: dict[str, Any],
    ) -> str:
        image_content = final_hop.dst_content or {}
        source = final_hop_summary.get("source") or cls._compress_hop_prompt_label(
            final_hop.src_content,
            fallback=final_hop.src_node_id,
        )
        answer = cls._prompt_text_value(raw_target_ask.get("answer") or "")
        lines = [
            f"statement1: {cls._prompt_text_value(final_hop_summary.get('statement') or '')}",
            f"question: {cls._prompt_text_value(raw_target_ask.get('ask_target') or '')}",
            f"answer: {answer}",
            f"source: {cls._prompt_text_value(source)}",
            f"mid-image: {cls._mid_image_prompt_value(image_content, fallback=final_hop.dst_node_id)}",
        ]
        return "\n".join(lines)

    def _run_polish_subtask(
        self,
        *,
        task_name: str,
        system_prompt: str,
        payload: dict[str, Any],
        image_url: str | None,
    ) -> dict[str, Any]:
        try:
            parsed = self._generate_json(
                system=system_prompt,
                user_payload=payload,
                trace_label=f"polish_{task_name}",
                image_url=image_url,
            )
            return {
                "task_name": task_name,
                "input_payload": payload,
                "image_attached": bool(image_url),
                "parsed": parsed,
                "error": None,
            }
        except Exception as exc:
            return {
                "task_name": task_name,
                "input_payload": payload,
                "image_attached": bool(image_url),
                "parsed": None,
                "error": exc,
            }

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
        user_payload: Any,
        *,
        image_url: str | None = None,
    ) -> str | list[dict[str, Any]]:
        if isinstance(user_payload, str):
            prompt_text = user_payload
        else:
            prompt_text = json.dumps(user_payload, ensure_ascii=False, indent=2)
        if not image_url:
            return prompt_text
        resolved_image_url = QuestionWriter._resolve_multimodal_image_url(image_url)
        return [
            {"type": "text", "text": prompt_text},
            {"type": "image_url", "image_url": {"url": resolved_image_url}},
        ]

    @staticmethod
    def _resolve_multimodal_image_url(image_url: str) -> str:
        normalized = str(image_url or "").strip()
        if not normalized:
            return normalized
        lowered = normalized.lower()
        if lowered.startswith(("http://", "https://", "data:")):
            return normalized
        if lowered.startswith("file://"):
            local_path = Path(normalized[7:])
        else:
            local_path = Path(normalized)
        if not local_path.exists() or not local_path.is_file():
            return normalized
        mime_type, _ = mimetypes.guess_type(local_path.name)
        if not mime_type:
            mime_type = "application/octet-stream"
        encoded = base64.b64encode(local_path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

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
        node_source = node.get("source") or {}
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
            payload["source_type"] = (
                node_source.get("source_type") if isinstance(node_source, dict) else None
            )
            payload["image_origin"] = metadata.get("image_origin") if isinstance(metadata, dict) else None
            payload["source_text_node_id"] = (
                metadata.get("source_text_node_id") if isinstance(metadata, dict) else None
            )
            payload["visual_target"] = metadata.get("visual_target") if isinstance(metadata, dict) else None
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

    @staticmethod
    def _fallback_hide_image_terminal_ask(text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return ""
        rewritten = normalized
        removal_patterns = [
            r"\b(?:shown|visible|seen)\s+in\s+(?:the|this|that)\s+(?:image|photo|picture)\b",
            r"\bin\s+(?:the|this|that)\s+(?:image|photo|picture)\b",
            r"\bfrom\s+(?:the|this|that)\s+(?:image|photo|picture)\b",
        ]
        for pattern in removal_patterns:
            rewritten = re.sub(pattern, "", rewritten, flags=re.IGNORECASE)
        rewritten = re.sub(r"\s+", " ", rewritten).strip(" ,")
        rewritten = re.sub(r"\s+([?.!,;:])", r"\1", rewritten)
        return QuestionWriter._ensure_question(rewritten or normalized)

    @staticmethod
    def _ensure_declarative_statement(text: str) -> str:
        normalized = re.sub(r"\s+", " ", str(text or "")).strip()
        if not normalized:
            return ""
        if normalized.endswith("?"):
            normalized = normalized.rstrip("?").rstrip()
        if not re.search(r"[.?!]$", normalized):
            normalized = normalized.rstrip(" ,;:") + "."
        return normalized

    @staticmethod
    def _normalized_compact_text(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()

    @staticmethod
    def _strip_image_title_prefix(text: Any) -> str:
        normalized = str(text or "").strip()
        if not normalized:
            return ""
        prefix, sep, remainder = normalized.partition(":")
        if sep and prefix.strip().lower() == "image":
            return remainder.strip()
        return normalized


    @staticmethod
    def _prompt_text_value(text: Any) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip()

    @classmethod
    def _mid_image_prompt_value(cls, content: dict[str, Any], *, fallback: str) -> str:
        label = cls._strip_image_title_prefix(
            cls._prompt_text_value(cls._compress_hop_prompt_label(content, fallback=fallback))
        )
        if not label:
            label = cls._prompt_text_value(fallback)
        return f"Image: {label}" if label else "Image"

    @classmethod
    def _image_search_query(cls, content: dict[str, Any]) -> str:
        title = cls._strip_image_title_prefix(content.get("title"))
        search_query = str(content.get("search_query") or "").strip()
        if title and search_query and cls._normalized_compact_text(title) == cls._normalized_compact_text(search_query):
            return title
        return search_query or title

    @classmethod
    def _hop_image_retrieval_query(cls, hop: HopContext) -> str:
        if hop.dst_modality != "image":
            return ""
        return cls._image_search_query(hop.dst_content)

    @staticmethod
    def _looks_like_image_phrase(text: str) -> bool:
        normalized = str(text or "").strip().lower()
        if not normalized:
            return False
        return normalized.startswith(
            (
                "photo ",
                "photo of",
                "portrait ",
                "portrait of",
                "painting ",
                "painting of",
                "poster ",
                "poster of",
                "cover ",
                "cover of",
                "map ",
                "map of",
                "screenshot ",
                "screenshot of",
                "image ",
                "image of",
            )
        )

    @classmethod
    def _image_target_phrase(cls, target_label: str) -> str:
        normalized = str(target_label or "").strip()
        if not normalized:
            return ""
        lowered = normalized.lower()
        if lowered.startswith(("a ", "an ", "the ", "this ", "that ")):
            return normalized
        return f"a {normalized}"

    @classmethod
    def _compress_hop_prompt_label(cls, content: dict[str, Any], *, fallback: str) -> str:
        if content.get("node_type") == "image":
            search_query = cls._shorten_text(cls._image_search_query(content), limit=180)
            if search_query:
                return search_query
            caption = cls._shorten_text(content.get("caption"), limit=180)
            if caption:
                return caption
        title = cls._strip_image_title_prefix(content.get("title"))
        if title:
            return title
        return str(content.get("caption") or fallback)

    @classmethod
    def _hop_anchor_label(cls, content: dict[str, Any], *, fallback: str) -> str:
        if content.get("node_type") == "image":
            search_query = cls._shorten_text(cls._image_search_query(content), limit=180)
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

    @staticmethod
    def _difficulty_enhancement_payload(*, question: str, hops: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "question": question,
            "hops": [
                {
                    "hop_index": item.get("hop_index"),
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "statement": item.get("statement"),
                    "relation": item.get("relation"),
                    "retrieval_query": item.get("retrieval_query"),
                }
                for item in hops
            ],
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
                return (
                    f"An image associated with {source_clue} can be located using the clue "
                    f"\"{cleaned_query}\"."
                )
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

    def _fallback_opening_package(
        self,
        *,
        source_node: dict[str, Any],
        first_hop_summary: dict[str, Any],
        forbidden_labels: list[str],
    ) -> dict[str, Any]:
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
        target_label = target_label or QuestionWriter._compress_hop_prompt_label(hop.dst_content, fallback=hop.dst_node_id)
        if hop.src_modality == "text" and hop.dst_modality == "text":
            statement = f"{source_label} {relation} {target_label}".strip()
        elif hop.src_modality == "text" and hop.dst_modality == "image":
            retrieval_query = QuestionWriter._hop_image_retrieval_query(hop)
            if relation and not QuestionWriter._looks_like_image_phrase(relation):
                statement = f"{source_label} is related to a photo that shows {relation}."
            elif target_label:
                statement = f"{source_label} is related to {QuestionWriter._image_target_phrase(target_label)}."
            elif retrieval_query:
                statement = (
                    f"{source_label} is related to a photo that can be located using the clue: "
                    f"{retrieval_query}."
                )
            elif relation:
                statement = f"{source_label} is related to {relation}."
            else:
                statement = f"{source_label} is related to an unspecified photo target."
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
        raw_target_ask: dict[str, Any] | None = None,
        raw_hop_summaries: list[dict[str, Any]] | None = None,
        image_bridge_normalization: list[dict[str, Any]] | None = None,
        image_target_terminal_normalization: dict[str, Any] | None = None,
        question_terminal_bridge: dict[str, Any] | None = None,
        starting_image_url: str | None = None,
        writer_context: dict[str, Any] | None = None,
    ) -> QuestionDraft:
        raw_target_ask = dict(raw_target_ask or target_ask)
        question_target_ask = dict(target_ask)
        remaining_hops = hop_summaries[1:] if opening_package.get("packaged_first_hop") else hop_summaries
        hop_text = " ".join(item.get("statement", "") for item in remaining_hops if item.get("statement"))
        ask_target = str(question_target_ask.get("ask_target") or "What is the final answer?")
        answer = str(raw_target_ask.get("answer") or question_target_ask.get("answer") or "unknown")
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
        metadata: dict[str, Any] = {
            "opening_package": opening_package,
            "target_ask": raw_target_ask,
            "question_target_ask": question_target_ask,
        }
        if raw_hop_summaries is not None:
            metadata["raw_hop_summaries"] = raw_hop_summaries
        if image_bridge_normalization is not None:
            metadata["image_bridge_normalization"] = image_bridge_normalization
        if image_target_terminal_normalization is not None:
            metadata["image_target_terminal_normalization"] = image_target_terminal_normalization
        if question_terminal_bridge is not None:
            metadata["question_terminal_bridge"] = question_terminal_bridge
        if starting_image_url:
            metadata["starting_image_url"] = starting_image_url
        if writer_context is not None:
            metadata["writer_context"] = writer_context
        return QuestionDraft(
            question=question,
            answer=answer,
            answer_type=answer_type,
            reasoning_steps=hop_summaries,
            used_evidence_ids=[
                item.get("edge_id", "")
                for item in (raw_hop_summaries or hop_summaries)
                if item.get("edge_id")
            ],
            metadata=metadata,
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
        "--sampler-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for LLM-guided next-hop selection.",
    )
    parser.add_argument(
        "--history-exposure-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for sampler history-exposure filtering.",
    )
    parser.add_argument(
        "--compress-hop-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for compress_hop.",
    )
    parser.add_argument(
        "--image-bridge-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for hidden image-bridge normalization.",
    )
    parser.add_argument(
        "--image-target-ask-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for hidden final-image target-ask normalization.",
    )
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
    args = parser.parse_args()

    store = JsonlGraphStore(args.graph_dir)
    config = SamplerConfiguration(
        min_hops=args.min_hops,
        max_hops=args.max_hops,
        random_seed=args.seed,
        edge_penalty_alpha=args.edge_penalty_alpha,
        hop_sampling_strategy=args.hop_sampling_strategy,
        neighbor_selection_strategy=args.neighbor_selection_strategy,
        llm_candidate_count=args.llm_candidate_count,
        llm_score_temperature=args.llm_score_temperature,
        max_samples=1,
    )
    graph = GraphView(store, allowed_edge_types=set(config.allowed_edge_types))
    sampler = RandomPathSampler(
        graph=graph,
        config=config,
        model_client=LLM_WORKER if args.sampler_model_alias and args.neighbor_selection_strategy == "llm_guided" else None,
        model=args.sampler_model_alias,
        history_exposure_model_client=LLM_WORKER if args.history_exposure_model_alias else None,
        history_exposure_model=args.history_exposure_model_alias,
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
        compress_hop_model_client=LLM_WORKER if args.compress_hop_model_alias else None,
        compress_hop_model=args.compress_hop_model_alias,
        image_bridge_model_client=LLM_WORKER if args.image_bridge_model_alias else None,
        image_bridge_model=args.image_bridge_model_alias,
        image_target_ask_model_client=LLM_WORKER if args.image_target_ask_model_alias else None,
        image_target_ask_model=args.image_target_ask_model_alias,
    )
    context = writer.build_writer_context(path=path, graph=graph)
    raw_hop_summaries = [writer.compress_hop(hop=hop) for hop in context.hops]
    normalized_hop_summaries, image_bridge_normalization = writer._normalize_question_hops(
        path=path,
        context=context,
        hop_summaries=raw_hop_summaries,
    )
    debug_hop_summaries = [
        {
            key: value
            for key, value in item.items()
            if key not in {"edge_id", "src_node_id", "dst_node_id"}
        }
        for item in raw_hop_summaries
    ]
    debug_normalized_hop_summaries = [
        {
            key: value
            for key, value in item.items()
            if key not in {"edge_id", "src_node_id", "dst_node_id"}
        }
        for item in normalized_hop_summaries
    ]
    opening_package = writer.select_opening_package(context=context, hop_summaries=normalized_hop_summaries)
    draft = writer.compose_question(path=path, graph=graph, context=context)
    raw_target_ask = draft.metadata.get("target_ask") if isinstance(draft.metadata, dict) else None
    question_target_ask = draft.metadata.get("question_target_ask") if isinstance(draft.metadata, dict) else None
    question_terminal_bridge = draft.metadata.get("question_terminal_bridge") if isinstance(draft.metadata, dict) else None
    image_target_terminal_normalization = (
        draft.metadata.get("image_target_terminal_normalization") if isinstance(draft.metadata, dict) else None
    )
    debug_question_hop_summaries = [
        {
            key: value
            for key, value in item.items()
            if key not in {"edge_id", "src_node_id", "dst_node_id"}
        }
        for item in (draft.reasoning_steps or [])
    ]
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
    print(f"sampler_model: {args.sampler_model_alias or 'fallback(no llm)'}")
    print(f"image_bridge_model: {args.image_bridge_model_alias or args.model_alias or 'fallback(no llm)'}")
    print(
        f"image_target_ask_model: "
        f"{args.image_target_ask_model_alias or args.image_bridge_model_alias or args.model_alias or 'fallback(no llm)'}"
    )
    print(f"neighbor_selection_strategy: {args.neighbor_selection_strategy}")
    print("raw_hop_summaries:")
    print(json.dumps(debug_hop_summaries, ensure_ascii=False, indent=2))
    print("bridge_normalized_hop_summaries:")
    print(json.dumps(debug_normalized_hop_summaries, ensure_ascii=False, indent=2))
    print("question_hop_summaries:")
    print(json.dumps(debug_question_hop_summaries, ensure_ascii=False, indent=2))
    print("image_bridge_normalization:")
    print(json.dumps(image_bridge_normalization, ensure_ascii=False, indent=2))
    print("opening_package:")
    print(json.dumps(opening_package, ensure_ascii=False, indent=2))
    print("raw_target_ask:")
    print(json.dumps(raw_target_ask or {}, ensure_ascii=False, indent=2))
    print("question_target_ask:")
    print(json.dumps(question_target_ask or {}, ensure_ascii=False, indent=2))
    print("question_terminal_bridge:")
    print(json.dumps(question_terminal_bridge or {}, ensure_ascii=False, indent=2))
    print("image_target_terminal_normalization:")
    print(json.dumps(image_target_terminal_normalization or {}, ensure_ascii=False, indent=2))
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


def _difficulty_only_main() -> None:
    """
    python -m synthesis.vqa.question_writer \
  --model-alias gpt54_internal_azure \
  --question "The trophies in this image are from a specific year of a major tennis tournament..." \
  --answer "Knight Frank; Charles was ...; Francis was ..." \
  --hops-json '[{"hop_index":0,"source":"...","target":"...","statement":"...","relation":"...","retrieval_query":""}]'
    """
    parser = argparse.ArgumentParser(description="Debug difficulty enhancement on a manually supplied question.")
    parser.add_argument(
        "--model-alias",
        required=True,
        help="Model alias registered in synthesis/models.json for difficulty enhancement.",
    )
    parser.add_argument(
        "--question",
        required=True,
        help="Question to be passed into enhance_difficulty.",
    )
    parser.add_argument(
        "--answer",
        default="",
        help="Optional answer carried through the QuestionDraft.",
    )
    parser.add_argument(
        "--answer-type",
        default="other",
        help="Optional answer_type carried through the QuestionDraft.",
    )
    parser.add_argument(
        "--hops-json",
        default="[]",
        help="JSON string containing the hop chain list for difficulty enhancement.",
    )
    parser.add_argument(
        "--image-url",
        default=None,
        help="Optional image URL to attach for image-start questions.",
    )
    args = parser.parse_args()

    try:
        hops = json.loads(args.hops_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--hops-json is not valid JSON: {exc}") from exc
    if not isinstance(hops, list):
        raise SystemExit("--hops-json must decode to a JSON list.")

    draft = QuestionDraft(
        question=args.question,
        answer=args.answer,
        answer_type=args.answer_type,
        reasoning_steps=[item for item in hops if isinstance(item, dict)],
        used_evidence_ids=[],
        metadata={},
    )
    writer = QuestionWriter(
        model_client=LLM_WORKER,
        model=args.model_alias,
    )
    enhanced = writer.enhance_difficulty_direct(
        draft=draft,
        starting_image_url=args.image_url,
    )

    print("input_question:")
    print(args.question)
    print("input_hops:")
    print(json.dumps(draft.reasoning_steps, ensure_ascii=False, indent=2))
    print("enhanced_output:")
    print(
        json.dumps(
            {
                "question": enhanced.question,
                "answer": enhanced.answer,
                "answer_type": enhanced.answer_type,
                "difficulty_enhancement_result": enhanced.metadata.get("difficulty_enhancement_result"),
                "writer_warnings": enhanced.metadata.get("writer_warnings") or [],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    # _debug_main()
    _difficulty_only_main()
