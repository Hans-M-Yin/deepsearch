"""LLM-backed question writer for graph trajectories.

The writer now works directly from ``PathCandidate + GraphView`` instead of a
separate evidence-builder stage. Internally it follows a seven-step process:

1. compress each hop into a short statement
2. normalize hidden image bridges when they are only evidence carriers
3. build or normalize hop 0 as the entry hop
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
from .path_sampler import DEFAULT_HISTORY_EXPOSURE_MODEL, RandomPathSampler, SamplerConfiguration
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


PROMPT_SELECT_TEXT_TARGET = """
Now you are an expert designer of knowledge-based Q&A questions. You will design a quiz question for students to test their knowledge of a particular person’s life and background. You will be given a specific target object together with its complete profile, as well as a series of predecessor objects used to conceal that target. The final question should be designed so that the user must first start from the predecessor objects, reason step by step, and ultimately infer the specific target object, and then answer the final question based on information about that target. You only need to design this final question.
You need to design the question based on the complete profile of the given target object. From the full material, you should select one or more detailed, non-obvious factual pieces of information, and then organize the extracted information into a complete question. The information you extract will serve as the corresponding answer. The question may concern any aspect of the target, including identity, life experience, major events, detailed knowledge, relevant numbers, or dates. However, you must ensure that the question is not trivial or common-sense, and that it requires reasoning and knowledge retrieval to answer correctly.
If the series of predecessor objects used to conceal the target share a common topic, you may choose factual information related to that topic when constructing the question for the given target, as this will make the designed question more natural.However, you must ensure that the question is not overly simple or answerable through common sense alone; instead, it should require reasoning and knowledge retrieval to answer correctly, while also avoiding excessive length, obscurity, or too many constraints and answer conditions.
You must ensure that the question is not trivial or based on common sense, but it must still ask about an objective fact rather than a subjective or open-ended matter. The question should be elegant and natural, rather than merely obscure or niche.

Requirements:

1. Use only the information provided to you about the object.
2. The question must have a clear, unambiguous answer supported by evidence.
3. The factual information you select must not be revealed in the question, and the question must not tell students that any material exists.
4. Do not fabricate or add any information. Everything mentioned in your answer must appear in the original text.
5. If the content of your question contains highly distinctive markers that could reveal the target entity, you should make the question more ambiguous. For example, in the question “When was a certain politician’s slogan ‘Make America Great Again’ introduced?”, the highly distinctive slogan should be blurred. It can be rewritten as: “When was a certain politician’s own campaign slogan introduced?”
6. NOTICE: Try to avoid asking highly distinctive questions that would easily reveal the target’s identity, or wrap the content of the question in a way that ensures the subject being asked about cannot be inferred from the question itself.

Generate exactly 5 diverse candidate questions. Do not generate several near-duplicates
that ask about the same fact in slightly different wording.

Output format: JSON, containing the following fields:

{
  "candidates": [
    {
      "candidate_id": "candidate_1",
      "ask_target": "A complete question about the target node",
      "answer": "A standard answer that fully answers every part of ask_target",
      "supporting_facts": ["The exact original facts needed to answer the question"],
      "reasoning": "A concise derivation from supporting_facts to the answer",
      "support": "A brief explanation of why the question is evidence-supported and unambiguous"
    }
  ]
}
"""


PROMPT_ANSWER_TEXT_TARGET_CANDIDATES = """Answer every candidate question using only your own pre-existing knowledge.

You are not given a target profile, a predecessor chain, search results, images, or any other
reference material. Do not browse, retrieve, or infer hidden context. If you cannot answer a
question reliably from ordinary background knowledge, return an empty answer rather than guessing.

Return valid JSON with exactly this structure:
{
  "answers": [
    {
      "candidate_id": "candidate_1",
      "answer": "concise answer, or empty string when not answerable",
      "reason": "brief reason",
      "answerable": true
    }
  ]
}
"""


PROMPT_JUDGE_TEXT_TARGET_ANSWERS = """Judge whether each closed-book model answer is semantically correct relative to its gold answer.

Minor wording differences, synonyms, and equivalent formatting count as correct. Empty, uncertain,
contradictory, incomplete, or materially different answers count as incorrect.

Return valid JSON with exactly this structure:
{
  "evaluations": [
    {
      "candidate_id": "candidate_1",
      "correct": false,
      "reason": "brief comparison against the gold answer"
    }
  ]
}
"""


PROMPT_EVALUATE_TEXT_TARGET_CANDIDATES = """Select the highest-quality final knowledge question from the supplied candidates.

All supplied candidates have already failed a closed-book answer attempt, so do not reject a
candidate merely because it is not common knowledge. Select the one that is most natural, precise,
objectively answerable, and clearly supported by its supporting facts. Prefer a question that fits
the predecessor chain naturally and needs ordinary information retrieval after the target has been
identified. Do not rewrite candidates. If no candidate is acceptable, return reject_all.

Return valid JSON with exactly this structure:
{
  "decision": "select | reject_all",
  "selected_candidate_id": "candidate_id or null",
  "evaluations": [
    {
      "candidate_id": "candidate_1",
      "valid": true,
      "reason": "brief assessment"
    }
  ]
}
"""


PROMPT_SELECT_IMAGE_TARGET = """You are a professional designer of visual search questions, responsible for generating several candidate visual web-search questions.
** Task Setting
1. The solver will not receive the reference image you see. Instead, they will only be given an approximate description of the image’s content. Based on this description, the solver will search for relevant images and answer the final question by inspecting visual evidence in the retrieved images.
2. Because the images found by the solver may differ from the reference image in shooting angle, timing, composition, and so on, the reference image is only one possible search result and may not be the one the solver ultimately finds. Therefore, the questions you write must ensure that the answer derived from images matching the search description is unique or stable.

** Goals
1. Every question must genuinely require inspection of visual information. Avoid choosing facts that can be answered reliably using only ordinary background knowledge or directly from the search query itself. For example, if a person in the image has a white pocket square in the breast pocket of a suit, you should not ask about the object in the pocket, because that answer matches common real-world expectations and could be answered correctly without inspecting the image.
2. Prefer concrete visual details centered on the described scene or event (and the reference image). This includes not only low-level visual features but also higher-level semantic information, as long as the answer must still come from observing the image. Examples include:
    - actions, interactions, and object states;
    - clothing, equipment, accessories, signs, logos, labels, numbers, and visible text;
    - event-environment details, such as advertising boards, stage displays, or nearby equipment;
    - architectural, geographic, physical, or functional relations;
    - multi-step details that require first locating one entity and then inspecting something associated with it.
3. Use qualifiers to ensure answer uniqueness. Since image search from a text description can be ambiguous, use necessary qualifiers to constrain what the question refers to and avoid referential ambiguity. For example, when asking about an advertising board in a stadium, if multiple brands are present, you should specify which advertising board you mean.
4. Avoid vague expressions such as “that advertising board,” “that person,” or “that building,” especially when multiple instances may exist.
5. Avoid choosing details that are obviously incidental to a single photograph, such as unrelated bystanders, random vehicles, temporary objects, or arbitrary camera-relative positions.
6. If the query points to a fixed visual work—such as a particular album cover, poster, painting, logo, manuscript page, or iconic photograph—then composition-related locators such as “the upper-left corner” or “the second person from the left” are allowed.
7. Prioritize diversity. Do not generate multiple similar questions that differ only in the object or color being asked about. Explore different kinds of visual information and question types.
8. Use only details supported by the provided image evidence. Do not invent facts.

Please generate 5 candidate questions. Each candidate question must have an objective and concise gold answer. Return valid JSON, and you must follow exactly this structure:

** Example 1: A specific event photographed from multiple viewpoints
Image description:
Gonzalo Montiel taking the final penalty in the 2022 FIFA World Cup final shootout.

Reasonable questions:
1. Which foot did Montiel use to strike the ball?
2. Toward which side of the goal did the goalkeeper dive?
3. What brand appeared on the advertising board directly behind the goalkeeper?
4. When facing the goal, which player stood at the left end of the line of Argentine players with their arms linked?

These questions use actions or scene-grounded qualifiers to identify stable visual evidence. The relevant detail may require finding an appropriate view, but the wording identifies what must be inspected.

Unreasonable questions:
1. Who appears in the upper-right corner of the image?
2. What brand appeared on an advertising board?

The first depends on the photographer's viewpoint and composition. The second does not identify which of the many advertising boards is being asked about.

** Example 2: Another specific event photographed at different moments
Image description:
Steve Jobs unveiling the original iPhone during the Macworld 2007 keynote.

Reasonable questions:
1. When Jobs held the iPhone toward the audience with its home screen visible, which four application icons appeared in the bottom dock?
2. During the onstage demonstration of scrolling through a list by touch, which finger did Jobs use to operate the phone?

These questions use a specific action and moment within the event to locate the required visual evidence, rather than assuming that every photograph from the keynote shows the same content.

Unreasonable questions:
1. What was displayed on the presentation screen behind Jobs?
2. What color was the shirt of the audience member closest to the stage?

The first does not specify a moment even though the presentation screen changed throughout the event. The second asks about an incidental attendee who is not constrained by the event description.

** Example 3: A landmark with many possible photographs
Image description:
The Temple of Heaven in Beijing on a sunny day.

Reasonable questions:
1. What color are the roof tiles of the landmark's main circular hall?
2. What repeated decorative shapes appear beneath the roof of the main hall?

These questions concern stable architectural details of an explicitly identified part of the landmark.

Unreasonable questions:
1. What color is the clothing of the person closest to the camera?
2. What color is the roof of the building to the left of the Temple of Heaven?

The first asks about an incidental visitor. The second uses an undefined camera-relative direction, so different photographs may refer to different buildings.

** Example 4: A fixed visual work
Image description:
The Abbey Road album cover by the Beatles.

Reasonable questions:
1. What color suit is worn by the second person from the left?
2. Which member of the group is not wearing shoes?

Composition-relative wording is acceptable here because the description identifies a fixed visual work whose content and arrangement remain consistent across valid search results.

Unreasonable question:
What color is the car closest to the photographer in an Abbey Road street photo?

This refers to an arbitrary street photograph rather than a stable detail of the specified album cover.

Return valid JSON with exactly this structure:
{
  "candidates": [
    {
      "candidate_id": "candidate_1",
      "question_type": "action | interaction | equipment | clothing | text_or_symbol | event_environment | spatial_relation | functional_relation | composition | other",
      "ask_target": "one complete visual web-search question",
      "answer": "the objective and concise gold answer",
      "visual_locator": "the words or relations that locate the intended visual evidence",
      "visual_reasoning": ["the ordered visual localization and inspection steps"],
      "supporting_facts": ["the exact visual facts from the image that support the answer"]
    }
  ]
}
"""


PROMPT_EVALUATE_IMAGE_TARGET_CANDIDATES = """You are an expert evaluator selecting one valid visual web-search question from several candidates.

You will receive a base image-search query, one reference image with metadata, and candidate questions with proposed answers. The solver will not receive the reference image; they will search for relevant images using the base query.

Evaluate every candidate in two stages.

## Stage 1: Query-level uniqueness

Reason only from the base query and candidate question. Do not use the reference image to resolve ambiguity.

1. Determine whether the query identifies a specific event moment, an extended event, a general subject, or a fixed visual work.
2. Identify the visual referent asked about and reject the candidate if multiple people, signs, buildings, objects, or regions may satisfy its wording.
3. Check whether the referent has one stable answer under reasonable changes in viewpoint and shooting time.
   - Reject camera-relative references such as "on the left side of the image", "in the upper-right corner", "in the foreground", or "closest to the camera" when different viewpoints may change their meaning.
   - Scene-grounded relations such as "behind the goalkeeper", "on the player's right sleeve", "west of the main building", or "the leftmost player when facing the goal" may be stable.
   - For a specific event moment, accept facts fixed by that moment even across viewpoints.
   - If no precise moment is specified, reject facts that may change over time, including participant positions, display content, nearby people, vehicles, weather, or temporary objects.
   - For a fixed cover, poster, painting, logo, manuscript page, or iconic photograph, composition-relative positions may be stable.

Not every matching image must show the evidence. The solver may inspect multiple images. However, changing viewpoint or time must not change the underlying referent or produce a different answer.

## Stage 2: Image-level correctness

Now inspect the reference image. Check whether the referent is visible and uniquely located by the question, the proposed answer is visually correct, and the supporting facts are directly supported. Reject absent, unclear, incorrect, invented, or non-visual claims. Passing Stage 2 cannot override failure in Stage 1.

## Final selection

A candidate is valid only if it passes both stages. Among valid candidates, prefer stronger visual dependence, deeper reasoning, precise wording, interesting details, and lower text-only answerability. Do not automatically prefer simple color, count, or identity questions. Do not rewrite candidates. If none pass, return reject_all.

Return valid JSON with exactly this structure:
{
  "decision": "select | reject_all",
  "selected_candidate_id": "candidate_id or null",
  "evaluations": [
    {
      "candidate_id": "candidate_1",
      "query_analysis": {
        "query_type": "specific_moment | extended_event | general_subject | fixed_visual_work | unclear",
        "referent": "the underlying visual referent",
        "referent_unique": true,
        "answer_stable_across_viewpoints": true,
        "answer_stable_across_time": true,
        "valid": true,
        "rejection_reasons": []
      },
      "image_analysis": {
        "referent_visible": true,
        "referent_unique_in_image": true,
        "answer_correct": true,
        "supporting_facts_correct": true,
        "valid": true,
        "rejection_reasons": []
      },
      "valid": true
    }
  ]
}
"""


PROMPT_BUILD_TEXT_OPENING_HOP = """You are creating the first hop of a multi-hop retrieval chain for a text-start path.

The first hop has no preceding entity. It must identify the supplied `target` from an evidence-supported description of that target.

You will receive:
- `target`: the exact first entity that this opening hop must identify;
- `source_description`: the source text node's description, which is the only factual evidence you may use for the identifying description;
- `target_aliases`: names and aliases of the target;
- `path_statements`: the later hop statements, provided only to show the topic and direction of the downstream reasoning chain;
- `target_ask`: the final question, also provided only as downstream thematic context.

Use `path_statements` to choose a source fact that connects naturally with the later path. Do not copy facts from `path_statements` into the opening description unless they are also explicitly supported by `source_description`. Do not reveal a later target or create a shortcut into a later hop.

Write one complete declarative identification statement with this semantic form:

[an evidence-supported description of the target] + [a natural identification relation] + [the exact target]

Requirements:
1. The output must be a complete statement, not only a noun phrase.
2. The exact `target` must appear once as the entity being identified.
3. The descriptive portion before the identification must not contain the target name, an alias, abbreviation, initialism, canonical id, or near-copy surface form.
4. Prefer a set of facts that is concise, not so famous as to make the target obvious, supports web retrieval, and is semantically relevant to the downstream path.
5. Use only facts from `source_description`.
6. Do not mention that a profile, description, prompt, or source material was provided.
7. Do not write a question.

Good example:
`A 20th-century Romanian sculptor who created a war memorial ensemble in Targu Jiu was Constantin Brâncuși.`

Bad example:
`A 20th-century Romanian sculptor who created a war memorial ensemble in Targu Jiu.`
Reason: this is only a clue, not a complete identification hop.

Bad example:
`Constantin Brâncuși, a Romanian sculptor, created a war memorial ensemble in Targu Jiu.`
Reason: the target is exposed before the identifying description and the sentence reads as biography rather than an identification hop.

Return valid JSON with exactly these fields:
{
  "opening_statement": "one complete identification statement ending in the exact target",
  "supporting_facts": ["the exact facts from source_description used in the identifying description"],
  "why_relevant": "why the selected description is recoverable and connects naturally to the downstream path"
}
"""

PROMPT_ANSWER_VISUAL_TARGET_CANDIDATES = """Answer every candidate question using the attached reference image.

Inspect the image carefully. Do not use the proposed gold answers, which are intentionally not provided.
If the image does not support a confident answer, return an empty answer and explain why.

Return valid JSON with exactly this structure:
{
  "answers": [
    {
      "candidate_id": "candidate_1",
      "answer": "concise answer, or empty string when not answerable",
      "reason": "brief image-grounded reason",
      "answerable": true
    }
  ]
}
"""

PROMPT_ANSWER_TEXT_ONLY_TARGET_CANDIDATES = """Answer every candidate question without access to the reference image.

The image is not provided. Answer only from ordinary background knowledge or clues exposed by the
question. Do not assume hidden visual details. If you cannot answer confidently, return an empty answer.

Return valid JSON with exactly this structure:
{
  "answers": [
    {
      "candidate_id": "candidate_1",
      "answer": "concise answer, or empty string when not answerable",
      "reason": "brief reason",
      "answerable": true
    }
  ]
}
"""

PROMPT_JUDGE_VISUAL_TARGET_ANSWERS = """Judge whether each model answer is semantically correct relative to its gold answer.

For each candidate, separately judge:
1. whether the answer produced with the image is correct;
2. whether the answer produced without the image is correct.

Minor wording differences, synonyms, and equivalent formatting count as correct. Empty, uncertain,
contradictory, incomplete, or materially different answers count as incorrect.
A candidate passes only when the with-image answer is correct AND the without-image answer is incorrect.

Return valid JSON with exactly this structure:
{
  "evaluations": [
    {
      "candidate_id": "candidate_1",
      "with_image_correct": true,
      "without_image_correct": false,
      "pass": true,
      "reason": "brief comparison against the gold answer"
    }
  ]
}
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
PROMPT_COMPOSE_QUESTION = """You are an expert at composing multi-hop retrieval questions. Below, I will provide you with the specific structure of each hop in the data, and your task is to merge these scattered pieces of information into a deep reasoning question. This question should hide the intermediate reasoning steps and be presented to the user for them to answer.

The input includes:
	1. each hop in the reasoning chain, including:
	   - `source`: the starting point of this hop;
	   - `target`: the target of this hop;
	   - `statement`: a statement describing the relationship between the source and the target;
	   - `mark`: optional metadata; `"image"` means the hop depends on a specific image or scene.
	2. `target_ask`, a knowledge question about the final target.

The first hop is a special entry hop whose `source` is `-`. For a text-start path, its statement identifies the first real entity from a textual description. For an image-start path, its statement identifies the first real entity from the attached image and may refer to `this image`.

The `source` of each later hop is the `target` of the previous hop, so the whole hop sequence can be connected end-to-end into a complete multi-hop question, with the reasoning chain represented as A -> B -> C ...: A -> B is the first transition, where the user must infer B from A; B -> C is the next transition; and so on.

For each hop's `target` (which is also the next hop's `source`), it must not appear directly in the question you write. Instead, given the known `source`, the user should have to infer what the `target` is by reasoning from the relationship expressed in that hop's `statement`.

For the final hop's `target`, we additionally provide `target_ask`. After the user has inferred the final target entity step by step, they must then answer a knowledge question about that entity.

## Rules

1. After writing the question, check whether any source or target names appear in the question.

2. You need to use the information provided in the `statement` fields to bridge the source and target. Multiple statements may be compressed or merged to make the question more natural and concise.

3. Do not explicitly list the reasoning steps, and do not use expressions such as "starting from...", "then...", "next...", or "based on this clue...".

4. You must check for and remove redundant clues yourself to ensure there are no shortcuts. For example, if the relationship referred to by a statement is too famous, you should make that relationship more implicit.

5. If any hop is ambiguous, you should flexibly add a restricting modifier.
For example, suppose the true reasoning result is a club that a certain player once played for, and you ask, "Who was the first captain of that club in 2023?" In that case, the club may be ambiguous. The restriction should be derived naturally from the true target while still depending on the source.
For instance, if the true target is FC Barcelona, a better phrasing would be: "the first international club this player played for."
Do not use a restriction such as "the club that won the 2011 UEFA Champions League," because that would allow the target to be inferred without depending on the previous source, creating a shortcut.

6. If a statement in a given hop refers to a specific scene or image, that hop will be marked as `"image"`. In the rewritten question, preserve the description of that scene or image—especially the visual details—and only apply slight obfuscation to the entities within it.

7. Please ensure that the reasoning order in the question follows the order of the hops.

## Tips

1. Merge multiple statements so that the question is compact and natural.

2. In addition to source and target not being allowed to appear directly, other widely known or obvious entities or relations appearing in the statements may also be blurred. When choosing such wording, aim for language that is neither too explicit nor too ambiguous; choose a relational expression that is connected to the previous source, or one that can still support smooth downstream reasoning once the previous source has been inferred.

3. Do not arbitrarily alter the details of a statement, and make sure the question content is factually accurate. For example, consider the statement: “In 1814, the British general who was killed during the battle involving the 175th Infantry Regiment (United States) at Bread and Cheese Creek was Robert Ross (British Army officer).” This does not explicitly specify Robert Ross’s side in the battle, so it must not be further interpreted as: “This general was killed in 1814 while fighting the latter at Bread and Cheese Creek.”

## Example 1

Input:
{
  "hop_facts": [
    {
      "hop_index": 0,
      "source": "-",
      "target": "Brent Scowcroft",
      "statement": "The man on the left in this image is Brent Scowcroft."
    },
    {
      "hop_index": 1,
      "source": "Brent Scowcroft",
      "target": "Barbara Bush",
      "statement": "When he received the Presidential Medal of Freedom from President George H. W. Bush at the White House in 1991, the woman standing next to him in a red-and-white polka-dot dress was Barbara Bush."
    }
  ],
  "target_ask": {
    "ask_target": "What type of necklace was Barbara Bush wearing when she attended the Wellesley College commencement with another country's first lady on June 1, 1990?"
  }
}

Bad output:
"Please look at the man on the left in this image. In 1991, he received the Presidential Medal of Freedom from President George H. W. Bush at the White House. A woman in a red-and-white polka-dot dress was standing beside him. That woman later attended a famous commencement ceremony at Wellesley College. What type of necklace was she wearing?"

Good output:
"In the ceremony where the man on the left in this image received a certain civilian honor at the White House in 1991, the woman standing beside him in a red-and-white polka-dot dress attended a women's college commencement with a certain country's first lady the previous year. What type of necklace was she wearing?"

Why it is good:
It hides overly explicit information: it obscures the nationality of the other first lady, preventing the user from directly searching that first lady's schedule to infer the identity of the woman in the dress. At the same time, it preserves the fact that she was a first lady, so it is not made too vague to identify.
It restates time information across clauses naturally: 1991 is kept at the start, and 1990 is later described as "the previous year," making the sentence more natural.
It removes overly obvious clues: "Presidential Medal of Freedom" is blurred into "a certain civilian honor," and the detail that the award was presented by the sitting president is omitted. This ensures that the user must first infer Scowcroft and then search his biography to deduce that the award was the Presidential Medal of Freedom, and only then search for the award ceremony image. The degree of obfuscation matches the source appropriately.
The sentence structure is compact and clean, rather than directly restating multiple isolated sentences.
It does not introduce extra information or other shortcuts. The question is clean and non-redundant.

## Example 2

Input:
{
  "hop_facts": [
    {
      "hop_index": 0,
      "source": "-",
      "target": "Constantin Brâncuși",
      "statement": "A 20th-century Romanian sculptor who created a war memorial ensemble in Targu Jiu was Constantin Brâncuși."
    },
    {
      "hop_index": 1,
      "source": "Constantin Brâncuși",
      "target": "Bird in Space",
      "statement": "In a photograph of Constantin Brâncuși's Paris studio taken by Edward Steichen in the 1920s, the slender sculpture at the center of the image is Bird in Space."
    },
    {
      "hop_index": 2,
      "source": "Bird in Space",
      "target": "National Gallery of Art",
      "statement": "The National Gallery of Art holds both a marble version and a bronze version of the sculpture Bird in Space."
    },
    {
      "hop_index": 3,
      "source": "National Gallery of Art",
      "target": "David E. Finley, Jr.",
      "statement": "The museum's director from 1938 to 1956 was David E. Finley, Jr."
    }
  ],
  "target_ask": {
    "ask_target": "Where did David E. Finley, Jr. earn his professional degree, and in what field was that degree?"
  }
}

Bad output:
"A 20th-century Romanian sculptor created a war memorial ensemble in Targu Jiu. He was photographed in his Paris studio in 1920. In the image, the sculpture in the center is Bird in Space. The National Gallery of Art holds two versions of the sculpture. Its director from 1938 to 1956 was David E. Finley, Jr. Where did he earn his professional degree, and in what field?"

Good output:
"A 20th-century Romanian sculptor who created a war memorial ensemble in Targu Jiu was photographed in his Paris studio by Edward Steichen in the 1920s. Where did the 1938-56 director of the museum that holds both marble and bronze versions of the slender sculpture at the center of that studio photograph earn his professional degree, and in what field?"

Why it is good:
It uses two sentences because they form natural semantic units, not because the path contains multiple hops.
It preserves the path order while compressing the middle transitions.
It hides the sculpture, museum, and director names without losing the facts needed to recover them.
It retains dates and material types because they perform useful identifying functions.

## Example 3

Input:
{
  "hop_facts": [
    {
      "hop_index": 0,
      "source": "-",
      "target": "Du Liniang",
      "statement": "The female protagonist of the play The Peony Pavilion who dies from lovesickness after dreaming of a lover and is later resurrected is Du Liniang."
    },
    {
      "hop_index": 1,
      "source": "Du Liniang",
      "target": "Metropolitan Museum of Art",
      "statement": "The venue where a 2012 outdoor production of The Peony Pavilion featuring Du Liniang's story was performed in the Astor Court is the Metropolitan Museum of Art."
    },
    {
      "hop_index": 2,
      "source": "Metropolitan Museum of Art",
      "target": "Astor Court (Metropolitan Museum of Art)",
      "statement": "The Ming Dynasty-style garden courtyard with a pavilion and rock formations in the Asian Art wing of the Metropolitan Museum of Art is the Astor Court."
    }
  ],
  "target_ask": {
    "ask_target": "The Metropolitan Museum of Art's Astor Court was built using traditional Chinese materials. What rare tree species was felled under special permission for its wooden columns, and what three ingredients made up the hand-mixed adhesive used to secure its terracotta floor tiles?"
  }
}

Bad output:
"A 2012 outdoor production of The Peony Pavilion was performed in a Ming Dynasty-style garden courtyard; the play's heroine dies of lovesickness and is later resurrected. What rare tree species was felled under special permission for the courtyard's wooden columns, and what three ingredients made up the hand-mixed adhesive used to secure its terracotta floor tiles?"

Good output:
"At the venue of a 2012 performance adapted from the story of the Peony Pavilion heroine who dies from lovesickness after dreaming of a lover and is later resurrected, what rare tree species was felled under special permission for the wooden columns of the Ming Dynasty-style garden courtyard in its Asian Art wing, and what three ingredients made up the hand-mixed adhesive used to secure the terracotta floor tiles?"

Why it is good:
It retains The Peony Pavilion as the opening clue while hiding all intermediate source and target names.
It removes the premature mention of Astor Court from hop 1, so the courtyard must be inferred only after the venue has been identified.
It is compact and preserves the same dependency and reasoning order as the hop chain.

Now compose the question for the provided input.
Return valid JSON with exactly these fields:
{
  "analysis": "brief explanation of how the hops were merged without exposing intermediate targets or creating shortcuts",
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

PROMPT_DIFFICULTY_ENHANCEMENT = """You are a difficulty-enhancement editor for multi-hop retrieval questions. You will be given a multi-hop question together with its underlying reasoning chain. Your task is to revise the question so that the descriptions of intermediate entities and clues become subtler and harder for strong models to short-circuit, while keeping the original answer, factual relations, and core reasoning chain unchanged. Revise as much as is needed to close every shortcut — do NOT under-edit — but never so much that the question loses unique solvability. The rewritten question must have the same answer, remain uniquely solvable, and be verifiable.

Goal: The purpose is not to make the question longer, nor to blindly make all clues vaguer. Based on the provided `reasoning_chain`, analyze which entities and clues let a strong model reach an entity (including the final answer) WITHOUT traversing the earlier hops, and neutralize exactly those. Keep the question fluent, natural, and benchmark-like.

### Core verification test (apply this to EVERY hop — it is your single most important check)

Model the chain as a sequence of hops of the form **"entity A + description D → next entity B"**. For each hop, the wording is correct only if BOTH hold:

- **(Forward safety)** When A is still unknown, D on its own must be too underspecified to reveal A or to jump directly to B. If D alone already pins down A or B, it is a shortcut — blur it.
- **(Backward sufficiency)** Once A is known, D must lead to B **uniquely and unambiguously**. If D could point to several candidates given A, it is too vague — add a neutral relational qualifier until B is unique.

In short: a well-tuned description is **useless without its predecessor, and decisive with it.** Every keep/blur decision you make should be justified by this test.

### Shortcut taxonomy — scan the draft for every one of these

Treat each of the following as a **shortcut candidate**, whether it sits on an intermediate bridge entity, the final entity, or the answer itself:

- **Explicit years and precise dates** ("in 1958", "since 2007", "the 1966 Uniform Time Act").
- **Unique official titles or positions** ("head of government", "president of country X").
- **Distinctive, near-unique career or life events** ("was forced to resign", "was stripped of", "the only person to ...").
- **Superlative / "the last / first / only" phrasing** ("the last crewed Moon landing", "the highest peak").
- **Signature slogans, signature works, iconic named titles, acts, or awards.**
- **Evaluative or reputation modifiers that fingerprint an entity** ("highly prestigious", "world-famous", "landmark").
- **Precise numeric specifications** that pin a single entity.

**Policy:** For each candidate, keep it ONLY if it is (a) the deliberate entry point at the *head of the reasoning chain*, or (b) strictly required to keep the question uniquely solvable or to define the final answer. Otherwise blur it via the verification test above (e.g., "in 1958" → "that same year, later on"; "after being forced to resign as head of government" → "after leaving his government post"; "the last crewed Moon landing" → "a crewed lunar mission of that era").

### Requirements

1. Do not change the core question, and do not change the final answer.
2. Reduce the salience of intermediate entities according to the reasoning chain. After revision, each intermediate entity must be inferable only after reasoning through the previous one — enforce this with the Core verification test above.
3. **Preserve the entry point.** The clue(s) at the *beginning of the reasoning chain* (not necessarily the surface start of the sentence) are the question's only foothold and should be kept relatively explicit — do NOT apply heavy obfuscation to them. Blur only when a later hop can re-establish the same entity; the head of the chain has no predecessor to lean on.
4. **Shortcuts are not confined to intermediate bridge entities.** Also scan the FINAL hop and the clues attached to the answer: if a dense terminal clue (e.g., a unique title + a specific year + a distinctive event) lets a strong model jump straight to the answer, that bypasses the chain — blur or thin it out just as aggressively.
5. Do not repeat the same qualifier for one entity. Introduce a description once, then refer back with a short anaphor; delete redundant restatements.
6. The question may include an image. Preserve the connection between the question and the image; descriptions of a scene/image marked `"image"` in the reasoning_chain should be preserved (especially visual details) and only lightly obfuscated at the entity level, never deleted.
7. Do not fabricate information. If an entity is not explicitly revealed in the current wording, do not invent a way to reveal it.
8. Delete any redundant information that is only weakly related to the main reasoning chain.

### Self-check (before finalizing)

Enumerate every explicit year, date, unique title, superlative, signature term, and proper noun remaining in your draft. For each, state in the analysis whether you **keep or blur** it, and justify with the Core verification test (entry point / uniqueness anchor → keep; otherwise → blur). Only declare "no revision needed" if this scan leaves nothing removable AND no unjustified year/title/superlative remains.

### JSON output format
Return exactly one valid JSON object and no other text.
{
  "analysis": "hop-by-hop application of the Core verification test, the shortcut scan, and the revision plan",
  "question": "the improved question"
}

### Techniques
1. Prefer relational, structural, or contextual constraints over highly salient signals (famous titles, person names, signature works, unique achievements, explicit years, reputation adjectives, iconic named acts/papers).
2. When blurring creates ambiguity, disambiguate with a neutral relational qualifier, not with a fresh salient hint.
3. Keep the question natural, concise, and compact — not a pile of stitched-together hints.

### Examples

**Example 1**
question: The man shown in this image later became nationally prominent for his handling of the devastating 1927 flood, a development that helped lead to his 1928 presidential nomination by the political party that narrowly carried three long-Democratic "blue wall" states in 2016. Which three states were they, and what broader effect did that victory have on that party?
{
  "analysis": "Chain: [image man] → [the party that nominated him for president in 1928] → [the three states that party flipped in 2016]. Entry point = the man in the image, which must stay explicit (Requirement 3) — it has no predecessor to lean on. Hop test on each clue: (1) The '1927 flood → 1928 nomination' clause is a redundant biographical hop that is not on the path to the answer and acts as a strong fingerprint of the man; by the forward-safety test it lets a solver identify him without the image, so remove it. (2) Terminal shortcut: explicitly naming the 'Democratic Party' and 'blue wall' fails the forward-safety test — D alone reveals the 2016 states with no need to identify the man or his party first. Blur to a relational description ('three states that had long supported its rival party'). Backward-sufficiency check: once the man and his 1928 nominating party are fixed, that relational phrase still resolves to exactly Michigan, Wisconsin, and Pennsylvania — unique. Also merge the fragmented clauses for compactness.",
  "question": "The political party that nominated the man shown in this image as its 1928 presidential candidate narrowly carried three states that had long supported its rival party in 2016. Which three states were they, and what broader effect did that victory have on that party?"
}

**Example 2**
question: The 2015 Copa Libertadores champion was an Argentine giant whose home ground is the Estadio Monumental. In River Plate's 2024–25 squad list, the player wearing number 29 took the final penalty in the 2022 World Cup final. The provided photo shows Montiel taking that last penalty. Which side of the goal did he send the ball to?
{
  "analysis": "Chain: [the 2015 Copa Libertadores champion] → [the number 29 player in that club’s 2024–25 squad] → [that player’s penalty in a World Cup final] → [the shooting direction read from the image]. Entry point = ‘the 2015 Copa Libertadores champion’, kept (Requirement 3): it already identifies River Plate uniquely, so the added clue ‘whose home ground is the Estadio Monumental’ is redundant reinforcement and should be removed. Hop test: (2) the club name ‘River Plate’ is restated in the second hop, re-exposing an entity the solver was supposed to derive — remove the name and replace it with a relational anaphor (‘that club’). (3) ‘the 2022 World Cup final’ is an explicit-year shortcut; by forward safety it over-specifies the event, while by backward sufficiency, changing it to ‘a World Cup final’ still leaves the event uniquely determined once the player is identified, so the year can be blurred. (3) Once Montiel has been inferred, the penalty can only be the final one, so to reduce salience, the modifier on the penalty can be removed; under the intended reasoning order the question remains solvable and does not introduce ambiguity, while the shortcut is weakened. (4) The photo is not presented at the top of the question, and ‘Montiel’ directly names the target player; both should be hidden — the solver should answer through the visual clue without being given the name or an explicit identification of the pictured player. Finally, reorder and compress the clues.",
  "question": "The number 29 player who was with the 2015 Copa Libertadores-winning club in 2024–25 once took a penalty in a World Cup final. Which side of the goal did he send the ball to?"
}

**Example 3**
question: A 20th-century Romanian sculptor who created a war memorial ensemble in Targu Jiu was photographed in his Paris studio by Edward Steichen in the 1920s. Where did the 1938-56 director of the museum that holds both marble and bronze versions of the slender sculpture at the center of that studio photograph earn his professional degree, and in what field?
{
  "analysis": "Chain: [the sculptor described obliquely as the creator of the Targu Jiu war memorial ensemble] → [the studio photograph] → [the slender sculpture at the center of that photograph] → [the museum holding both marble and bronze versions of that sculpture] → [the museum’s director during 1938–56] → [his degree information]. This is a case where only light revision is needed. Entry point: the sculptor is introduced relationally through the Targu Jiu war memorial ensemble plus a photograph taken by Edward Steichen, rather than by name. (1) Brâncuși’s name and signature works are not stated directly, but salience can be reduced further by removing the nationality. (2) The chain points to a photograph taken by Edward Steichen, which is an exposed proper name; however, Edward Steichen photographed many people and scenes, so mentioning him does not by itself expose Brâncuși. Once Brâncuși is derived from the earlier clue, the photograph taken by Steichen can be uniquely fixed without retaining the redundant time and place details, so the decade and Paris location can be removed. This adds some local ambiguity in isolation, but that ambiguity disappears once the predecessor is known. (3) The slender sculpture does not directly expose Bird in Space, and the fact that the museum holds both marble and bronze versions is the necessary disambiguating bridge to the next hop, so it should be kept. (4) The explicit date range 1938–56 is the unique anchor for the director hop and does not itself expose who the director is, so it should be kept. (5) Beyond these blurring and deletion decisions, the wording should also be compressed for fluency.",
  "question": "The 20th-century sculptor who created the Targu Jiu war memorial ensemble was photographed in his studio by Edward Steichen. Where did the 1938–56 director of the museum that holds both marble and bronze versions of the slender sculpture at the center of that studio photograph earn his professional degree, and in what field?"
}

**Example 4**
question: In Jacques-Louis David's painting of the Tennis Court Oath, the man standing on a table at the center of the crowd with his arm raised later wrote a eulogy for an astronomer; that oath took place in the city that hosted the 8th summit of an intergovernmental forum in 1982. While conducting an arc measurement at the Cape of Good Hope, that astronomer incorrectly concluded that the Earth was prolate. Decades later, who first proposed that this error was caused by the gravitational pull of nearby mountains, and who later confirmed the theory through new measurements?
{
  "analysis": "Chain: [the city that hosted the 8th summit of an intergovernmental forum in 1982] → [the historical event that took place in that city] → [David’s painting of that event] → [the man standing on a table at the center of the crowd with his arm raised] → [the astronomer for whom he later wrote a eulogy] → [that astronomer’s mistaken conclusion about the shape of the Earth] → [who proposed that nearby mountains caused the error, and who later confirmed the theory]. Entry point = the 1982 summit-city clue, kept explicit (Requirement 3). Shortcut scan: (1) directly naming the Tennis Court Oath fails forward safety — it exposes the event and its location before the solver derives them from the summit-city clue, so it should be replaced with ‘an event that took place in that city’. (2) Terminal-region shortcut (Requirement 4): ‘an arc measurement at the Cape of Good Hope’ together with ‘concluded that the Earth was prolate’ would let a strong model identify the astronomer directly, bypassing the painting and eulogy hops; under the hop test, that description is already decisive on its own, so the location should be blurred to ‘an arc measurement carried out somewhere’, and ‘prolate’ should be softened to ‘an incorrect conclusion about the Earth’s shape’. Backward sufficiency still holds: once the man in the painting is identified, ‘the astronomer for whom he wrote a eulogy’ still resolves uniquely. The painter’s name and the raised-arm visual detail should be kept, because once the city is derived, they are still needed to determine the painting and the man in it. (5) There is also some redundancy that does not help the core reasoning chain, such as ‘decades later’, so it should be removed.",
  "question": "In Jacques-Louis David’s painting of an event that took place in the city that hosted the 8th summit of an intergovernmental forum in 1982, the man standing on a table at the center of the crowd with his arm raised later wrote a eulogy for an astronomer, who reached an incorrect conclusion about the Earth’s shape during an arc measurement carried out somewhere. Who first proposed that this shape error was caused by nearby mountains, and who later confirmed the theory through new measurements?"
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
    ask_target_verify_model_client: ModelWorkerClient | None = None
    ask_target_verify_model: str | None = None
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
            fallback = self._fallback_select_target(context.target_node)
            if context.target_node.get("node_type") == "image":
                fallback["image_target_candidates"] = []
                fallback["image_target_candidate_evaluation"] = {
                    "decision": "fallback",
                    "selected_candidate_id": None,
                    "evaluations": [],
                    "reason": "no_model_client",
                }
            return fallback
        target_node_type = str(context.target_node.get("node_type") or "")
        if target_node_type == "image":
            return self._select_image_target_ask(context=context)

        return self._select_text_target_ask(context=context)

    def _select_text_target_ask(self, *, context: WriterContext) -> dict[str, Any]:
        user_payload = {
            "target_node": context.target_node,
            "predecessor_chain": self._format_predecessor_chain(context),
        }
        try:
            parsed = self._generate_json(
                system=PROMPT_SELECT_TEXT_TARGET,
                user_payload=user_payload,
                trace_label="select_target_ask_text_candidates",
                max_tokens=max(self.max_tokens, 2400),
            )
        except Exception as exc:
            fallback = self._fallback_select_target(context.target_node)
            fallback["writer_warning"] = self._writer_warning_entry(
                stage="select_target_ask_text_candidates",
                error=exc,
            )
            return fallback

        candidates = self._normalize_text_target_candidates(parsed.get("candidates"))
        if not candidates:
            fallback = self._fallback_select_target(context.target_node)
            fallback["text_target_candidates"] = []
            fallback["text_target_candidate_verification"] = {
                "decision": "fallback",
                "reason": "no_valid_generated_candidates",
                "kept_candidate_ids": [],
                "evaluations": [],
            }
            return fallback

        verified_candidates, verification = self._verify_text_target_candidates(candidates=candidates)
        if not verified_candidates:
            fallback = self._fallback_select_target(context.target_node)
            fallback["text_target_candidates"] = candidates
            fallback["text_target_candidate_verification"] = verification
            fallback["text_target_candidate_evaluation"] = {
                "decision": "fallback",
                "selected_candidate_id": None,
                "evaluations": [],
                "reason": "all_candidates_closed_book_solvable",
            }
            return fallback

        evaluation_payload = {
            "target_node": context.target_node,
            "predecessor_chain": self._format_predecessor_chain(context),
            "candidates": verified_candidates,
        }
        try:
            evaluation = self._generate_json(
                system=PROMPT_EVALUATE_TEXT_TARGET_CANDIDATES,
                user_payload=evaluation_payload,
                trace_label="evaluate_text_target_candidates",
                max_tokens=max(self.max_tokens, 1800),
            )
        except Exception as exc:
            fallback = self._fallback_select_target(context.target_node)
            fallback["text_target_candidates"] = candidates
            fallback["text_target_candidate_verification"] = verification
            fallback["text_target_candidate_evaluation"] = {
                "decision": "fallback",
                "selected_candidate_id": None,
                "evaluations": [],
                "reason": "candidate_evaluation_error",
            }
            fallback["writer_warning"] = self._writer_warning_entry(
                stage="evaluate_text_target_candidates",
                error=exc,
            )
            return fallback

        selected_candidate = self._selected_target_candidate(
            candidates=verified_candidates,
            evaluation=evaluation,
        )
        if selected_candidate is None:
            fallback = self._fallback_select_target(context.target_node)
            fallback["text_target_candidates"] = candidates
            fallback["text_target_candidate_verification"] = verification
            fallback["text_target_candidate_evaluation"] = evaluation
            fallback["writer_warning"] = self._writer_warning_entry(
                stage="evaluate_text_target_candidates_selection",
                error=ValueError("Text target evaluator did not select a valid generated candidate."),
            )
            return fallback

        result = dict(selected_candidate)
        result["text_target_candidates"] = candidates
        result["text_target_candidate_verification"] = verification
        result["text_target_candidate_evaluation"] = evaluation
        return result

    def _select_image_target_ask(self, *, context: WriterContext) -> dict[str, Any]:
        target_image_url = self._target_image_url(context.target_node)
        generation_payload = {
            "base_search_query": self._image_search_query(context.target_node),
            "target_node": context.target_node,
        }
        try:
            generated = self._generate_json(
                system=PROMPT_SELECT_IMAGE_TARGET,
                user_payload=generation_payload,
                trace_label="select_target_ask_image_candidates",
                image_url=target_image_url,
                max_tokens=max(self.max_tokens, 2400),
            )
        except Exception as exc:
            fallback = self._fallback_select_target(context.target_node)
            fallback["image_target_candidates"] = []
            fallback["image_target_candidate_evaluation"] = {
                "decision": "fallback",
                "selected_candidate_id": None,
                "evaluations": [],
                "reason": "candidate_generation_error",
            }
            fallback["writer_warning"] = self._writer_warning_entry(
                stage="select_target_ask_image_candidates",
                error=exc,
            )
            return fallback

        candidates = self._normalize_image_target_candidates(generated.get("candidates"))
        if not candidates:
            fallback = self._fallback_select_target(context.target_node)
            fallback["image_target_candidates"] = []
            fallback["image_target_candidate_evaluation"] = {
                "decision": "fallback",
                "selected_candidate_id": None,
                "evaluations": [],
                "reason": "no_valid_generated_candidates",
                "raw_generation": generated,
            }
            fallback["writer_warning"] = self._writer_warning_entry(
                stage="select_target_ask_image_candidates_parse",
                error=ValueError("Image target candidate generation returned no usable candidates."),
            )
            return fallback

        verified_candidates, visual_verification = self._verify_image_target_candidates(
            candidates=candidates,
            image_url=target_image_url,
        )
        evaluation_payload = {
            "base_search_query": self._image_search_query(context.target_node),
            "target_node": context.target_node,
            "candidates": verified_candidates,
        }
        try:
            evaluation = self._generate_json(
                system=PROMPT_EVALUATE_IMAGE_TARGET_CANDIDATES,
                user_payload=evaluation_payload,
                trace_label="evaluate_image_target_candidates",
                image_url=target_image_url,
                max_tokens=max(self.max_tokens, 2400),
            )
        except Exception as exc:
            fallback = self._fallback_select_target(context.target_node)
            fallback["image_target_candidates"] = candidates
            fallback["image_target_candidate_verification"] = visual_verification
            fallback["image_target_candidate_evaluation"] = {
                "decision": "fallback",
                "selected_candidate_id": None,
                "evaluations": [],
                "reason": "candidate_evaluation_error",
            }
            fallback["writer_warning"] = self._writer_warning_entry(
                stage="evaluate_image_target_candidates",
                error=exc,
            )
            return fallback

        selected_candidate = self._selected_target_candidate(
            candidates=verified_candidates,
            evaluation=evaluation,
        )
        if selected_candidate is None:
            fallback = self._fallback_select_target(context.target_node)
            fallback["image_target_candidates"] = candidates
            fallback["image_target_candidate_verification"] = visual_verification
            fallback["image_target_candidate_evaluation"] = evaluation
            fallback["writer_warning"] = self._writer_warning_entry(
                stage="evaluate_image_target_candidates_selection",
                error=ValueError("Image target evaluator did not select a valid generated candidate."),
            )
            return fallback

        result = dict(selected_candidate)
        result["image_target_candidates"] = candidates
        result["image_target_candidate_verification"] = visual_verification
        result["image_target_candidate_evaluation"] = evaluation
        return result

    def _verify_image_target_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
        image_url: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Keep questions that require the image and are answerable from it."""
        model_client = self.ask_target_verify_model_client
        model = self.ask_target_verify_model
        if model_client is None or not image_url:
            return candidates, {
                "decision": "skip",
                "reason": "no_verification_model" if model_client is None else "no_target_image",
                "kept_candidate_ids": [item["candidate_id"] for item in candidates],
                "evaluations": [],
            }

        questions = [
            {
                "candidate_id": item["candidate_id"],
                "question": item["ask_target"],
            }
            for item in candidates
        ]
        try:
            with_image = self._generate_json(
                system=PROMPT_ANSWER_VISUAL_TARGET_CANDIDATES,
                user_payload={"candidates": questions},
                trace_label="verify_image_target_answers_with_image",
                image_url=image_url,
                model_client=model_client,
                model=model,
                max_tokens=max(self.max_tokens, 1600),
            )
            without_image = self._generate_json(
                system=PROMPT_ANSWER_TEXT_ONLY_TARGET_CANDIDATES,
                user_payload={
                    "instruction": "图片未提供，请根据常识作答。",
                    "candidates": questions,
                },
                trace_label="verify_image_target_answers_without_image",
                model_client=model_client,
                model=model,
                max_tokens=max(self.max_tokens, 1600),
            )
            judgment = self._generate_json(
                system=PROMPT_JUDGE_VISUAL_TARGET_ANSWERS,
                user_payload={
                    "candidates": [
                        {
                            "candidate_id": item["candidate_id"],
                            "question": item["ask_target"],
                            "gold_answer": item["answer"],
                        }
                        for item in candidates
                    ],
                    "with_image_answers": with_image.get("answers") or [],
                    "without_image_answers": without_image.get("answers") or [],
                },
                trace_label="judge_image_target_answerability",
                model_client=model_client,
                model=model,
                max_tokens=max(self.max_tokens, 1800),
            )
        except Exception as exc:
            return candidates, {
                "decision": "skip",
                "reason": "verification_error",
                "error": f"{exc.__class__.__name__}: {exc}",
                "kept_candidate_ids": [item["candidate_id"] for item in candidates],
                "evaluations": [],
            }

        evaluations = judgment.get("evaluations") or []
        if not isinstance(evaluations, list):
            evaluations = []
        passed_ids = {
            str(item.get("candidate_id") or "").strip()
            for item in evaluations
            if isinstance(item, dict)
            and item.get("with_image_correct") is True
            and item.get("without_image_correct") is False
            and item.get("pass") is True
        }
        filtered = [item for item in candidates if item["candidate_id"] in passed_ids]
        filtered_candidate_ids = [
            item["candidate_id"]
            for item in candidates
            if item["candidate_id"] not in passed_ids
        ]
        skipped = not filtered
        kept = candidates if skipped else filtered
        return kept, {
            "decision": "skip_all_filtered" if skipped else "filter",
            "reason": "all_candidates_filtered; original candidates retained" if skipped else "visual_answerability_filter",
            "kept_candidate_ids": [item["candidate_id"] for item in kept],
            "filtered_candidate_ids": filtered_candidate_ids,
            "with_image_answers": with_image.get("answers") or [],
            "without_image_answers": without_image.get("answers") or [],
            "evaluations": evaluations,
        }

    def _verify_text_target_candidates(
        self,
        *,
        candidates: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Remove final-text questions answerable by the configured closed-book model."""

        model_client = self.ask_target_verify_model_client
        model = self.ask_target_verify_model
        if model_client is None or not model:
            return candidates, {
                "decision": "skip",
                "reason": "no_verification_model",
                "kept_candidate_ids": [item["candidate_id"] for item in candidates],
                "filtered_candidate_ids": [],
                "answers": [],
                "evaluations": [],
            }

        questions = [
            {
                "candidate_id": item["candidate_id"],
                "question": item["ask_target"],
            }
            for item in candidates
        ]
        try:
            answers = self._generate_json(
                system=PROMPT_ANSWER_TEXT_TARGET_CANDIDATES,
                user_payload={"candidates": questions},
                trace_label="verify_text_target_answers_closed_book",
                model_client=model_client,
                model=model,
                max_tokens=max(self.max_tokens, 1600),
            )
            judgment = self._generate_json(
                system=PROMPT_JUDGE_TEXT_TARGET_ANSWERS,
                user_payload={
                    "candidates": [
                        {
                            "candidate_id": item["candidate_id"],
                            "question": item["ask_target"],
                            "gold_answer": item["answer"],
                        }
                        for item in candidates
                    ],
                    "closed_book_answers": answers.get("answers") or [],
                },
                trace_label="judge_text_target_closed_book_answers",
                model_client=model_client,
                model=model,
                max_tokens=max(self.max_tokens, 1600),
            )
        except Exception as exc:
            return candidates, {
                "decision": "skip",
                "reason": "verification_error",
                "error": f"{exc.__class__.__name__}: {exc}",
                "kept_candidate_ids": [item["candidate_id"] for item in candidates],
                "filtered_candidate_ids": [],
                "answers": [],
                "evaluations": [],
            }

        evaluations = judgment.get("evaluations") or []
        if not isinstance(evaluations, list):
            evaluations = []
        solved_ids = {
            str(item.get("candidate_id") or "").strip()
            for item in evaluations
            if isinstance(item, dict) and item.get("correct") is True
        }
        kept = [item for item in candidates if item["candidate_id"] not in solved_ids]
        return kept, {
            "decision": "filter",
            "reason": "closed_book_shortcut_filter",
            "kept_candidate_ids": [item["candidate_id"] for item in kept],
            "filtered_candidate_ids": [item["candidate_id"] for item in candidates if item["candidate_id"] in solved_ids],
            "answers": answers.get("answers") or [],
            "evaluations": evaluations,
        }

    @classmethod
    def _normalize_image_target_candidates(cls, raw_candidates: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_candidates, list):
            return []
        candidates: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for index, raw_candidate in enumerate(raw_candidates, start=1):
            if not isinstance(raw_candidate, dict):
                continue
            ask_target = cls._ensure_question(str(raw_candidate.get("ask_target") or "").strip())
            answer = str(raw_candidate.get("answer") or "").strip()
            if not ask_target or not answer:
                continue
            candidate_id = str(raw_candidate.get("candidate_id") or f"candidate_{index}").strip()
            if not candidate_id or candidate_id in used_ids:
                candidate_id = f"candidate_{index}"
            used_ids.add(candidate_id)
            supporting_facts = raw_candidate.get("supporting_facts") or []
            if not isinstance(supporting_facts, list):
                supporting_facts = []
            visual_reasoning = raw_candidate.get("visual_reasoning") or []
            if not isinstance(visual_reasoning, list):
                visual_reasoning = []
            candidate = dict(raw_candidate)
            candidate.update(
                {
                    "candidate_id": candidate_id,
                    "ask_target": ask_target,
                    "answer": answer,
                    "supporting_facts": [
                        str(item).strip()
                        for item in supporting_facts
                        if str(item).strip()
                    ],
                    "visual_reasoning": [
                        str(item).strip()
                        for item in visual_reasoning
                        if str(item).strip()
                    ],
                }
            )
            candidates.append(candidate)
        return candidates

    @classmethod
    def _normalize_text_target_candidates(cls, raw_candidates: Any) -> list[dict[str, Any]]:
        if not isinstance(raw_candidates, list):
            return []
        candidates: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for index, raw_candidate in enumerate(raw_candidates, start=1):
            if not isinstance(raw_candidate, dict):
                continue
            ask_target = cls._ensure_question(str(raw_candidate.get("ask_target") or "").strip())
            answer = str(raw_candidate.get("answer") or "").strip()
            if not ask_target or not answer:
                continue
            candidate_id = str(raw_candidate.get("candidate_id") or f"candidate_{index}").strip()
            if not candidate_id or candidate_id in used_ids:
                candidate_id = f"candidate_{index}"
            used_ids.add(candidate_id)
            supporting_facts = raw_candidate.get("supporting_facts") or []
            if not isinstance(supporting_facts, list):
                supporting_facts = []
            candidate = dict(raw_candidate)
            candidate.update(
                {
                    "candidate_id": candidate_id,
                    "ask_target": ask_target,
                    "answer": answer,
                    "supporting_facts": [
                        str(item).strip()
                        for item in supporting_facts
                        if str(item).strip()
                    ],
                    "reasoning": str(raw_candidate.get("reasoning") or "").strip(),
                    "support": str(raw_candidate.get("support") or "").strip(),
                }
            )
            candidates.append(candidate)
        return candidates

    @staticmethod
    def _selected_image_target_candidate(
        *,
        candidates: list[dict[str, Any]],
        evaluation: dict[str, Any],
    ) -> dict[str, Any] | None:
        if str(evaluation.get("decision") or "").strip().lower() != "select":
            return None
        selected_id = str(evaluation.get("selected_candidate_id") or "").strip()
        if not selected_id:
            return None
        evaluations = evaluation.get("evaluations") or []
        if isinstance(evaluations, list):
            selected_evaluation = next(
                (
                    item
                    for item in evaluations
                    if isinstance(item, dict)
                    and str(item.get("candidate_id") or "").strip() == selected_id
                ),
                None,
            )
            if selected_evaluation is not None and selected_evaluation.get("valid") is not True:
                return None
        return next(
            (
                dict(candidate)
                for candidate in candidates
                if str(candidate.get("candidate_id") or "").strip() == selected_id
            ),
            None,
        )

    @staticmethod
    def _selected_target_candidate(
        *,
        candidates: list[dict[str, Any]],
        evaluation: dict[str, Any],
    ) -> dict[str, Any] | None:
        if str(evaluation.get("decision") or "").strip().lower() != "select":
            return None
        selected_id = str(evaluation.get("selected_candidate_id") or "").strip()
        if not selected_id:
            return None
        evaluations = evaluation.get("evaluations") or []
        if isinstance(evaluations, list):
            selected_evaluation = next(
                (
                    item
                    for item in evaluations
                    if isinstance(item, dict)
                    and str(item.get("candidate_id") or "").strip() == selected_id
                ),
                None,
            )
            if selected_evaluation is not None and selected_evaluation.get("valid") is not True:
                return None
        return next(
            (
                dict(candidate)
                for candidate in candidates
                if str(candidate.get("candidate_id") or "").strip() == selected_id
            ),
            None,
        )

    @classmethod
    def _format_predecessor_chain(cls, context: WriterContext) -> str:
        """Format the sampled path as ``object --relation--> next object``."""
        if not context.hops:
            return ""

        first_hop = context.hops[0]
        parts = [
            cls._compress_hop_prompt_label(
                first_hop.src_content,
                fallback=first_hop.src_node_id,
            )
        ]
        for hop in context.hops:
            relation = str(hop.relation or hop.edge_type or "is connected to").strip()
            target = cls._compress_hop_prompt_label(
                hop.dst_content,
                fallback=hop.dst_node_id,
            )
            parts.append(f"--{relation}--> {target}")
        return " ".join(parts)

    def build_entry_hop(
        self,
        *,
        path: PathCandidate,
        context: WriterContext,
        hop_summaries: list[dict[str, Any]],
        target_ask: dict[str, Any],
    ) -> dict[str, Any]:
        """Return the question-facing hop 0 for either a text or image entry."""
        if not context.hops or not hop_summaries:
            target = str(context.target_node.get("title") or context.target_node.get("node_id") or "the target").strip()
            source_description = str(context.target_node.get("description") or context.target_node.get("summary") or "").strip()
            clue = self._shorten_text(source_description, limit=180) or "the described starting subject"
            return {
                "hop_index": 0,
                "source": "-",
                "target": target,
                "statement": self._ensure_declarative_statement(f"{clue} was {target}"),
                "relation": "is",
                "retrieval_query": "",
                "edge_id": "",
                "src_node_id": None,
                "dst_node_id": context.target_node.get("node_id"),
                "entry_kind": "text",
                "supporting_facts": [clue] if clue else [],
                "why_relevant": "Fallback entry hop generated because no path hop was available.",
            }

        first_hop = context.hops[0]
        first_summary = dict(hop_summaries[0])
        if path.trajectory.starts_with_image:
            statement = self._select_image_entry_statement(first_summary)
            return {
                **first_summary,
                "hop_index": 0,
                "source": "-",
                "statement": statement,
                "entry_kind": "image",
                "mark": "image",
            }

        source_node = first_hop.src_content
        target = str(source_node.get("title") or first_summary.get("source") or first_hop.src_node_id).strip()
        target_aliases = self._forbidden_source_labels(source_node)
        path_statements = [
            {
                "hop_index": item.get("hop_index"),
                "statement": item.get("statement"),
            }
            for item in hop_summaries
            if isinstance(item, dict) and item.get("statement")
        ]
        source_description = str(source_node.get("description") or source_node.get("summary") or "").strip()
        payload = {
            "target": target,
            "source_description": source_description,
            "target_aliases": target_aliases,
            "path_statements": path_statements,
            "target_ask": str(target_ask.get("ask_target") or "").strip(),
        }
        if self.model_client is None:
            return self._fallback_text_entry_hop(
                first_hop=first_hop,
                target=target,
                source_node=source_node,
                target_aliases=target_aliases,
            )
        try:
            parsed = self._generate_json(
                system=PROMPT_BUILD_TEXT_OPENING_HOP,
                user_payload=payload,
                trace_label="build_text_entry_hop",
            )
        except Exception as exc:
            fallback = self._fallback_text_entry_hop(
                first_hop=first_hop,
                target=target,
                source_node=source_node,
                target_aliases=target_aliases,
            )
            fallback["writer_warning"] = self._writer_warning_entry(stage="build_text_entry_hop", error=exc)
            return fallback

        statement = self._ensure_declarative_statement(str(parsed.get("opening_statement") or "").strip())
        supporting_facts = parsed.get("supporting_facts") or []
        if not isinstance(supporting_facts, list):
            supporting_facts = []
        supporting_facts = [str(item).strip() for item in supporting_facts if str(item).strip()]
        why_relevant = str(parsed.get("why_relevant") or "").strip()
        if not self._valid_text_entry_statement(statement=statement, target=target, target_aliases=target_aliases):
            fallback = self._fallback_text_entry_hop(
                first_hop=first_hop,
                target=target,
                source_node=source_node,
                target_aliases=target_aliases,
            )
            fallback["writer_warning"] = self._writer_warning_entry(
                stage="build_text_entry_hop_validation",
                error=ValueError("Opening statement did not identify the target exactly once after an alias-free clue."),
            )
            return fallback
        return {
            "hop_index": 0,
            "source": "-",
            "target": target,
            "statement": statement,
            "relation": "is",
            "retrieval_query": "",
            "edge_id": "",
            "src_node_id": None,
            "dst_node_id": first_hop.src_node_id,
            "entry_kind": "text",
            "supporting_facts": supporting_facts,
            "why_relevant": why_relevant,
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
        question_hop_summaries, image_bridge_normalization = self._normalize_question_hops(
            path=path,
            context=context,
            hop_summaries=raw_hop_summaries,
        )
        entry_context_hops = list(question_hop_summaries)
        raw_target_ask = self.select_target_ask(context=context)
        question_hop_summaries, question_target_ask, question_terminal_bridge, image_target_terminal_normalization = self._normalize_question_terminal_step(
            path=path,
            context=context,
            hop_summaries=question_hop_summaries,
            raw_target_ask=raw_target_ask,
        )
        entry_hop = self.build_entry_hop(
            path=path,
            context=context,
            hop_summaries=entry_context_hops,
            target_ask=question_target_ask,
        )
        if path.trajectory.starts_with_image:
            compose_hops = [entry_hop, *question_hop_summaries[1:]]
        else:
            compose_hops = [entry_hop, *question_hop_summaries]
        compose_hops = self._renumber_question_hops(compose_hops)
        draft_warnings = self._collect_writer_warnings(
            raw_hop_summaries,
            question_hop_summaries,
            image_bridge_normalization,
            entry_hop,
            raw_target_ask,
            image_target_terminal_normalization,
        )
        answer_type = self._default_answer_type(context.target_node)
        starting_image_url = self._starting_image_url(path=path, graph=graph)
        if self.model_client is None:
            return self._draft_with_writer_warnings(
                self._fallback_compose_question(
                    path=path,
                    hop_summaries=compose_hops,
                    target_ask=question_target_ask,
                    answer_type=answer_type,
                    raw_target_ask=raw_target_ask,
                    raw_hop_summaries=raw_hop_summaries,
                    image_bridge_normalization=image_bridge_normalization,
                    image_target_terminal_normalization=image_target_terminal_normalization,
                    question_terminal_bridge=question_terminal_bridge,
                    entry_hop=entry_hop,
                    starting_image_url=starting_image_url,
                    writer_context=context.to_dict(),
                ),
                warnings=draft_warnings,
            )
        compose_payload = self._compose_question_payload(
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
                    hop_summaries=compose_hops,
                    target_ask=question_target_ask,
                    answer_type=answer_type,
                    raw_target_ask=raw_target_ask,
                    raw_hop_summaries=raw_hop_summaries,
                    image_bridge_normalization=image_bridge_normalization,
                    image_target_terminal_normalization=image_target_terminal_normalization,
                    question_terminal_bridge=question_terminal_bridge,
                    entry_hop=entry_hop,
                    starting_image_url=starting_image_url,
                    writer_context=context.to_dict(),
                ),
                warnings=draft_warnings,
            )
        question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        answer = str(raw_target_ask.get("answer") or "").strip()
        if not question or not answer or self._looks_like_chain_narration(question):
            try:
                rewritten = self._rewrite_chain_narration(
                    hop_summaries=compose_hops,
                    target_ask=question_target_ask,
                    image_url=starting_image_url,
                )
            except Exception as exc:
                draft_warnings.append(self._writer_warning_entry(stage="rewrite_chain_narration", error=exc))
                rewritten = None
            if rewritten is not None:
                question = rewritten
        if not question or not answer:
            return self._draft_with_writer_warnings(
                self._fallback_compose_question(
                    path=path,
                    hop_summaries=compose_hops,
                    target_ask=question_target_ask,
                    answer_type=answer_type,
                    raw_target_ask=raw_target_ask,
                    raw_hop_summaries=raw_hop_summaries,
                    image_bridge_normalization=image_bridge_normalization,
                    image_target_terminal_normalization=image_target_terminal_normalization,
                    question_terminal_bridge=question_terminal_bridge,
                    entry_hop=entry_hop,
                    starting_image_url=starting_image_url,
                    writer_context=context.to_dict(),
                ),
                warnings=draft_warnings,
            )
        return self._draft_with_writer_warnings(
            QuestionDraft(
                question=question,
                answer=answer,
                answer_type=answer_type,
                reasoning_steps=compose_hops,
                used_evidence_ids=[hop.edge_id for hop in context.hops],
                metadata={
                    "path_id": path.path_id,
                    "entry_hop": entry_hop,
                    "compose_payload": compose_payload,
                    "compose_result": {
                        "raw_response": parsed,
                        "analysis": str(parsed.get("analysis") or "").strip(),
                        "question": question,
                    },
                    "raw_hop_summaries": raw_hop_summaries,
                    "image_bridge_normalization": image_bridge_normalization,
                    "starting_image_url": starting_image_url,
                    "target_ask": raw_target_ask,
                    "question_target_ask": question_target_ask,
                    "question_terminal_bridge": question_terminal_bridge,
                    "image_target_terminal_normalization": image_target_terminal_normalization,
                    "writer_context": context.to_dict(),
                    **(
                        {
                            "image_target_candidates": list(
                                raw_target_ask.get("image_target_candidates") or []
                            ),
                            "image_target_candidate_verification": dict(
                                raw_target_ask.get("image_target_candidate_verification") or {}
                            ),
                            "image_target_candidate_evaluation": dict(
                                raw_target_ask.get("image_target_candidate_evaluation") or {}
                            ),
                        }
                        if context.target_node.get("node_type") == "image"
                        else {}
                    ),
                    **(
                        {
                            "text_target_candidates": list(
                                raw_target_ask.get("text_target_candidates") or []
                            ),
                            "text_target_candidate_verification": dict(
                                raw_target_ask.get("text_target_candidate_verification") or {}
                            ),
                            "text_target_candidate_evaluation": dict(
                                raw_target_ask.get("text_target_candidate_evaluation") or {}
                            ),
                        }
                        if context.target_node.get("node_type") == "text"
                        else {}
                    ),
                },
            ),
            warnings=draft_warnings,
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
            answer=draft.answer,
            hops=draft.reasoning_steps,
            target_ask=draft.metadata.get("question_target_ask"),
            question_terminal_bridge=draft.metadata.get("question_terminal_bridge"),
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
        max_tokens: int | None = None,
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
                max_tokens=self.max_tokens if max_tokens is None else max_tokens,
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
            "mark": "image",
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
        model_client = (
            self.ask_target_verify_model_client
            or self.image_bridge_model_client
            or self.model_client
        )
        model = self.ask_target_verify_model or self.image_bridge_model or self.model
        question_target_ask = dict(raw_target_ask)
        target_value = str(raw_target_ask.get("answer") or final_hop_summary.get("target") or "").strip()
        diagnostic: dict[str, Any] = {
            "hop_index": final_hop.hop_index,
            "image_node_id": final_hop.dst_node_id,
            "image_label": image_label,
            "model_alias": model,
            "decision": "keep_image",
            "reason": "no_ask_target_verify_model_available",
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
        question_target_ask["mark"] = "image"
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
            "mark": "image",
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
        hop_summaries: list[dict[str, Any]],
        target_ask: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "hop_facts": [
                {
                    "hop_index": item.get("hop_index"),
                    "source": item.get("source"),
                    "target": item.get("target"),
                    "statement": item.get("statement"),
                    "mark": item.get("mark"),
                }
                for item in hop_summaries
            ],
            "target_ask": {
                "ask_target": target_ask.get("ask_target"),
                "mark": target_ask.get("mark"),
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
    def _difficulty_enhancement_payload(
        *,
        question: str,
        answer: str,
        hops: list[dict[str, Any]],
        target_ask: dict[str, Any] | None = None,
        question_terminal_bridge: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        reasoning_chain = [
            {
                "hop_index": item.get("hop_index"),
                "source": item.get("source"),
                "target": item.get("target"),
                "statement": item.get("statement"),
                "relation": item.get("relation"),
                "retrieval_query": item.get("retrieval_query"),
                "mark": item.get("mark"),
            }
            for item in hops
        ]
        target_ask = target_ask if isinstance(target_ask, dict) else {}
        question_terminal_bridge = (
            question_terminal_bridge
            if isinstance(question_terminal_bridge, dict)
            else {}
        )
        if target_ask.get("mark") == "image":
            removed_hop = question_terminal_bridge.get("removed_question_hop") or {}
            if not isinstance(removed_hop, dict):
                removed_hop = {}
            reasoning_chain.append(
                {
                    "hop_index": removed_hop.get("hop_index", len(reasoning_chain)),
                    "source": question_terminal_bridge.get("source"),
                    "target": question_terminal_bridge.get("target_image"),
                    "statement": target_ask.get("ask_target"),
                    "relation": removed_hop.get("relation"),
                    "retrieval_query": removed_hop.get("retrieval_query"),
                    "mark": "image",
                    "terminal_question": True,
                }
            )
        return {
            "question": question,
            "answer": answer,
            "reasoning_chain": reasoning_chain,
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

    def _fallback_text_entry_hop(
        self,
        *,
        first_hop: HopContext,
        target: str,
        source_node: dict[str, Any],
        target_aliases: list[str],
    ) -> dict[str, Any]:
        clue = self._fallback_select_source(source_node, forbidden_labels=target_aliases)
        statement = self._ensure_declarative_statement(f"{clue} was {target}")
        return {
            "hop_index": 0,
            "source": "-",
            "target": target,
            "statement": statement,
            "relation": "is",
            "retrieval_query": "",
            "edge_id": "",
            "src_node_id": None,
            "dst_node_id": first_hop.src_node_id,
            "entry_kind": "text",
            "supporting_facts": [clue] if clue else [],
            "why_relevant": "Fallback entry hop generated from the source node description.",
        }

    @classmethod
    def _valid_text_entry_statement(
        cls,
        *,
        statement: str,
        target: str,
        target_aliases: list[str],
    ) -> bool:
        if not statement or not target:
            return False
        normalized_statement = cls._normalize_label(statement)
        normalized_target = cls._normalize_label(target)
        if not normalized_statement or not normalized_target:
            return False
        if len(re.findall(rf"(?<!\w){re.escape(normalized_target)}(?!\w)", normalized_statement)) != 1:
            return False
        target_start = normalized_statement.find(normalized_target)
        clue_prefix = normalized_statement[:target_start].strip()
        if not clue_prefix:
            return False
        for alias in target_aliases:
            normalized_alias = cls._normalize_label(alias)
            if normalized_alias and re.search(rf"(?<!\w){re.escape(normalized_alias)}(?!\w)", clue_prefix):
                return False
        return True

    @staticmethod
    def _renumber_question_hops(hops: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{**dict(hop), "hop_index": index} for index, hop in enumerate(hops)]

    @classmethod
    def _fallback_image_entry_statement(cls, first_hop_summary: dict[str, Any]) -> str:
        statement = str(first_hop_summary.get("statement") or "").strip()
        if not statement:
            return "This image provides the next clue."
        entry_statement = cls._normalize_image_reference(statement)
        if not re.search(r"[.?!]$", entry_statement):
            entry_statement = entry_statement.rstrip(" ,;:") + "."
        return entry_statement

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

    def _select_image_entry_statement(self, first_hop_summary: dict[str, Any]) -> str:
        fallback = self._fallback_image_entry_statement(first_hop_summary)
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
        target_ask: dict[str, Any],
        answer_type: str,
        raw_target_ask: dict[str, Any] | None = None,
        raw_hop_summaries: list[dict[str, Any]] | None = None,
        image_bridge_normalization: list[dict[str, Any]] | None = None,
        image_target_terminal_normalization: dict[str, Any] | None = None,
        question_terminal_bridge: dict[str, Any] | None = None,
        entry_hop: dict[str, Any] | None = None,
        starting_image_url: str | None = None,
        writer_context: dict[str, Any] | None = None,
    ) -> QuestionDraft:
        del path
        raw_target_ask = dict(raw_target_ask or target_ask)
        question_target_ask = dict(target_ask)
        hop_text = " ".join(str(item.get("statement") or "").strip() for item in hop_summaries if item.get("statement"))
        ask_target = str(question_target_ask.get("ask_target") or "What is the final answer?")
        answer = str(raw_target_ask.get("answer") or question_target_ask.get("answer") or "unknown")
        question = QuestionWriter._clean_composed_question(f"{hop_text} {ask_target}")
        metadata: dict[str, Any] = {
            "entry_hop": dict(entry_hop or {}),
            "target_ask": raw_target_ask,
            "question_target_ask": question_target_ask,
        }
        if "image_target_candidates" in raw_target_ask:
            metadata["image_target_candidates"] = list(raw_target_ask.get("image_target_candidates") or [])
        if "image_target_candidate_verification" in raw_target_ask:
            metadata["image_target_candidate_verification"] = dict(
                raw_target_ask.get("image_target_candidate_verification") or {}
            )
        if "image_target_candidate_evaluation" in raw_target_ask:
            metadata["image_target_candidate_evaluation"] = dict(
                raw_target_ask.get("image_target_candidate_evaluation") or {}
            )
        if "text_target_candidates" in raw_target_ask:
            metadata["text_target_candidates"] = list(raw_target_ask.get("text_target_candidates") or [])
        if "text_target_candidate_verification" in raw_target_ask:
            metadata["text_target_candidate_verification"] = dict(
                raw_target_ask.get("text_target_candidate_verification") or {}
            )
        if "text_target_candidate_evaluation" in raw_target_ask:
            metadata["text_target_candidate_evaluation"] = dict(
                raw_target_ask.get("text_target_candidate_evaluation") or {}
            )
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
        hop_summaries: list[dict[str, Any]],
        target_ask: dict[str, Any],
        image_url: str | None = None,
    ) -> str | None:
        if self.model_client is None:
            return None
        parsed = self._generate_json(
            system=(
                "You are rewriting a bad multi-hop question draft.\n\n"
                "Rewrite the supplied hop facts and target ask into one natural search question.\n"
                "Do not narrate the chain or use phrases like 'starting with', 'then', "
                "'following that clue', or 'using that clue'.\n"
                "Hop 0 has source '-' and establishes the entry entity from either a textual description "
                "or the attached image. Hide every intermediate target name in the final wording.\n"
                "Keep the latent dependency structure and produce a clear final ask.\n\n"
                "Return valid JSON with exactly this field:\n"
                '{"question": "..."}'
            ),
            user_payload=self._compose_question_payload(
                hop_summaries=hop_summaries,
                target_ask=target_ask,
            ),
            trace_label="rewrite_chain_narration",
            image_url=image_url,
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
        "--sampler-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for LLM-guided next-hop selection.",
    )
    parser.add_argument(
        "--history-exposure-model-alias",
        default=os.environ.get("VQA_HISTORY_EXPOSURE_MODEL") or DEFAULT_HISTORY_EXPOSURE_MODEL,
        help="Model alias for sampler history-exposure filtering. Defaults to VQA_HISTORY_EXPOSURE_MODEL or multimodal_process.",
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
        "--ask-target-verify-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for image/text target-ask verification.",
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
        history_exposure_model_client=LLM_WORKER,
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
        ask_target_verify_model_client=LLM_WORKER if args.ask_target_verify_model_alias else None,
        ask_target_verify_model=args.ask_target_verify_model_alias,
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
        f"ask_target_verify_model: "
        f"{args.ask_target_verify_model_alias or 'not configured (verification skipped)'}"
    )
    print(f"neighbor_selection_strategy: {args.neighbor_selection_strategy}")
    print("raw_hop_summaries:")
    print(json.dumps(debug_hop_summaries, ensure_ascii=False, indent=2))
    print("bridge_normalized_hop_summaries:")
    print(json.dumps(debug_normalized_hop_summaries, ensure_ascii=False, indent=2))
    print("question_hop_summaries:")
    print(json.dumps(debug_question_hop_summaries, ensure_ascii=False, indent=2))
    print("entry_hop:")
    print(json.dumps((draft.metadata or {}).get("entry_hop") or {}, ensure_ascii=False, indent=2))
    print("image_bridge_normalization:")
    print(json.dumps(image_bridge_normalization, ensure_ascii=False, indent=2))
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
