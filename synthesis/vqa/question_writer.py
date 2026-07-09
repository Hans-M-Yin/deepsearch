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

PROMPT_COMPOSE_QUESTION = """
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
4. While reducing salience, you must preserve or introduce enough non-shortcut constraints to keep the question uniquely solvable. For example, you may retain the name of one or two objects mentioned at the beginning of the original question to provide an entry point for identification and reasoning.
5. If the original question is artificially tied to a specific source framing (“according to profile X,” “in source Y’s description,” etc.) but the answer is really a real-world fact rather than a document-specific wording question, remove or naturalize that framing instead of keeping it mechanically.
6. If there is an image attached to the question, keep the connection between the question and the image content. Note that the original question may contain a cue sentence like "In the provided image." In the final version of the question, however, the user will only be given the image if it appears at the beginning. So if such a cue does not appear at the beginning, or if the image has not been provided to you, that means the user would need to search for the image online themselves. In that case, to increase the difficulty of the question, you should hide obvious image-related cues. The goal is to avoid signaling in the question that an image search is needed, while still making the rewritten question depend on a particular image in order to be solved. See the examples below.

Preferred rewriting strategies:
1. Use relational, structural, or contextual constraints instead of highly distinctive signals including famous titles, people names, signature works, unique achievements, strong year markers, or iconic paper titles.
2. Remove features that directly expose intermediate entities, but replace them with weaker contextual descriptions rather than simply deleting them.
3. If obfuscation introduces ambiguity, add non-shortcut constraints to eliminate wrong candidates.
4. Avoid an overly explicit hop-by-hop reasoning structure; the order of clues should not mechanically mirror the order of inference steps.
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
        target_label = self._hop_anchor_label(hop.dst_content, fallback=hop.dst_node_id)
        model_client = self.compress_hop_model_client or self.model_client
        model = self.compress_hop_model or self.model
        if model_client is None:
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
        try:
            parsed = self._generate_json(
                system=PROMPT_COMPRESS_HOP,
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
        hop_summaries = self._compress_hops(context.hops)
        opening_package = self.select_opening_package(context=context, hop_summaries=hop_summaries)
        target_ask = self.select_target_ask(context=context)
        draft_warnings = self._collect_writer_warnings(hop_summaries, opening_package, target_ask)
        opening_mode = "image_start" if path.trajectory.starts_with_image else "text_start"
        answer_type = self._default_answer_type(context.target_node)
        if self.model_client is None:
            return self._draft_with_writer_warnings(self._fallback_compose_question(
                path=path,
                hop_summaries=hop_summaries,
                opening_package=opening_package,
                target_ask=target_ask,
                opening_mode=opening_mode,
                answer_type=answer_type,
            ), warnings=draft_warnings)
        compose_hops = hop_summaries
        compose_payload = self._compose_question_payload(
            opening_mode=opening_mode,
            opening_package=opening_package,
            hop_summaries=compose_hops,
            target_ask=target_ask,
        )
        starting_image_url = self._starting_image_url(path=path, graph=graph)
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
                    hop_summaries=hop_summaries,
                    opening_package=opening_package,
                    target_ask=target_ask,
                    opening_mode=opening_mode,
                    answer_type=answer_type,
                ),
                warnings=draft_warnings,
            )
        question = self._clean_composed_question(str(parsed.get("question") or "").strip())
        answer = str(target_ask.get("answer") or "").strip()
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
                    target_ask=target_ask,
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
                hop_summaries=hop_summaries,
                opening_package=opening_package,
                target_ask=target_ask,
                opening_mode=opening_mode,
                answer_type=answer_type,
            ), warnings=draft_warnings)
        return self._draft_with_writer_warnings(QuestionDraft(
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
        target_ask = draft.metadata.get("target_ask") or {}
        obfuscation_payload = self._obfuscation_question_payload(
            question=draft.question,
            hops=draft.reasoning_steps,
            final_ask=str(target_ask.get("ask_target") or "").strip(),
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
        user_payload: dict[str, Any],
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
            return {"task_name": task_name, "parsed": parsed, "error": None}
        except Exception as exc:
            return {"task_name": task_name, "parsed": None, "error": exc}

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
        "--sampler-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for LLM-guided next-hop selection.",
    )
    parser.add_argument(
        "--compress-hop-model-alias",
        default=None,
        help="Optional model alias registered in synthesis/models.json for compress_hop.",
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
    print(f"sampler_model: {args.sampler_model_alias or 'fallback(no llm)'}")
    print(f"neighbor_selection_strategy: {args.neighbor_selection_strategy}")
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
