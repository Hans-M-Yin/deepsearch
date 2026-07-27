"""Image discovery strategy layer for visual search plans.

This module sits above the low-level image search clients. It runs one or more
text-to-image queries, records search traces, applies cheap candidate filters,
creates graph records, and leaves one image_check hook for future MLLM checks.
"""

from __future__ import annotations

import base64
import html
from io import BytesIO
import json
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from hashlib import sha256
from pathlib import Path
import sys
import traceback
from typing import Any
from urllib.parse import urlparse
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "synthesis"

from .edges import Edge, EdgeSource, EdgeType, EvidenceRef
from .evidence import (
    Asset,
    AssetType,
    Evidence,
    EvidenceType,
    RecordStatus,
    SearchEngine,
    SearchSnapshot,
)
from .model_worker import LLM_WORKER, ModelMessage, ModelRequest, ModelResponse, ModelWorkerClient
from .nodes import ImageNode, ImageVariant, NodeType, TextNode
from .search_client import ImageSearchResult, SearchClient, SearchResponse
from .store import JsonlGraphStore
from .visual_planner import SearchQuerySpec, VisualSearchPlan
from .wiki_text_builder import EnhancedReaderClient


def _trace_timing_enabled() -> bool:
    return os.environ.get("SYNTHESIS_TRACE_TIMING", "0") != "0"


def _trace_timing(message: str) -> None:
    if _trace_timing_enabled():
        print(f"[trace]{message}", file=sys.stderr, flush=True)
from .wiki_entity_resolver import WikiEntityResolver


def _jsonify(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


def _short_debug_text(text: str | None, limit: int = 240) -> str | None:
    if text is None:
        return None
    compact = re.sub(r"\s+", " ", str(text)).strip()
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _image_source_kind(url: str | None) -> str:
    if not url:
        return "missing"
    if str(url).startswith("data:"):
        return "data_url"
    return "remote_url"


def _format_debug_image_source(url: str | None) -> str | None:
    if not url:
        return None
    raw = str(url)
    if raw.startswith("data:"):
        header, _, payload = raw.partition(",")
        return f"{header},<payload_len={len(payload)}>"
    return _short_debug_text(raw, limit=240)


def _log_image_debug(label: str, **payload: Any) -> None:
    del label, payload
    return


class ImageCandidateStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"


PROMPT_IMAGE_CHECK = """You are checking whether a candidate image is useful visual evidence for a multimodal deep-search target.

Judge primarily from the image content. Candidate metadata can help disambiguate, but it must not override what is visible in the image.

Accept if the image visibly matches the target or is a useful intermediate visual clue for the target.
Reject if the image is generic, unrelated, too ambiguous, only textually related, a placeholder, or an icon/logo when the target is not asking for one.

Output exactly one block:
<check>
decision: accept|reject
confidence: 0.0-1.0
reason: short reason
visual_fact: visible fact 1
visual_fact: visible fact 2
</check>
"""


PROMPT_IMAGE_RETRIEVAL_CONSISTENCY = """You are judging whether several image-search results converge on the same visual content.

All candidate images were returned for one search query and have already passed an individual check that they are relevant to the query. However, relevance alone is not enough. Keep this query only when at least half of the candidate images depict approximately the same main visual content.

For an event photograph, images count as consistent only when they show the same particular event and approximately the same sub-scene, main people/group, and action. Images from the same broad event but different moments, locations, crowds, performances, or camera situations do not count as consistent.

For a canonical object, artwork, building, album cover, or product, images count as consistent when the same main object is clearly depicted. Different normal views may count only if the object itself remains the stable main content.

Examples:

Query: healthcare workers in protective clothing during the 2003 Hong Kong SARS outbreak
Candidate images: doctors in different hospitals, at different dates, and in different protective-clothing scenes.
Answer: FALSE
Reason: They concern the same outbreak but do not depict one consistent visual scene.

Query: 12 Years a Slave cast and producers accepting the Best Picture Oscar at the 86th Academy Awards in 2014
Candidate images: several photographs of the film's group on the Oscar stage during the acceptance.
Answer: TRUE
Consistent images: 1, 2, 4
Reason: A majority depict the same award-acceptance scene with the same main group and stage context.

Output exactly:
<thinking>Your reasoning process</thinking>
<answer>TRUE/FALSE</answer>
<consistent_images>comma-separated candidate numbers, or none</consistent_images>
<reason>short reason</reason>
"""


PROMPT_IMAGE_RECOVERY_SELECTION = """You are selecting the single best recovered image from one source page for a search query.

These recovered images come from the same webpage and may include duplicates, different preview sizes of the same image, unrelated media from the page, thumbnails, UI graphics, or low-information images.

Selection priorities:
1. Choose the image whose visible content best matches the search query.
2. If multiple images depict the same target equally well, prefer the higher-quality image with the larger visible resolution.
3. Prefer a real scene/image relevant to the query over generic page assets, logos, icons, avatars, or decorative graphics.
4. Always choose exactly one candidate.

Output exactly:
<selection>
decision: select
candidate_index: integer
reason: short reason
</selection>
"""


PROMPT_IMAGE_GROUND = """You are analyzing an accepted image for multimodal graph construction.

Task:
Describe the image and ground only unique, searchable entities visible in or clearly represented by the image. Entity grounding is for linking this image to existing text/entity nodes.

Keep entities only if they are named or uniquely identifiable, such as a person, landmark, movie, book, album, artwork, product, brand, team, organization, event, document, map, or logo. Do not output generic objects such as person, woman, car, building, crowd, red shirt, tree.

You may receive webpage context associated with the image. Use that context only to disambiguate what is visible. Do not invent entities that are not visually supported by the image.

Important grounding rules:
1. Include indirect but clearly visible searchable entities when they are visually grounded.
   - Examples: an Adidas or Nike logo on clothing, a team crest on a jersey, a visible brand mark on an object. These marks point to a unique brand.
2. For every output entity, ensure the entity name is unambiguous and can be used directly to look up the correct Wikipedia page.
   - Prefer full canonical names over short or ambiguous surface forms.
   - Prefer "Kobe Bryant" over "Kobe", "Los Angeles Lakers" over "Lakers", and "Eiffel Tower" over "the tower".
   
Output guidance:
1. The second field is a scene-centric or object-centric locator, not an image-centric description.
   Treat the image only as a carrier of evidence, not as the reference frame.
2. It should help a user identify the entity through scene semantics, object semantics, or stable local structure.
   Prefer locators based on:
   - role or identity in the scene
   - action or interaction
   - distinctive clothing, pose, held object, or visible mark
   - relation to another visible person or object
   - stable position within a visible group, row, lineup, or formation
3. If positional language is needed, anchor it to another visible entity or to a stable group structure, not to the image frame.
   Good examples:
   - woman standing beside Cristiano Ronaldo
   - player immediately to the left of the trophy holder
   - front-row center person in the group
   - child sitting on the man's shoulders
   - logo on the front of the jersey
4. DO NOT use any frame-anchored locators such as:
   - left side of the image
   - right side of the image
   - top-right corner if the image
   - upper part of the picture
   - foreground of the image
   - background of the image
   - center of the image
5. Avoid salience-anchored or media-anchored locators such as:
   - main character
   - main subject
   - central figure
   - person in the photo
   - shown in the image
   - depicted here
   These rely too much on the image as an image, rather than on the depicted scene or object structure.
6. For image artifacts such as posters, album covers, screenshots, documents, or maps, still prefer semantic or structural parts over frame coordinates.
   Good examples:
   - face at the far right of the four-person lineup
   - logo on the front of the jersey
   - text in the app header bar
   - emblem on the shield held by the knight
7. Avoid abstract or non-localizable relations such as:
   - depicted in image
   - shown in image
   - associated with image
   - represented in image
   These are too generic and do not help locate the entity.
8. If the image contains multiple people or objects, the locator must disambiguate the target.
   The locator must be concrete and specifically localizable; avoid vague descriptions such as `sponsor logo in the background`.
9. If no stable scene-centric or object-centric locator is available, omit the entity rather than using a vague image-centric relation.
10. `evidence` should be one short sentence explaining the visible cue that supports the grounding.
11. If two surface forms refer to the same entity, output only the canonical one and mention the alias/handle inside `evidence`.
12. Do not ground watermarks, publisher logos, channel bugs, UI overlays, copyright marks...
13. Avoid grounding entities that do not actually require the image and could be identified through text-only search alone—for example, grounding the director from a movie poster when the director’s name can be found directly by searching the film title in text. You should focus on the visual relationships between entities rather than knowledge-based inference jumps.
NOTICE:If there are more than 5 entities, keep only the 5 clearest, most salient, and most certain entities, and ignore the rest.

Examples:
- For a 2025 G20 summit group photo:
  `entity: Emmanuel Macron | front-row center person in the group | visible as the suited male figure standing in the middle of the front row`
- For the Queen II album cover:
  `entity: Roger Taylor | face at the far right of the four-person lineup | visible as the rightmost face in the four-person arrangement`
- For a John Wick 4 poster:
  `entity: Eiffel Tower | landmark behind the man in the black suit | visible rising behind the standing man in the black suit`

Before the final grounding block, output one brief analysis block. For each
entity you plan to ground, explain why the entity is visually supported by the
image and why the webpage context is only being used for disambiguation rather
than to guess the entity. Do not introduce entities in this analysis that you
do not include in the final grounding block.

Output exactly these two blocks, in this order:
<analysis>
Brief grounding rationale for each retained entity.
</analysis>
<ground>
caption: one concise image caption
entity: name | locator | evidence
entity: name | locator | evidence
</ground>
"""


PROMPT_IMAGE_QUERY_ENTITY_FILTER = """You are filtering grounded image entities for multi-hop graph expansion.

Goal:
We only want image-derived entities that add new information beyond the visual query itself.

Task:
Given:
- the source text node title
- the visual query text
- a list of grounded candidate entities from the image

Decide for each candidate whether it should be blocked because it is already mentioned in the query, or is just an alias / handle / surface form of an entity already mentioned in the query.

Block an entity if:
- it is the same entity as one already mentioned or implied in the query
- any form of its name is explicitly present in the query, even when it appears
  inside a larger phrase; do not keep it merely because it is still a distinct
  geographic, organizational, or other real-world entity

Keep an entity if:
- it is a new entity not already present in the query
- it is related to the query subject but still introduces a distinct new entity

Important:  
- Do not block entities merely because they are associated with the query subject.
- Example: if the query mentions Lionel Messi, block "Messi" or "leomessi", but keep "Argentina national football team" unless the query already mentions it.

Output exactly one block:
<filter>
<entity>candidate name | block/keep | short reason</entity>
<entity>candidate name | block/keep | short reason</entity>
</filter>

Every candidate must be written inside its own <entity>...</entity> tag. Do not
use Markdown bullets, a field prefix such as "entity:", or any text outside
the <filter> block.
"""


PROMPT_TEXT_TO_IMAGE_RELATION_REWRITE = """You are rewriting an image-search query into a source-aware graph relation.

Goal:
- The original search query already identifies one unique image or visual scene.
- Rewrite it into a short relation phrase that connects the source text node to that image node more naturally.
- The relation should explain why this image is a meaningful visual neighbor of the source, not merely restate what the image shows.

Requirements:
1. Make only minimal edits to the original search query.
2. Preserve every uniqueness-bearing detail, including the main entity, event, action, date, place, and distinctive scene details.
3. Make the phrase read naturally as a relation from the source node to the image.
4. Prefer replacing repeated mentions of the source title with a pronoun or possessive when this is natural and unambiguous.
5. Do not broaden the query, drop key details, or add unsupported facts.
6. Do not output a full sentence. Output a short phrase only.
7. Avoid vague scaffolds such as "related to a photo", "related to an image", "associated with a picture", or "image that shows" when they do not explain the source-image connection.
8. If the image is tied to the source through an intermediate work, product, label, organization, event, location, award, or other bridge entity, include that bridge explicitly instead of hiding it behind "related to".
9. Pronouns and possessives such as "its", "his", or "her" are encouraged when they clearly refer to the source. For indirect source-image connections, do not let a pronoun hide the bridge; name the bridge explicitly with wording such as "through ..." or "via ...".

Examples:
Source title: Kobe Bryant
Original search query: Kobe Bryant giving his farewell "Mamba Out" speech at center court after his final NBA game on April 13, 2016
Relation: photo of Kobe Bryant giving his farewell "Mamba Out" speech at center court after his final NBA game on April 13, 2016

Source title: Lionel Messi
Original search query: Lionel Messi sleeping while hugging the World Cup trophy after the 2022 FIFA World Cup final
Relation: photo of Lionel Messi sleeping while hugging the World Cup trophy after the 2022 FIFA World Cup final

Source title: Warner Music Group
Original search query: Fleetwood Mac 1977 Rumours album cover released by Warner Bros. Records
Relation: the cover image of Rumours, the 1977 Fleetwood Mac album released through Warner Music Group's Warner Bros. Records label

Source title: The United States Constitution
Original search query: first handwritten page of the United States Constitution beginning with We the People
Relation: image of the United States Constitution's first handwritten page beginning with We the People

Source title: Southern Methodist University
Original search query: five U.S. presidents at the dedication ceremony for the George W. Bush Presidential Center on April 25, 2013
Relation: image from the dedication ceremony for the George W. Bush Presidential Center on its campus on April 25, 2013

Return valid JSON with exactly this field:
{
  "relation": "..."
}
"""


PROMPT_IMAGE_ENTITY_RESOLUTION = """You are selecting the best Wikipedia candidate for linking one image-grounded entity into the multimodal graph.

Given one grounded entity from an image and a candidate list, decide whether one candidate can be confidently selected.

The goal is not to force a match.
Only select when the grounded entity and the candidate clearly refer to the same real-world target.
If the candidate list is ambiguous, too broad, partially related, or insufficiently supported by the visual evidence, return none.

You may receive:
- the grounded entity name
- the grounded entity type
- the locator phrase and visual evidence
- the image caption
- the source text node title
- the source query text
- a candidate list, where some candidates may already exist as local text nodes in the graph

Selection rules:
- Prefer candidates that denote the exact same canonical Wikipedia target as the grounded entity.
- Use the visual evidence and image caption as the primary evidence.
- Use the source node title and source query text only as disambiguation hints.
- Be conservative. If multiple candidates remain plausible, return none.
- Do not select a candidate that is only loosely related to the grounded entity.
- Do not invent a candidate outside the provided list.
- If a selected candidate already exists in the local graph, it will be linked to that existing node.
- If a selected candidate does not exist in the local graph, it will be queued for text-node expansion.

Examples:
- Grounded entity: Meta Orion
  Candidate 0: Ray-Ban Meta
  Candidate 1: Orion (mythology)
  decision: none
  reason: No candidate is clearly the same AR product shown in the image.

- Grounded entity: Los Angeles Lakers
  Candidate 0: Los Angeles Lakers
  Candidate 1: Lakers
  decision: select
  candidate_index: 0
  reason: Exact canonical team entity supported by the grounded name and image evidence.

Output exactly one block:
<selection>
decision: select|none
candidate_index: integer or none
reason: short reason
</selection>
"""


PROMPT_WIKI_INLINE_IMAGE_QUESTION = """
I’m determining whether a user recognizes a particular image or its main content, and I need you to help me come up with a question to ask the user.
You will be given the image, a brief description of the image, and the Wikipedia page the image comes from. The question you propose should focus on the semantic content of the image rather than purely visual details. In other words, it should require world knowledge to answer, such as identifying people, objects, and so on.

Requirements:
1. The question must point to a clearly defined target and be unambiguous.
2. Since you may also be unable to directly identify the people, objects or other entities that have unique names in the image, you should use the provided image description and the subject of the associated Wikipedia page to make a reliable inference about the people and objects shown.
3. Output format: please follow the format below exactly:
<thinking>Your reasoning process</thinking>
<question>The question you propose</question>
<answer>The answer to the question</answer>

Example 1:
(a photo of Trump taking the oath of office)
Wikipedia: Donald Trump
description: Taking the presidential oath of office, administered by Chief Justice John Roberts, on January 20, 2017
<thinking>Based on the image, the man in the foreground wearing a red tie can be identified as Trump. Standing next to him is his wife Melania, and his youngest son Barron, wearing a dark blue tie, is also beside them. The man with a balding hairstyle, seen from behind and administering the oath, can be inferred from the description and world knowledge to be John Roberts. Now I can ask a question to determine whether the user recognizes the main figures shown in the image.</thinking>
<question>Who is the man in the foreground wearing the red tie, and who is the balding man seen from behind?</question>
<answer>Donald Trump; John Roberts</answer>

Example 2:
(a gameplay screenshot)
Wikipedia: City-building game
description: Lincity is a city-building game.
<thinking>The image shows a video game screenshot that appears to be from a city-building simulation game. The description states that “Lincity is a city-building game,” which suggests that this screenshot is from Lincity. So I can directly ask the user what game this is.</thinking>
<question>Which game is shown in this image?</question>
<answer>Lincity, a city-building simulation game</answer>

Example 3:
(a photo of a bus with 'R.I.P Kobe')
Wikipedia: Kobe Byrant
description: Metro Bus in Los Angeles with "RIP Kobe" banner, January 2020
<thinking> This image shows a metro bus in LA, with a 'RIP Kobe' banner. This indicates that the bus is mourning Kobe’s death, so you can directly ask who the bus in this image is commemorating.</thinking>
<question>Who is the bus in the image mourning?</question>
<answer>Kobe Byrant</answer>
"""


PROMPT_WIKI_INLINE_IMAGE_ANSWER = """You are answering a question about an image content.

Rules:.
- Answer as specifically as possible.
- Do not explain your reasoning.
- If you are unsure, output exactly: UNKNOWN
"""


PROMPT_WIKI_INLINE_IMAGE_JUDGE = """
You are judging whether an answer correctly identifies the key information in an image caption. The question and the answer are both provided to you, but the corresponding image is not. Please determine whether a user's response is correct. If the meaning is the same or mostly the same, it should be considered TRUE; the wording does not need to match exactly. If the user's response clearly shows that they do not recognize one of the people, objects, or events in the image, then output FALSE.
Output format:
<thinking>Your reasoning process</thinking>
<answer>TRUE/FALSE</answer>
"""


PROMPT_WIKI_INLINE_ENTITY_UNIQUENESS_FILTER = """
You are filtering grounded entities from a Wikipedia inline image for graph retention.

Goal:
Keep the image only if it contains at least one unique canonical entity that can anchor a useful graph node.
Drop the image if all grounded entities are only generic classes, types, roles, phenomena, species, materials, scene categories, or other non-unique concepts.

Definitions:
- A unique canonical entity has one stable referent in a knowledge graph.
  Examples: Aeron chair, Herman Miller, HMS Beagle, Eiffel Tower, Apollo 11, Queen II.
- A non-unique category is a class or type rather than one stable referent.
  Examples: cumulonimbus cloud, overshooting top, thunderstorm, cathedral, basketball player, dog, steam locomotive.
- Technical or fine-grained terms are still non-unique if they denote a category, phenomenon, or visual pattern rather than one canonical entity.

Task:
You will receive:
- the Wikipedia page title
- a short image description
- grounded candidate entities from the image, each with locator/evidence text

For each candidate entity, decide whether to keep or block it.

Keep an entity only if:
- it denotes a unique canonical entity rather than a generic category; and
- the caption / grounding evidence makes that identification plausible.

Block an entity if:
- it is only a generic class, type, role, phenomenon, species, material, scene category, or part;
- it is a technical term but still denotes a category rather than one canonical entity;
- uniqueness is unclear or doubtful.

Important rules:
1. Do not keep an entity only because it matches the Wikipedia page title. The page itself may describe a generic concept.
2. Human-created named works, products, organizations, vehicles, buildings, artworks, and named historical entities can still be unique even if many copies or instances exist.
3. Natural kinds, cloud types, biological taxa, weather formations, astronomical classes, and generic object categories should usually be blocked unless they clearly denote one named canonical entity.
4. Be conservative. If you are unsure whether the term denotes one stable canonical entity, block it.
5. The final image decision is KEEP only if at least one entity is kept.

Examples:

Wikipedia: Cumulonimbus cloud
description: An anvil-topped thundercloud with a protruding dome above the top.
Grounded candidate entities:
- Cumulonimbus cloud | locator: main storm cloud filling the frame | evidence: the image shows the classic anvil-topped thundercloud form
- Overshooting top | locator: dome above the cloud top | evidence: a protruding dome rises above the anvil top

<filter>
overall_decision: drop
reason: all grounded entities are generic atmospheric categories rather than unique canonical entities
entity: Cumulonimbus cloud | block | weather cloud type, not a unique canonical entity
entity: Overshooting top | block | cloud feature category, not a unique canonical entity
</filter>

Wikipedia: Aeron chair
description: Office chair designed by Don Chadwick and Bill Stumpf.
Grounded candidate entities:
- Aeron chair | locator: office chair in the foreground | evidence: the distinctive mesh-backed chair matches the named product model
- Herman Miller | locator: brand mark on the chair base | evidence: the visible branding supports the named company

<filter>
overall_decision: keep
reason: at least one grounded entity is a unique canonical entity
entity: Aeron chair | keep | named product model with one stable canonical referent
entity: Herman Miller | keep | named company uniquely identified by the grounded evidence
</filter>

Output exactly one block:
<filter>
overall_decision: keep|drop
reason: short reason
entity: candidate name | keep|block | short reason
entity: candidate name | keep|block | short reason
</filter>
"""


PROMPT_WIKI_INLINE_IMAGE_TITLE_CHECK = """You are checking whether a Wikipedia inline image is visually relevant to the subject of a Wikipedia page.

You will receive:
- the Wikipedia page title
- the image itself

Important rules:
1. Judge from the visible image content first.
2. Do not rely on caption text, alt text, surrounding prose, or other metadata that is not visually shown.
3. Accept if the image is directly about the page subject, OR if it provides clear and informative visual context that is still meaningfully related to the page subject.
4. Keep informative contextual images when a reasonable reader would say the image is about this title or helps illustrate this title, even if the subject itself is not the only thing visible.
5. Reject only when the image has little or no visual information, is a tiny/icon-like/decorative asset, is effectively empty, or has no clear visual relation to the page title.
6. For a person page, keep images that show the person, the person's recognizable representation, or a clearly related scene/object strongly associated with that person.
7. For an organization, place, product, vehicle, artwork, event, or topic page, keep images that directly depict the title or provide clearly relevant visual context for it.

Output exactly one block:
<check>
decision: accept|reject
confidence: 0.0-1.0
reason: short reason
visual_fact: visible fact 1
visual_fact: visible fact 2
</check>
"""


@dataclass(slots=True)
class ImageDiscoveryConfig:
    """Cheap gates and retrieval limits for image discovery."""

    per_query_limit: int = 10
    max_images_per_plan: int = 8
    enable_retrieval_consistency_check: bool = True
    retrieval_consistency_max_images: int = 6
    retrieval_consistency_min_images: int = 2
    persist_search_snapshots: bool = False
    min_width: int | None = 120
    min_height: int | None = 120
    allowed_content_types: set[str] | None = None
    rejected_extensions: set[str] = field(default_factory=lambda: {".svg"})
    store_rejected: bool = True
    force_accept_images: bool = False
    precheck_image_urls: bool = True
    precheck_timeout_s: float = 15.0
    precheck_max_bytes: int = 262144
    model_image_max_bytes: int | None = None
    model_image_max_edge: int | None = 2000
    precheck_retries: int = 3
    host_min_interval_s: float = 0.35
    wikimedia_host_min_interval_s: float = 1.25
    wikimedia_429_retry_after_s: float = 15.0
    user_agent: str | None = None
    cache_dir: str | None = None
    upload_cached_images: bool = True
    try_source_page_recovery: bool = True
    source_page_timeout_s: float = 20.0
    image_grounding_context_backend: str = "source_page_reader"
    image_grounding_reader_base_url: str = "http://127.0.0.1:8004"
    image_grounding_reader_timeout_s: float = 40.0
    image_grounding_max_context_chars: int = 6000
    enable_image_entity_queue_verification: bool = True
    image_entity_queue_verify_prepare_model: str | None = None
    image_entity_queue_verify_judge_model: str | None = None
    image_entity_queue_verify_max_reference_images: int = 6
    enable_wiki_inline_entity_uniqueness_filter: bool = True
    enable_visual_plan_post_grounding_filter: bool = True
    visual_plan_min_expandable_entities: int = 2
    visual_plan_self_qa_entity_count: int = 2
    expandable_entity_types: set[str] = field(
        default_factory=lambda: {
            "person",
            "team",
            "organization",
            "event",
            "movie",
            "book",
            "album",
            "brand",
            "product",
            "landmark",
            "document",
            "artwork",
        }
    )


@dataclass(slots=True)
class ImageValidationResult:
    """Result returned by the image_check function."""

    status: ImageCandidateStatus
    confidence: float | None = None
    reason: str | None = None
    drop_candidate: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


@dataclass(slots=True)
class ImageSearchCandidate:
    """One retrieved image candidate before/after validation."""

    candidate_id: str
    source_query: SearchQuerySpec
    source_snapshot: SearchSnapshot
    search_result: ImageSearchResult
    validation: ImageValidationResult
    used_fallback: bool = False
    is_primary: bool = False
    grounded_entities: list[dict[str, Any]] = field(default_factory=list)
    grounded_caption: str | None = None
    visual_facts: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "source_query": self.source_query.to_dict(),
            "source_snapshot": self.source_snapshot.to_dict(),
            "search_result": self.search_result.to_dict(),
            "validation": self.validation.to_dict(),
            "used_fallback": self.used_fallback,
            "is_primary": self.is_primary,
            "grounded_entities": _jsonify(self.grounded_entities),
            "grounded_caption": self.grounded_caption,
            "visual_facts": list(self.visual_facts),
        }


@dataclass(slots=True)
class ResolvedImageAsset:
    cache_key: str
    original_url: str | None
    resolved_url: str | None
    source_page_url: str | None
    model_url: str
    asset_uri: str
    cache_path: str | None
    content_type: str | None
    width: int | None = None
    height: int | None = None
    strategy: str = "direct"

    def to_metadata(self) -> dict[str, Any]:
        return {
            "cache_key": self.cache_key,
            "original_url": self.original_url,
            "resolved_url": self.resolved_url,
            "source_page_url": self.source_page_url,
            "asset_uri": self.asset_uri,
            "cache_path": self.cache_path,
            "content_type": self.content_type,
            "width": self.width,
            "height": self.height,
            "strategy": self.strategy,
        }


@dataclass(slots=True)
class ImageDiscoveryResult:
    """All records produced for one visual search plan."""

    plan_id: str
    image_node: ImageNode | None = None
    edge: Edge | None = None
    image_evidence: Evidence | None = None
    search_evidence: Evidence | None = None
    grounded_edges: list[Edge] = field(default_factory=list)
    candidates: list[ImageSearchCandidate] = field(default_factory=list)
    queued_tasks: list[dict[str, Any]] = field(default_factory=list)
    snapshots: list[SearchSnapshot] = field(default_factory=list)
    fallback_used: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def accepted_images(self) -> list[ImageSearchCandidate]:
        return [
            image
            for image in self.candidates
            if image.validation.status == ImageCandidateStatus.ACCEPTED
        ]

    def usable_images(self) -> list[ImageSearchCandidate]:
        return [
            image
            for image in self.candidates
            if image.validation.status == ImageCandidateStatus.ACCEPTED
        ]

    def primary_image(self) -> ImageSearchCandidate | None:
        for image in self.candidates:
            if image.is_primary:
                return image
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan_id,
            "image_node": self.image_node.to_dict() if self.image_node else None,
            "edge": self.edge.to_dict() if self.edge else None,
            "image_evidence": self.image_evidence.to_dict() if self.image_evidence else None,
            "search_evidence": self.search_evidence.to_dict() if self.search_evidence else None,
            "grounded_edges": [edge.to_dict() for edge in self.grounded_edges],
            "candidates": [image.to_dict() for image in self.candidates],
            "queued_tasks": _jsonify(self.queued_tasks),
            "snapshots": [snapshot.to_dict() for snapshot in self.snapshots],
            "fallback_used": self.fallback_used,
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ImageGroundingContext:
    """Prompt-side context provider output for image grounding."""

    provider: str
    prompt_text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))


class ImageDiscoveryBuilder:
    """Run image discovery for a visual target and persist graph records."""

    builder_name = "image_discovery_builder"

    def __init__(
        self,
        *,
        store: JsonlGraphStore | None = None,
        search_client: SearchClient,
        config: ImageDiscoveryConfig | None = None,
        model_client: ModelWorkerClient | None = None,
        image_check_model_alias: str | None = None,
        wiki_resolver: WikiEntityResolver | None = None,
    ) -> None:
        self.store = store
        self.search_client = search_client
        self.config = config or ImageDiscoveryConfig()
        self.model_client = model_client or LLM_WORKER
        self.image_check_model_alias = image_check_model_alias
        self.wiki_resolver = wiki_resolver or WikiEntityResolver()
        self.reader = EnhancedReaderClient(
            base_url=self.config.image_grounding_reader_base_url,
            timeout_s=self.config.image_grounding_reader_timeout_s,
        )
        self._resolved_image_cache: dict[str, ResolvedImageAsset] = {}
        self._transient_image_cache: dict[str, ResolvedImageAsset] = {}
        self._grounding_context_cache: dict[str, ImageGroundingContext] = {}
        self._host_not_before: dict[str, float] = {}
        self._host_locks: dict[str, threading.Lock] = {}
        self._download_lock = threading.Lock()

    def discover_for_plan(
        self,
        plan: VisualSearchPlan,
        *,
        run_id: str | None = None,
        persist: bool = True,
    ) -> ImageDiscoveryResult:
        """Discover images for one visual plan."""

        total_started = time.perf_counter()
        result = ImageDiscoveryResult(plan_id=plan.plan_id)
        seen_keys: set[str] = set()
        decision_log: list[dict[str, Any]] = []
        _trace_timing(f"[image-discovery] phase=start plan_id={plan.plan_id} queries={len(plan.queries)}")

        started = time.perf_counter()
        result.candidates = self._discover_with_client(
            client=self.search_client,
            plan=plan,
            run_id=run_id,
            seen_keys=seen_keys,
            persist=persist,
            snapshots=result.snapshots,
            decision_log=decision_log,
        )
        _trace_timing(
            f"[image-discovery] stage=search_and_check plan_id={plan.plan_id} elapsed_s={time.perf_counter() - started:.3f} candidates={len(result.candidates)}"
        )
        try:
            result.candidates = result.candidates[: self.config.max_images_per_plan]
            content_checked_candidates = [candidate.to_dict() for candidate in result.candidates]
            consistency = self._apply_retrieval_consistency_check(
                plan=plan,
                candidates=result.candidates,
            )
            result.fallback_used = any(candidate.used_fallback for candidate in result.candidates)
            primary_candidate = self._select_primary_candidate(result.candidates)
            if primary_candidate is not None:
                started = time.perf_counter()
                self._materialize_primary_candidate(
                    result=result,
                    plan=plan,
                    candidate=primary_candidate,
                    run_id=run_id,
                    persist=persist,
                )
                _trace_timing(
                    f"[image-discovery] stage=materialize_primary plan_id={plan.plan_id} elapsed_s={time.perf_counter() - started:.3f} primary_title={primary_candidate.search_result.title!r}"
                )
            result.metadata.update(
                {
                    "query_count": len(plan.queries),
                    "image_count": len(result.candidates),
                    "usable_image_count": len(result.usable_images()),
                    "accepted_image_count": len(result.accepted_images()),
                    "queued_task_count": len(result.queued_tasks),
                    "candidate_decisions": decision_log,
                    "content_checked_candidates": content_checked_candidates,
                    "retrieval_consistency": consistency,
                }
            )
            if persist and self.store is not None:
                self.store.maybe_flush()
            _trace_timing(
                f"[image-discovery] phase=done plan_id={plan.plan_id} elapsed_s={time.perf_counter() - total_started:.3f} accepted={len(result.accepted_images())} kept={'yes' if result.image_node is not None else 'no'}"
            )
            return result
        finally:
            self._clear_transient_assets(result.candidates)

    def discover_for_wiki_inline_image(
        self,
        plan: VisualSearchPlan,
        *,
        search_result: ImageSearchResult,
        run_id: str | None = None,
        persist: bool = True,
    ) -> ImageDiscoveryResult:
        total_started = time.perf_counter()
        result = ImageDiscoveryResult(plan_id=plan.plan_id)
        validation: ImageValidationResult | None = None
        try:
            validation = self._wiki_inline_image_check(
                plan=plan,
                search_result=search_result,
                run_id=run_id,
                persist_asset=False,
            )
            snapshot = self._snapshot_from_wiki_inline_result(search_result, plan=plan, run_id=run_id)
            result.snapshots.append(snapshot)
            if persist:
                self._persist_snapshot(snapshot)

            if not validation.drop_candidate:
                candidate = ImageSearchCandidate(
                    candidate_id=self._candidate_record_id(search_result),
                    source_query=(plan.queries or [SearchQuerySpec.create("", plan.target.evidence_id)])[0],
                    source_snapshot=snapshot,
                    search_result=search_result,
                    validation=validation,
                    used_fallback=False,
                )
                result.candidates = [candidate]
                if validation.status == ImageCandidateStatus.ACCEPTED:
                    self._materialize_primary_candidate(
                        result=result,
                        plan=plan,
                        candidate=candidate,
                        run_id=run_id,
                        persist=False,
                        create_source_edge=False,
                    )
                    provisional_grounded_edge_count = len(result.grounded_edges)
                    provisional_queued_task_count = len(result.queued_tasks)
                    keep_in_graph = self._wiki_inline_result_has_expandable_targets(result)
                    result.metadata.update(
                        {
                            "wiki_inline_keep_in_graph": keep_in_graph,
                            "wiki_inline_grounded_edge_count": provisional_grounded_edge_count,
                            "wiki_inline_queued_task_count": provisional_queued_task_count,
                        }
                    )
                    if keep_in_graph and persist:
                        self._materialize_primary_candidate(
                            result=result,
                            plan=plan,
                            candidate=candidate,
                            run_id=run_id,
                            persist=True,
                            create_source_edge=False,
                        )
                    if keep_in_graph:
                        self._annotate_wiki_inline_materialized_result(
                            result=result,
                            plan=plan,
                        )
                    else:
                        uniqueness_summary = result.metadata.get("wiki_inline_entity_uniqueness_filter") or {}
                        if (
                            isinstance(uniqueness_summary, dict)
                            and uniqueness_summary.get("applied")
                            and int(uniqueness_summary.get("kept_entity_count") or 0) == 0
                        ):
                            result.metadata["wiki_inline_skip_reason"] = "no_unique_canonical_grounded_entities"
                        else:
                            result.metadata["wiki_inline_skip_reason"] = "no_expandable_grounded_entities"
                        self._discard_materialized_result(result)
            result.metadata.update(
                {
                    "query_count": len(plan.queries),
                    "image_count": len(result.candidates),
                    "usable_image_count": len(result.usable_images()),
                    "accepted_image_count": len(result.accepted_images()),
                    "queued_task_count": len(result.queued_tasks),
                    "candidate_decisions": [
                        {
                            "kind": "wiki_inline_image",
                            "query": (plan.queries[0].query if plan.queries else ""),
                            "title": search_result.title,
                            "url": search_result.image_url,
                            "status": validation.status.value,
                            "reason": validation.reason,
                            "check": (validation.metadata or {}).get("check"),
                        }
                    ],
                }
            )
            if persist and self.store is not None:
                self.store.maybe_flush()
            _trace_timing(
                f"[image-discovery] phase=done_wiki_inline plan_id={plan.plan_id} elapsed_s={time.perf_counter() - total_started:.3f} accepted={len(result.accepted_images())} kept={'yes' if result.image_node is not None else 'no'}"
            )
            return result
        finally:
            if result.candidates:
                self._clear_transient_assets(result.candidates)
            elif validation is not None:
                self._clear_transient_asset_from_validation(validation)

    @staticmethod
    def _wiki_inline_result_has_expandable_targets(result: ImageDiscoveryResult) -> bool:
        return bool(result.grounded_edges or result.queued_tasks)

    @staticmethod
    def _is_wiki_inline_plan(plan: VisualSearchPlan) -> bool:
        metadata = plan.metadata or {}
        return plan.planner == "wikipedia_inline_image_planner" or metadata.get("plan_source") == "wikipedia_inline_image"

    @staticmethod
    def _annotate_wiki_inline_materialized_result(
        *,
        result: ImageDiscoveryResult,
        plan: VisualSearchPlan,
    ) -> None:
        if result.image_node is not None and result.image_node.source is not None:
            result.image_node.source.source_type = "wikipedia_inline_image"
        if result.image_node is not None:
            result.image_node.metadata = dict(result.image_node.metadata or {})
            result.image_node.metadata.update(
                {
                    "image_origin": "wikipedia_inline",
                    "source_text_node_id": plan.source_node_id,
                    "source_evidence_ids": list(plan.source_evidence_ids),
                }
            )

    @staticmethod
    def _discard_materialized_result(result: ImageDiscoveryResult) -> None:
        result.image_node = None
        result.edge = None
        result.image_evidence = None
        result.search_evidence = None
        result.grounded_edges = []
        result.queued_tasks = []

    def _apply_visual_plan_post_grounding_filter(
        self,
        *,
        plan: VisualSearchPlan,
        candidate: ImageSearchCandidate,
        search_result: ImageSearchResult,
        image_node: ImageNode,
        grounded_edges: list[Edge],
        queued_tasks: list[dict[str, Any]],
        run_id: str | None,
    ) -> tuple[bool, dict[str, Any]]:
        if self._is_wiki_inline_plan(plan) or not self.config.enable_visual_plan_post_grounding_filter:
            return True, {}

        grounded_edge_count = len(grounded_edges)
        queued_task_count = len(queued_tasks)
        expandable_entity_count = grounded_edge_count + queued_task_count
        filter_summary: dict[str, Any] = {
            "expandable_entity_count": expandable_entity_count,
            "grounded_edge_count": grounded_edge_count,
            "queued_task_count": queued_task_count,
            "min_expandable_entities": int(self.config.visual_plan_min_expandable_entities),
            "self_qa_entity_count": int(self.config.visual_plan_self_qa_entity_count),
            "self_qa_applied": False,
            "kept_in_graph": True,
            "filter_reason": None,
        }

        if expandable_entity_count < int(self.config.visual_plan_min_expandable_entities):
            filter_summary["kept_in_graph"] = False
            filter_summary["filter_reason"] = "expandable_entity_count_below_threshold"
            candidate.validation = ImageValidationResult(
                status=ImageCandidateStatus.REJECTED,
                confidence=candidate.validation.confidence,
                reason="expandable_entity_count_below_threshold",
                drop_candidate=True,
                metadata={
                    **dict(candidate.validation.metadata or {}),
                    "visual_plan_post_grounding_filter": filter_summary,
                },
            )
            candidate.is_primary = False
            return False, filter_summary

        # Intentionally do not run self-QA when exactly two expandable entities
        # are found.  Visual plans are now filtered only when fewer than the
        # configured minimum (default: two) can be linked or queued.

        if candidate.validation.metadata:
            candidate.validation.metadata["visual_plan_post_grounding_filter"] = filter_summary
        else:
            candidate.validation.metadata = {"visual_plan_post_grounding_filter": filter_summary}
        return True, filter_summary

    def _discover_with_client(
        self,
        *,
        client: SearchClient,
        plan: VisualSearchPlan,
        run_id: str | None,
        seen_keys: set[str],
        persist: bool,
        snapshots: list[SearchSnapshot],
        decision_log: list[dict[str, Any]],
    ) -> list[ImageSearchCandidate]:
        discovered: list[ImageSearchCandidate] = []
        for query in plan.queries:
            try:
                started = time.perf_counter()
                response = client.search_image(query.query, limit=self.config.per_query_limit)
                _trace_timing(
                    f"[image-discovery] stage=search_query plan_id={plan.plan_id} query={query.query!r} elapsed_s={time.perf_counter() - started:.3f} returned={len(response.results)}"
                )
            except Exception as exc:
                snapshot = self._snapshot_from_error(
                    client=client,
                    query=query.query,
                    error=exc,
                    run_id=run_id,
                )
                snapshots.append(snapshot)
                if persist:
                    self._persist_snapshot(snapshot)
                decision_log.append(
                    {
                        "kind": "query_error",
                        "query": query.query,
                        "reason": f"{exc.__class__.__name__}: {exc}",
                    }
                )
                continue

            snapshot = self._snapshot_from_response(response, run_id=run_id)
            snapshots.append(snapshot)
            if persist:
                self._persist_snapshot(snapshot)
            used_fallback = bool(response.metadata.get("fallback_used"))
            decision_log.append(
                {
                    "kind": "query_results",
                    "query": query.query,
                    "returned": len(response.results),
                    "fallback_used": used_fallback,
                }
            )

            for result_index, search_result in enumerate(response.results, start=1):
                if not isinstance(search_result, ImageSearchResult):
                    decision_log.append(
                        {
                            "kind": "candidate_skip",
                            "query": query.query,
                            "result_index": result_index,
                            "reason": "non_image_search_result",
                        }
                    )
                    self._log_image_result_fate(
                        plan_id=plan.plan_id,
                        query=query.query,
                        result_index=result_index,
                        search_result=None,
                        fate="skipped",
                        reason="non_image_search_result",
                    )
                    continue
                key = self._candidate_key(search_result)
                if not key or key in seen_keys:
                    decision_log.append(
                        self._candidate_decision_record(
                            kind="candidate_skip",
                            query=query.query,
                            search_result=search_result,
                            reason="missing_or_duplicate_candidate_key",
                            result_index=result_index,
                        )
                    )
                    self._log_image_result_fate(
                        plan_id=plan.plan_id,
                        query=query.query,
                        result_index=result_index,
                        search_result=search_result,
                        fate="skipped",
                        reason="missing_or_duplicate_candidate_key",
                    )
                    continue
                seen_keys.add(key)

                validation = self.image_check(
                    plan=plan,
                    query=query,
                    search_result=search_result,
                    run_id=run_id,
                )
                if validation.drop_candidate:
                    self._clear_transient_asset_from_validation(validation)
                    decision_log.append(
                        self._candidate_decision_record(
                            kind="candidate_drop",
                            query=query.query,
                            search_result=search_result,
                            reason=validation.reason or "drop_candidate",
                            result_index=result_index,
                            validation=validation,
                        )
                    )
                    self._log_image_result_fate(
                        plan_id=plan.plan_id,
                        query=query.query,
                        result_index=result_index,
                        search_result=search_result,
                        fate="dropped",
                        reason=validation.reason or "drop_candidate",
                        raw_model_output=(validation.metadata or {}).get("raw_model_output"),
                    )
                    continue
                if (
                    validation.status == ImageCandidateStatus.REJECTED
                    and not self.config.store_rejected
                ):
                    self._clear_transient_asset_from_validation(validation)
                    decision_log.append(
                        self._candidate_decision_record(
                            kind="candidate_skip",
                            query=query.query,
                            search_result=search_result,
                            reason=validation.reason or "rejected_not_stored",
                            result_index=result_index,
                            validation=validation,
                        )
                    )
                    self._log_image_result_fate(
                        plan_id=plan.plan_id,
                        query=query.query,
                        result_index=result_index,
                        search_result=search_result,
                        fate="skipped",
                        reason=validation.reason or "rejected_not_stored",
                        raw_model_output=(validation.metadata or {}).get("raw_model_output"),
                    )
                    continue

                discovered.append(
                    ImageSearchCandidate(
                        candidate_id=self._candidate_record_id(search_result),
                        source_query=query,
                        source_snapshot=snapshot,
                        search_result=search_result,
                        validation=validation,
                        used_fallback=used_fallback,
                    )
                )
                decision_log.append(
                    self._candidate_decision_record(
                        kind="candidate_kept",
                        query=query.query,
                        search_result=search_result,
                        reason=validation.reason or validation.status.value,
                        result_index=result_index,
                        status=validation.status.value,
                        bundle_count=len(discovered),
                        validation=validation,
                    )
                )
                self._log_image_result_fate(
                    plan_id=plan.plan_id,
                    query=query.query,
                    result_index=result_index,
                    search_result=search_result,
                    fate=(
                        "accepted"
                        if validation.status == ImageCandidateStatus.ACCEPTED
                        else "rejected"
                    ),
                    reason=validation.reason or validation.status.value,
                    raw_model_output=(validation.metadata or {}).get("raw_model_output"),
                )
                if len(discovered) >= self.config.max_images_per_plan:
                    decision_log.append(
                        {
                            "kind": "query_limit_reached",
                            "query": query.query,
                            "limit": self.config.max_images_per_plan,
                        }
                    )
                    return discovered
        return discovered

    @staticmethod
    def _candidate_decision_record(
        *,
        kind: str,
        query: str,
        search_result: ImageSearchResult,
        reason: str,
        result_index: int | None = None,
        status: str | None = None,
        bundle_count: int | None = None,
        validation: ImageValidationResult | None = None,
    ) -> dict[str, Any]:
        payload = {
            "kind": kind,
            "query": query,
            "rank": search_result.rank,
            "title": search_result.title,
            "url": search_result.image_url,
            "reason": reason,
        }
        if result_index is not None:
            payload["result_index"] = result_index
        if status is not None:
            payload["status"] = status
        if bundle_count is not None:
            payload["bundle_count"] = bundle_count
        if validation is not None:
            metadata = validation.metadata or {}
            if metadata.get("check") is not None:
                payload["check"] = metadata.get("check")
            if metadata.get("raw_model_output") is not None:
                payload["raw_model_output"] = metadata.get("raw_model_output")
            if metadata.get("visual_facts") is not None:
                payload["visual_facts"] = metadata.get("visual_facts")
        return payload

    @staticmethod
    def _candidate_record_id(search_result: ImageSearchResult) -> str:
        return ImageVariant.make_id(
            search_result.image_url,
            search_result.source_page_url,
            search_result.title,
        )

    def _apply_retrieval_consistency_check(
        self,
        *,
        plan: VisualSearchPlan,
        candidates: list[ImageSearchCandidate],
    ) -> dict[str, Any]:
        if not self.config.enable_retrieval_consistency_check:
            return {"check": "retrieval_consistency", "decision": "disabled"}
        accepted = [
            candidate
            for candidate in candidates
            if candidate.validation.status == ImageCandidateStatus.ACCEPTED
        ][: self.config.retrieval_consistency_max_images]
        required_count = max(
            self.config.retrieval_consistency_min_images,
            (len(accepted) + 1) // 2,
        )
        metadata: dict[str, Any] = {
            "check": "retrieval_consistency",
            "candidate_count": len(accepted),
            "required_consistent_count": required_count,
        }
        if len(accepted) < self.config.retrieval_consistency_min_images:
            self._reject_candidates_for_consistency(
                accepted,
                reason="retrieval_consistency_insufficient_content_matched_images",
                metadata=metadata,
            )
            metadata["decision"] = "reject"
            return metadata

        resolved_candidates: list[tuple[ImageSearchCandidate, ResolvedImageAsset]] = []
        for candidate in accepted:
            asset = self._transient_asset_for_candidate(candidate)
            if asset is None:
                asset, error = self._resolve_image_asset(
                    candidate.search_result,
                    persist_asset=False,
                    recovery_query=candidate.source_query.query,
                )
                if asset is None:
                    candidate.validation.status = ImageCandidateStatus.REJECTED
                    candidate.validation.reason = f"retrieval_consistency_image_unavailable:{error or 'unknown'}"
                    continue
                candidate.validation.metadata = dict(candidate.validation.metadata or {})
                candidate.validation.metadata["transient_image_key"] = asset.cache_key
            resolved_candidates.append((candidate, asset))

        if len(resolved_candidates) < self.config.retrieval_consistency_min_images:
            self._reject_candidates_for_consistency(
                accepted,
                reason="retrieval_consistency_insufficient_resolved_images",
                metadata=metadata,
            )
            metadata["decision"] = "reject"
            return metadata

        required_count = max(
            self.config.retrieval_consistency_min_images,
            (len(resolved_candidates) + 1) // 2,
        )
        metadata["candidate_count"] = len(resolved_candidates)
        metadata["required_consistent_count"] = required_count
        image_parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Search query:\n{plan.target.content or ''}\n\n"
                    f"There are {len(resolved_candidates)} content-matched candidate images. "
                    f"At least {required_count} must depict the same main visual content."
                ),
            }
        ]
        for index, (candidate, asset) in enumerate(resolved_candidates, start=1):
            image_parts.append(
                {
                    "type": "text",
                    "text": (
                        f"Candidate image {index}:\n"
                        f"title: {candidate.search_result.title or ''}\n"
                        f"caption: {candidate.search_result.snippet or ''}"
                    ),
                }
            )
            image_parts.append({"type": "image_url", "image_url": {"url": asset.model_url}})

        model_alias = (
            os.environ.get("IMAGE_RETRIEVAL_CONSISTENCY_MODEL")
            or self.image_check_model_alias
            or os.environ.get("IMAGE_CHECK_MODEL")
        )
        if not model_alias:
            self._reject_candidates_for_consistency(
                accepted,
                reason="missing_image_retrieval_consistency_model",
                metadata=metadata,
            )
            metadata["decision"] = "reject"
            return metadata

        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_IMAGE_RETRIEVAL_CONSISTENCY),
                        ModelMessage(role="user", content=image_parts),
                    ],
                    metadata={"trace_label": f"image_retrieval_consistency:{plan.plan_id}"},
                )
            )
            decision, consistent_indexes, judge_reason = self._parse_retrieval_consistency_response(response.content)
        except Exception as exc:
            self._reject_candidates_for_consistency(
                accepted,
                reason=f"retrieval_consistency_model_error:{exc.__class__.__name__}:{exc}",
                metadata=metadata,
            )
            metadata["decision"] = "reject"
            return metadata

        consistent_indexes = {
            index
            for index in consistent_indexes
            if 1 <= index <= len(resolved_candidates)
        }
        metadata.update(
            {
                "model_alias": model_alias,
                "decision": decision,
                "consistent_indexes": sorted(consistent_indexes),
                "judge_reason": judge_reason,
                "raw_model_output": response.content,
            }
        )
        if decision != "true" or len(consistent_indexes) < required_count:
            self._reject_candidates_for_consistency(
                accepted,
                reason="retrieval_consistency_not_converged",
                metadata=metadata,
            )
            metadata["decision"] = "reject"
            return metadata

        selected_ids = {
            resolved_candidates[index - 1][0].candidate_id
            for index in consistent_indexes
        }
        for candidate in accepted:
            candidate.validation.metadata = dict(candidate.validation.metadata or {})
            candidate.validation.metadata["retrieval_consistency"] = metadata
            if candidate.candidate_id not in selected_ids:
                candidate.validation.status = ImageCandidateStatus.REJECTED
                candidate.validation.reason = "retrieval_consistency_outlier"
        metadata["decision"] = "accept"
        return metadata

    @staticmethod
    def _parse_retrieval_consistency_response(text: str) -> tuple[str, set[int], str | None]:
        answer_match = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
        images_match = re.search(r"<consistent_images>(.*?)</consistent_images>", text, flags=re.DOTALL | re.IGNORECASE)
        reason_match = re.search(r"<reason>(.*?)</reason>", text, flags=re.DOTALL | re.IGNORECASE)
        answer = answer_match.group(1).strip().lower() if answer_match else ""
        image_text = images_match.group(1) if images_match else ""
        indexes = {int(value) for value in re.findall(r"\d+", image_text)}
        reason = re.sub(r"\s+", " ", reason_match.group(1)).strip() if reason_match else None
        return answer, indexes, reason

    @staticmethod
    def _reject_candidates_for_consistency(
        candidates: list[ImageSearchCandidate],
        *,
        reason: str,
        metadata: dict[str, Any],
    ) -> None:
        for candidate in candidates:
            candidate.validation.status = ImageCandidateStatus.REJECTED
            candidate.validation.reason = reason
            candidate.validation.metadata = dict(candidate.validation.metadata or {})
            candidate.validation.metadata["retrieval_consistency"] = dict(metadata)

    def _transient_asset_for_candidate(self, candidate: ImageSearchCandidate) -> ResolvedImageAsset | None:
        key = (candidate.validation.metadata or {}).get("transient_image_key")
        return self._transient_image_cache.get(key) if key else None

    def _clear_transient_assets(self, candidates: list[ImageSearchCandidate]) -> None:
        for candidate in candidates:
            self._clear_transient_asset_from_validation(candidate.validation)

    def _clear_transient_asset_from_validation(self, validation: ImageValidationResult) -> None:
        key = (validation.metadata or {}).get("transient_image_key")
        if key:
            self._transient_image_cache.pop(key, None)

    def _bind_resolved_asset_to_validation(
        self,
        validation: ImageValidationResult,
        resolved_asset: ResolvedImageAsset,
        *,
        persist_asset: bool,
    ) -> None:
        metadata = dict(validation.metadata or {})
        old_transient_key = metadata.get("transient_image_key")
        if persist_asset:
            if old_transient_key:
                self._transient_image_cache.pop(old_transient_key, None)
            metadata.pop("transient_image_key", None)
            metadata["resolved_image_key"] = resolved_asset.cache_key
        else:
            metadata.pop("resolved_image_key", None)
            metadata["transient_image_key"] = resolved_asset.cache_key
        metadata["resolved_image"] = resolved_asset.to_metadata()
        validation.metadata = metadata

    @staticmethod
    def _select_primary_candidate(candidates: list[ImageSearchCandidate]) -> ImageSearchCandidate | None:
        accepted = [
            candidate
            for candidate in candidates
            if candidate.validation.status == ImageCandidateStatus.ACCEPTED
        ]
        if not accepted:
            return None
        accepted.sort(
            key=lambda candidate: (
                -(candidate.validation.confidence if candidate.validation.confidence is not None else 0.0),
                candidate.used_fallback,
                candidate.search_result.rank if candidate.search_result.rank is not None else 10**9,
            )
        )
        primary = accepted[0]
        primary.is_primary = True
        return primary

    def _materialize_primary_candidate(
        self,
        *,
        result: ImageDiscoveryResult,
        plan: VisualSearchPlan,
        candidate: ImageSearchCandidate,
        run_id: str | None,
        persist: bool,
        create_source_edge: bool = True,
    ) -> None:
        resolved_asset = self._resolved_image_from_validation(
            candidate.validation,
            include_transient=not persist,
        )
        if resolved_asset is None and self.config.precheck_image_urls:
            resolved_asset, _ = self._resolve_image_asset(
                candidate.search_result,
                persist_asset=persist,
                recovery_query=candidate.source_query.query,
            )
            if resolved_asset is not None:
                self._bind_resolved_asset_to_validation(
                    candidate.validation,
                    resolved_asset,
                    persist_asset=persist,
                )
        provisional_node = self._image_node_from_result(
            candidate.search_result,
            run_id=run_id,
            resolved_asset=resolved_asset,
        )
        grounding = self.image_ground(
            plan=plan,
            search_result=candidate.search_result,
            image_node=provisional_node,
            validation=candidate.validation,
            run_id=run_id,
            persist_asset=persist,
        )
        candidate.grounded_entities = list(grounding.get("grounded_entities", []))
        candidate.grounded_caption = grounding.get("caption")
        candidate.visual_facts = list(grounding.get("visual_facts", []))

        variants = [
            self._variant_from_candidate(
                item,
                is_primary=item.candidate_id == candidate.candidate_id,
            )
            for item in result.candidates
        ]
        source_node_title = self._source_node_title(plan.source_node_id) or plan.target.content
        primary_caption = candidate.grounded_caption or provisional_node.caption or candidate.search_result.snippet
        primary_query = (
            candidate.source_query.query
            or plan.target.content
            or candidate.search_result.title
            or ""
        )
        primary_image_uri = (
            resolved_asset.asset_uri
            if resolved_asset is not None
            else candidate.search_result.image_url or candidate.search_result.source_page_url or candidate.search_result.title or ""
        )
        image_node = ImageNode.from_bundle(
            primary_image_uri,
            primary_image_id=candidate.candidate_id,
            image_variants=variants,
            source_page_url=candidate.search_result.source_page_url,
            caption=primary_caption,
            title=primary_query or candidate.search_result.title,
            width=resolved_asset.width if resolved_asset is not None and resolved_asset.width is not None else candidate.search_result.width,
            height=resolved_asset.height if resolved_asset is not None and resolved_asset.height is not None else candidate.search_result.height,
            content_type=resolved_asset.content_type if resolved_asset is not None else self._content_type(candidate.search_result),
            run_id=run_id,
            metadata={
                "search_query": primary_query,
                "candidate_count": len(result.candidates),
                "visual_target": plan.target.content,
                "resolved_image": resolved_asset.to_metadata() if resolved_asset is not None else None,
            },
        )
        self._apply_grounding_to_image_node(image_node, grounding)

        wiki_inline_uniqueness_filter: dict[str, Any] = {}
        filtered_grounded_entities, wiki_inline_uniqueness_filter = self._filter_wiki_inline_grounded_entities(
            plan=plan,
            search_result=candidate.search_result,
            image_node=image_node,
            grounded_entities=candidate.grounded_entities,
            run_id=run_id,
        )
        if wiki_inline_uniqueness_filter:
            candidate.validation.metadata = dict(candidate.validation.metadata or {})
            candidate.validation.metadata["wiki_inline_entity_uniqueness_filter"] = dict(wiki_inline_uniqueness_filter)
            result.metadata = dict(result.metadata or {})
            result.metadata["wiki_inline_entity_uniqueness_filter"] = dict(wiki_inline_uniqueness_filter)
            image_node.metadata = dict(image_node.metadata or {})
            image_node.metadata["wiki_inline_entity_uniqueness_filter"] = dict(wiki_inline_uniqueness_filter)
        candidate.grounded_entities = list(filtered_grounded_entities)
        image_node.metadata = dict(image_node.metadata or {})
        image_node.metadata["grounded_entities"] = list(candidate.grounded_entities)

        edge_relation, relation_rewrite_metadata = self._rewrite_text_to_image_relation(
            source_node_title=source_node_title,
            search_query=primary_query,
            image_node=image_node,
            resolved_asset=resolved_asset,
        )

        original_asset = self._image_asset(
            candidate.search_result,
            image_node=image_node,
            resolved_asset=resolved_asset,
        )
        thumb_asset = self._thumbnail_asset(candidate.search_result)
        asset_ids = [original_asset.asset_id]
        if thumb_asset:
            asset_ids.append(thumb_asset.asset_id)

        search_evidence = Evidence.create(
            EvidenceType.SEARCH_RESULT,
            content=candidate.search_result.title or candidate.search_result.snippet,
            node_ids=[image_node.node_id],
            url=candidate.search_result.source_page_url or candidate.search_result.image_url,
            source_snapshot_id=candidate.source_snapshot.snapshot_id if self.config.persist_search_snapshots else None,
            extractor=self.builder_name,
            confidence=candidate.validation.confidence,
            metadata={
                "query_id": candidate.source_query.query_id,
                "query": candidate.source_query.query,
                "rank": candidate.search_result.rank,
                "engine": candidate.source_snapshot.engine.value,
                "snapshot_id": candidate.source_snapshot.snapshot_id,
                "used_fallback": candidate.used_fallback,
                "validation": candidate.validation.to_dict(),
            },
            evidence_key=f"{candidate.source_snapshot.snapshot_id}:{candidate.source_query.query_id}:{self._candidate_key(candidate.search_result)}",
        )
        image_evidence = Evidence.create(
            EvidenceType.IMAGE,
            content=primary_caption or candidate.search_result.title,
            node_ids=[image_node.node_id],
            asset_ids=asset_ids,
            url=candidate.search_result.image_url,
            source_snapshot_id=candidate.source_snapshot.snapshot_id if self.config.persist_search_snapshots else None,
            extractor=self.builder_name,
            confidence=candidate.validation.confidence,
            metadata={
                "source_page_url": candidate.search_result.source_page_url,
                "thumbnail_url": candidate.search_result.thumbnail_url,
                "snapshot_id": candidate.source_snapshot.snapshot_id,
                "query_id": candidate.source_query.query_id,
                "target_evidence_id": plan.target.evidence_id,
                "validation": candidate.validation.to_dict(),
                "primary_candidate_id": candidate.candidate_id,
            },
            evidence_key=f"image_bundle:{candidate.candidate_id}",
        )

        edge = None
        if create_source_edge:
            edge = self._edge_from_plan_to_image(
                plan=plan,
                query=candidate.source_query,
                image_node=image_node,
                search_evidence=search_evidence,
                image_evidence=image_evidence,
                search_result=candidate.search_result,
                run_id=run_id,
                used_fallback=candidate.used_fallback,
                relation=edge_relation,
                relation_metadata=relation_rewrite_metadata,
            )
        grounded_edges, queued_tasks = self._link_or_queue_grounded_entities(
            image_node=image_node,
            grounded_entities=candidate.grounded_entities,
            image_evidence=image_evidence,
            run_id=run_id,
            source_node_title=source_node_title,
            source_query_text=candidate.source_query.query,
        )

        keep_materialized_result, post_grounding_filter = self._apply_visual_plan_post_grounding_filter(
            plan=plan,
            candidate=candidate,
            search_result=candidate.search_result,
            image_node=image_node,
            grounded_edges=grounded_edges,
            queued_tasks=queued_tasks,
            run_id=run_id,
        )
        if post_grounding_filter:
            result.metadata = dict(result.metadata or {})
            result.metadata["visual_plan_post_grounding_filter"] = dict(post_grounding_filter)
        if not keep_materialized_result:
            return

        search_evidence.metadata = dict(search_evidence.metadata or {})
        search_evidence.metadata["validation"] = candidate.validation.to_dict()
        image_evidence.metadata = dict(image_evidence.metadata or {})
        image_evidence.metadata["validation"] = candidate.validation.to_dict()

        if persist:
            self._persist_records(
                image_node=image_node,
                original_asset=original_asset,
                thumb_asset=thumb_asset,
                search_evidence=search_evidence,
                image_evidence=image_evidence,
                edge=edge,
                grounded_edges=grounded_edges,
            )

        result.image_node = image_node
        result.edge = edge
        result.image_evidence = image_evidence
        result.search_evidence = search_evidence
        result.grounded_edges = grounded_edges
        result.queued_tasks = queued_tasks

    @staticmethod
    def _variant_from_candidate(candidate: ImageSearchCandidate, *, is_primary: bool) -> ImageVariant:
        return ImageVariant(
            variant_id=candidate.candidate_id,
            image_url=candidate.search_result.image_url,
            source_page_url=candidate.search_result.source_page_url,
            thumbnail_url=candidate.search_result.thumbnail_url,
            title=candidate.search_result.title,
            search_caption=candidate.search_result.snippet,
            width=candidate.search_result.width,
            height=candidate.search_result.height,
            source=candidate.search_result.source,
            rank=candidate.search_result.rank,
            validation_status=candidate.validation.status.value,
            validation_confidence=candidate.validation.confidence,
            validation_reason=candidate.validation.reason,
            used_fallback=candidate.used_fallback,
            is_primary=is_primary,
            metadata={
                "query": candidate.source_query.query,
                "snapshot_id": candidate.source_snapshot.snapshot_id,
                "visual_facts": list(candidate.visual_facts),
                "resolved_image": (candidate.validation.metadata or {}).get("resolved_image"),
            },
        )

    def image_ground(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        image_node: ImageNode,
        validation: ImageValidationResult,
        run_id: str | None,
        persist_asset: bool = True,
    ) -> dict[str, Any]:
        """Analyze an accepted image and ground unique visible entities."""

        model_alias = os.environ.get("IMAGE_GROUND_MODEL")
        if not model_alias:
            grounding = {
                "caption": image_node.caption,
                "grounded_entities": [],
                "check": "not_configured",
                "context": None,
            }
            self._apply_grounding_to_image_node(image_node, grounding)
            return grounding

        resolved_asset = self._resolved_image_from_validation(
            validation,
            include_transient=not persist_asset,
        )
        precheck_error: str | None = None
        if self.config.precheck_image_urls and resolved_asset is None:
            resolved_asset, precheck_error = self._resolve_image_asset(
                search_result,
                persist_asset=persist_asset,
                recovery_query=image_node.caption or search_result.title or search_result.snippet,
            )
            if resolved_asset is not None:
                self._bind_resolved_asset_to_validation(
                    validation,
                    resolved_asset,
                    persist_asset=persist_asset,
                )
        if self.config.precheck_image_urls and resolved_asset is None:
            precheck_error = precheck_error or "missing_resolved_image_asset"
            self._log_invalid_image_url(search_result.image_url, precheck_error, stage="image_ground")
            grounding = {
                "caption": image_node.caption,
                "grounded_entities": [],
                "check": "image_url_precheck_failed",
                "raw_model_output": None,
                "run_id": run_id,
                "context": None,
            }
            image_node.metadata = dict(image_node.metadata or {})
            image_node.metadata["image_ground_error"] = precheck_error
            self._apply_grounding_to_image_node(image_node, grounding)
            return grounding

        if self.config.precheck_image_urls and resolved_asset is not None:
            image_node.metadata = dict(image_node.metadata or {})
            image_node.metadata["resolved_image"] = resolved_asset.to_metadata()

        grounding_context = self._build_image_grounding_context(search_result)
        image_node.metadata = dict(image_node.metadata or {})
        image_node.metadata["image_grounding_context"] = grounding_context.to_dict()
        image_node.metadata["image_grounding_prompt"] = {
            "system": PROMPT_IMAGE_GROUND,
            "user_text": grounding_context.prompt_text,
        }

        try:
            model_image_url = resolved_asset.model_url if resolved_asset is not None else search_result.image_url
            self._log_image_model_call(
                stage="image_ground",
                when="before",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=model_image_url,
                resolved_asset=resolved_asset,
            )
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_IMAGE_GROUND),
                        ModelMessage(
                            role="user",
                            content=[
                                {
                                    "type": "text",
                                    "text": grounding_context.prompt_text,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": model_image_url},
                                },
                            ],
                        ),
                    ],
                    metadata={"trace_label": f"image_ground:{plan.plan_id}:{search_result.title or ''}"},
                )
            )
            self._log_image_model_call(
                stage="image_ground",
                when="after",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=model_image_url,
                resolved_asset=resolved_asset,
                model_output=response.content,
            )
        except Exception as exc:
            error = f"{exc.__class__.__name__}: {exc}"
            self._log_invalid_image_url(search_result.image_url, error, stage="image_ground")
            grounding = {
                "caption": image_node.caption,
                "grounded_entities": [],
                "check": "mllm_grounding_failed",
                "raw_model_output": error,
                "run_id": run_id,
                "context": grounding_context.to_dict(),
                "debug_prompt_system": PROMPT_IMAGE_GROUND,
                "debug_prompt_user_text": grounding_context.prompt_text,
            }
            image_node.metadata = dict(image_node.metadata or {})
            image_node.metadata["image_ground_error"] = error
            self._apply_grounding_to_image_node(image_node, grounding)
            return grounding

        grounding = self._parse_image_ground_response(
            response.content,
            run_id=run_id,
            model_alias=model_alias,
            usage=response.usage,
        )
        grounding["context"] = grounding_context.to_dict()
        grounding["debug_prompt_system"] = PROMPT_IMAGE_GROUND
        grounding["debug_prompt_user_text"] = grounding_context.prompt_text
        self._apply_grounding_to_image_node(image_node, grounding)
        return grounding

    def _build_image_grounding_context(self, search_result: ImageSearchResult) -> ImageGroundingContext:
        backend = (self.config.image_grounding_context_backend or "source_page_reader").strip().lower()
        cache_key = f"{backend}::{search_result.source_page_url or ''}::{search_result.title or ''}"
        cached = self._grounding_context_cache.get(cache_key)
        if cached is not None:
            return cached

        if backend == "source_page_reader":
            context = self._build_source_page_reader_grounding_context(search_result)
        elif backend == "title_only":
            context = self._build_title_only_grounding_context(search_result)
        else:
            context = self._build_title_only_grounding_context(
                search_result,
                fallback_reason=f"unsupported_backend:{backend}",
            )

        self._grounding_context_cache[cache_key] = context
        return context

    def _build_source_page_reader_grounding_context(self, search_result: ImageSearchResult) -> ImageGroundingContext:
        source_page_url = (search_result.source_page_url or "").strip()
        if not source_page_url:
            return self._build_title_only_grounding_context(search_result, fallback_reason="missing_source_page_url")

        page_title = ""
        page_content = ""
        try:
            document = self.reader.read(source_page_url)
        except Exception as exc:
            fallback_reason = f"reader_error:{exc.__class__.__name__}"
        else:
            page_title = (document.title or "").strip()
            page_content = self._trim_grounding_context_text(document.content)
            fallback_reason = None if (page_title or page_content) else "reader_empty"

        return ImageGroundingContext(
            provider="source_page_reader",
            prompt_text=self._format_image_grounding_prompt_text(
                image_title=search_result.title,
                image_snippet=search_result.snippet,
                source_page_title=page_title,
                source_page_content=page_content,
            ),
            metadata={
                "source_page_url": source_page_url,
                "image_title": (search_result.title or "").strip() or None,
                "image_snippet": (search_result.snippet or "").strip() or None,
                "page_title": page_title or None,
                "content_chars": len(page_content),
                "fallback_reason": fallback_reason,
            },
        )

    def _build_title_only_grounding_context(
        self,
        search_result: ImageSearchResult,
        *,
        fallback_reason: str | None = None,
    ) -> ImageGroundingContext:
        title = (search_result.title or "").strip()
        snippet = (search_result.snippet or "").strip()
        return ImageGroundingContext(
            provider="title_only",
            prompt_text=self._format_image_grounding_prompt_text(
                image_title=title,
                image_snippet=snippet,
                source_page_title="",
                source_page_content="",
            ),
            metadata={
                "title": title or None,
                "image_snippet": snippet or None,
                "fallback_reason": fallback_reason,
            },
        )

    @staticmethod
    def _format_image_grounding_prompt_text(
        *,
        image_title: str | None,
        image_snippet: str | None,
        source_page_title: str | None,
        source_page_content: str | None,
    ) -> str:
        return (
            "Use the text fields below only to help identify entities that are actually visible in the image.\n"
            "If the text mentions entities not shown in the image, do not output them.\n"
            "For every output entity, ensure the entity name is unambiguous and can be used directly to look up the correct Wikipedia page.\n\n"
            f"Image Title: {(image_title or '').strip()}\n"
            f"Image Snippet: {(image_snippet or '').strip()}\n"
            f"Source Page Title: {(source_page_title or '').strip()}\n"
            "Source Page Content:\n"
            f"{(source_page_content or '').strip()}"
        )

    def _trim_grounding_context_text(self, text: str | None) -> str:
        normalized = re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", (text or "").strip()))
        if not normalized:
            return ""
        limit = max(256, int(self.config.image_grounding_max_context_chars))
        if len(normalized) <= limit:
            return normalized
        trimmed = normalized[:limit]
        last_break = max(trimmed.rfind("\n\n"), trimmed.rfind(". "))
        if last_break >= 256:
            trimmed = trimmed[: last_break + 1]
        return trimmed.rstrip()

    @staticmethod
    def _parse_image_ground_response(
        text: str,
        *,
        run_id: str | None,
        model_alias: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        match = re.search(r"<ground>(.*?)</ground>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        grounding: dict[str, Any] = {
            "caption": None,
            "grounded_entities": [],
            "raw_model_output": text,
            "run_id": run_id,
            "model_alias": model_alias,
            "usage": usage,
            "check": "mllm_grounding",
        }
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not value:
                continue
            if key == "caption":
                grounding["caption"] = value
            elif key == "entity":
                entity = ImageDiscoveryBuilder._parse_grounded_entity(value)
                if entity is not None:
                    grounding["grounded_entities"].append(entity)
        return grounding

    @staticmethod
    def _parse_grounded_entity(value: str) -> dict[str, Any] | None:
        parts = [part.strip() for part in value.split("|")]
        if not parts or not parts[0]:
            return None
        return {
            "name": parts[0],
            "relation_to_image": parts[1] if len(parts) > 1 and parts[1] else "depicted in image",
            "evidence": parts[2] if len(parts) > 2 else None,
        }

    @staticmethod
    def _apply_grounding_to_image_node(image_node: ImageNode, grounding: dict[str, Any]) -> None:
        caption = grounding.get("caption")
        if caption:
            image_node.caption = caption
            image_node.summary = caption
        image_node.metadata = dict(image_node.metadata or {})
        image_node.metadata["grounded_entities"] = grounding.get("grounded_entities", [])
        context = grounding.get("context") or image_node.metadata.get("image_grounding_context")
        if context is not None:
            image_node.metadata["image_grounding_context"] = context
        image_node.metadata["image_grounding"] = {
            "check": grounding.get("check"),
            "model_alias": grounding.get("model_alias"),
            "usage": grounding.get("usage"),
            "raw_model_output": grounding.get("raw_model_output"),
            "run_id": grounding.get("run_id"),
            "context": context,
            "debug_prompt_system": grounding.get("debug_prompt_system"),
            "debug_prompt_user_text": grounding.get("debug_prompt_user_text"),
        }

    def image_check(
        self,
        *,
        plan: VisualSearchPlan,
        query: SearchQuerySpec,
        search_result: ImageSearchResult,
        run_id: str | None,
    ) -> ImageValidationResult:
        """Check one candidate image.

        This single function owns both cheap deterministic gates and future MLLM
        semantic validation. Keeping them together makes the discovery flow only
        depend on one accept/reject decision.
        """
        resolved_asset, rejection = self._precheck_candidate_for_processing(
            search_result,
            stage="image_check",
            persist_asset=False,
            recovery_query=query.query,
        )
        if rejection is not None:
            return rejection

        model_alias = self.image_check_model_alias or os.environ.get("IMAGE_CHECK_MODEL")

        if self.config.force_accept_images:
            metadata: dict[str, Any] = {
                "check": "force_accept_images",
                "debug_force_accept_images": True,
            }
            result = ImageValidationResult(
                status=ImageCandidateStatus.ACCEPTED,
                confidence=1.0,
                reason="force_accept_images",
                metadata=metadata,
            )
            if resolved_asset is not None:
                self._bind_resolved_asset_to_validation(
                    result,
                    resolved_asset,
                    persist_asset=False,
                )
            return result

        if model_alias:
            try:
                result = self._image_check_with_mllm(
                    plan=plan,
                    search_result=search_result,
                    model_alias=model_alias,
                    run_id=run_id,
                    resolved_asset=resolved_asset,
                )
                if resolved_asset is not None:
                    self._bind_resolved_asset_to_validation(
                        result,
                        resolved_asset,
                        persist_asset=False,
                    )
                return result
            except Exception as exc:
                error = f"{exc.__class__.__name__}: {exc}"
                self._log_invalid_image_url(search_result.image_url, error, stage="image_check")
                return self._reject(
                    f"image_check_model_error:{error}",
                    drop_candidate=True,
                )

        del query, run_id
        return ImageValidationResult(
            status=ImageCandidateStatus.ACCEPTED,
            confidence=None,
            metadata={"check": "basic_url_format_size"},
        )

    def _precheck_candidate_for_processing(
        self,
        search_result: ImageSearchResult,
        *,
        stage: str,
        persist_asset: bool = True,
        recovery_query: str | None = None,
    ) -> tuple[ResolvedImageAsset | None, ImageValidationResult | None]:
        if not search_result.image_url:
            return None, self._reject("missing_image_url")
        extension = self._extension(search_result.image_url)
        if extension and extension in self.config.rejected_extensions:
            return None, self._reject(f"rejected_extension:{extension}")
        if (
            self.config.min_width is not None
            and search_result.width is not None
            and search_result.width < self.config.min_width
        ):
            return None, self._reject(f"width_below_min:{search_result.width}")
        if (
            self.config.min_height is not None
            and search_result.height is not None
            and search_result.height < self.config.min_height
        ):
            return None, self._reject(f"height_below_min:{search_result.height}")
        content_type = self._content_type(search_result)
        if self.config.allowed_content_types and content_type and content_type not in self.config.allowed_content_types:
            return None, self._reject(f"content_type_not_allowed:{content_type}")

        resolved_asset: ResolvedImageAsset | None = None
        if self.config.precheck_image_urls:
            resolved_asset, precheck_error = self._resolve_image_asset(
                search_result,
                persist_asset=persist_asset,
                recovery_query=recovery_query,
            )
            if precheck_error is not None or resolved_asset is None:
                self._log_invalid_image_url(search_result.image_url, precheck_error, stage=stage)
                return None, self._reject(
                    f"image_url_precheck_failed:{precheck_error}",
                    drop_candidate=True,
                )
            if (
                self.config.min_width is not None
                and resolved_asset.width is not None
                and resolved_asset.width < self.config.min_width
            ):
                if not persist_asset:
                    self._transient_image_cache.pop(resolved_asset.cache_key, None)
                return None, self._reject(
                    f"resolved_width_below_min:{resolved_asset.width}",
                    drop_candidate=True,
                )
            if (
                self.config.min_height is not None
                and resolved_asset.height is not None
                and resolved_asset.height < self.config.min_height
            ):
                if not persist_asset:
                    self._transient_image_cache.pop(resolved_asset.cache_key, None)
                return None, self._reject(
                    f"resolved_height_below_min:{resolved_asset.height}",
                    drop_candidate=True,
                )
        return resolved_asset, None

    def _wiki_inline_image_check(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        run_id: str | None,
        persist_asset: bool = False,
    ) -> ImageValidationResult:
        resolved_asset, rejection = self._precheck_candidate_for_processing(
            search_result,
            stage="wiki_inline_image_check",
            persist_asset=persist_asset,
            recovery_query=plan.target.content or search_result.title or search_result.snippet,
        )
        if rejection is not None:
            return rejection

        wiki_title = (self._source_node_title(plan.source_node_id) or "").strip()
        if not wiki_title:
            return self._reject("missing_wiki_inline_title", drop_candidate=True)

        model_alias = self._wiki_inline_model_alias()
        if not model_alias:
            return self._reject("missing_wiki_inline_image_model", drop_candidate=True)

        try:
            image_for_model = resolved_asset.model_url if resolved_asset is not None else search_result.image_url
            self._log_image_model_call(
                stage="wiki_inline_image_check",
                when="before",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=image_for_model,
                resolved_asset=resolved_asset,
            )
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_WIKI_INLINE_IMAGE_TITLE_CHECK),
                        ModelMessage(
                            role="user",
                            content=[
                                {
                                    "type": "text",
                                    "text": self._wiki_inline_title_check_prompt_input(wikipedia_title=wiki_title),
                                },
                                {"type": "image_url", "image_url": {"url": image_for_model}},
                            ],
                        ),
                    ],
                    metadata={"trace_label": f"wiki_inline_title_check:{plan.plan_id}"},
                )
            )
        except Exception as exc:
            print(
                "[wiki-inline-image] model request failed "
                f"plan_id={plan.plan_id} "
                f"model_alias={model_alias!r}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            return self._reject(
                f"wiki_inline_image_model_error:{exc.__class__.__name__}:{exc}",
                drop_candidate=True,
            )
        self._log_image_model_call(
            stage="wiki_inline_image_check",
            when="after",
            model_alias=model_alias,
            plan_id=plan.plan_id,
            search_result=search_result,
            model_image_url=image_for_model,
            resolved_asset=resolved_asset,
            model_output=response.content,
        )

        result = self._parse_image_check_response(
            response.content,
            run_id=run_id,
            model_alias=model_alias,
            usage=response.usage,
        )
        result.metadata = dict(result.metadata or {})
        result.metadata.update(
            {
                "check": "wiki_inline_title_relevance",
                "wikipedia_title": wiki_title,
                "search_result_title": search_result.title,
            }
        )
        if resolved_asset is not None:
            self._bind_resolved_asset_to_validation(
                result,
                resolved_asset,
                persist_asset=persist_asset,
            )
        if result.status == ImageCandidateStatus.REJECTED:
            result.drop_candidate = True
            return result
        return self._wiki_inline_self_qa_check(
            plan=plan,
            search_result=search_result,
            validation=result,
            run_id=run_id,
        )

    def _resolve_image_asset(
        self,
        search_result: ImageSearchResult,
        *,
        persist_asset: bool = True,
        recovery_query: str | None = None,
    ) -> tuple[ResolvedImageAsset | None, str | None]:
        image_url = search_result.image_url
        source_page_url = search_result.source_page_url
        selection_hint = self._recovery_selection_hint(search_result, recovery_query)
        cache_key = self._resolved_image_cache_key(image_url, source_page_url, selection_hint)
        if persist_asset:
            cached = self._resolved_image_cache.get(cache_key)
            if cached is not None:
                _log_image_debug(
                    "image-resolve-cache-hit",
                    image_url=image_url,
                    source_page_url=source_page_url,
                    persist_asset=persist_asset,
                    selection_hint=_short_debug_text(selection_hint),
                    strategy=cached.strategy,
                    resolved_url=cached.resolved_url,
                    model_image_source_kind=_image_source_kind(cached.model_url),
                )
                return cached, None
        else:
            cached = self._transient_image_cache.get(cache_key)
            if cached is not None:
                _log_image_debug(
                    "image-resolve-cache-hit",
                    image_url=image_url,
                    source_page_url=source_page_url,
                    persist_asset=persist_asset,
                    selection_hint=_short_debug_text(selection_hint),
                    strategy=cached.strategy,
                    resolved_url=cached.resolved_url,
                    model_image_source_kind=_image_source_kind(cached.model_url),
                )
                return cached, None

        attempted_errors: list[str] = []
        direct_asset, direct_error = self._download_and_prepare_image_asset(
            image_url,
            source_page_url=source_page_url,
            strategy="direct",
            cache_key=cache_key,
            persist_asset=persist_asset,
        )
        if direct_asset is not None:
            (self._resolved_image_cache if persist_asset else self._transient_image_cache)[cache_key] = direct_asset
            _log_image_debug(
                "image-resolve-success",
                image_url=image_url,
                source_page_url=source_page_url,
                persist_asset=persist_asset,
                selection_hint=_short_debug_text(selection_hint),
                strategy=direct_asset.strategy,
                resolved_url=direct_asset.resolved_url,
                content_type=direct_asset.content_type,
                width=direct_asset.width,
                height=direct_asset.height,
                model_image_source_kind=_image_source_kind(direct_asset.model_url),
            )
            return direct_asset, None
        if direct_error:
            attempted_errors.append(direct_error)

        if self.config.try_source_page_recovery and source_page_url:
            recovered_assets: list[ResolvedImageAsset] = []
            for recovered_url in self._recover_candidate_image_urls(search_result):
                recovered_asset, recovered_error = self._download_and_prepare_image_asset(
                    recovered_url,
                    source_page_url=source_page_url,
                    strategy="source_page_recovery",
                    cache_key=cache_key,
                    persist_asset=False,
                )
                if recovered_asset is not None:
                    self._log_recovered_image_url(
                        original_url=image_url,
                        recovered_url=recovered_url,
                        source_page_url=source_page_url,
                    )
                    recovered_assets.append(recovered_asset)
                    continue
                if recovered_error:
                    attempted_errors.append(recovered_error)

            if recovered_assets:
                selected_asset, selection_metadata = self._select_best_recovered_asset(
                    search_result=search_result,
                    recovery_query=selection_hint,
                    recovered_assets=recovered_assets,
                )
                final_asset = selected_asset
                if persist_asset:
                    persisted_asset, persisted_error = self._download_and_prepare_image_asset(
                        selected_asset.resolved_url,
                        source_page_url=source_page_url,
                        strategy=selected_asset.strategy,
                        cache_key=cache_key,
                        persist_asset=True,
                    )
                    if persisted_asset is not None:
                        final_asset = persisted_asset
                    elif persisted_error:
                        attempted_errors.append(f"persist_selected_asset_failed:{persisted_error}")
                (self._resolved_image_cache if persist_asset else self._transient_image_cache)[cache_key] = final_asset
                _log_image_debug(
                    "image-recovery-selection",
                    image_url=image_url,
                    source_page_url=source_page_url,
                    persist_asset=persist_asset,
                    selection_hint=_short_debug_text(selection_hint),
                    candidate_count=len(recovered_assets),
                    selection=selection_metadata,
                    candidates=[
                        {
                            "resolved_url": asset.resolved_url,
                            "width": asset.width,
                            "height": asset.height,
                            "content_type": asset.content_type,
                        }
                        for asset in recovered_assets
                    ],
                )
                _log_image_debug(
                    "image-resolve-success",
                    image_url=image_url,
                    source_page_url=source_page_url,
                    persist_asset=persist_asset,
                    selection_hint=_short_debug_text(selection_hint),
                    strategy=final_asset.strategy,
                    resolved_url=final_asset.resolved_url,
                    content_type=final_asset.content_type,
                    width=final_asset.width,
                    height=final_asset.height,
                    model_image_source_kind=_image_source_kind(final_asset.model_url),
                )
                return final_asset, None

        final_error = " | ".join(attempted_errors) if attempted_errors else "unresolved_image_asset"
        _log_image_debug(
            "image-resolve-failed",
            image_url=image_url,
            source_page_url=source_page_url,
            persist_asset=persist_asset,
            selection_hint=_short_debug_text(selection_hint),
            error=final_error,
        )
        return None, final_error

    @staticmethod
    def _recovery_selection_hint(
        search_result: ImageSearchResult,
        recovery_query: str | None,
    ) -> str:
        return (
            (recovery_query or "").strip()
            or (search_result.title or "").strip()
            or (search_result.snippet or "").strip()
            or (search_result.source_page_url or "").strip()
            or (search_result.image_url or "").strip()
        )

    def _select_best_recovered_asset(
        self,
        *,
        search_result: ImageSearchResult,
        recovery_query: str,
        recovered_assets: list[ResolvedImageAsset],
    ) -> tuple[ResolvedImageAsset, dict[str, Any]]:
        if len(recovered_assets) == 1:
            return recovered_assets[0], {
                "mode": "single_candidate",
                "selected_index": 0,
                "reason": "only_recovered_candidate",
            }

        model_alias = self.image_check_model_alias or os.environ.get("IMAGE_CHECK_MODEL")
        if not model_alias:
            return recovered_assets[0], {
                "mode": "fallback_first_candidate",
                "selected_index": 0,
                "reason": "missing_image_check_model",
            }

        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_IMAGE_RECOVERY_SELECTION),
                        ModelMessage(
                            role="user",
                            content=self._build_recovered_image_selection_input(
                                search_result=search_result,
                                recovery_query=recovery_query,
                                recovered_assets=recovered_assets,
                            ),
                        ),
                    ],
                    metadata={
                        "trace_label": (
                            f"image_recovery_selection:{search_result.title or ''}:{recovery_query[:80]}"
                        )
                    },
                )
            )
            decision = self._parse_recovered_image_selection_response(response.content)
            index = decision.get("candidate_index")
            if isinstance(index, int) and 0 <= index < len(recovered_assets):
                return recovered_assets[index], {
                    "mode": "llm_selected",
                    "selected_index": index,
                    "reason": decision.get("reason") or "",
                    "raw_model_output": response.content,
                    "model_alias": model_alias,
                }
            return recovered_assets[0], {
                "mode": "fallback_first_candidate",
                "selected_index": 0,
                "reason": "invalid_llm_selection_index",
                "raw_model_output": response.content,
                "model_alias": model_alias,
            }
        except Exception as exc:
            return recovered_assets[0], {
                "mode": "fallback_first_candidate",
                "selected_index": 0,
                "reason": f"image_recovery_selection_model_error:{exc.__class__.__name__}:{exc}",
                "model_alias": model_alias,
            }

    @staticmethod
    def _build_recovered_image_selection_input(
        *,
        search_result: ImageSearchResult,
        recovery_query: str,
        recovered_assets: list[ResolvedImageAsset],
    ) -> list[dict[str, Any]]:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Search query:\n{recovery_query}\n\n"
                    f"Original search-result title: {search_result.title or ''}\n"
                    f"Original search-result caption: {search_result.snippet or ''}\n"
                    f"Source page URL: {search_result.source_page_url or ''}\n\n"
                    "Choose the single best recovered image by candidate_index."
                ),
            }
        ]
        for index, asset in enumerate(recovered_assets):
            content.append(
                {
                    "type": "text",
                    "text": (
                        f"Candidate [{index}]\n"
                        f"resolved_url: {asset.resolved_url or ''}\n"
                        f"content_type: {asset.content_type or ''}\n"
                        f"size: {asset.width}x{asset.height}"
                    ),
                }
            )
            content.append({"type": "image_url", "image_url": {"url": asset.model_url}})
        return content

    @staticmethod
    def _parse_recovered_image_selection_response(text: str) -> dict[str, Any]:
        match = re.search(r"<selection>(.*?)</selection>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        fields: dict[str, str] = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
        raw_index = (fields.get("candidate_index") or "").strip().lower()
        candidate_index: int | None = None
        if raw_index:
            try:
                candidate_index = int(raw_index)
            except ValueError:
                candidate_index = None
        return {
            "decision": (fields.get("decision") or "").strip().lower(),
            "candidate_index": candidate_index,
            "reason": fields.get("reason") or "",
        }

    def _download_and_prepare_image_asset(
        self,
        image_url: str | None,
        *,
        source_page_url: str | None,
        strategy: str,
        cache_key: str,
        persist_asset: bool = True,
    ) -> tuple[ResolvedImageAsset | None, str | None]:
        if not image_url:
            return None, "missing_image_url"

        download_result = self._download_image_payload(
            image_url,
            max_bytes=self.config.model_image_max_bytes,
        )
        if isinstance(download_result, str):
            return None, f"{image_url} -> {download_result}"
        payload, content_type = download_result
        if not payload:
            return None, f"{image_url} -> empty_response_body"

        normalized_content_type = (content_type or "").lower()
        sniffed_content_type = self._sniff_content_type(payload)
        if normalized_content_type and not normalized_content_type.startswith("image/"):
            if not sniffed_content_type:
                return None, f"{image_url} -> non_image_content_type:{content_type}"

        try:
            width, height = self._image_dimensions(payload, verify=True)
        except Exception as exc:
            return None, f"{image_url} -> decode_error:{exc.__class__.__name__}:{exc}"

        content_type = (
            sniffed_content_type
            if normalized_content_type == "application/octet-stream" and sniffed_content_type
            else content_type or sniffed_content_type or "image/jpeg"
        )
        resized_content_type, resized_payload = self._prepare_model_payload(
            payload=payload,
            content_type=content_type,
            max_edge=self.config.model_image_max_edge,
        )
        resized_width, resized_height = self._image_dimensions(resized_payload)
        if resized_width is not None:
            width = resized_width
        if resized_height is not None:
            height = resized_height

        cache_path = None
        asset_uri = image_url
        if persist_asset:
            cache_path = self._write_image_cache_file(cache_key, resized_payload, resized_content_type)
            asset_uri = (
                self._maybe_upload_cached_image(cache_path, cache_key)
                if self.config.upload_cached_images
                else None
            ) or cache_path
        model_url = self._data_url(resized_content_type, resized_payload)
        _log_image_debug(
            "image-asset-ready",
            strategy=strategy,
            image_url=image_url,
            source_page_url=source_page_url,
            persist_asset=persist_asset,
            content_type=resized_content_type,
            width=width,
            height=height,
            cache_path=cache_path,
            asset_uri=asset_uri,
            model_image_source_kind=_image_source_kind(model_url),
            model_image_source=_format_debug_image_source(model_url),
        )
        return (
            ResolvedImageAsset(
                cache_key=cache_key,
                original_url=image_url,
                resolved_url=image_url,
                source_page_url=source_page_url,
                model_url=model_url,
                asset_uri=asset_uri,
                cache_path=cache_path,
                content_type=resized_content_type,
                width=width,
                height=height,
                strategy=strategy,
            ),
            None,
        )

    def _download_image_payload(
        self,
        image_url: str,
        *,
        max_bytes: int | None,
    ) -> tuple[bytes, str | None] | str:
        host = urlparse(image_url).netloc.lower()
        last_error: str | None = None
        for attempt in range(1, max(1, self.config.precheck_retries) + 1):
            host_lock = self._host_lock(host)
            with host_lock:
                self._wait_for_host_slot(host)
                request = Request(
                    image_url,
                    headers={
                        "Accept": "image/*,*/*;q=0.8",
                        "User-Agent": self._user_agent(),
                    },
                )
                started_at = time.perf_counter()
                try:
                    with urlopen(request, timeout=self.config.precheck_timeout_s) as response:
                        content_type = response.headers.get("Content-Type", "")
                        payload = response.read() if not max_bytes or max_bytes <= 0 else response.read(max_bytes)
                    elapsed_s = time.perf_counter() - started_at
                    self._mark_host_slot(host, success=True)
                    self._log_image_download(
                        image_url=image_url,
                        byte_count=len(payload),
                        elapsed_s=elapsed_s,
                        content_type=content_type,
                        attempt=attempt,
                        max_bytes=max_bytes,
                    )
                    return payload, content_type
                except HTTPError as exc:
                    elapsed_s = time.perf_counter() - started_at
                    self._log_image_download_failure(
                        image_url=image_url,
                        reason=f"http_{exc.code}",
                        elapsed_s=elapsed_s,
                        attempt=attempt,
                    )
                    retry_after = self._retry_after_seconds(exc)
                    if exc.code == 429 and attempt < self.config.precheck_retries:
                        backoff_s = retry_after or self._default_retry_after_seconds(host, attempt)
                        self._mark_host_slot(host, retry_after=backoff_s)
                        last_error = "http_429"
                        continue
                    return f"http_{exc.code}"
                except URLError as exc:
                    elapsed_s = time.perf_counter() - started_at
                    self._log_image_download_failure(
                        image_url=image_url,
                        reason=f"url_error:{exc.reason}",
                        elapsed_s=elapsed_s,
                        attempt=attempt,
                    )
                    last_error = f"url_error:{exc.reason}"
                except TimeoutError:
                    elapsed_s = time.perf_counter() - started_at
                    self._log_image_download_failure(
                        image_url=image_url,
                        reason=f"timeout_after_{self.config.precheck_timeout_s}s",
                        elapsed_s=elapsed_s,
                        attempt=attempt,
                    )
                    last_error = f"timeout_after_{self.config.precheck_timeout_s}s"
                except Exception as exc:
                    elapsed_s = time.perf_counter() - started_at
                    self._log_image_download_failure(
                        image_url=image_url,
                        reason=f"download_error:{exc.__class__.__name__}:{exc}",
                        elapsed_s=elapsed_s,
                        attempt=attempt,
                    )
                    last_error = f"download_error:{exc.__class__.__name__}:{exc}"
            if attempt < self.config.precheck_retries:
                time.sleep(min(6.0, attempt * 1.5))
        return last_error or "download_failed"

    def _recover_candidate_image_urls(self, search_result: ImageSearchResult) -> list[str]:
        source_page_url = search_result.source_page_url
        if not source_page_url:
            return []
        html_text = self._fetch_source_page_html(source_page_url)
        if html_text is None:
            return []

        patterns = [
            r'<meta[^>]+property=["\']og:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'<meta[^>]+name=["\']twitter:image["\'][^>]+content=["\']([^"\']+)["\']',
            r'"display_url"\s*:\s*"([^"]+)"',
            r'"image_url"\s*:\s*"([^"]+)"',
        ]
        candidates: list[str] = []
        seen = set()
        for pattern in patterns:
            for match in re.findall(pattern, html_text, flags=re.IGNORECASE):
                candidate = html.unescape(match).replace("\\u0026", "&").replace("\\/", "/").strip()
                if not candidate.startswith(("http://", "https://")) or candidate in seen:
                    continue
                seen.add(candidate)
                candidates.append(candidate)
        return candidates

    def _fetch_source_page_html(self, source_page_url: str) -> str | None:
        request = Request(
            source_page_url,
            headers={
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "User-Agent": self._user_agent(),
            },
        )
        try:
            with urlopen(request, timeout=self.config.source_page_timeout_s) as response:
                payload = response.read(min(self.config.precheck_max_bytes * 4, 1048576))
            return payload.decode("utf-8", errors="ignore")
        except Exception:
            return None

    def _resolved_image_from_validation(
        self,
        validation: ImageValidationResult,
        *,
        include_transient: bool = False,
    ) -> ResolvedImageAsset | None:
        metadata = validation.metadata or {}
        key = metadata.get("resolved_image_key")
        if key:
            asset = self._resolved_image_cache.get(key)
            if asset is not None:
                return asset
        if not include_transient:
            return None
        transient_key = metadata.get("transient_image_key")
        if not transient_key:
            return None
        return self._transient_image_cache.get(transient_key)

    @staticmethod
    def _resolved_image_cache_key(
        image_url: str | None,
        source_page_url: str | None,
        selection_hint: str | None = None,
    ) -> str:
        payload = f"{image_url or ''}||{source_page_url or ''}||{selection_hint or ''}"
        return sha256(payload.encode("utf-8")).hexdigest()[:24]

    def _write_image_cache_file(self, cache_key: str, payload: bytes, content_type: str) -> str:
        cache_dir = self._cache_dir()
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{cache_key}{self._suffix_for_content_type(content_type)}"
        if not path.exists():
            path.write_bytes(payload)
        return str(path.resolve())

    def _cache_dir(self) -> Path:
        configured = self.config.cache_dir
        if configured:
            return Path(configured)
        return Path(__file__).resolve().parent / ".image_cache"

    @staticmethod
    def _suffix_for_content_type(content_type: str | None) -> str:
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
            "image/avif": ".avif",
        }
        return mapping.get((content_type or "").lower(), ".img")

    @staticmethod
    def _data_url(content_type: str, payload: bytes) -> str:
        return f"data:{content_type};base64,{base64.b64encode(payload).decode('ascii')}"

    @staticmethod
    def _sniff_content_type(payload: bytes) -> str | None:
        if payload.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if payload.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if payload[:6] in (b"GIF87a", b"GIF89a"):
            return "image/gif"
        if payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
            return "image/webp"
        return None

    @staticmethod
    def _prepare_model_payload(
        *,
        payload: bytes,
        content_type: str,
        max_edge: int | None,
    ) -> tuple[str, bytes]:
        if not max_edge or max_edge <= 0:
            return content_type, payload
        try:
            from PIL import Image
        except ImportError:
            return content_type, payload
        try:
            with Image.open(BytesIO(payload)) as image:
                image.load()
                width, height = image.size
                if max(width, height) <= max_edge:
                    return content_type, payload
                resized = image.copy()
                resized.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                has_alpha = resized.mode in ("RGBA", "LA") or (
                    resized.mode == "P" and "transparency" in resized.info
                )
                output = BytesIO()
                if has_alpha:
                    resized.save(output, format="PNG", optimize=True)
                    return "image/png", output.getvalue()
                resized = resized.convert("RGB")
                resized.save(output, format="JPEG", quality=90, optimize=True)
                return "image/jpeg", output.getvalue()
        except Exception:
            return content_type, payload

    @staticmethod
    def _image_dimensions(payload: bytes, *, verify: bool = False) -> tuple[int | None, int | None]:
        try:
            from PIL import Image
        except ImportError:
            return None, None
        with Image.open(BytesIO(payload)) as image:
            width, height = image.size
            if verify:
                image.verify()
            return width, height

    def _maybe_upload_cached_image(self, cache_path: str, cache_key: str) -> str | None:
        try:
            from PIL import Image
            from opensearch_vl.opensearch_infer import cos_upload
        except Exception:
            return None
        if not cos_upload.upload_available():
            return None
        try:
            with Image.open(cache_path) as pil_image:
                pil_copy = pil_image.copy()
        except Exception:
            return None
        return cos_upload.upload_pil_image(
            pil_copy,
            filename_prefix="synthesis",
            case_idx=0,
            turn_num=0,
            tool_name=f"image_cache_{cache_key}",
        )

    def _wait_for_host_slot(self, host: str) -> None:
        if not host:
            return
        with self._download_lock:
            not_before = self._host_not_before.get(host, 0.0)
        now = time.time()
        if not_before > now:
            time.sleep(not_before - now)

    def _mark_host_slot(self, host: str, retry_after: float | None = None, *, success: bool = False) -> None:
        if not host:
            return
        with self._download_lock:
            min_interval_s = self._host_min_interval_seconds(host)
            delay = min_interval_s if success else max(
                min_interval_s,
                retry_after or min_interval_s,
            )
            self._host_not_before[host] = time.time() + delay

    def _host_lock(self, host: str) -> threading.Lock:
        with self._download_lock:
            lock = self._host_locks.get(host)
            if lock is None:
                lock = threading.Lock()
                self._host_locks[host] = lock
            return lock

    def _host_min_interval_seconds(self, host: str) -> float:
        if self._is_wikimedia_host(host):
            return self.config.wikimedia_host_min_interval_s
        return self.config.host_min_interval_s

    def _default_retry_after_seconds(self, host: str, attempt: int) -> float:
        if self._is_wikimedia_host(host):
            return max(self.config.wikimedia_429_retry_after_s, float(attempt) * self.config.wikimedia_429_retry_after_s)
        return float(attempt) * 2.0

    @staticmethod
    def _is_wikimedia_host(host: str) -> bool:
        normalized = (host or "").lower()
        return normalized.endswith((".wikimedia.org", ".wikipedia.org", ".mediawiki.org"))

    def _user_agent(self) -> str:
        configured = (
            self.config.user_agent
            or os.environ.get("SYNTHESIS_USER_AGENT")
            or os.environ.get("WIKIMEDIA_USER_AGENT")
        )
        if configured:
            return configured
        return "DeepSearchBot/0.1 (https://github.com/shawn0728/OpenSearch-VL; automated research image fetcher)"

    @staticmethod
    def _retry_after_seconds(error: HTTPError) -> float | None:
        value = error.headers.get("Retry-After") if error.headers else None
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    @staticmethod
    def _log_invalid_image_url(image_url: str | None, reason: str, *, stage: str) -> None:
        _log_image_debug(
            "image-invalid-url",
            stage=stage,
            image_url=image_url,
            reason=reason,
        )

    @staticmethod
    def _log_image_download(
        *,
        image_url: str,
        byte_count: int,
        elapsed_s: float,
        content_type: str | None,
        attempt: int,
        max_bytes: int | None,
    ) -> None:
        _log_image_debug(
            "image-download-ok",
            image_url=image_url,
            byte_count=byte_count,
            byte_count_human=ImageDiscoveryBuilder._format_byte_count(byte_count),
            elapsed_s=round(elapsed_s, 3),
            content_type=content_type,
            attempt=attempt,
            max_bytes=max_bytes,
        )

    @staticmethod
    def _log_image_download_failure(
        *,
        image_url: str,
        reason: str,
        elapsed_s: float,
        attempt: int,
    ) -> None:
        _log_image_debug(
            "image-download-failed",
            image_url=image_url,
            reason=reason,
            elapsed_s=round(elapsed_s, 3),
            attempt=attempt,
        )

    @staticmethod
    def _format_byte_count(size: int) -> str:
        value = float(max(0, size))
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024.0 or unit == "GB":
                return f"{value:.1f}{unit}" if unit != "B" else f"{int(value)}B"
            value /= 1024.0
        return f"{int(value)}B"

    @staticmethod
    def _log_recovered_image_url(
        *,
        original_url: str | None,
        recovered_url: str,
        source_page_url: str | None,
    ) -> None:
        _log_image_debug(
            "image-url-recovered",
            original_url=original_url,
            recovered_url=recovered_url,
            source_page_url=source_page_url,
        )

    @staticmethod
    def _log_image_result_fate(
        *,
        plan_id: str,
        query: str,
        result_index: int | None,
        search_result: ImageSearchResult | None,
        fate: str,
        reason: str,
        raw_model_output: str | None = None,
    ) -> None:
        _log_image_debug(
            "image-result-fate",
            plan_id=plan_id,
            query=query,
            result_index=result_index,
            title=search_result.title if search_result is not None else None,
            image_url=search_result.image_url if search_result is not None else None,
            source_page_url=search_result.source_page_url if search_result is not None else None,
            thumbnail_url=search_result.thumbnail_url if search_result is not None else None,
            fate=fate,
            reason=reason,
            raw_model_output_preview=_short_debug_text(raw_model_output, limit=240),
        )

    def _image_check_with_mllm(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        model_alias: str,
        run_id: str | None,
        resolved_asset: ResolvedImageAsset | None = None,
    ) -> ImageValidationResult:
        if not search_result.image_url:
            return self._reject("missing_image_url_for_mllm_check")
        image_for_model = resolved_asset.model_url if resolved_asset is not None else search_result.image_url
        self._log_image_model_call(
            stage="image_check",
            when="before",
            model_alias=model_alias,
            plan_id=plan.plan_id,
            search_result=search_result,
            model_image_url=image_for_model,
            resolved_asset=resolved_asset,
        )
        response = self.model_client.generate(
            ModelRequest(
                model=model_alias,
                messages=[
                    ModelMessage(role="system", content=PROMPT_IMAGE_CHECK),
                    ModelMessage(
                        role="user",
                        content=[
                            {
                                "type": "text",
                                "text": self._image_check_prompt_input(
                                    plan=plan,
                                    search_result=search_result,
                                ),
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_for_model},
                            },
                        ],
                    ),
                ],
                metadata={"trace_label": f"image_check:{plan.plan_id}:{search_result.title or ''}"},
            )
        )
        self._log_image_model_call(
            stage="image_check",
            when="after",
            model_alias=model_alias,
            plan_id=plan.plan_id,
            search_result=search_result,
            model_image_url=image_for_model,
            resolved_asset=resolved_asset,
            model_output=response.content,
        )
        return self._parse_image_check_response(
            response.content,
            run_id=run_id,
            model_alias=model_alias,
            usage=response.usage,
        )

    @staticmethod
    def _log_image_model_call(
        *,
        stage: str,
        when: str,
        model_alias: str,
        plan_id: str,
        search_result: ImageSearchResult,
        model_image_url: str | None,
        resolved_asset: ResolvedImageAsset | None = None,
        model_output: str | None = None,
    ) -> None:
        _log_image_debug(
            "image-llm-call",
            stage=stage,
            when=when,
            model_alias=model_alias,
            plan_id=plan_id,
            image_attached=bool(model_image_url),
            model_image_source_kind=_image_source_kind(model_image_url),
            model_image_source=_format_debug_image_source(model_image_url),
            search_result_image_url=search_result.image_url,
            search_result_source_page_url=search_result.source_page_url,
            search_result_thumbnail_url=search_result.thumbnail_url,
            resolved_strategy=resolved_asset.strategy if resolved_asset is not None else None,
            resolved_url=resolved_asset.resolved_url if resolved_asset is not None else None,
            resolved_content_type=resolved_asset.content_type if resolved_asset is not None else None,
            resolved_cache_path=resolved_asset.cache_path if resolved_asset is not None else None,
            model_output_preview=_short_debug_text(model_output, limit=240),
        )

    @staticmethod
    def _image_check_prompt_input(
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
    ) -> str:
        return (
            f"Target:\n{plan.target.content or ''}\n\n"
            "Candidate metadata:\n"
            f"title: {search_result.title or ''}\n"
            f"caption/snippet: {search_result.snippet or ''}\n"
            f"source_page_url: {search_result.source_page_url or ''}\n"
        )

    @staticmethod
    def _wiki_inline_question_prompt_input(
        *,
        caption: str,
        wikipedia_title: str,
    ) -> str:
        return (
            f"Wikipedia: {wikipedia_title or ''}\n"
            f"description: {caption or ''}"
        )

    @staticmethod
    def _wiki_inline_title_check_prompt_input(*, wikipedia_title: str) -> str:
        return f"Wikipedia page title: {wikipedia_title or ''}"

    @staticmethod
    def _wiki_inline_judge_prompt_input(
        *,
        question: str,
        reference_answer: str,
        model_answer: str,
    ) -> str:
        return (
            f"Question: {question or ''}\n"
            f"Reference answer: {reference_answer or ''}\n"
            f"User answer: {model_answer or ''}"
        )

    def _wiki_inline_model_alias(self, *, preferred_env: str | None = None) -> str | None:
        if preferred_env:
            alias = os.environ.get(preferred_env)
            if alias:
                return alias
        return (
            os.environ.get("WIKI_INLINE_IMAGE_CHECK_MODEL")
            or self.image_check_model_alias
            or os.environ.get("IMAGE_CHECK_MODEL")
            or os.environ.get("TEXT_PROCESS_MODEL")
            or os.environ.get("IMAGE_GROUND_MODEL")
        )

    @staticmethod
    def _parse_image_check_response(
        text: str,
        *,
        run_id: str | None,
        model_alias: str | None = None,
        usage: dict[str, Any] | None = None,
    ) -> ImageValidationResult:
        match = re.search(r"<check>(.*?)</check>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        fields: dict[str, Any] = {"visual_facts": []}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not value:
                continue
            if key == "visual_fact":
                fields["visual_facts"].append(value)
            else:
                fields[key] = value

        decision = str(fields.get("decision", "")).lower()
        status = (
            ImageCandidateStatus.ACCEPTED
            if decision == "accept"
            else ImageCandidateStatus.REJECTED
        )
        confidence = ImageDiscoveryBuilder._parse_confidence(fields.get("confidence"))
        return ImageValidationResult(
            status=status,
            confidence=confidence,
            reason=fields.get("reason"),
            metadata={
                "check": "mllm_semantic",
                "model_alias": model_alias,
                "usage": usage,
                "visual_facts": fields.get("visual_facts", []),
                "raw_model_output": text,
                "run_id": run_id,
            },
        )

    @staticmethod
    def _parse_wiki_inline_question(text: str) -> dict[str, str]:
        payload = str(text or "")
        question_match = re.search(r"<question>(.*?)</question>", payload, flags=re.DOTALL | re.IGNORECASE)
        answer_match = re.search(r"<answer>(.*?)</answer>", payload, flags=re.DOTALL | re.IGNORECASE)
        question = re.sub(r"\s+", " ", question_match.group(1)).strip() if question_match else ""
        answer = re.sub(r"\s+", " ", answer_match.group(1)).strip() if answer_match else ""
        if not question:
            for raw_line in payload.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if line.lower().startswith("question:"):
                    question = line.split(":", 1)[1].strip()
                    if question:
                        break
                if not question and line.endswith("?"):
                    question = line
        if not answer:
            for raw_line in payload.splitlines():
                line = raw_line.strip()
                if line.lower().startswith("answer:"):
                    answer = line.split(":", 1)[1].strip()
                    if answer:
                        break
        if not question:
            raise ValueError(f"Failed to parse wiki-inline question from: {text[:500]}")
        result = {"question": question}
        if answer:
            result["answer"] = answer
        return result

    @staticmethod
    def _parse_wiki_inline_answer(text: str) -> str:
        answer = re.sub(r"\s+", " ", str(text or "")).strip()
        if not answer:
            raise ValueError("Empty wiki-inline answer output.")
        return answer

    @staticmethod
    def _parse_wiki_inline_judge(text: str) -> tuple[str, str | None]:
        payload = str(text or "")
        answer_match = re.search(r"<answer>(.*?)</answer>", payload, flags=re.DOTALL | re.IGNORECASE)
        decision = re.sub(r"\s+", " ", answer_match.group(1)).strip() if answer_match else ""
        reason: str | None = None
        thinking_match = re.search(r"<thinking>(.*?)</thinking>", payload, flags=re.DOTALL | re.IGNORECASE)
        if thinking_match:
            reason = re.sub(r"\s+", " ", thinking_match.group(1)).strip() or None
        if not decision:
            for raw_line in payload.splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                upper = line.upper()
                if upper in {"TRUE", "FALSE"}:
                    decision = upper
                    break
                if ":" in line:
                    key, value = line.split(":", 1)
                    if key.strip().lower() == "answer":
                        decision = value.strip()
                        break
        normalized = decision.lower()
        if normalized not in {"true", "false"}:
            raise ValueError(f"Failed to parse wiki-inline judge decision from: {text[:500]}")
        return normalized, reason

    @staticmethod
    def _parse_confidence(value: Any) -> float | None:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, confidence))

    def _run_image_self_qa_check(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        validation: ImageValidationResult,
        run_id: str | None,
        subject_title: str,
        caption: str,
        metadata_key: str,
        preferred_env: str | None,
        missing_model_reason: str,
        error_reason_prefix: str,
        trace_prefix: str,
    ) -> ImageValidationResult:
        if validation.status != ImageCandidateStatus.ACCEPTED:
            return validation

        model_alias = self._wiki_inline_model_alias(preferred_env=preferred_env)
        if not model_alias:
            return self._reject(missing_model_reason, drop_candidate=True)

        resolved_asset = self._resolved_image_from_validation(validation, include_transient=True)
        image_for_model = resolved_asset.model_url if resolved_asset is not None else search_result.image_url
        prompt_input = self._wiki_inline_question_prompt_input(
            caption=caption,
            wikipedia_title=subject_title,
        )

        question_stage = f"{trace_prefix}_question"
        answer_stage = f"{trace_prefix}_answer"
        judge_stage = f"{trace_prefix}_judge"

        try:
            self._log_image_model_call(
                stage=question_stage,
                when="before",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=image_for_model,
                resolved_asset=resolved_asset,
            )
            question_response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_WIKI_INLINE_IMAGE_QUESTION),
                        ModelMessage(
                            role="user",
                            content=[
                                {"type": "text", "text": prompt_input},
                                {"type": "image_url", "image_url": {"url": image_for_model}},
                            ],
                        ),
                    ],
                    metadata={"trace_label": f"{question_stage}:{plan.plan_id}"},
                )
            )
            self._log_image_model_call(
                stage=question_stage,
                when="after",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=image_for_model,
                resolved_asset=resolved_asset,
                model_output=question_response.content,
            )
            question_payload = self._parse_wiki_inline_question(question_response.content)
            generated_question = question_payload["question"]
            reference_answer = question_payload.get("answer", "").strip()
            if not reference_answer:
                raise ValueError("Missing self-QA reference answer output.")

            self._log_image_model_call(
                stage=answer_stage,
                when="before",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=image_for_model,
                resolved_asset=resolved_asset,
            )
            answer_response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_WIKI_INLINE_IMAGE_ANSWER),
                        ModelMessage(
                            role="user",
                            content=[
                                {"type": "text", "text": generated_question},
                                {"type": "image_url", "image_url": {"url": image_for_model}},
                            ],
                        ),
                    ],
                    metadata={"trace_label": f"{answer_stage}:{plan.plan_id}"},
                )
            )
            self._log_image_model_call(
                stage=answer_stage,
                when="after",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=image_for_model,
                resolved_asset=resolved_asset,
                model_output=answer_response.content,
            )
            model_answer = self._parse_wiki_inline_answer(answer_response.content)

            judge_input = self._wiki_inline_judge_prompt_input(
                question=generated_question,
                reference_answer=reference_answer,
                model_answer=model_answer,
            )
            self._log_image_model_call(
                stage=judge_stage,
                when="before",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=None,
                resolved_asset=resolved_asset,
            )
            judge_response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_WIKI_INLINE_IMAGE_JUDGE),
                        ModelMessage(role="user", content=judge_input),
                    ],
                    metadata={"trace_label": f"{judge_stage}:{plan.plan_id}"},
                )
            )
            self._log_image_model_call(
                stage=judge_stage,
                when="after",
                model_alias=model_alias,
                plan_id=plan.plan_id,
                search_result=search_result,
                model_image_url=None,
                resolved_asset=resolved_asset,
                model_output=judge_response.content,
            )
            judge_decision, judge_reason = self._parse_wiki_inline_judge(judge_response.content)
        except Exception as exc:
            print(
                f"[{trace_prefix}] self-qa request failed "
                f"plan_id={plan.plan_id} "
                f"model_alias={model_alias!r}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            return self._reject(
                f"{error_reason_prefix}:{exc.__class__.__name__}:{exc}",
                drop_candidate=True,
            )

        merged_metadata = dict(validation.metadata or {})
        merged_metadata[metadata_key] = {
            "question": generated_question,
            "reference_answer": reference_answer,
            "model_answer": model_answer,
            "judge_decision": judge_decision,
            "judge_reason": judge_reason,
            "model_alias": model_alias,
            "question_usage": question_response.usage,
            "answer_usage": answer_response.usage,
            "judge_usage": judge_response.usage,
            "question_raw_model_output": question_response.content,
            "answer_raw_model_output": answer_response.content,
            "judge_raw_model_output": judge_response.content,
            "caption": caption,
            "wikipedia_title": subject_title,
            "filter_reason": (
                "model_answered_generated_question"
                if judge_decision == "true"
                else "model_failed_generated_question"
            ),
        }
        if judge_decision == "true":
            merged_metadata["check"] = metadata_key
            merged_metadata["model_alias"] = model_alias
            merged_metadata["raw_model_output"] = judge_response.content
            return ImageValidationResult(
                status=ImageCandidateStatus.REJECTED,
                confidence=validation.confidence,
                reason="model_answered_generated_question",
                drop_candidate=True,
                metadata=merged_metadata,
            )

        return ImageValidationResult(
            status=validation.status,
            confidence=validation.confidence,
            reason=validation.reason,
            drop_candidate=validation.drop_candidate,
            metadata=merged_metadata,
        )

    def _wiki_inline_self_qa_check(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        validation: ImageValidationResult,
        run_id: str | None,
    ) -> ImageValidationResult:
        if validation.status != ImageCandidateStatus.ACCEPTED:
            return validation
        wiki_title = (self._source_node_title(plan.source_node_id) or "").strip()
        if not wiki_title:
            return self._reject("missing_wiki_inline_title", drop_candidate=True)
        caption = (search_result.snippet or plan.target.content or "").strip()
        return self._run_image_self_qa_check(
            plan=plan,
            search_result=search_result,
            validation=validation,
            run_id=run_id,
            subject_title=wiki_title,
            caption=caption,
            metadata_key="wiki_inline_self_qa",
            preferred_env="WIKI_INLINE_IMAGE_JUDGE_MODEL",
            missing_model_reason="missing_wiki_inline_judge_model",
            error_reason_prefix="wiki_inline_self_qa_model_error",
            trace_prefix="wiki_inline",
        )

    def _visual_plan_self_qa_check(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        image_node: ImageNode,
        validation: ImageValidationResult,
        run_id: str | None,
    ) -> ImageValidationResult:
        if validation.status != ImageCandidateStatus.ACCEPTED:
            return validation
        source_title = (self._source_node_title(plan.source_node_id) or plan.target.content or "").strip()
        if not source_title:
            return self._reject("missing_visual_plan_title", drop_candidate=True)
        caption = (image_node.caption or search_result.snippet or plan.target.content or "").strip()
        return self._run_image_self_qa_check(
            plan=plan,
            search_result=search_result,
            validation=validation,
            run_id=run_id,
            subject_title=source_title,
            caption=caption,
            metadata_key="visual_plan_self_qa",
            preferred_env="VISUAL_PLAN_IMAGE_JUDGE_MODEL",
            missing_model_reason="missing_visual_plan_judge_model",
            error_reason_prefix="visual_plan_self_qa_model_error",
            trace_prefix="visual_plan",
        )


    @staticmethod
    def _wiki_inline_entity_uniqueness_prompt_input(
        *,
        wikipedia_title: str,
        caption: str,
        grounded_entities: list[dict[str, Any]],
    ) -> str:
        lines = [
            f"Wikipedia: {wikipedia_title or ''}",
            f"description: {caption or ''}",
            "Grounded candidate entities:",
        ]
        for entity in grounded_entities:
            lines.append(
                "- "
                f"{entity.get('name') or ''} | "
                f"locator: {entity.get('relation_to_image') or ''} | "
                f"evidence: {entity.get('evidence') or ''}"
            )
        return "\n".join(lines)

    def _parse_wiki_inline_entity_uniqueness_filter(self, text: str) -> dict[str, Any]:
        match = re.search(r"<filter>(.*?)</filter>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        parsed: dict[str, Any] = {
            "overall_decision": None,
            "reason": None,
            "entities": {},
            "raw_model_output": text,
        }
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().lower()
            value = value.strip()
            if not value:
                continue
            if key == "overall_decision":
                decision = value.lower()
                if decision in {"keep", "drop"}:
                    parsed["overall_decision"] = decision
            elif key == "reason":
                parsed["reason"] = value
            elif key == "entity":
                parts = [part.strip() for part in value.split("|")]
                if len(parts) < 2 or not parts[0]:
                    continue
                decision = parts[1].lower()
                if decision not in {"keep", "block"}:
                    continue
                normalized = self._normalize_entity_label(parts[0])
                if not normalized:
                    continue
                parsed["entities"][normalized] = {
                    "name": parts[0],
                    "decision": decision,
                    "reason": parts[2] if len(parts) > 2 else "",
                }
        return parsed

    def _filter_wiki_inline_grounded_entities(
        self,
        *,
        plan: VisualSearchPlan,
        search_result: ImageSearchResult,
        image_node: ImageNode,
        grounded_entities: list[dict[str, Any]],
        run_id: str | None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        entities = list(grounded_entities or [])
        if not self._is_wiki_inline_plan(plan) or not self.config.enable_wiki_inline_entity_uniqueness_filter:
            return entities, {}

        base_summary: dict[str, Any] = {
            "applied": False,
            "input_entity_count": len(entities),
            "kept_entity_count": len(entities),
            "blocked_entity_count": 0,
            "wikipedia_title": (self._source_node_title(plan.source_node_id) or "").strip(),
            "caption": (image_node.caption or search_result.snippet or plan.target.content or "").strip(),
        }
        if not entities:
            base_summary["skip_reason"] = "no_grounded_entities"
            return entities, base_summary

        wiki_title = base_summary["wikipedia_title"]
        if not wiki_title:
            base_summary["skip_reason"] = "missing_wiki_inline_title"
            return entities, base_summary

        model_alias = self._wiki_inline_model_alias(preferred_env="WIKI_INLINE_IMAGE_ENTITY_FILTER_MODEL")
        if not model_alias:
            base_summary["skip_reason"] = "missing_wiki_inline_entity_filter_model"
            return entities, base_summary

        prompt_input = self._wiki_inline_entity_uniqueness_prompt_input(
            wikipedia_title=wiki_title,
            caption=base_summary["caption"],
            grounded_entities=entities,
        )
        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_WIKI_INLINE_ENTITY_UNIQUENESS_FILTER),
                        ModelMessage(role="user", content=prompt_input),
                    ],
                    metadata={
                        "trace_label": f"wiki_inline_entity_uniqueness:{plan.plan_id}:{wiki_title[:80]}",
                        "run_id": run_id,
                    },
                )
            )
            parsed = self._parse_wiki_inline_entity_uniqueness_filter(response.content)
        except Exception as exc:
            print(
                "[wiki-inline-entity-uniqueness] model request failed "
                f"plan_id={plan.plan_id} "
                f"model_alias={model_alias!r}",
                file=sys.stderr,
                flush=True,
            )
            traceback.print_exc(file=sys.stderr)
            base_summary["skip_reason"] = f"model_error:{exc.__class__.__name__}:{exc}"
            return entities, base_summary

        decisions = parsed.get("entities") or {}
        kept_entities: list[dict[str, Any]] = []
        kept_entity_summaries: list[dict[str, Any]] = []
        blocked_entity_summaries: list[dict[str, Any]] = []
        for entity in entities:
            label = (entity.get("name") or "").strip()
            normalized = self._normalize_entity_label(label)
            decision_record = decisions.get(normalized) if normalized else None
            decision = (decision_record or {}).get("decision")
            reason = ((decision_record or {}).get("reason") or "").strip()
            if decision == "keep":
                kept_entities.append(entity)
                kept_entity_summaries.append(
                    {
                        "name": label,
                        "reason": reason or "kept_by_model",
                    }
                )
                continue
            blocked_entity_summaries.append(
                {
                    **entity,
                    "status": "blocked_by_uniqueness_filter" if decision == "block" else "missing_model_decision",
                    "filter_reason": (
                        reason
                        or "no keep/block decision returned for entity"
                    ),
                }
            )

        summary = {
            "applied": True,
            "model_alias": model_alias,
            "wikipedia_title": wiki_title,
            "caption": base_summary["caption"],
            "input_entity_count": len(entities),
            "kept_entity_count": len(kept_entities),
            "blocked_entity_count": len(blocked_entity_summaries),
            "overall_decision": parsed.get("overall_decision") or ("keep" if kept_entities else "drop"),
            "reason": parsed.get("reason") or (
                "at least one grounded entity is a unique canonical entity"
                if kept_entities
                else "all grounded entities were blocked as non-unique or insufficiently grounded"
            ),
            "kept_entities": kept_entity_summaries,
            "blocked_entities": blocked_entity_summaries,
            "usage": response.usage,
            "raw_model_output": response.content,
        }
        return kept_entities, summary

    @staticmethod
    def _reject(reason: str, *, drop_candidate: bool = False) -> ImageValidationResult:
        return ImageValidationResult(
            status=ImageCandidateStatus.REJECTED,
            confidence=0.0,
            reason=reason,
            drop_candidate=drop_candidate,
        )

    @staticmethod
    def _snapshot_from_wiki_inline_result(
        search_result: ImageSearchResult,
        *,
        plan: VisualSearchPlan,
        run_id: str | None,
    ) -> SearchSnapshot:
        query = (plan.queries[0].query if plan.queries else search_result.snippet or search_result.title or search_result.image_url or "")
        return SearchSnapshot.create(
            SearchEngine.WIKIPEDIA,
            query=query,
            request={
                "query": query,
                "engine": "wikipedia_inline_image",
                "source_page_url": search_result.source_page_url,
            },
            response_preview=repr(search_result.to_dict()),
            result_count=1,
            status_code=200,
            run_id=run_id,
            metadata={"raw_engine": "wikipedia_inline_image"},
        )

    @staticmethod
    def _candidate_key(result: ImageSearchResult) -> str | None:
        return result.image_url or result.source_page_url or result.title

    @staticmethod
    def _extension(url: str | None) -> str | None:
        if not url:
            return None
        path = url.split("?", 1)[0].split("#", 1)[0].lower()
        if "." not in path:
            return None
        return "." + path.rsplit(".", 1)[-1]

    @staticmethod
    def _content_type(result: ImageSearchResult) -> str | None:
        imageinfo = result.raw.get("imageinfo") if result.raw else None
        if not isinstance(imageinfo, list) or not imageinfo:
            return None
        first = imageinfo[0]
        if not isinstance(first, dict):
            return None
        mime = first.get("mime")
        return mime if isinstance(mime, str) else None

    @staticmethod
    def _snapshot_engine(response: SearchResponse) -> SearchEngine:
        engine = response.engine.lower()
        if "commons" in engine:
            return SearchEngine.WIKIMEDIA_COMMONS
        if "serpapi" in engine and "image" in engine:
            return SearchEngine.SERPAPI_IMAGE
        if "serpapi" in engine:
            return SearchEngine.SERPAPI_TEXT
        if "serper" in engine or "image" in engine:
            return SearchEngine.SERPER_IMAGE
        return SearchEngine.OTHER

    def _snapshot_from_response(
        self,
        response: SearchResponse,
        *,
        run_id: str | None,
    ) -> SearchSnapshot:
        return SearchSnapshot.create(
            self._snapshot_engine(response),
            query=response.query,
            request={"query": response.query, "engine": response.engine},
            response_preview=self._response_preview(response),
            result_count=len(response.results),
            status_code=response.status_code,
            run_id=run_id,
            metadata={
                "raw_engine": response.engine,
                "response_metadata": response.metadata,
            },
        )

    def _snapshot_from_error(
        self,
        *,
        client: SearchClient,
        query: str,
        error: Exception,
        run_id: str | None,
    ) -> SearchSnapshot:
        return SearchSnapshot.create(
            self._engine_from_client(client),
            query=query,
            request={
                "query": query,
                "client": client.__class__.__name__,
                "limit": self.config.per_query_limit,
            },
            result_count=0,
            error=f"{error.__class__.__name__}: {error}",
            run_id=run_id,
            status=RecordStatus.FAILED,
        )

    @staticmethod
    def _engine_from_client(client: SearchClient) -> SearchEngine:
        name = client.__class__.__name__.lower()
        if "commons" in name:
            return SearchEngine.WIKIMEDIA_COMMONS
        if "serpapi" in name:
            return SearchEngine.SERPAPI_IMAGE
        if "serper" in name:
            return SearchEngine.SERPER_IMAGE
        return SearchEngine.OTHER

    @staticmethod
    def _response_preview(response: SearchResponse, *, limit: int = 5) -> str:
        preview = [item.to_dict() for item in response.results[:limit]]
        return repr(preview)

    @staticmethod
    def _image_node_from_result(
        result: ImageSearchResult,
        *,
        run_id: str | None,
        resolved_asset: ResolvedImageAsset | None = None,
    ) -> ImageNode:
        metadata = {
            "search_source": result.source,
            "thumbnail_url": result.thumbnail_url,
            "rank": result.rank,
            "raw": result.raw,
        }
        if resolved_asset is not None:
            metadata["resolved_image"] = resolved_asset.to_metadata()
        return ImageNode.from_url(
            (
                resolved_asset.asset_uri
                if resolved_asset is not None
                else result.image_url or result.source_page_url or result.title or ""
            ),
            source_page_url=result.source_page_url,
            caption=result.snippet,
            title=result.title,
            run_id=run_id,
            metadata=metadata,
        )

    @staticmethod
    def _image_asset(
        result: ImageSearchResult,
        *,
        image_node: ImageNode,
        resolved_asset: ResolvedImageAsset | None = None,
    ) -> Asset:
        uri = (
            resolved_asset.asset_uri
            if resolved_asset is not None
            else result.image_url or image_node.image_url or image_node.node_id
        )
        return Asset.create(
            AssetType.IMAGE_ORIGINAL,
            uri,
            original_url=result.image_url,
            content_type=resolved_asset.content_type if resolved_asset is not None else ImageDiscoveryBuilder._content_type(result),
            metadata={
                "source_page_url": result.source_page_url,
                "width": result.width,
                "height": result.height,
                "storage_status": image_node.storage_status,
                "resolved_url": resolved_asset.resolved_url if resolved_asset is not None else None,
                "cache_path": resolved_asset.cache_path if resolved_asset is not None else None,
                "resolution_strategy": resolved_asset.strategy if resolved_asset is not None else None,
            },
        )

    @staticmethod
    def _thumbnail_asset(result: ImageSearchResult) -> Asset | None:
        if not result.thumbnail_url:
            return None
        return Asset.create(
            AssetType.IMAGE_THUMBNAIL,
            result.thumbnail_url,
            original_url=result.thumbnail_url,
            metadata={
                "source_page_url": result.source_page_url,
                "original_image_url": result.image_url,
            },
        )

    def _link_or_queue_grounded_entities(
        self,
        *,
        image_node: ImageNode,
        grounded_entities: list[dict[str, Any]],
        image_evidence: Evidence,
        run_id: str | None,
        source_node_title: str | None,
        source_query_text: str | None,
    ) -> tuple[list[Edge], list[dict[str, Any]]]:
        if self.store is None or not grounded_entities:
            return [], []

        edges: list[Edge] = []
        unresolved: list[dict[str, Any]] = []
        queued_tasks: list[dict[str, Any]] = []
        blocked_query_entities = self._query_implied_entity_labels(
            source_query_text,
            source_node_title=source_node_title,
            grounded_entities=grounded_entities,
        )
        query_overlap_entities: list[dict[str, Any]] = []
        for entity in grounded_entities:
            if not self._should_expand_entity(entity):
                unresolved.append({**entity, "status": "filtered_out"})
                continue
            query_overlap_entity = self._is_query_implied_entity(entity, blocked_query_entities)
            if query_overlap_entity:
                query_overlap_entities.append(
                    {
                        **entity,
                        "status": "query_overlap_entity",
                    }
                )
            resolution = self._resolve_grounded_entity_link_target(
                entity,
                source_node_title=source_node_title,
                source_query_text=source_query_text,
                image_caption=image_node.caption,
            )
            if resolution is None:
                unresolved.append({**entity, "status": "unresolved"})
                continue
            matched_node = resolution.get("matched_node")
            resolved_target = resolution.get("resolved_target")
            if matched_node is None:
                if resolved_target is None:
                    unresolved.append({**entity, "status": "unresolved"})
                    continue
                queue_verification = self._verify_image_entity_before_queue(
                    image_node=image_node,
                    entity=entity,
                    resolved_target=resolved_target,
                    source_type="image_grounding_delayed",
                )
                if queue_verification is not None:
                    # Mutate the original grounding record as well as the
                    # pending-link copy: image-node metadata already holds
                    # these entity dictionaries for later debugging/stats.
                    entity["queue_verification"] = queue_verification
                    if queue_verification.get("decision") == "contradict":
                        unresolved.append({**entity, "status": "verification_contradict"})
                        continue
                queued_tasks.append(
                    {
                        "url": resolved_target["url"],
                        "title": resolved_target.get("title") or entity.get("name"),
                        "pending_link": {
                            "link_type": "image_entity",
                            "parent_node_id": image_node.node_id,
                            "source_evidence_id": image_evidence.evidence_id,
                            "entity": entity,
                            "resolved_target": resolved_target,
                            "query_overlap_entity": query_overlap_entity,
                            "entity_resolution": resolution.get("debug"),
                        },
                    }
                )
                continue
            relation = entity.get("relation_to_image") or "depicts"
            edge = Edge.create(
                image_node.node_id,
                matched_node["node_id"],
                edge_type=EdgeType.IMAGE_DEPICTS,
                relation=relation,
                src_node_type=NodeType.IMAGE.value,
                dst_node_type=NodeType.TEXT.value,
                evidence_refs=[
                    EvidenceRef(
                        evidence_id=image_evidence.evidence_id,
                        quote=entity.get("evidence"),
                        metadata={
                            "grounded_entity": entity,
                            "matched_title": matched_node.get("title"),
                            "query_overlap_entity": query_overlap_entity,
                        },
                    )
                ],
                source=EdgeSource(
                    source_type="image_grounding",
                    url=image_node.image_url,
                    run_id=run_id,
                    builder=self.builder_name,
                ),
                extractor=self.builder_name,
                metadata={
                    "entity_name": entity.get("name"),
                    "entity_type": entity.get("type"),
                    "match_method": matched_node.get("_match_method"),
                    "query_overlap_entity": query_overlap_entity,
                    "entity_resolution": resolution.get("debug"),
                },
                evidence_key=f"{image_evidence.evidence_id}:{entity.get('name')}:{matched_node['node_id']}",
            )
            edges.append(edge)

        if unresolved or query_overlap_entities:
            image_node.metadata = dict(image_node.metadata or {})
            if unresolved:
                image_node.metadata["unresolved_grounded_entities"] = unresolved
            if query_overlap_entities:
                image_node.metadata["query_overlap_grounded_entities"] = query_overlap_entities
        return edges, queued_tasks

    def _verify_image_entity_before_queue(
        self,
        *,
        image_node: ImageNode,
        entity: dict[str, Any],
        resolved_target: dict[str, Any],
        source_type: str,
    ) -> dict[str, Any] | None:
        """Run the existing post-process verifier before a new text task is queued.

        Only a confident ``contradict`` result blocks the queue.  Evidence
        shortages and operational failures intentionally fail open so transient
        Reader/reference-image problems do not remove recall.
        """
        if not self.config.enable_image_entity_queue_verification:
            return None
        prepare_model = (
            self.config.image_entity_queue_verify_prepare_model
            or os.environ.get("IMAGE_ENTITY_QUEUE_VERIFY_PREPARE_MODEL")
            or os.environ.get("IMAGE_EDGE_VERIFY_PREPARE_MODEL")
            or os.environ.get("TEXT_PROCESS_MODEL")
            or ""
        )
        judge_model = (
            self.config.image_entity_queue_verify_judge_model
            or os.environ.get("IMAGE_ENTITY_QUEUE_VERIFY_JUDGE_MODEL")
            or os.environ.get("IMAGE_EDGE_VERIFY_JUDGE_MODEL")
            or os.environ.get("IMAGE_GROUND_MODEL")
            or os.environ.get("IMAGE_CHECK_MODEL")
            or ""
        )
        target_url = str(resolved_target.get("url") or "").strip()
        target_title = str(resolved_target.get("title") or entity.get("name") or "").strip()
        base_record = {
            "enabled": True,
            "target_url": target_url or None,
            "target_title": target_title or None,
            "prepare_model_alias": prepare_model or None,
            "judge_model_alias": judge_model or None,
        }
        if not target_url or not prepare_model or not judge_model:
            return {
                **base_record,
                "decision": "insufficient",
                "error_type": "verifier_not_configured",
                "reason": "missing resolved target URL or verifier model alias",
            }

        # The post-process verifier is intentionally reused here: it prepares
        # target-page evidence, filters Wiki reference images, and judges the
        # exact grounding relation against the graph image.
        try:
            from .post_process.verify_image_text_edges import (
                _compact_reference_images_for_output,
                _extract_reference_images,
                _judge_edge,
                _prepare_entity_context,
                _prepare_reference_image,
                _resolve_image_node_for_model,
                _resolve_reference_image,
            )

            reader = EnhancedReaderClient(
                base_url=self.config.image_grounding_reader_base_url,
                timeout_s=self.config.image_grounding_reader_timeout_s,
            )
            wiki_doc = reader.read(target_url).to_dict()
            text_node = {
                "node_id": f"pending:{resolved_target.get('canonical_id') or target_url}",
                "title": target_title,
                "aliases": [entity.get("name")] if entity.get("name") else [],
                "url": target_url,
                "source": {"url": target_url},
            }
            edge = {
                "edge_id": f"pending:{image_node.node_id}:{target_url}",
                "relation": entity.get("relation_to_image") or "depicts",
                "source": {"source_type": source_type},
            }
            image_record = image_node.to_dict()
            prepared_context = _prepare_entity_context(
                self.model_client,
                prepare_model,
                text_node=text_node,
                wiki_document=wiki_doc,
                image_title=str(image_node.title or ""),
            )
            raw_reference_images = _extract_reference_images(
                wiki_doc.get("raw_markdown") or wiki_doc.get("content") or "",
                target_url,
                max(1, int(self.config.image_entity_queue_verify_max_reference_images)),
            )
            kept_reference_images: list[dict[str, Any]] = []
            for image_item in raw_reference_images:
                resolved = _resolve_reference_image(
                    self,
                    image_item=image_item,
                    page_title=str(wiki_doc.get("title") or target_title),
                    entity_title=target_title,
                )
                if resolved is None:
                    continue
                reference_decision = _prepare_reference_image(
                    self.model_client,
                    prepare_model,
                    entity_title=target_title,
                    visual_profile=list(prepared_context.get("visual_profile") or []),
                    event_context=list(prepared_context.get("event_context") or []),
                    image_item=resolved,
                )
                if reference_decision.get("keep") is True:
                    kept_reference_images.append({**resolved, **reference_decision})
                if len(kept_reference_images) >= max(1, int(self.config.image_entity_queue_verify_max_reference_images)):
                    break
            judged = _judge_edge(
                self.model_client,
                judge_model,
                image_node=image_record,
                text_node=text_node,
                edge=edge,
                grounded_entity=entity,
                prepared_context=prepared_context,
                reference_images=kept_reference_images,
                primary_image_model_url=_resolve_image_node_for_model(self, image_node=image_record),
            )
            return {
                **base_record,
                "decision": str(judged.get("decision") or "insufficient"),
                "error_type": str(judged.get("error_type") or "insufficient_evidence"),
                "confidence": judged.get("confidence"),
                "reason": str(judged.get("reason") or ""),
                "evidence_for": list(judged.get("evidence_for") or []),
                "evidence_against": list(judged.get("evidence_against") or []),
                "kept_reference_image_count": len(kept_reference_images),
                "reference_images": _compact_reference_images_for_output(kept_reference_images),
            }
        except Exception as exc:
            return {
                **base_record,
                "decision": "insufficient",
                "error_type": "verification_error",
                "reason": f"{exc.__class__.__name__}: {exc}",
            }

    def _query_implied_entity_labels(
        self,
        query_text: str | None,
        *,
        source_node_title: str | None = None,
        grounded_entities: list[dict[str, Any]] | None = None,
    ) -> set[str]:
        if not query_text:
            return set()
        blocked_from_llm = self._query_implied_entity_labels_with_llm(
            query_text,
            source_node_title=source_node_title,
            grounded_entities=grounded_entities or [],
        )
        if self.store is None:
            return blocked_from_llm
        normalized_query = self._normalize_entity_label(query_text)
        if not normalized_query:
            return set()

        blocked: set[str] = set()
        query_tokens = set(normalized_query.split())
        source_title_label = self._normalize_entity_label(source_node_title or "")
        if source_title_label and (
            source_title_label == normalized_query
            or source_title_label in normalized_query
            or set(source_title_label.split()).issubset(query_tokens)
        ):
            blocked.add(source_title_label)
        for node in self.store.list_nodes():
            if node.get("node_type") != NodeType.TEXT.value:
                continue
            labels = [node.get("title") or "", *(node.get("aliases") or [])]
            for label in labels:
                normalized_label = self._normalize_entity_label(label)
                if not normalized_label or len(normalized_label) < 4:
                    continue
                label_tokens = set(normalized_label.split())
                if not label_tokens:
                    continue
                if (
                    normalized_label == normalized_query
                    or normalized_label in normalized_query
                    or label_tokens.issubset(query_tokens)
                    or (len(label_tokens) == 1 and next(iter(label_tokens)) in query_tokens)
                ):
                    blocked.add(normalized_label)
        # Explicit lexical matches are hard filters: an LLM "keep" must not
        # re-enable an entity that the search query already names. The LLM can
        # still add semantic matches such as aliases and alternate names.
        return blocked | blocked_from_llm

    def _query_implied_entity_labels_with_llm(
        self,
        query_text: str,
        *,
        source_node_title: str | None,
        grounded_entities: list[dict[str, Any]],
    ) -> set[str]:
        if not grounded_entities:
            return set()
        model_alias = (
            os.environ.get("IMAGE_QUERY_ENTITY_FILTER_MODEL")
            or os.environ.get("IMAGE_GROUND_MODEL")
            or self.image_check_model_alias
        )
        if not model_alias:
            return set()
        candidate_names = []
        seen: set[str] = set()
        for entity in grounded_entities:
            label = (entity.get("name") or "").strip()
            normalized = self._normalize_entity_label(label)
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            candidate_names.append(label)
        if not candidate_names:
            return set()
        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_IMAGE_QUERY_ENTITY_FILTER),
                        ModelMessage(
                            role="user",
                            content=(
                                f"Source text node title:\n{source_node_title or ''}\n\n"
                                f"Visual query text:\n{query_text}\n\n"
                                "Grounded candidate entities:\n"
                                + "\n".join(f"- {name}" for name in candidate_names)
                            ),
                        ),
                    ],
                    metadata={"trace_label": f"image_query_entity_filter:{source_node_title or ''}:{query_text[:80]}"},
                )
            )
        except Exception:
            return set()
        return self._parse_query_entity_filter_response(response.content)

    def _parse_query_entity_filter_response(self, text: str) -> set[str]:
        match = re.search(r"<filter>(.*?)</filter>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        blocked: set[str] = set()
        tagged_entities = re.findall(r"<entity>\s*(.*?)\s*</entity>", block, flags=re.DOTALL | re.IGNORECASE)
        # Accept legacy / malformed untagged records as a compatibility fallback.
        records = tagged_entities or [line.strip() for line in block.splitlines()]
        for record in records:
            value = record.strip()
            if value.lower().startswith("entity:"):
                value = value.split(":", 1)[1].strip()
            parts = [part.strip() for part in value.split("|")]
            if len(parts) < 2:
                continue
            name = parts[0]
            decision = parts[1].lower()
            if decision == "block":
                normalized = self._normalize_entity_label(name)
                if normalized:
                    blocked.add(normalized)
        return blocked

    def _is_query_implied_entity(self, entity: dict[str, Any], blocked_query_entities: set[str]) -> bool:
        label = self._normalize_entity_label(entity.get("name") or "")
        if not label:
            return False
        return label in blocked_query_entities

    def _resolve_grounded_entity(
        self,
        entity: dict[str, Any],
        *,
        source_node_title: str | None,
        image_caption: str | None,
    ) -> dict[str, Any] | None:
        resolution = self._resolve_grounded_entity_link_target(
            entity,
            source_node_title=source_node_title,
            source_query_text=None,
            image_caption=image_caption,
        )
        if resolution is None:
            return None
        if resolution.get("resolved_target") is not None:
            return resolution["resolved_target"]
        matched_node = resolution.get("matched_node")
        if matched_node is None:
            return None
        source = matched_node.get("source") or {}
        url = source.get("url") if isinstance(source, dict) else None
        if not url:
            return None
        return {
            "title": matched_node.get("title") or entity.get("name"),
            "url": url,
            "canonical_id": matched_node.get("canonical_id") or f"wikipedia:{matched_node.get('title') or entity.get('name')}",
        }

    def _resolve_grounded_entity_link_target(
        self,
        entity: dict[str, Any],
        *,
        source_node_title: str | None,
        source_query_text: str | None,
        image_caption: str | None,
    ) -> dict[str, Any] | None:
        label = (entity.get("name") or "").strip()
        if not label:
            return None
        context_parts = [part for part in (entity.get("evidence"), image_caption, source_node_title) if part]
        context = " ".join(context_parts)
        wiki_candidates = self.wiki_resolver.search_candidates(
            label,
            entity_type=entity.get("type"),
            source_title=source_node_title,
            context=context,
        )
        if not wiki_candidates:
            return None

        local_candidates = self._find_text_nodes_by_candidate_urls(wiki_candidates)
        selection = self._select_entity_resolution_candidate(
            entity=entity,
            source_node_title=source_node_title,
            source_query_text=source_query_text,
            image_caption=image_caption,
            wiki_candidates=wiki_candidates,
            local_candidates=local_candidates,
        )
        if selection is None:
            return None
        matched_node = selection.get("matched_node")
        if matched_node is not None:
            matched = dict(matched_node)
            matched["_match_method"] = "wiki_url_llm_select_existing"
            return {
                "matched_node": matched,
                "resolved_target": None,
                "debug": selection.get("debug"),
            }
        resolved_target = selection.get("resolved_target")
        if resolved_target is None:
            return None
        return {
            "matched_node": None,
            "resolved_target": resolved_target,
            "debug": selection.get("debug"),
        }

    def _find_text_nodes_by_candidate_urls(
        self,
        candidates: list[Any],
    ) -> list[dict[str, Any]]:
        if self.store is None or not candidates:
            return []
        candidate_by_url = {
            candidate.url: candidate
            for candidate in candidates
            if getattr(candidate, "url", None)
        }
        if not candidate_by_url:
            return []
        matches: list[dict[str, Any]] = []
        for node in self.store.list_nodes():
            if node.get("node_type") != NodeType.TEXT.value:
                continue
            source = node.get("source") or {}
            url = source.get("url") if isinstance(source, dict) else None
            if not url or url not in candidate_by_url:
                continue
            candidate = candidate_by_url[url]
            matches.append(
                {
                    "node": dict(node),
                    "url": url,
                    "candidate": candidate.to_dict(),
                }
            )
        return matches

    def _select_entity_resolution_candidate(
        self,
        *,
        entity: dict[str, Any],
        source_node_title: str | None,
        source_query_text: str | None,
        image_caption: str | None,
        wiki_candidates: list[Any],
        local_candidates: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        model_alias = (
            os.environ.get("IMAGE_ENTITY_RESOLVE_MODEL")
            or os.environ.get("TEXT_PROCESS_MODEL")
            or os.environ.get("IMAGE_GROUND_MODEL")
            or self.image_check_model_alias
        )
        if not model_alias:
            return None

        user_text = self._build_entity_resolution_prompt_input(
            entity=entity,
            source_node_title=source_node_title,
            source_query_text=source_query_text,
            image_caption=image_caption,
            wiki_candidates=wiki_candidates,
            local_candidates=local_candidates,
        )
        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    messages=[
                        ModelMessage(role="system", content=PROMPT_IMAGE_ENTITY_RESOLUTION),
                        ModelMessage(role="user", content=user_text),
                    ],
                    metadata={
                        "trace_label": f"image_entity_resolution:{entity.get('name') or ''}:{source_node_title or ''}"
                    },
                )
            )
        except Exception:
            return None
        decision = self._parse_entity_resolution_response(response.content)
        local_candidate_by_url = {
            item.get("url"): item
            for item in local_candidates
            if item.get("url")
        }
        debug = self._build_entity_resolution_debug_payload(
            model_alias=model_alias,
            decision=decision,
            wiki_candidates=wiki_candidates,
            local_candidates=local_candidates,
            local_candidate_by_url=local_candidate_by_url,
        )
        if decision["decision"] != "select":
            return None
        index = decision["candidate_index"]
        if index is None:
            return None
        if index < 0 or index >= len(wiki_candidates):
            return None
        chosen = wiki_candidates[index]
        matched_local = local_candidate_by_url.get(getattr(chosen, "url", None))
        if matched_local is not None:
            return {
                "matched_node": dict(matched_local["node"]),
                "resolved_target": None,
                "debug": debug,
            }
        return {
            "matched_node": None,
            "resolved_target": chosen.to_dict(),
            "debug": debug,
        }

    @staticmethod
    def _compact_entity_resolution_candidate(candidate: Any) -> dict[str, Any]:
        raw = candidate.to_dict() if hasattr(candidate, "to_dict") else dict(candidate or {})
        return {
            "title": raw.get("title") or "",
            "url": raw.get("url") or "",
            "canonical_id": raw.get("canonical_id") or "",
            "qid": raw.get("qid") or "",
            "score": raw.get("score"),
            "source": raw.get("source") or "",
        }

    @staticmethod
    def _compact_local_resolution_match(item: dict[str, Any] | None) -> dict[str, Any] | None:
        if not item:
            return None
        node = item.get("node") or {}
        return {
            "node_id": node.get("node_id") or "",
            "title": node.get("title") or "",
            "url": item.get("url") or "",
        }

    def _build_entity_resolution_debug_payload(
        self,
        *,
        model_alias: str,
        decision: dict[str, Any],
        wiki_candidates: list[Any],
        local_candidates: list[dict[str, Any]],
        local_candidate_by_url: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        selected_candidate = None
        selected_local_node = None
        index = decision.get("candidate_index")
        if isinstance(index, int) and 0 <= index < len(wiki_candidates):
            chosen = wiki_candidates[index]
            selected_candidate = self._compact_entity_resolution_candidate(chosen)
            selected_local_node = self._compact_local_resolution_match(
                local_candidate_by_url.get(getattr(chosen, "url", None))
            )
        return {
            "model_alias": model_alias,
            "decision": decision,
            "candidate_count": len(wiki_candidates),
            "local_candidate_count": len(local_candidates),
            "selected_candidate": selected_candidate,
            "selected_local_node": selected_local_node,
        }

    @staticmethod
    def _build_entity_resolution_prompt_input(
        *,
        entity: dict[str, Any],
        source_node_title: str | None,
        source_query_text: str | None,
        image_caption: str | None,
        wiki_candidates: list[Any],
        local_candidates: list[dict[str, Any]],
    ) -> str:
        local_candidate_by_url = {
            item.get("url"): item
            for item in local_candidates
            if item.get("url")
        }
        lines = [
            "Grounded entity:",
            f"name: {entity.get('name') or ''}",
            f"type: {entity.get('type') or ''}",
            f"relation_to_image: {entity.get('relation_to_image') or ''}",
            f"evidence: {entity.get('evidence') or ''}",
            "",
            "Context:",
            f"source node title: {source_node_title or ''}",
            f"source query text: {source_query_text or ''}",
            f"image caption: {image_caption or ''}",
            "",
            "Candidate list (choose by candidate_index):",
        ]
        for idx, candidate in enumerate(wiki_candidates):
            local_item = local_candidate_by_url.get(getattr(candidate, "url", None))
            local_node = (local_item or {}).get("node") or {}
            lines.extend(
                [
                    f"[{idx}] title: {candidate.title or ''}",
                    f"    url: {candidate.url or ''}",
                    f"    snippet: {candidate.snippet or ''}",
                    f"    score: {candidate.score}",
                    f"    exists_in_local_graph: {'yes' if local_item is not None else 'no'}",
                    f"    local_node_id: {local_node.get('node_id') or ''}",
                    f"    local_title: {local_node.get('title') or ''}",
                ]
            )
        return "\n".join(lines)

    @staticmethod
    def _parse_entity_resolution_response(text: str) -> dict[str, Any]:
        match = re.search(r"<selection>(.*?)</selection>", text, flags=re.DOTALL | re.IGNORECASE)
        block = match.group(1) if match else text
        fields: dict[str, str] = {}
        for raw_line in block.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            fields[key.strip().lower()] = value.strip()
        decision = (fields.get("decision") or "").strip().lower()
        raw_index = (fields.get("candidate_index") or "").strip().lower()
        candidate_index: int | None = None
        if raw_index and raw_index != "none":
            try:
                candidate_index = int(raw_index)
            except ValueError:
                candidate_index = None
        return {
            "decision": decision,
            "candidate_index": candidate_index,
            "reason": fields.get("reason") or "",
        }

    def _find_text_node_by_url(self, url: str | None) -> dict[str, Any] | None:
        if self.store is None or not url:
            return None
        for node in self.store.list_nodes():
            if node.get("node_type") != NodeType.TEXT.value:
                continue
            source = node.get("source") or {}
            if isinstance(source, dict) and source.get("url") == url:
                return dict(node)
        return None

    def _source_node_title(self, node_id: str | None) -> str | None:
        if self.store is None or not node_id:
            return None
        record = self.store.get_node(node_id)
        if record is None:
            return None
        return record.get("title") or record.get("canonical_id")

    def _should_expand_entity(self, entity: dict[str, Any]) -> bool:
        label = (entity.get("name") or "").strip()
        if len(label) < 2:
            return False
        entity_type = self._normalize_entity_type(entity.get("type"))
        if entity_type and entity_type not in self.config.expandable_entity_types:
            return False
        return True

    def _match_text_node(self, label: str | None) -> dict[str, Any] | None:
        if self.store is None or not label:
            return None
        needle = self._normalize_entity_label(label)
        if not needle:
            return None

        exact_matches: list[tuple[dict[str, Any], str]] = []
        contains_matches: list[tuple[dict[str, Any], str]] = []
        for node in self.store.list_nodes():
            if node.get("node_type") != NodeType.TEXT.value:
                continue
            title = node.get("title") or ""
            aliases = node.get("aliases") or []
            labels = [title, *aliases]
            normalized_labels = [self._normalize_entity_label(item) for item in labels if item]
            if needle in normalized_labels:
                exact_matches.append((node, "exact_or_alias"))
                continue
            for normalized_label in normalized_labels:
                if self._is_unique_contains_match(needle, normalized_label):
                    contains_matches.append((node, "unique_contains"))
                    break

        if len(exact_matches) == 1:
            node, method = exact_matches[0]
            matched = dict(node)
            matched["_match_method"] = method
            return matched
        if len(contains_matches) == 1:
            node, method = contains_matches[0]
            matched = dict(node)
            matched["_match_method"] = method
            return matched
        return None

    @staticmethod
    def _normalize_entity_label(label: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z\u4e00-\u9fff]+", " ", label).lower()).strip()

    @staticmethod
    def _is_unique_contains_match(needle: str, candidate: str) -> bool:
        if not needle or not candidate or needle == candidate:
            return False
        if len(needle) < 4:
            return False
        needle_tokens = set(needle.split())
        candidate_tokens = set(candidate.split())
        return needle_tokens.issubset(candidate_tokens)

    @staticmethod
    def _normalize_entity_type(entity_type: str | None) -> str | None:
        normalized = re.sub(r"\s+", " ", re.sub(r"[^0-9a-zA-Z]+", " ", (entity_type or "").lower())).strip()
        return normalized or None

    @staticmethod
    def _normalize_relation_rewrite_key(text: str | None) -> str:
        return re.sub(r"\s+", " ", str(text or "")).strip().lower()

    def _rewrite_text_to_image_relation(
        self,
        *,
        source_node_title: str | None,
        search_query: str | None,
        image_node: ImageNode,
        resolved_asset: ResolvedImageAsset | None,
    ) -> tuple[str, dict[str, Any]]:
        raw_query = str(search_query or "").strip()
        if not raw_query:
            return "retrieved_image_for_visual_target", {
                "relation_rewrite_applied": False,
                "relation_rewrite_reason": "missing_search_query",
            }

        model_alias = (
            os.environ.get("IMAGE_RELATION_REWRITE_MODEL")
            or os.environ.get("IMAGE_GROUND_MODEL")
            or self.image_check_model_alias
            or os.environ.get("IMAGE_CHECK_MODEL")
        )
        if not model_alias:
            return raw_query, {
                "relation_rewrite_applied": False,
                "relation_rewrite_reason": "missing_model_alias",
            }

        user_parts: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Source text node title:\n{source_node_title or ''}\n\n"
                    f"Original search query:\n{raw_query}\n\n"
                    f"Image node title:\n{image_node.title or ''}\n\n"
                    f"Grounded image caption:\n{image_node.caption or ''}"
                ),
            }
        ]
        if resolved_asset is not None and resolved_asset.model_url:
            user_parts.append({"type": "image_url", "image_url": {"url": resolved_asset.model_url}})

        try:
            response = self.model_client.generate(
                ModelRequest(
                    model=model_alias,
                    response_format={"type": "json_object"},
                    messages=[
                        ModelMessage(role="system", content=PROMPT_TEXT_TO_IMAGE_RELATION_REWRITE),
                        ModelMessage(role="user", content=user_parts),
                    ],
                    metadata={
                        "trace_label": f"text_to_image_relation_rewrite:{source_node_title or ''}:{raw_query[:80]}"
                    },
                )
            )
            parsed = json.loads(response.content)
        except Exception as exc:
            return raw_query, {
                "relation_rewrite_applied": False,
                "relation_rewrite_reason": f"model_error:{exc.__class__.__name__}",
            }

        relation = str(parsed.get("relation") or "").strip()
        if not relation:
            return raw_query, {
                "relation_rewrite_applied": False,
                "relation_rewrite_reason": "empty_relation",
                "relation_rewrite_model": model_alias,
            }

        applied = self._normalize_relation_rewrite_key(relation) != self._normalize_relation_rewrite_key(raw_query)
        return relation, {
            "relation_rewrite_applied": applied,
            "relation_rewrite_reason": "model_output",
            "relation_rewrite_model": model_alias,
        }

    def _edge_from_plan_to_image(
        self,
        *,
        plan: VisualSearchPlan,
        query: SearchQuerySpec,
        image_node: ImageNode,
        search_evidence: Evidence,
        image_evidence: Evidence,
        search_result: ImageSearchResult,
        run_id: str | None,
        used_fallback: bool,
        relation: str | None = None,
        relation_metadata: dict[str, Any] | None = None,
    ) -> Edge | None:
        if not plan.source_node_id:
            return None
        edge_relation = relation or query.query or "retrieved_image_for_visual_target"
        metadata = {
            "query_id": query.query_id,
            "query": query.query,
            "used_fallback": used_fallback,
        }
        if relation_metadata:
            metadata.update({key: value for key, value in relation_metadata.items() if value is not None})
        return Edge.create(
            plan.source_node_id,
            image_node.node_id,
            edge_type=EdgeType.SEARCH_RETRIEVED,
            relation=edge_relation,
            src_node_type=NodeType.TEXT.value,
            dst_node_type=NodeType.IMAGE.value,
            evidence_refs=[
                EvidenceRef(evidence_id=plan.target.evidence_id),
                EvidenceRef(evidence_id=search_evidence.evidence_id),
                EvidenceRef(evidence_id=image_evidence.evidence_id),
            ],
            source=EdgeSource(
                source_type="image_search",
                url=search_result.source_page_url or search_result.image_url,
                run_id=run_id,
                builder=self.builder_name,
            ),
            extractor=self.builder_name,
            metadata=metadata,
            evidence_key=f"{query.query_id}:{image_node.node_id}",
        )

    def _persist_snapshot(self, snapshot: SearchSnapshot) -> None:
        if self.store is not None and self.config.persist_search_snapshots:
            self.store.upsert_search_snapshot(snapshot)

    def _persist_records(
        self,
        *,
        image_node: ImageNode,
        original_asset: Asset,
        thumb_asset: Asset | None,
        search_evidence: Evidence,
        image_evidence: Evidence,
        edge: Edge | None,
        grounded_edges: list[Edge] | None = None,
    ) -> None:
        if self.store is None:
            return
        self.store.upsert_node(image_node)
        self.store.upsert_asset(original_asset)
        if thumb_asset is not None:
            self.store.upsert_asset(thumb_asset)
        self.store.upsert_evidence(search_evidence)
        self.store.upsert_evidence(image_evidence)
        if edge is not None:
            self.store.upsert_edge(edge)
        for grounded_edge in grounded_edges or []:
            self.store.upsert_edge(grounded_edge)


def _smoke_test() -> None:
    import os
    import tempfile

    class MockImageSearchClient:
        def search_text(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
            del limit, kwargs
            return SearchResponse(query=query, engine="mock:text", results=[])

        def search_image(self, query: str, *, limit: int = 10, **kwargs: Any) -> SearchResponse:
            del limit, kwargs
            return SearchResponse(
                query=query,
                engine="mock:image",
                results=[
                    ImageSearchResult(
                        title="Kobe Bryant final game",
                        image_url="https://example.com/kobe-final-game.jpg",
                        source_page_url="https://example.com/kobe",
                        snippet="Kobe Bryant in final game uniform",
                        width=640,
                        height=480,
                    )
                ],
            )

    class MockWikiResolver:
        def search_candidates(
            self,
            label: str,
            *,
            entity_type: str | None = None,
            source_title: str | None = None,
            context: str | None = None,
            limit: int = 5,
        ) -> list[Any]:
            del entity_type, source_title, context, limit
            normalized = label.strip().lower()
            records = {
                "los angeles lakers": {
                    "title": "Los Angeles Lakers",
                    "url": "https://en.wikipedia.org/wiki/Los_Angeles_Lakers",
                    "canonical_id": "wikidata:Q121783",
                    "snippet": "American professional basketball team based in Los Angeles.",
                },
                "national basketball association": {
                    "title": "National Basketball Association",
                    "url": "https://en.wikipedia.org/wiki/National_Basketball_Association",
                    "canonical_id": "wikidata:Q155223",
                    "snippet": "Professional basketball league in North America.",
                },
            }
            record = records.get(normalized)
            if record is None:
                return []
            return [
                type(
                    "MockCandidate",
                    (),
                    {
                        "title": record["title"],
                        "url": record["url"],
                        "canonical_id": record["canonical_id"],
                        "snippet": record["snippet"],
                        "score": 5.0,
                        "to_dict": lambda self: {
                            "title": self.title,
                            "url": self.url,
                            "canonical_id": self.canonical_id,
                            "snippet": self.snippet,
                            "score": self.score,
                        },
                    },
                )()
            ]

    class MockModel:
        def generate(self, request: ModelRequest) -> ModelResponse:
            system = request.messages[0].content
            if "Wikipedia inline image is visually relevant" in system:
                return ModelResponse(
                    content=(
                        "<check>\n"
                        "decision: accept\n"
                        "confidence: 0.9\n"
                        "reason: clearly related to the page subject\n"
                        "visual_fact: Kobe Bryant is visible\n"
                        "</check>"
                    )
                )
            if "I’m determining whether a user recognizes a particular image" in system:
                return ModelResponse(
                    content=(
                        "<thinking>The page and caption suggest Kobe Bryant is the subject.</thinking>\n"
                        "<question>Who is shown courtside in this image?</question>\n"
                        "<answer>Kobe Bryant</answer>"
                    )
                )
            if "You are answering a question about an image content." in system:
                return ModelResponse(content="UNKNOWN")
            if "You are judging whether an answer correctly identifies the key information" in system:
                return ModelResponse(
                    content="<thinking>The answer does not identify the subject.</thinking>\n<answer>FALSE</answer>"
                )
            if "filtering grounded entities from a Wikipedia inline image" in system:
                return ModelResponse(
                    content="""<filter>
overall_decision: keep
reason: the image grounds named basketball entities with stable canonical referents
entity: Los Angeles Lakers | keep | named NBA team with a stable canonical identity
entity: National Basketball Association | keep | named sports league with a stable canonical identity
</filter>"""
                )
            if "checking whether a candidate image" in system:
                return ModelResponse(
                    content="""<check>
decision: accept
confidence: 0.9
reason: visible player in uniform
visual_fact: Kobe Bryant is visible
</check>"""
                )
            if "several image-search results converge" in system:
                return ModelResponse(
                    content="""<thinking>Both candidates show the same final-game scene.</thinking>
<answer>TRUE</answer>
<consistent_images>1, 2</consistent_images>
<reason>Both images depict the same main visual content.</reason>"""
                )
            if "selecting the single best recovered image" in system:
                return ModelResponse(
                    content="""<selection>
decision: select
candidate_index: 1
reason: candidate 1 is the closest content match and has the higher usable resolution
</selection>"""
                )
            if "selecting the best Wikipedia candidate" in system:
                return ModelResponse(
                    content="""<selection>
decision: select
candidate_index: 0
reason: exact local Wikipedia match
</selection>"""
                )
            if "rewriting an image-search query into a source-aware graph relation" in system:
                return ModelResponse(content=json.dumps({"relation": "his final game uniform photo"}))
            return ModelResponse(
                content="""<ground>
caption: Kobe Bryant in his final game
visual_fact: basketball uniform
entity: Los Angeles Lakers | jersey logo | visible team branding on the uniform
entity: National Basketball Association | league branding | NBA league branding is visible in the arena context
</ground>"""
            )

    class MockAnswerableInlineModel(MockModel):
        def generate(self, request: ModelRequest) -> ModelResponse:
            system = request.messages[0].content
            if "You are answering a question about an image content." in system:
                return ModelResponse(content="Kobe Bryant")
            if "You are judging whether an answer correctly identifies the key information" in system:
                return ModelResponse(
                    content="<thinking>The answer matches the reference answer.</thinking>\n<answer>TRUE</answer>"
                )
            return super().generate(request)

    class MockSingleEntityGroundModel(MockModel):
        def generate(self, request: ModelRequest) -> ModelResponse:
            system = request.messages[0].content
            if "Wikipedia inline image is visually relevant" in system:
                return super().generate(request)
            if "I’m determining whether a user recognizes a particular image" in system:
                return super().generate(request)
            if "You are answering a question about an image content." in system:
                return super().generate(request)
            if "You are judging whether an answer correctly identifies the key information" in system:
                return super().generate(request)
            if "checking whether a candidate image" in system:
                return super().generate(request)
            if "selecting the single best recovered image" in system:
                return super().generate(request)
            if "selecting the best Wikipedia candidate" in system:
                return super().generate(request)
            if "rewriting an image-search query into a source-aware graph relation" in system:
                return super().generate(request)
            return ModelResponse(
                content="""<ground>
caption: Kobe Bryant in his final game
visual_fact: basketball uniform
entity: Los Angeles Lakers | jersey logo | visible team branding on the uniform
</ground>"""
            )

    class MockNoEntityGroundModel(MockModel):
        def generate(self, request: ModelRequest) -> ModelResponse:
            system = request.messages[0].content
            if "Wikipedia inline image is visually relevant" in system:
                return super().generate(request)
            if "I’m determining whether a user recognizes a particular image" in system:
                return super().generate(request)
            if "You are answering a question about an image content." in system:
                return super().generate(request)
            if "You are judging whether an answer correctly identifies the key information" in system:
                return super().generate(request)
            if "checking whether a candidate image" in system:
                return super().generate(request)
            if "selecting the single best recovered image" in system:
                return super().generate(request)
            if "selecting the best Wikipedia candidate" in system:
                return super().generate(request)
            if "rewriting an image-search query into a source-aware graph relation" in system:
                return super().generate(request)
            return ModelResponse(
                content="""<ground>
caption: Kobe Bryant courtside photo
</ground>"""
            )

    class MockGenericCategoryInlineModel(MockModel):
        def generate(self, request: ModelRequest) -> ModelResponse:
            system = request.messages[0].content
            if "filtering grounded entities from a Wikipedia inline image" in system:
                return ModelResponse(
                    content="""<filter>
overall_decision: drop
reason: all grounded entities are generic atmospheric categories rather than unique canonical entities
entity: Cumulonimbus cloud | block | weather cloud type, not a unique canonical entity
entity: Overshooting top | block | cloud feature category, not a unique canonical entity
</filter>"""
                )
            if "Wikipedia inline image is visually relevant" in system:
                return super().generate(request)
            if "I’m determining whether a user recognizes a particular image" in system:
                return super().generate(request)
            if "You are answering a question about an image content." in system:
                return super().generate(request)
            if "You are judging whether an answer correctly identifies the key information" in system:
                return super().generate(request)
            if "checking whether a candidate image" in system:
                return super().generate(request)
            if "selecting the single best recovered image" in system:
                return super().generate(request)
            if "selecting the best Wikipedia candidate" in system:
                return super().generate(request)
            if "rewriting an image-search query into a source-aware graph relation" in system:
                return super().generate(request)
            return ModelResponse(
                content="""<ground>
caption: An anvil-topped thundercloud with a protruding dome above the top
entity: Cumulonimbus cloud | main storm cloud filling the frame | the image shows the classic anvil-topped thundercloud form
entity: Overshooting top | dome above the cloud top | a protruding dome rises above the anvil top
</ground>"""
            )

    old_check = os.environ.get("IMAGE_CHECK_MODEL")
    old_ground = os.environ.get("IMAGE_GROUND_MODEL")
    os.environ["IMAGE_CHECK_MODEL"] = "mock_image"
    os.environ["IMAGE_GROUND_MODEL"] = "mock_image"
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            store = JsonlGraphStore(tmpdir)
            text_node = TextNode.from_wiki_entity(
                "Q25369",
                "Kobe Bryant",
                aliases=["Kobe"],
                source_url="https://en.wikipedia.org/wiki/Kobe_Bryant",
            )
            lakers_node = TextNode.from_wiki_entity(
                "Q121783",
                "Los Angeles Lakers",
                aliases=["Lakers"],
                source_url="https://en.wikipedia.org/wiki/Los_Angeles_Lakers",
            )
            store.upsert_node(text_node)
            store.upsert_node(lakers_node)

            def make_seed_store(name: str) -> JsonlGraphStore:
                child_store = JsonlGraphStore(Path(tmpdir) / name)
                child_store.upsert_node(text_node)
                child_store.upsert_node(lakers_node)
                return child_store

            target = Evidence.create(
                EvidenceType.VISUAL_TARGET,
                content="Kobe Bryant final game uniform",
                node_ids=[text_node.node_id],
                metadata={"expected_visual": "Kobe Bryant in a Lakers uniform"},
            )
            query = SearchQuerySpec.create(
                "Kobe Bryant final game uniform photo",
                target.evidence_id,
                expected_visual="Kobe Bryant in a Lakers uniform",
            )
            plan = VisualSearchPlan.create(
                target,
                queries=[query],
                source_node_id=text_node.node_id,
                source_evidence_ids=["evidence_text"],
            )
            builder = ImageDiscoveryBuilder(
                store=store,
                search_client=MockImageSearchClient(),
                config=ImageDiscoveryConfig(
                    per_query_limit=1,
                    max_images_per_plan=1,
                    enable_retrieval_consistency_check=False,
                    precheck_image_urls=False,
                ),
                model_client=MockModel(),
                wiki_resolver=MockWikiResolver(),
            )
            result = builder.discover_for_plan(plan, run_id="run_smoke")
            assert len(result.accepted_images()) == 1
            image = result.primary_image()
            assert image is not None
            assert result.image_node is not None
            assert result.image_node.title == "Image: Kobe Bryant final game uniform photo"
            assert result.image_node.caption == "Kobe Bryant in his final game"
            assert result.edge is not None
            assert result.edge.relation == "his final game uniform photo"
            assert result.edge.metadata.get("query") == "Kobe Bryant final game uniform photo"
            assert result.edge.metadata.get("relation_rewrite_applied") is True
            assert result.grounded_edges
            assert len(result.grounded_edges) == 1
            assert len(result.queued_tasks) == 1
            assert result.metadata.get("visual_plan_post_grounding_filter", {}).get("self_qa_applied") is False
            assert result.metadata.get("visual_plan_post_grounding_filter", {}).get("kept_in_graph") is True
            assert result.image_node.metadata.get("image_grounding", {}).get("context") is not None
            assert store.stats()["nodes"] == 3

            answerable_store = make_seed_store("answerable_visual_plan")
            answerable_builder = ImageDiscoveryBuilder(
                store=answerable_store,
                search_client=MockImageSearchClient(),
                config=ImageDiscoveryConfig(
                    per_query_limit=1,
                    max_images_per_plan=1,
                    enable_retrieval_consistency_check=False,
                    precheck_image_urls=False,
                ),
                model_client=MockAnswerableInlineModel(),
                wiki_resolver=MockWikiResolver(),
            )
            answerable_visual = answerable_builder.discover_for_plan(plan, run_id="run_smoke_answerable_visual")
            assert answerable_visual.image_node is not None
            assert answerable_visual.primary_image() is not None
            assert len(answerable_visual.accepted_images()) == 1
            assert answerable_visual.metadata.get("visual_plan_post_grounding_filter", {}).get("self_qa_applied") is False
            assert answerable_visual.metadata.get("visual_plan_post_grounding_filter", {}).get("filter_reason") is None
            assert answerable_store.stats()["nodes"] == 3

            single_entity_store = make_seed_store("single_entity_visual_plan")
            single_entity_builder = ImageDiscoveryBuilder(
                store=single_entity_store,
                search_client=MockImageSearchClient(),
                config=ImageDiscoveryConfig(
                    per_query_limit=1,
                    max_images_per_plan=1,
                    enable_retrieval_consistency_check=False,
                    precheck_image_urls=False,
                ),
                model_client=MockSingleEntityGroundModel(),
                wiki_resolver=MockWikiResolver(),
            )
            single_entity_visual = single_entity_builder.discover_for_plan(plan, run_id="run_smoke_single_entity_visual")
            assert single_entity_visual.image_node is None
            assert single_entity_visual.primary_image() is None
            assert len(single_entity_visual.accepted_images()) == 0
            assert single_entity_visual.candidates[0].validation.reason == "expandable_entity_count_below_threshold"
            assert single_entity_visual.metadata.get("visual_plan_post_grounding_filter", {}).get("expandable_entity_count") == 1
            assert single_entity_visual.metadata.get("visual_plan_post_grounding_filter", {}).get("filter_reason") == "expandable_entity_count_below_threshold"
            assert single_entity_store.stats()["nodes"] == 2

            first_candidate = result.candidates[0]
            second_candidate = ImageSearchCandidate(
                candidate_id="candidate_same_scene",
                source_query=first_candidate.source_query,
                source_snapshot=first_candidate.source_snapshot,
                search_result=ImageSearchResult(
                    title="Kobe Bryant final game second photo",
                    image_url="https://example.com/kobe-final-game-second.jpg",
                    source_page_url="https://example.com/kobe",
                    snippet="Kobe Bryant in the same final game uniform",
                ),
                validation=ImageValidationResult(
                    status=ImageCandidateStatus.ACCEPTED,
                    confidence=0.8,
                ),
            )
            transient_candidates = [first_candidate, second_candidate]
            for index, candidate in enumerate(transient_candidates, start=1):
                key = f"smoke_transient_{index}"
                candidate.validation.metadata = {"transient_image_key": key}
                builder._transient_image_cache[key] = ResolvedImageAsset(
                    cache_key=key,
                    original_url=candidate.search_result.image_url,
                    resolved_url=candidate.search_result.image_url,
                    source_page_url=candidate.search_result.source_page_url,
                    model_url="data:image/jpeg;base64,AA==",
                    asset_uri="",
                    cache_path=None,
                    content_type="image/jpeg",
                )
            builder.config.enable_retrieval_consistency_check = True
            consistency = builder._apply_retrieval_consistency_check(
                plan=plan,
                candidates=transient_candidates,
            )
            assert consistency["decision"] == "accept"
            assert len([item for item in transient_candidates if item.validation.status == ImageCandidateStatus.ACCEPTED]) == 2
            builder._clear_transient_assets(transient_candidates)
            assert not builder._transient_image_cache

            recovered_urls = [
                "https://example.com/recovered-small.jpg",
                "https://example.com/recovered-large.jpg",
            ]
            recovery_search_result = ImageSearchResult(
                title="Recovered source page image",
                image_url="https://example.com/recovery-original.jpg",
                source_page_url="https://example.com/source-page",
                snippet="Kobe Bryant standing on the scorer's table",
            )

            def fake_recover_candidate_image_urls(search_result: ImageSearchResult) -> list[str]:
                assert search_result.source_page_url == recovery_search_result.source_page_url
                return list(recovered_urls)

            def fake_download_and_prepare_image_asset(
                image_url: str | None,
                *,
                source_page_url: str | None,
                strategy: str,
                cache_key: str,
                persist_asset: bool = True,
            ) -> tuple[ResolvedImageAsset | None, str | None]:
                del cache_key
                if image_url == recovery_search_result.image_url:
                    return None, "direct_unavailable"
                if image_url == recovered_urls[0]:
                    return (
                        ResolvedImageAsset(
                            cache_key="recovered_small",
                            original_url=image_url,
                            resolved_url=image_url,
                            source_page_url=source_page_url,
                            model_url="data:image/jpeg;base64,AA==",
                            asset_uri=image_url,
                            cache_path=None,
                            content_type="image/jpeg",
                            width=240,
                            height=300,
                            strategy=strategy,
                        ),
                        None,
                    )
                if image_url == recovered_urls[1]:
                    return (
                        ResolvedImageAsset(
                            cache_key="recovered_large",
                            original_url=image_url,
                            resolved_url=image_url,
                            source_page_url=source_page_url,
                            model_url="data:image/jpeg;base64,AQ==",
                            asset_uri=image_url,
                            cache_path=None,
                            content_type="image/jpeg",
                            width=600,
                            height=750,
                            strategy=strategy,
                        ),
                        None,
                    )
                return None, f"unexpected_url:{image_url}"

            builder._recover_candidate_image_urls = fake_recover_candidate_image_urls  # type: ignore[method-assign]
            builder._download_and_prepare_image_asset = fake_download_and_prepare_image_asset  # type: ignore[method-assign]
            selected_asset, selected_error = builder._resolve_image_asset(
                recovery_search_result,
                persist_asset=False,
                recovery_query="Kobe Bryant standing on the scorer's table",
            )
            assert selected_error is None
            assert selected_asset is not None
            assert selected_asset.resolved_url == recovered_urls[1]

            wiki_inline_target = Evidence.create(
                EvidenceType.VISUAL_TARGET,
                content="Kobe Bryant courtside photo",
                node_ids=[text_node.node_id],
                metadata={"expected_visual": "Kobe Bryant courtside photo"},
            )
            wiki_inline_query = SearchQuerySpec.create(
                "Kobe Bryant courtside photo",
                wiki_inline_target.evidence_id,
                expected_visual="Kobe Bryant courtside photo",
            )
            wiki_inline_plan = VisualSearchPlan.create(
                wiki_inline_target,
                queries=[wiki_inline_query],
                source_node_id=text_node.node_id,
                source_evidence_ids=["evidence_text"],
                planner="wikipedia_inline_image_planner",
                metadata={"plan_source": "wikipedia_inline_image"},
            )
            wiki_inline_search_result = ImageSearchResult(
                title="Kobe Bryant courtside photo",
                image_url="https://example.com/wiki-inline-kobe.jpg",
                source_page_url="https://en.wikipedia.org/wiki/Kobe_Bryant",
                snippet="Kobe Bryant courtside",
                source="wikipedia_inline",
            )
            inline_builder = ImageDiscoveryBuilder(
                store=store,
                search_client=MockImageSearchClient(),
                config=ImageDiscoveryConfig(
                    per_query_limit=1,
                    max_images_per_plan=1,
                    enable_retrieval_consistency_check=False,
                    precheck_image_urls=True,
                    upload_cached_images=False,
                    try_source_page_recovery=False,
                ),
                model_client=MockModel(),
                wiki_resolver=MockWikiResolver(),
            )
            persist_calls: list[bool] = []

            def fake_inline_download_and_prepare_image_asset(
                image_url: str | None,
                *,
                source_page_url: str | None,
                strategy: str,
                cache_key: str,
                persist_asset: bool = True,
            ) -> tuple[ResolvedImageAsset | None, str | None]:
                persist_calls.append(persist_asset)
                if not image_url:
                    return None, "missing_image_url"
                asset_uri = f"/tmp/{cache_key}.jpg" if persist_asset else image_url
                cache_path = f"/tmp/{cache_key}.jpg" if persist_asset else None
                return (
                    ResolvedImageAsset(
                        cache_key=cache_key,
                        original_url=image_url,
                        resolved_url=image_url,
                        source_page_url=source_page_url,
                        model_url="data:image/jpeg;base64,AA==",
                        asset_uri=asset_uri,
                        cache_path=cache_path,
                        content_type="image/jpeg",
                        width=640,
                        height=480,
                        strategy=strategy,
                    ),
                    None,
                )

            inline_builder._download_and_prepare_image_asset = fake_inline_download_and_prepare_image_asset  # type: ignore[method-assign]

            provisional_inline = inline_builder.discover_for_wiki_inline_image(
                wiki_inline_plan,
                search_result=wiki_inline_search_result,
                run_id="run_smoke",
                persist=False,
            )
            assert provisional_inline.image_node is not None
            assert persist_calls and set(persist_calls) == {False}
            assert not inline_builder._resolved_image_cache
            assert not inline_builder._transient_image_cache
            assert store.stats()["nodes"] == 3

            previous_call_count = len(persist_calls)
            kept_inline = inline_builder.discover_for_wiki_inline_image(
                wiki_inline_plan,
                search_result=wiki_inline_search_result,
                run_id="run_smoke",
                persist=True,
            )
            assert kept_inline.image_node is not None
            assert kept_inline.image_node.source is not None
            assert kept_inline.image_node.source.source_type == "wikipedia_inline_image"
            kept_call_slice = persist_calls[previous_call_count:]
            assert False in kept_call_slice
            assert True in kept_call_slice
            assert inline_builder._resolved_image_cache
            assert not inline_builder._transient_image_cache
            assert store.stats()["nodes"] == 4

            answerable_inline_builder = ImageDiscoveryBuilder(
                store=store,
                search_client=MockImageSearchClient(),
                config=ImageDiscoveryConfig(
                    per_query_limit=1,
                    max_images_per_plan=1,
                    enable_retrieval_consistency_check=False,
                    precheck_image_urls=True,
                    upload_cached_images=False,
                    try_source_page_recovery=False,
                ),
                model_client=MockAnswerableInlineModel(),
                wiki_resolver=MockWikiResolver(),
            )
            answerable_persist_calls: list[bool] = []

            def fake_answerable_inline_download_and_prepare_image_asset(
                image_url: str | None,
                *,
                source_page_url: str | None,
                strategy: str,
                cache_key: str,
                persist_asset: bool = True,
            ) -> tuple[ResolvedImageAsset | None, str | None]:
                answerable_persist_calls.append(persist_asset)
                if not image_url:
                    return None, "missing_image_url"
                asset_uri = f"/tmp/{cache_key}.jpg" if persist_asset else image_url
                cache_path = f"/tmp/{cache_key}.jpg" if persist_asset else None
                return (
                    ResolvedImageAsset(
                        cache_key=cache_key,
                        original_url=image_url,
                        resolved_url=image_url,
                        source_page_url=source_page_url,
                        model_url="data:image/jpeg;base64,AA==",
                        asset_uri=asset_uri,
                        cache_path=cache_path,
                        content_type="image/jpeg",
                        width=640,
                        height=480,
                        strategy=strategy,
                    ),
                    None,
                )

            answerable_inline_builder._download_and_prepare_image_asset = fake_answerable_inline_download_and_prepare_image_asset  # type: ignore[method-assign]
            before_answerable_node_count = store.stats()["nodes"]
            answerable_inline = answerable_inline_builder.discover_for_wiki_inline_image(
                wiki_inline_plan,
                search_result=wiki_inline_search_result,
                run_id="run_smoke",
                persist=True,
            )
            assert answerable_inline.image_node is None
            assert answerable_inline.metadata.get("query_count") == 1
            assert answerable_inline.metadata.get("candidate_decisions", [])[0].get("reason") == "model_answered_generated_question"
            assert answerable_persist_calls and set(answerable_persist_calls) == {False}
            assert not answerable_inline_builder._resolved_image_cache
            assert not answerable_inline_builder._transient_image_cache
            assert store.stats()["nodes"] == before_answerable_node_count

            drop_inline_builder = ImageDiscoveryBuilder(
                store=store,
                search_client=MockImageSearchClient(),
                config=ImageDiscoveryConfig(
                    per_query_limit=1,
                    max_images_per_plan=1,
                    enable_retrieval_consistency_check=False,
                    precheck_image_urls=True,
                    upload_cached_images=False,
                    try_source_page_recovery=False,
                ),
                model_client=MockNoEntityGroundModel(),
                wiki_resolver=MockWikiResolver(),
            )
            drop_persist_calls: list[bool] = []

            def fake_drop_inline_download_and_prepare_image_asset(
                image_url: str | None,
                *,
                source_page_url: str | None,
                strategy: str,
                cache_key: str,
                persist_asset: bool = True,
            ) -> tuple[ResolvedImageAsset | None, str | None]:
                drop_persist_calls.append(persist_asset)
                if not image_url:
                    return None, "missing_image_url"
                asset_uri = f"/tmp/{cache_key}.jpg" if persist_asset else image_url
                cache_path = f"/tmp/{cache_key}.jpg" if persist_asset else None
                return (
                    ResolvedImageAsset(
                        cache_key=cache_key,
                        original_url=image_url,
                        resolved_url=image_url,
                        source_page_url=source_page_url,
                        model_url="data:image/jpeg;base64,AA==",
                        asset_uri=asset_uri,
                        cache_path=cache_path,
                        content_type="image/jpeg",
                        width=640,
                        height=480,
                        strategy=strategy,
                    ),
                    None,
                )

            drop_inline_builder._download_and_prepare_image_asset = fake_drop_inline_download_and_prepare_image_asset  # type: ignore[method-assign]
            before_drop_node_count = store.stats()["nodes"]
            dropped_inline = drop_inline_builder.discover_for_wiki_inline_image(
                wiki_inline_plan,
                search_result=wiki_inline_search_result,
                run_id="run_smoke",
                persist=True,
            )
            assert dropped_inline.image_node is None
            assert dropped_inline.metadata.get("wiki_inline_keep_in_graph") is False
            assert dropped_inline.metadata.get("wiki_inline_skip_reason") == "no_expandable_grounded_entities"
            assert drop_persist_calls and set(drop_persist_calls) == {False}
            assert not drop_inline_builder._resolved_image_cache
            assert not drop_inline_builder._transient_image_cache
            assert store.stats()["nodes"] == before_drop_node_count

            cloud_node = TextNode.from_wiki_entity(
                "Q183165",
                "Cumulonimbus cloud",
                source_url="https://en.wikipedia.org/wiki/Cumulonimbus_cloud",
            )
            cloud_store = JsonlGraphStore(Path(tmpdir) / "generic_inline")
            cloud_store.upsert_node(cloud_node)
            generic_inline_target = Evidence.create(
                EvidenceType.VISUAL_TARGET,
                content="Cumulonimbus cloud formation photo",
                node_ids=[cloud_node.node_id],
                metadata={"expected_visual": "An anvil-topped cumulonimbus cloud"},
            )
            generic_inline_query = SearchQuerySpec.create(
                "Cumulonimbus cloud formation photo",
                generic_inline_target.evidence_id,
                expected_visual="An anvil-topped cumulonimbus cloud",
            )
            generic_inline_plan = VisualSearchPlan.create(
                generic_inline_target,
                queries=[generic_inline_query],
                source_node_id=cloud_node.node_id,
                source_evidence_ids=["evidence_text"],
                planner="wikipedia_inline_image_planner",
                metadata={"plan_source": "wikipedia_inline_image"},
            )
            generic_inline_search_result = ImageSearchResult(
                title="Cumulonimbus cloud illustration",
                image_url="https://example.com/wiki-inline-cloud.jpg",
                source_page_url="https://en.wikipedia.org/wiki/Cumulonimbus_cloud",
                snippet="Anvil-topped cumulonimbus cloud",
                source="wikipedia_inline",
            )
            generic_inline_builder = ImageDiscoveryBuilder(
                store=cloud_store,
                search_client=MockImageSearchClient(),
                config=ImageDiscoveryConfig(
                    per_query_limit=1,
                    max_images_per_plan=1,
                    enable_retrieval_consistency_check=False,
                    precheck_image_urls=True,
                    upload_cached_images=False,
                    try_source_page_recovery=False,
                ),
                model_client=MockGenericCategoryInlineModel(),
                wiki_resolver=MockWikiResolver(),
            )
            generic_persist_calls: list[bool] = []

            def fake_generic_inline_download_and_prepare_image_asset(
                image_url: str | None,
                *,
                source_page_url: str | None,
                strategy: str,
                cache_key: str,
                persist_asset: bool = True,
            ) -> tuple[ResolvedImageAsset | None, str | None]:
                generic_persist_calls.append(persist_asset)
                if not image_url:
                    return None, "missing_image_url"
                asset_uri = f"/tmp/{cache_key}.jpg" if persist_asset else image_url
                cache_path = f"/tmp/{cache_key}.jpg" if persist_asset else None
                return (
                    ResolvedImageAsset(
                        cache_key=cache_key,
                        original_url=image_url,
                        resolved_url=image_url,
                        source_page_url=source_page_url,
                        model_url="data:image/jpeg;base64,AA==",
                        asset_uri=asset_uri,
                        cache_path=cache_path,
                        content_type="image/jpeg",
                        width=640,
                        height=480,
                        strategy=strategy,
                    ),
                    None,
                )

            generic_inline_builder._download_and_prepare_image_asset = fake_generic_inline_download_and_prepare_image_asset  # type: ignore[method-assign]
            before_generic_node_count = cloud_store.stats()["nodes"]
            generic_inline = generic_inline_builder.discover_for_wiki_inline_image(
                generic_inline_plan,
                search_result=generic_inline_search_result,
                run_id="run_smoke",
                persist=True,
            )
            assert generic_inline.image_node is None
            assert generic_inline.metadata.get("wiki_inline_keep_in_graph") is False
            assert generic_inline.metadata.get("wiki_inline_skip_reason") == "no_unique_canonical_grounded_entities"
            assert generic_inline.metadata.get("wiki_inline_entity_uniqueness_filter", {}).get("applied") is True
            assert generic_inline.metadata.get("wiki_inline_entity_uniqueness_filter", {}).get("kept_entity_count") == 0
            assert generic_persist_calls and set(generic_persist_calls) == {False}
            assert not generic_inline_builder._resolved_image_cache
            assert not generic_inline_builder._transient_image_cache
            assert cloud_store.stats()["nodes"] == before_generic_node_count
    finally:
        if old_check is None:
            os.environ.pop("IMAGE_CHECK_MODEL", None)
        else:
            os.environ["IMAGE_CHECK_MODEL"] = old_check
        if old_ground is None:
            os.environ.pop("IMAGE_GROUND_MODEL", None)
        else:
            os.environ["IMAGE_GROUND_MODEL"] = old_ground
    print("image_discovery smoke test passed")


if __name__ == "__main__":
    _smoke_test()
