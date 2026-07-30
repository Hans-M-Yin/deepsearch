#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate the primary-image search-query refinement prompt on curated cases.

This is an online debug/evaluation script: each selected case sends one request
through the existing ``LLM_WORKER``. It does not override temperature, token
limits, or other sampling parameters; those come from ``synthesis/models.json``.

Examples:
    python debug/eval_image_query_refinement.py \
      --model multimodal_process --list-cases

    python debug/eval_image_query_refinement.py \
      --model multimodal_process --case broad_ceremony_speech

    python debug/eval_image_query_refinement.py \
      --model multimodal_process --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from synthesis.evidence import Evidence, EvidenceType, SearchEngine, SearchSnapshot
from synthesis.image_discovery import (
    ImageCandidateStatus,
    ImageDiscoveryBuilder,
    ImageDiscoveryConfig,
    ImageSearchCandidate,
    ImageValidationResult,
    ResolvedImageAsset,
)
from synthesis.model_worker import LLM_WORKER
from synthesis.search_client import ImageSearchResult
from synthesis.visual_planner import SearchQuerySpec, VisualSearchPlan


DEFAULT_MODEL = os.environ.get("IMAGE_QUERY_REFINEMENT_MODEL", "multimodal_process")


@dataclass(frozen=True)
class RefinementCase:
    case_id: str
    description: str
    original_query: str
    image_url: str
    image_title: str
    image_snippet: str
    visual_target: str
    expected_decisions: tuple[str, ...]
    expected_constraint_terms: tuple[str, ...] = ()
    forbidden_terms: tuple[str, ...] = ()
    note: str = ""


CASES: tuple[RefinementCase, ...] = (
    RefinementCase(
        case_id="broad_ceremony_speech",
        description="A ceremony-level query whose selected image shows the inductee speaking at a podium.",
        original_query=(
            "Rodney Harrison wearing his red jacket during his induction ceremony for the "
            "New England Patriots Hall of Fame in July 2019"
        ),
        image_url=(
            "https://bostonglobe-prod.cdn.arcpublishing.com/resizer/v2/"
            "WDP6VDVSKMI6TI4SAUR672ILPM.jpg?auth=9ad2d4e9a43d2782e6619d7d1090a07317d5379806614fe70f355bf0a4b949ae&width=1440"
        ),
        image_title="Rodney Harrison at his Patriots Hall of Fame induction ceremony",
        image_snippet="Rodney Harrison, in a red Hall of Fame jacket, speaks from the ceremony podium.",
        visual_target="Rodney Harrison during his July 2019 Patriots Hall of Fame induction ceremony",
        expected_decisions=("refine",),
        expected_constraint_terms=("speak", "podium", "lectern", "address"),
        forbidden_terms=("left side", "right side", "background", "foreground"),
        note="A good result adds one short action such as 'speaking at the podium'.",
    ),
    RefinementCase(
        case_id="already_precise_event",
        description="An event query already locked to a short, iconic action.",
        original_query=(
            "FIFA President Sepp Blatter being showered with fake money by comedian Simon "
            "Brodkin during a press conference in Zurich on July 20, 2015"
        ),
        image_url=(
            "https://ca-times.brightspotcdn.com/dims4/default/d909a3d/2147483647/strip/true/"
            "crop/2048x1365+0+0/resize/1200x800!/quality/75/?url=https%3A%2F%2Fcalifornia-times-"
            "brightspot.s3.amazonaws.com%2F6c%2Fbe%2F32c9546fadaa7bfaee136b800a34%2Fla-sp-sn-fifa-sepp-blatter-showered-with-money-001"
        ),
        image_title="Sepp Blatter showered with fake money",
        image_snippet="Simon Brodkin throws fake banknotes over Sepp Blatter during the press conference.",
        visual_target="The fake-money protest against Sepp Blatter at the July 20, 2015 press conference",
        expected_decisions=("keep",),
        forbidden_terms=("left side", "right side", "background", "foreground"),
        note="The original query already supplies people, date, event, and exact action.",
    ),
    RefinementCase(
        case_id="fixed_visual_work",
        description="A fixed painting that should never be expanded into a caption.",
        original_query=(
            'The 1843 double portrait painting "The Two Sisters" by Theodore Chasseriau, '
            "showing his sisters Adele and Aline"
        ),
        image_url=(
            "https://upload.wikimedia.org/wikipedia/commons/4/40/"
            "Th%C3%A9odore_Chass%C3%A9riau_-_Mesdemoiselles_Chass%C3%A9riau_%28Louvre_RF_2214%29_0000787160_OG.JPG"
        ),
        image_title="The Two Sisters by Theodore Chasseriau",
        image_snippet="The 1843 double portrait of Adele and Aline Chasseriau.",
        visual_target="The Two Sisters painting",
        expected_decisions=("keep",),
        forbidden_terms=("left side", "right side", "background", "foreground", "wearing matching"),
        note="The work is already a canonical fixed visual object.",
    ),
    RefinementCase(
        case_id="landmark_incidental_detail",
        description="A semi-unique landmark whose visitors, weather, and composition must not be added.",
        original_query="The Vasco da Gama Pillar monument on the coast of Malindi, Kenya",
        image_url="https://upload.wikimedia.org/wikipedia/commons/9/9e/Pillar_of_Vasco_da_Gama.jpg",
        image_title="Vasco da Gama Pillar in Malindi",
        image_snippet="The coastal monument overlooking the Indian Ocean.",
        visual_target="The Vasco da Gama Pillar monument",
        expected_decisions=("keep",),
        forbidden_terms=("person", "visitor", "left side", "right side", "foreground", "background"),
        note="The query remains semi-unique; refinement should not overfit this photograph.",
    ),
    RefinementCase(
        case_id="extended_parade_moment",
        description="An extended procession where the primary image may support one short event-phase constraint.",
        original_query=(
            "General de Gaulle and members of the French Forces of the Interior marching down "
            "the Champs-Elysees after the liberation of Paris on August 26, 1944"
        ),
        image_url=(
            "https://images.squarespace-cdn.com/content/v1/63fe4d3769a5dc31520d2d6f/"
            "1724357549655-16HCLNW2PZQ2F85M7E24/de+Gaulle+liberation+of+paris+1944.jpg"
        ),
        image_title="De Gaulle during the Liberation of Paris procession",
        image_snippet="De Gaulle walks at the head of the procession through a dense crowd on the Champs-Elysees.",
        visual_target="De Gaulle and FFI members in the August 26, 1944 Liberation of Paris procession",
        expected_decisions=("refine", "keep"),
        expected_constraint_terms=("head", "lead", "walk", "crowd", "procession"),
        forbidden_terms=("left side", "right side", "upper", "foreground", "background"),
        note="Refine is preferred only if the model can add one stable procession phase without captioning.",
    ),
)


class _UnusedSearchClient:
    pass


def _build_inputs(case: RefinementCase, model_alias: str):
    target = Evidence.create(
        EvidenceType.VISUAL_TARGET,
        content=case.visual_target,
        node_ids=["debug_text_node"],
    )
    query = SearchQuerySpec.create(case.original_query, target.evidence_id)
    plan = VisualSearchPlan.create(
        target,
        queries=[query],
        source_node_id="debug_text_node",
        source_evidence_ids=["debug_evidence"],
    )
    search_result = ImageSearchResult(
        title=case.image_title,
        image_url=case.image_url,
        source_page_url=case.image_url,
        snippet=case.image_snippet,
        rank=1,
    )
    snapshot = SearchSnapshot.create(
        SearchEngine.OTHER,
        query=case.original_query,
        request={"query": case.original_query},
        result_count=1,
    )
    candidate = ImageSearchCandidate(
        candidate_id=f"debug_{case.case_id}",
        source_query=query,
        source_snapshot=snapshot,
        search_result=search_result,
        validation=ImageValidationResult(
            status=ImageCandidateStatus.ACCEPTED,
            confidence=1.0,
        ),
        is_primary=True,
    )
    asset = ResolvedImageAsset(
        cache_key=f"debug_{case.case_id}",
        original_url=case.image_url,
        resolved_url=case.image_url,
        source_page_url=case.image_url,
        model_url=case.image_url,
        asset_uri=case.image_url,
        cache_path=None,
        content_type="image/jpeg",
    )
    builder = ImageDiscoveryBuilder(
        search_client=_UnusedSearchClient(),
        config=ImageDiscoveryConfig(
            precheck_image_urls=False,
            primary_query_refinement_model=model_alias,
        ),
        model_client=LLM_WORKER,
    )
    return builder, plan, candidate, asset


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    normalized = text.casefold()
    return any(term.casefold() in normalized for term in terms)


def evaluate_case(case: RefinementCase, *, model_alias: str) -> dict[str, Any]:
    builder, plan, candidate, asset = _build_inputs(case, model_alias)
    result = builder._refine_primary_search_query(
        plan=plan,
        candidate=candidate,
        resolved_asset=asset,
        run_id=f"debug_query_refinement:{case.case_id}",
    )
    decision = str(result.get("decision") or "")
    effective_query = str(result.get("refined_query") or "")
    proposed_query = str(result.get("proposed_refined_query") or effective_query)
    checks = {
        "decision_expected": decision in case.expected_decisions,
        "programmatic_validation_passed": not bool(result.get("validation_errors")),
        "no_forbidden_terms": not _contains_any(proposed_query, case.forbidden_terms),
        "constraint_term_present": (
            True
            if decision != "refine" or not case.expected_constraint_terms
            else _contains_any(proposed_query, case.expected_constraint_terms)
        ),
        "keep_preserves_original": decision != "keep" or effective_query == case.original_query,
    }
    return {
        "case": asdict(case),
        "result": result,
        "checks": checks,
        "passed": all(checks.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Registered multimodal LLM_WORKER model alias.")
    parser.add_argument("--case", action="append", default=[], help="Case ID to run; may be repeated.")
    parser.add_argument("--all", action="store_true", help="Run every built-in case.")
    parser.add_argument("--list-cases", action="store_true", help="List case IDs and exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    by_id = {case.case_id: case for case in CASES}
    if args.list_cases:
        for case in CASES:
            print(f"{case.case_id}\t{case.description}")
        return 0
    selected_ids = list(by_id) if args.all or not args.case else args.case
    unknown = sorted(set(selected_ids) - set(by_id))
    if unknown:
        raise SystemExit(f"error: unknown case IDs: {unknown}")

    outputs = []
    for index, case_id in enumerate(selected_ids, start=1):
        case = by_id[case_id]
        evaluated = evaluate_case(case, model_alias=args.model)
        outputs.append(evaluated)
        result = evaluated["result"]
        proposed_query = str(
            result.get("proposed_refined_query")
            or result.get("refined_query")
            or case.original_query
        ).strip()
        reason = str(result.get("reason") or "<missing>").strip()
        print(f"[{index}/{len(selected_ids)}] {case_id}")
        print(f"改动前：{case.original_query}")
        print(f"改动后：{proposed_query}")
        print(f"模型理由：{reason}")
        if index < len(selected_ids):
            print()

    failed = sum(not bool(item["passed"]) for item in outputs)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
