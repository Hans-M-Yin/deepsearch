from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from synthesis.edges import Edge, EdgeType, EvidenceRef
from synthesis.evidence import Evidence, EvidenceType
from synthesis.nodes import ImageNode, TextNode
from synthesis.store import JsonlGraphStore
from synthesis.vqa.graph_view import GraphView
from synthesis.vqa.repository_verifier import (
    OfflineGraphRepositoryVerifier,
    RepositoryAssembler,
    RepositoryVerificationConfig,
    build_question_only_shortcut_request,
    build_repository_answer_judge_request,
    build_repository_solver_request,
)


class FakeTextResponse:
    def __init__(self, content: str):
        self.content = content


class FakeJsonClient:
    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("No fake response queued for generate")
        response = self._responses.pop(0)
        if isinstance(response, str):
            if response.lstrip().startswith("{"):
                try:
                    parsed = json.loads(response)
                except json.JSONDecodeError:
                    return FakeTextResponse(response)
                self._responses.insert(0, parsed)
                self.requests.pop()
                return self.generate_json(request)
            return FakeTextResponse(response)
        return FakeTextResponse(json.dumps(response, ensure_ascii=False))

    def generate_json(self, request):
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("No fake response queued for generate_json")
        return self._responses.pop(0)


class FixtureData:
    def __init__(self, *, graph_dir: Path, vqa_dir: Path, question_record: dict, sample_record: dict):
        self.graph_dir = graph_dir
        self.vqa_dir = vqa_dir
        self.question_record = question_record
        self.sample_record = sample_record



def _write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")



def _build_fixture(tmp_path: Path) -> FixtureData:
    graph_dir = tmp_path / "graph"
    vqa_dir = graph_dir / "vqa" / "case_a"
    vqa_dir.mkdir(parents=True, exist_ok=True)
    store = JsonlGraphStore(graph_dir)

    source_node = TextNode.from_webpage("https://example.org/source", title="Source Page")
    mid_node = TextNode.from_webpage("https://example.org/mid", title="Mid Topic")
    target_node = TextNode.from_webpage("https://example.org/target", title="Target Entity")
    sibling_node = TextNode.from_webpage("https://example.org/sibling", title="Sibling Topic")
    extra_source = TextNode.from_webpage("https://example.org/extra-source", title="Extra Source")
    extra_target = TextNode.from_webpage("https://example.org/extra-target", title="Extra Target")

    relevant_image = ImageNode.from_url(
        "https://images.example.org/relevant.jpg",
        title="Relevant poster",
        caption="A commemorative poster with the distinctive emblem of Target Entity.",
        metadata={"source_text_node_id": mid_node.node_id, "image_origin": "visual_plan"},
    )
    sibling_image = ImageNode.from_url(
        "https://images.example.org/sibling.jpg",
        title="Sibling poster",
        caption="A sibling poster for an unrelated topic.",
        metadata={"source_text_node_id": mid_node.node_id, "image_origin": "visual_plan"},
    )
    random_image = ImageNode.from_url(
        "https://images.example.org/random.jpg",
        title="Random object",
        caption="A random object that is unrelated to the target entity.",
        metadata={"source_text_node_id": extra_source.node_id, "image_origin": "visual_plan"},
    )

    for node in [
        source_node,
        mid_node,
        target_node,
        sibling_node,
        extra_source,
        extra_target,
        relevant_image,
        sibling_image,
        random_image,
    ]:
        store.upsert_node(node)

    source_quote = "The source page says that Mid Topic is the official codename used for the project."
    sibling_quote = "The same source page also mentions the unrelated Sibling Topic in passing."
    image_ground_quote = "The emblem on the commemorative poster belongs to Target Entity."
    extra_quote = "A different page describes the unrelated Extra Target in detail."

    ev_source_quote = Evidence.create(EvidenceType.WEB_TEXT, content=source_quote, node_ids=[source_node.node_id])
    ev_sibling_quote = Evidence.create(EvidenceType.WEB_TEXT, content=sibling_quote, node_ids=[source_node.node_id])
    ev_visual_target = Evidence.create(
        EvidenceType.VISUAL_TARGET,
        content="A commemorative poster showing the emblem of Target Entity",
        node_ids=[relevant_image.node_id],
    )
    ev_search_result = Evidence.create(
        EvidenceType.SEARCH_RESULT,
        content="Poster associated with the target entity and its emblem",
        node_ids=[relevant_image.node_id],
    )
    ev_image = Evidence.create(
        EvidenceType.IMAGE,
        content="Poster with a distinctive emblem",
        node_ids=[relevant_image.node_id],
        url=relevant_image.image_url,
    )
    ev_image_ground = Evidence.create(
        EvidenceType.LLM_OUTPUT,
        content=image_ground_quote,
        node_ids=[relevant_image.node_id, target_node.node_id],
    )
    ev_extra = Evidence.create(EvidenceType.WEB_TEXT, content=extra_quote, node_ids=[extra_source.node_id])

    for evidence in [
        ev_source_quote,
        ev_sibling_quote,
        ev_visual_target,
        ev_search_result,
        ev_image,
        ev_image_ground,
        ev_extra,
    ]:
        store.upsert_evidence(evidence)

    edge_source_mid = Edge.create(
        source_node.node_id,
        mid_node.node_id,
        edge_type=EdgeType.WIKI_LINK,
        relation="official codename",
        src_node_type="text",
        dst_node_type="text",
        evidence_refs=[EvidenceRef(evidence_id=ev_source_quote.evidence_id, quote=source_quote)],
        evidence_key="source_mid",
    )
    edge_source_sibling = Edge.create(
        source_node.node_id,
        sibling_node.node_id,
        edge_type=EdgeType.WIKI_LINK,
        relation="also mentions",
        src_node_type="text",
        dst_node_type="text",
        evidence_refs=[EvidenceRef(evidence_id=ev_sibling_quote.evidence_id, quote=sibling_quote)],
        evidence_key="source_sibling",
    )
    edge_mid_image = Edge.create(
        mid_node.node_id,
        relevant_image.node_id,
        edge_type=EdgeType.SEARCH_RETRIEVED,
        relation="target emblem poster",
        src_node_type="text",
        dst_node_type="image",
        evidence_refs=[
            EvidenceRef(evidence_id=ev_visual_target.evidence_id),
            EvidenceRef(evidence_id=ev_search_result.evidence_id),
            EvidenceRef(evidence_id=ev_image.evidence_id),
        ],
        evidence_key="mid_image",
    )
    edge_image_target = Edge.create(
        relevant_image.node_id,
        target_node.node_id,
        edge_type=EdgeType.IMAGE_DEPICTS,
        relation="depicts the emblem of",
        src_node_type="image",
        dst_node_type="text",
        evidence_refs=[EvidenceRef(evidence_id=ev_image_ground.evidence_id, quote=image_ground_quote)],
        evidence_key="image_target",
    )
    edge_extra = Edge.create(
        extra_source.node_id,
        extra_target.node_id,
        edge_type=EdgeType.WIKI_LINK,
        relation="describes",
        src_node_type="text",
        dst_node_type="text",
        evidence_refs=[EvidenceRef(evidence_id=ev_extra.evidence_id, quote=extra_quote)],
        evidence_key="extra_edge",
    )

    for edge in [edge_source_mid, edge_source_sibling, edge_mid_image, edge_image_target, edge_extra]:
        store.upsert_edge(edge)
    store.flush()

    sample_record = {
        "sample_id": "sample_path_1",
        "status": "verified",
        "path": {
            "path_id": "path_1",
            "node_ids": [source_node.node_id, mid_node.node_id, relevant_image.node_id, target_node.node_id],
            "edge_ids": [edge_source_mid.edge_id, edge_mid_image.edge_id, edge_image_target.edge_id],
            "node_types": ["text", "text", "image", "text"],
            "edge_types": ["wiki_link", "search_retrieved", "image_depicts"],
            "relations": ["official codename", "target emblem poster", "depicts the emblem of"],
        },
        "hop_chain": [
            {"hop_index": 0, "edge_id": edge_source_mid.edge_id, "src_node_id": source_node.node_id, "dst_node_id": mid_node.node_id},
            {"hop_index": 1, "edge_id": edge_mid_image.edge_id, "src_node_id": mid_node.node_id, "dst_node_id": relevant_image.node_id},
            {"hop_index": 2, "edge_id": edge_image_target.edge_id, "src_node_id": relevant_image.node_id, "dst_node_id": target_node.node_id},
        ],
        "entry_hop": {
            "hop_index": 0,
            "source": "-",
            "target": "Source Page",
            "statement": "The source page clue identifies Source Page.",
            "supporting_facts": ["Opening clue fact from the source page."],
            "why_relevant": "The opening clue identifies the source without naming it.",
            "entry_kind": "text",
        },
        "target_ask": {
            "ask_target": "Which entity is identified by the emblem?",
            "answer": "Target Entity",
            "supporting_facts": ["The emblem identifies Target Entity."],
            "reasoning": "The emblem uniquely maps to Target Entity.",
            "support": "The target answer is evidence-supported.",
        },
    }
    question_record = {
        "question_id": "q_000001",
        "sample_id": "sample_path_1",
        "path_id": "path_1",
        "status": "verified",
        "question": "Which entity is identified by the emblem shown in the commemorative poster connected to the official codename mentioned on the source page?",
        "answer": "Target Entity",
    }

    _write_jsonl(vqa_dir / "questions.jsonl", [question_record])
    _write_jsonl(vqa_dir / "samples.jsonl", [sample_record])
    return FixtureData(graph_dir=graph_dir, vqa_dir=vqa_dir, question_record=question_record, sample_record=sample_record)



def _build_assembler(graph_dir: Path, *, random_seed: int = 0) -> RepositoryAssembler:
    graph = GraphView(JsonlGraphStore(graph_dir))
    return RepositoryAssembler(graph=graph, config=RepositoryVerificationConfig(random_seed=random_seed))



def _pick_labels(bundle):
    relevant_doc = next(item.label for item in bundle.items if item.is_relevant and item.item_type == "doc")
    relevant_image = next(item.label for item in bundle.items if item.is_relevant and item.item_type == "image")
    distractor = next(item.label for item in bundle.items if not item.is_relevant)
    return relevant_doc, relevant_image, distractor


class RepositoryVerifierTests(unittest.TestCase):
    def _fixture(self) -> FixtureData:
        tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(tempdir.cleanup)
        return _build_fixture(Path(tempdir.name))

    def test_repository_assembly_mixes_relevant_and_distractors(self):
        fixture = self._fixture()
        assembler = _build_assembler(fixture.graph_dir, random_seed=7)

        bundle = assembler.build_bundle(
            question_record=fixture.question_record,
            sample_record=fixture.sample_record,
        )

        self.assertTrue(bundle.items)
        self.assertTrue(any(item.is_relevant and item.item_type == "doc" for item in bundle.items))
        self.assertTrue(any(item.is_relevant and item.item_type == "image" for item in bundle.items))
        self.assertTrue(any((not item.is_relevant) and item.item_type == "doc" for item in bundle.items))
        self.assertTrue(any((not item.is_relevant) and item.item_type == "image" for item in bundle.items))
        self.assertTrue(any(item.selection_reason == "sibling_image_distractor" for item in bundle.items if not item.is_relevant))
        self.assertTrue(any(item.selection_reason == "sibling_distractor_edge_quote" for item in bundle.items if not item.is_relevant and item.item_type == "doc"))
        self.assertTrue(any(item.selection_reason == "writer_entry_supporting_fact" for item in bundle.items))
        self.assertTrue(any(item.selection_reason == "writer_target_ask_supporting_fact" for item in bundle.items))
        self.assertGreater(bundle.metadata.get("writer_stage_doc_count") or 0, 0)

        blocks = assembler.build_solver_user_content(bundle=bundle)
        self.assertTrue(any(block.get("type") == "image_url" for block in blocks))
        self.assertTrue(any("[DOC " in block.get("text", "") for block in blocks if block.get("type") == "text"))
        self.assertTrue(any("Opening clue fact from the source page." in block.get("text", "") for block in blocks if block.get("type") == "text"))
        self.assertTrue(any("The emblem identifies Target Entity." in block.get("text", "") for block in blocks if block.get("type") == "text"))
        self.assertTrue(any("Caption: A commemorative poster with the distinctive emblem of Target Entity." in block.get("text", "") for block in blocks if block.get("type") == "text"))

    def test_solver_request_uses_multimodal_user_blocks(self):
        fixture = self._fixture()
        assembler = _build_assembler(fixture.graph_dir, random_seed=7)
        bundle = assembler.build_bundle(
            question_record=fixture.question_record,
            sample_record=fixture.sample_record,
        )

        request = build_repository_solver_request(
            bundle=bundle,
            answer_model_alias="answer-model",
            answer_max_tokens=321,
            user_content=assembler.build_solver_user_content(bundle=bundle),
        )

        payload = request.to_dict()
        self.assertEqual(payload["model"], "answer-model")
        self.assertIsNone(payload["response_format"])
        self.assertEqual(payload["max_tokens"], 321)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertIsInstance(payload["messages"][0]["content"], str)
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIsInstance(payload["messages"][1]["content"], list)
        self.assertTrue(any(block.get("type") == "text" for block in payload["messages"][1]["content"]))
        self.assertTrue(any(block.get("type") == "image_url" for block in payload["messages"][1]["content"]))
        self.assertIn("Final answer:", payload["messages"][0]["content"])

    def test_judge_request_uses_json_string_user_payload(self):
        request = build_repository_answer_judge_request(
            question="Q?",
            gold_answer="Gold",
            predicted_answer="Pred",
            judge_model_alias="judge-model",
            judge_max_tokens=123,
            question_id="q_1",
        )

        payload = request.to_dict()
        self.assertEqual(payload["model"], "judge-model")
        self.assertEqual(payload["response_format"], {"type": "json_object"})
        self.assertEqual(payload["max_tokens"], 123)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIsInstance(payload["messages"][1]["content"], str)
        judge_payload = json.loads(payload["messages"][1]["content"])
        self.assertEqual(judge_payload["question"], "Q?")
        self.assertEqual(judge_payload["gold_answer"], "Gold")
        self.assertEqual(judge_payload["predicted_answer"], "Pred")

    def test_question_only_request_uses_plain_text_question(self):
        request = build_question_only_shortcut_request(
            question="Who is this?",
            answer_model_alias="answer-model",
            answer_max_tokens=77,
            question_id="q_2",
        )

        payload = request.to_dict()
        self.assertEqual(payload["model"], "answer-model")
        self.assertIsNone(payload["response_format"])
        self.assertEqual(payload["max_tokens"], 77)
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIsInstance(payload["messages"][1]["content"], str)
        self.assertIn("Question:\nWho is this?", payload["messages"][1]["content"])

    def test_question_only_request_includes_attached_image_when_provided(self):
        request = build_question_only_shortcut_request(
            question="What logo is shown in this image?",
            answer_model_alias="answer-model",
            answer_max_tokens=88,
            question_id="q_3",
            image_url="https://images.example.org/question.jpg",
        )

        payload = request.to_dict()
        self.assertEqual(payload["model"], "answer-model")
        self.assertIsNone(payload["response_format"])
        self.assertEqual(payload["max_tokens"], 88)
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertIsInstance(payload["messages"][1]["content"], list)
        self.assertTrue(any(block.get("type") == "text" for block in payload["messages"][1]["content"]))
        self.assertTrue(any(block.get("type") == "image_url" for block in payload["messages"][1]["content"]))

    def test_repository_verifier_passes_when_answer_and_citations_are_correct(self):
        fixture = self._fixture()
        assembler = _build_assembler(fixture.graph_dir, random_seed=11)
        bundle = assembler.build_bundle(question_record=fixture.question_record, sample_record=fixture.sample_record)
        relevant_doc, relevant_image, _ = _pick_labels(bundle)

        client = FakeJsonClient(
            [
                (
                    f"The source page points to the official codename. [{relevant_doc}]\n\n"
                    f"The poster image identifies the target entity. [{relevant_image}]\n\n"
                    "Final answer: Target Entity"
                ),
                {
                    "correct": True,
                    "reason": "Semantic match.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "Target Entity",
                },
                "Cannot answer: No reliable shortcut in the wording.",
                {
                    "correct": False,
                    "reason": "No predicted answer.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "",
                },
            ]
        )
        verifier = OfflineGraphRepositoryVerifier(
            assembler=assembler,
            model_client=client,
            answer_model_alias="fake-answer",
            judge_model_alias="fake-judge",
        )

        record = verifier.verify_question_record(
            question_record=fixture.question_record,
            sample_record=fixture.sample_record,
            question_index=1,
            question_fingerprint="fingerprint",
        )

        self.assertTrue(record["final_keep"])
        self.assertTrue(record["checks"]["citations_exist"]["passed"])
        self.assertTrue(record["checks"]["citations_within_relevant_scope"]["passed"])
        self.assertTrue(record["checks"]["answer_judgment"]["correct"])
        self.assertTrue(record["checks"]["question_only_shortcut"]["passed"])
        self.assertEqual(record["question_only_solver_result"]["answer"], "")
        self.assertEqual(record["question_only_solver_result"]["cannot_answer_reason"], "No reliable shortcut in the wording.")
        self.assertTrue(any(block.get("type") == "image_url" for block in client.requests[0].messages[1].content))
        self.assertIsInstance(client.requests[2].messages[1].content, str)

    def test_repository_verifier_rejects_distractor_citation(self):
        fixture = self._fixture()
        assembler = _build_assembler(fixture.graph_dir, random_seed=13)
        bundle = assembler.build_bundle(question_record=fixture.question_record, sample_record=fixture.sample_record)
        relevant_doc, _, distractor = _pick_labels(bundle)

        client = FakeJsonClient(
            [
                f"Use one true clue and one false clue. [{relevant_doc}] [{distractor}]\n\nFinal answer: Target Entity",
                {
                    "correct": True,
                    "reason": "Answer itself is correct.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "Target Entity",
                },
                "Cannot answer: No shortcut.",
                {
                    "correct": False,
                    "reason": "No predicted answer.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "",
                },
            ]
        )
        verifier = OfflineGraphRepositoryVerifier(
            assembler=assembler,
            model_client=client,
            answer_model_alias="fake-answer",
            judge_model_alias="fake-judge",
        )

        record = verifier.verify_question_record(
            question_record=fixture.question_record,
            sample_record=fixture.sample_record,
            question_index=1,
            question_fingerprint="fingerprint",
        )

        self.assertFalse(record["final_keep"])
        self.assertFalse(record["checks"]["citations_within_relevant_scope"]["passed"])
        self.assertIn("used_distractor_or_out_of_scope_evidence", record["reject_reasons"])

    def test_repository_verifier_rejects_question_only_shortcut(self):
        fixture = self._fixture()
        assembler = _build_assembler(fixture.graph_dir, random_seed=19)
        bundle = assembler.build_bundle(question_record=fixture.question_record, sample_record=fixture.sample_record)
        relevant_doc, relevant_image, _ = _pick_labels(bundle)

        client = FakeJsonClient(
            [
                (
                    f"The source page points to the official codename. [{relevant_doc}]\n\n"
                    f"The poster image identifies the target entity. [{relevant_image}]\n\n"
                    "Final answer: Target Entity"
                ),
                {
                    "correct": True,
                    "reason": "Semantic match.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "Target Entity",
                },
                "The wording points directly to the emblem and the target entity.\n\nFinal answer: Target Entity",
                {
                    "correct": True,
                    "reason": "The shortcut answer is still correct.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "Target Entity",
                },
            ]
        )
        verifier = OfflineGraphRepositoryVerifier(
            assembler=assembler,
            model_client=client,
            answer_model_alias="fake-answer",
            judge_model_alias="fake-judge",
        )

        record = verifier.verify_question_record(
            question_record=fixture.question_record,
            sample_record=fixture.sample_record,
            question_index=1,
            question_fingerprint="fingerprint",
        )

        self.assertFalse(record["final_keep"])
        self.assertFalse(record["checks"]["question_only_shortcut"]["passed"])
        self.assertIn("closed_book_shortcut", record["reject_reasons"])
        self.assertEqual(record["question_only_solver_result"]["status"], "answered")

    def test_repository_verifier_run_writes_outputs(self):
        fixture = self._fixture()
        assembler = _build_assembler(fixture.graph_dir, random_seed=17)
        bundle = assembler.build_bundle(question_record=fixture.question_record, sample_record=fixture.sample_record)
        relevant_doc, relevant_image, _ = _pick_labels(bundle)

        client = FakeJsonClient(
            [
                (
                    f"The source page gives the codename clue. [{relevant_doc}]\n\n"
                    f"The poster image resolves the entity. [{relevant_image}]\n\n"
                    "Final answer: Target Entity"
                ),
                {
                    "correct": True,
                    "reason": "Correct answer.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "Target Entity",
                },
                "Cannot answer: No shortcut.",
                {
                    "correct": False,
                    "reason": "No predicted answer.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "",
                },
            ]
        )
        verifier = OfflineGraphRepositoryVerifier(
            assembler=assembler,
            model_client=client,
            answer_model_alias="fake-answer",
            judge_model_alias="fake-judge",
        )

        summary = verifier.run(vqa_dir=fixture.vqa_dir)

        self.assertEqual(summary["verified_total"], 1)
        self.assertEqual(summary["final_keep_total"], 1)
        self.assertEqual(summary["question_only_shortcut_total"], 0)
        results = [json.loads(line) for line in (fixture.vqa_dir / verifier.output_file_name).read_text(encoding="utf-8").splitlines() if line.strip()]
        self.assertEqual(len(results), 1)
        self.assertTrue(results[0]["final_keep"])
        self.assertTrue((fixture.vqa_dir / verifier.summary_file_name).exists())

    def test_combined_repository_verifier_wrapper_requires_model_aliases(self):
        fixture = self._fixture()
        repo_root = Path(__file__).resolve().parents[1]
        command = [
            sys.executable,
            "-m",
            "synthesis.vqa.run_repository_verifier_with_io",
            "--vqa-dir",
            str(fixture.vqa_dir),
            "--graph-dir",
            str(fixture.graph_dir),
            "--limit",
            "1",
        ]
        result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--run-verification requires both --answer-model-alias and --judge-model-alias", result.stderr or result.stdout)

    def test_debug_repository_verifier_script_prints_repository_bundle(self):
        fixture = self._fixture()
        repo_root = Path(__file__).resolve().parents[1]
        command = [
            sys.executable,
            "-m",
            "synthesis.vqa.debug.debug_repository_verifier",
            "--vqa-dir",
            str(fixture.vqa_dir),
            "--graph-dir",
            str(fixture.graph_dir),
            "--question-id",
            fixture.question_record["question_id"],
            "--limit",
            "1",
        ]
        result = subprocess.run(command, cwd=repo_root, capture_output=True, text=True, check=False)

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("Repository Bundle", result.stdout)
        self.assertIn(fixture.question_record["question_id"], result.stdout)
        self.assertIn("Items", result.stdout)
        self.assertIn("Answer Model Request", result.stdout)
        self.assertIn("Question-Only Shortcut Request", result.stdout)
        self.assertIn('"messages": [', result.stdout)
        self.assertIn('"response_format": null', result.stdout)

    def test_combined_repository_verifier_prints_io_summary(self):
        from synthesis.vqa.debug import debug_repository_verifier as debug_mod

        fixture = self._fixture()

        class FakeWorker(FakeJsonClient):
            pass

        fake_worker = FakeWorker(
            [
                "The source page gives the codename clue. [DOC 1]\n\nThe poster image resolves the entity. [IMG 3]\n\nFinal answer: Target Entity",
                {
                    "correct": True,
                    "reason": "Correct answer.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "Target Entity",
                },
                "Cannot answer: No shortcut.",
                {
                    "correct": False,
                    "reason": "No predicted answer.",
                    "normalized_gold_answer": "Target Entity",
                    "normalized_predicted_answer": "",
                },
            ]
        )
        argv = [
            "--vqa-dir",
            str(fixture.vqa_dir),
            "--graph-dir",
            str(fixture.graph_dir),
            "--question-id",
            fixture.question_record["question_id"],
            "--limit",
            "1",
            "--run-verification",
            "--answer-model-alias",
            "fake-answer",
            "--judge-model-alias",
            "fake-judge",
        ]
        with mock.patch("synthesis.model_worker.LLM_WORKER", fake_worker), mock.patch("sys.stdout") as fake_stdout:
            result = debug_mod.main(argv)

        self.assertEqual(result, 0)
        printed = "\n".join(str(call.args[0]) for call in fake_stdout.write.call_args_list if call.args)
        self.assertIn("Verification IO Summary", printed)
        self.assertIn("Repository-grounded:", printed)
        self.assertIn("Question-only:", printed)
        self.assertIn("  correct: 1", printed)
        self.assertIn("  cannot_answer_or_question_issue: 1", printed)

    def test_debug_repository_verifier_redacts_inline_image_data_urls(self):
        from synthesis.vqa.debug.debug_repository_verifier import _dump_stdout_safe_json

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Look at the image."},
                        {
                            "type": "image_url",
                            "image_url": {"url": "data:image/png;base64,QUJDREVGRw=="},
                        },
                    ],
                }
            ]
        }

        rendered = _dump_stdout_safe_json(payload)

        self.assertIn("<redacted inline image data URL: mime=image/png", rendered)
        self.assertNotIn("QUJDREVGRw==", rendered)

    def test_debug_repository_verifier_stdout_omits_local_image_base64(self):
        fixture = self._fixture()
        local_image = fixture.vqa_dir / "question.png"
        local_image.write_bytes(b"not a real png but enough for base64")
        fixture.question_record["image_url"] = str(local_image)
        (fixture.vqa_dir / "questions.jsonl").write_text(
            json.dumps(fixture.question_record, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

        from synthesis.vqa.debug import debug_repository_verifier as debug_mod

        argv = [
            "--vqa-dir",
            str(fixture.vqa_dir),
            "--graph-dir",
            str(fixture.graph_dir),
            "--question-id",
            fixture.question_record["question_id"],
            "--limit",
            "1",
        ]
        with mock.patch("sys.stdout") as fake_stdout:
            result = debug_mod.main(argv)

        self.assertEqual(result, 0)
        printed = "\n".join(str(call.args[0]) for call in fake_stdout.write.call_args_list if call.args)
        self.assertIn("Question-Only Shortcut Request", printed)
        self.assertIn("<redacted inline image data URL: mime=image/png", printed)
        self.assertNotIn("bm90IGEgcmVhbCBwbmcgYnV0IGVub3VnaCBmb3IgYmFzZTY0", printed)


if __name__ == "__main__":
    unittest.main()
