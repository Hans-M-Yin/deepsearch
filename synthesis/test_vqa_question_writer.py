import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from synthesis.model_worker import ModelResponse
from synthesis.vqa.batch_runner import VqaBatchRunner
from synthesis.vqa.filter_fallback_questions import filter_questions
from synthesis.vqa.question_writer import (
    HopContext,
    QuestionWriter,
    QuestionWriterGenerationError,
    WriterContext,
)


class MultimodalImageUrlTests(unittest.TestCase):
    def test_rewrites_configured_oss_host_to_accelerate_host(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "OSS_ENDPOINT": "oss-cn-beijing.aliyuncs.com",
                "OSS_BUCKET_NAME": "search-hans",
                "OSS_ENDPOINT_ACCELERATE": "oss-accelerate.aliyuncs.com",
            },
            clear=False,
        ):
            self.assertEqual(
                QuestionWriter._resolve_multimodal_image_url(
                    "https://search-hans.oss-cn-beijing.aliyuncs.com/vision/a.png?x=1"
                ),
                "https://search-hans.oss-accelerate.aliyuncs.com/vision/a.png?x=1",
            )

    def test_leaves_non_oss_url_unchanged(self) -> None:
        with patch.dict(
            "os.environ",
            {"OSS_ENDPOINT_ACCELERATE": "oss-accelerate.aliyuncs.com"},
            clear=False,
        ):
            url = "https://images.example.com/photo.png"
            self.assertEqual(QuestionWriter._resolve_multimodal_image_url(url), url)


def _hop(
    index: int,
    source_id: str,
    target_id: str,
    source_title: str,
    target_title: str,
    *,
    relation: str = "",
    edge_type: str = "wiki_link",
) -> HopContext:
    return HopContext(
        hop_index=index,
        src_node_id=source_id,
        dst_node_id=target_id,
        src_modality="text",
        dst_modality="text",
        edge_id=f"edge-{index}",
        edge_type=edge_type,
        relation=relation,
        src_content={"node_id": source_id, "node_type": "text", "title": source_title},
        dst_content={"node_id": target_id, "node_type": "text", "title": target_title},
    )


class PredecessorChainFormattingTests(unittest.TestCase):
    def test_uses_relations_in_path_order(self) -> None:
        context = WriterContext(
            path_id="path-1",
            trajectory={},
            hops=[
                _hop(0, "a", "b", "Object A", "Object B", relation="influenced"),
                _hop(1, "b", "c", "Object B", "Target C", relation="worked with"),
            ],
            target_node={"node_id": "c", "node_type": "text", "title": "Target C"},
        )

        self.assertEqual(
            QuestionWriter._format_predecessor_chain(context),
            "Object A --influenced--> Object B --worked with--> Target C",
        )

    def test_falls_back_to_edge_type(self) -> None:
        context = WriterContext(
            path_id="path-2",
            trajectory={},
            hops=[_hop(0, "a", "b", "Object A", "Target B")],
            target_node={"node_id": "b", "node_type": "text", "title": "Target B"},
        )

        self.assertEqual(
            QuestionWriter._format_predecessor_chain(context),
            "Object A --wiki_link--> Target B",
        )

    def test_is_empty_without_hops(self) -> None:
        context = WriterContext(
            path_id="path-empty",
            trajectory={},
            hops=[],
            target_node={"node_id": "a", "node_type": "text", "title": "Target A"},
        )

        self.assertEqual(QuestionWriter._format_predecessor_chain(context), "")


class _QueuedModelClient:
    def __init__(self, responses: list[dict]) -> None:
        self.responses = list(responses)
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return ModelResponse(content=__import__("json").dumps(self.responses.pop(0)))


class ImageTargetCandidateSelectionTests(unittest.TestCase):
    def test_generates_evaluates_and_keeps_all_image_candidates(self) -> None:
        candidates = [
            {
                "candidate_id": "candidate_1",
                "question_type": "color",
                "ask_target": "What color is the ball",
                "answer": "white",
                "visual_locator": "the ball",
                "visual_reasoning": ["locate the ball"],
                "supporting_facts": ["The ball is white."],
            },
            {
                "candidate_id": "candidate_2",
                "question_type": "event_environment",
                "ask_target": "What brand is shown directly behind the goalkeeper",
                "answer": "Brand B",
                "visual_locator": "directly behind the goalkeeper",
                "visual_reasoning": ["locate the goalkeeper", "read the board behind him"],
                "supporting_facts": ["Brand B is shown behind the goalkeeper."],
            },
        ]
        evaluation = {
            "decision": "select",
            "selected_candidate_id": "candidate_2",
            "evaluations": [
                {"candidate_id": "candidate_1", "valid": False},
                {"candidate_id": "candidate_2", "valid": True},
            ],
        }
        verification_client = _QueuedModelClient(
            [
                {
                    "answers": [
                        {"candidate_id": "candidate_1", "answer": "white", "answerable": True},
                        {"candidate_id": "candidate_2", "answer": "Brand B", "answerable": True},
                    ]
                },
                {
                    "answers": [
                        {"candidate_id": "candidate_1", "answer": "white", "answerable": True},
                        {"candidate_id": "candidate_2", "answer": "", "answerable": False},
                    ]
                },
                {
                    "evaluations": [
                        {
                            "candidate_id": "candidate_1",
                            "with_image_correct": True,
                            "without_image_correct": True,
                            "pass": False,
                        },
                        {
                            "candidate_id": "candidate_2",
                            "with_image_correct": True,
                            "without_image_correct": False,
                            "pass": True,
                        },
                    ]
                },
            ]
        )
        client = _QueuedModelClient([{"candidates": candidates}, evaluation])
        writer = QuestionWriter(
            model_client=client,
            model="writer",
            ask_target_verify_model_client=verification_client,
            ask_target_verify_model="visual-verifier",
        )
        context = WriterContext(
            path_id="path-image",
            trajectory={},
            hops=[],
            target_node={
                "node_id": "image-1",
                "node_type": "image",
                "title": "Penalty kick",
                "search_query": "Montiel taking the final penalty",
                "image_url": "https://example.com/image.jpg",
            },
        )

        selected = writer.select_target_ask(context=context)

        self.assertEqual(selected["ask_target"], "What brand is shown directly behind the goalkeeper?")
        self.assertEqual(selected["answer"], "Brand B")
        self.assertEqual(len(selected["image_target_candidates"]), 2)
        self.assertEqual(
            selected["image_target_candidate_verification"]["kept_candidate_ids"],
            ["candidate_2"],
        )
        self.assertEqual(selected["image_target_candidate_evaluation"], evaluation)
        self.assertEqual(len(client.requests), 2)
        self.assertTrue(all(request.max_tokens >= 2400 for request in client.requests))
        self.assertEqual(len(verification_client.requests), 3)
        self.assertTrue(all(request.model == "visual-verifier" for request in verification_client.requests))

    def test_visual_verification_keeps_all_when_every_candidate_is_filtered(self) -> None:
        candidates = [
            {"candidate_id": "candidate_1", "ask_target": "What color is it?", "answer": "red"},
            {"candidate_id": "candidate_2", "ask_target": "What number is shown?", "answer": "7"},
        ]
        client = _QueuedModelClient(
            [
                {"answers": []},
                {"answers": []},
                {
                    "evaluations": [
                        {
                            "candidate_id": item["candidate_id"],
                            "with_image_correct": False,
                            "without_image_correct": False,
                            "pass": False,
                        }
                        for item in candidates
                    ]
                },
            ]
        )
        writer = QuestionWriter(
            model_client=client,
            model="visual-verifier",
            ask_target_verify_model_client=client,
            ask_target_verify_model="visual-verifier",
        )

        kept, verification = writer._verify_image_target_candidates(
            candidates=candidates,
            image_url="https://example.com/image.jpg",
        )

        self.assertEqual(kept, candidates)
        self.assertEqual(verification["decision"], "skip_all_filtered")
        self.assertEqual(verification["kept_candidate_ids"], ["candidate_1", "candidate_2"])
        self.assertEqual(verification["filtered_candidate_ids"], ["candidate_1", "candidate_2"])

    def test_text_target_filters_closed_book_solvable_candidates_and_selects_best(self) -> None:
        candidates = [
            {
                "candidate_id": "candidate_1",
                "ask_target": "Where was the target born?",
                "answer": "Example City",
                "supporting_facts": ["Born in Example City."],
            },
            {
                "candidate_id": "candidate_2",
                "ask_target": "Which obscure institution granted the target a fellowship?",
                "answer": "Example Institute",
                "supporting_facts": ["The target received a fellowship from Example Institute."],
            },
        ]
        verification_client = _QueuedModelClient(
            [
                {
                    "answers": [
                        {"candidate_id": "candidate_1", "answer": "Example City", "answerable": True},
                        {"candidate_id": "candidate_2", "answer": "", "answerable": False},
                    ]
                },
                {
                    "evaluations": [
                        {"candidate_id": "candidate_1", "correct": True},
                        {"candidate_id": "candidate_2", "correct": False},
                    ]
                }
            ]
        )
        evaluation = {
            "decision": "select",
            "selected_candidate_id": "candidate_2",
            "evaluations": [{"candidate_id": "candidate_2", "valid": True}],
        }
        client = _QueuedModelClient([{"candidates": candidates}, evaluation])
        writer = QuestionWriter(
            model_client=client,
            model="writer",
            ask_target_verify_model_client=verification_client,
            ask_target_verify_model="ask-target-verifier",
        )
        context = WriterContext(
            path_id="path-text",
            trajectory={},
            hops=[],
            target_node={"node_id": "text-1", "node_type": "text", "title": "Target"},
        )

        selected = writer.select_target_ask(context=context)

        self.assertEqual(selected["ask_target"], "Which obscure institution granted the target a fellowship?")
        self.assertEqual(selected["answer"], "Example Institute")
        self.assertEqual(
            selected["text_target_candidate_verification"]["filtered_candidate_ids"],
            ["candidate_1"],
        )
        self.assertEqual(selected["text_target_candidate_evaluation"], evaluation)
        self.assertNotIn("image_target_candidates", selected)
        self.assertNotIn("image_target_candidate_evaluation", selected)
        self.assertEqual(len(client.requests), 2)
        self.assertEqual(len(verification_client.requests), 2)
        self.assertTrue(all(request.model == "ask-target-verifier" for request in verification_client.requests))

    def test_compact_record_persists_target_candidate_fields_when_present(self) -> None:
        base_sample = {
            "sample_id": "sample-1",
            "path": {},
            "draft": {
                "question": "Question?",
                "answer": "Answer",
                "metadata": {},
            },
        }
        text_record = VqaBatchRunner._compact_sample_record(base_sample)
        self.assertNotIn("image_target_candidates", text_record)
        self.assertNotIn("image_target_candidate_evaluation", text_record)
        self.assertNotIn("text_target_candidates", text_record)
        self.assertNotIn("text_target_candidate_evaluation", text_record)

        base_sample["draft"]["metadata"] = {
            "image_target_candidates": [{"candidate_id": "candidate_1"}],
            "image_target_candidate_evaluation": {
                "decision": "select",
                "selected_candidate_id": "candidate_1",
            },
            "image_target_candidate_verification": {
                "decision": "filter",
                "kept_candidate_ids": ["candidate_1"],
            },
        }
        image_record = VqaBatchRunner._compact_sample_record(base_sample)
        self.assertEqual(len(image_record["image_target_candidates"]), 1)
        self.assertEqual(
            image_record["image_target_candidate_evaluation"]["selected_candidate_id"],
            "candidate_1",
        )
        self.assertEqual(image_record["image_target_candidate_verification"]["decision"], "filter")

        base_sample["draft"]["metadata"] = {
            "text_target_candidates": [{"candidate_id": "candidate_2"}],
            "text_target_candidate_evaluation": {
                "decision": "select",
                "selected_candidate_id": "candidate_2",
            },
            "text_target_candidate_verification": {
                "decision": "filter",
                "kept_candidate_ids": ["candidate_2"],
            },
        }
        text_candidate_record = VqaBatchRunner._compact_sample_record(base_sample)
        self.assertEqual(len(text_candidate_record["text_target_candidates"]), 1)
        self.assertEqual(
            text_candidate_record["text_target_candidate_evaluation"]["selected_candidate_id"],
            "candidate_2",
        )
        self.assertEqual(text_candidate_record["text_target_candidate_verification"]["decision"], "filter")

    def test_compact_record_persists_compose_and_difficulty_analyses(self) -> None:
        sample = {
            "sample_id": "sample-1",
            "path": {},
            "draft": {
                "question": "Draft question?",
                "answer": "Answer",
                "metadata": {
                    "compose_payload": {"hops": [{"statement": "A leads to B."}]},
                    "compose_result": {
                        "analysis": "Merged the hops while hiding the intermediate target.",
                        "question": "Draft question?",
                    },
                },
            },
            "obfuscated": {
                "question": "Enhanced question?",
                "answer": "Answer",
                "metadata": {
                    "difficulty_enhancement_payload": {"question": "Draft question?"},
                    "difficulty_enhancement_result": {
                        "analysis": "Blurred the identifying clue.",
                        "question": "Enhanced question?",
                    },
                },
            },
        }

        record = VqaBatchRunner._compact_sample_record(sample)

        self.assertEqual(
            record["compose"]["result"]["analysis"],
            "Merged the hops while hiding the intermediate target.",
        )
        self.assertEqual(
            record["difficulty_enhancement"]["result"]["analysis"],
            "Blurred the identifying clue.",
        )


class DifficultyEnhancementImageMarkTests(unittest.TestCase):
    def test_image_entry_hop_is_marked(self) -> None:
        hop = HopContext(
            hop_index=0,
            src_node_id="image-1",
            dst_node_id="text-1",
            src_modality="image",
            dst_modality="text",
            edge_id="edge-1",
            edge_type="image_depicts",
            relation="depicts",
            src_content={"node_id": "image-1", "node_type": "image", "caption": "A scene"},
            dst_content={"node_id": "text-1", "node_type": "text", "title": "Target"},
        )
        context = WriterContext(
            path_id="path-image-entry",
            trajectory={},
            hops=[hop],
            target_node={"node_id": "text-1", "node_type": "text", "title": "Target"},
        )
        entry = QuestionWriter(
            model_client=_QueuedModelClient(
                [{"rewritten_statement": "This image shows a scene."}]
            ),
            model="test-model",
        ).build_entry_hop(
            path=SimpleNamespace(trajectory=SimpleNamespace(starts_with_image=True)),
            context=context,
            hop_summaries=[
                {
                    "hop_index": 0,
                    "source": "A scene",
                    "target": "Target",
                    "statement": "A scene depicts Target.",
                }
            ],
            target_ask={"ask_target": "What is the answer?"},
        )

        self.assertEqual(entry["mark"], "image")

    def test_terminal_image_ask_is_appended_to_difficulty_chain(self) -> None:
        payload = QuestionWriter._difficulty_enhancement_payload(
            question="A multi-hop question?",
            answer="Brand B",
            hops=[
                {
                    "hop_index": 0,
                    "source": "A",
                    "target": "B",
                    "statement": "A leads to B.",
                    "mark": "image",
                }
            ],
            target_ask={
                "ask_target": "What brand is directly behind the goalkeeper?",
                "mark": "image",
            },
            question_terminal_bridge={
                "source": "B",
                "target_image": "Penalty scene",
                "removed_question_hop": {
                    "hop_index": 1,
                    "relation": "is shown in",
                    "retrieval_query": "penalty scene",
                },
            },
        )

        self.assertEqual(payload["reasoning_chain"][0]["mark"], "image")
        terminal = payload["reasoning_chain"][1]
        self.assertEqual(terminal["mark"], "image")
        self.assertTrue(terminal["terminal_question"])
        self.assertEqual(
            terminal["statement"],
            "What brand is directly behind the goalkeeper?",
        )


class WriterFallbackTrackingTests(unittest.TestCase):
    def test_required_llm_stage_fails_instead_of_rule_composing(self) -> None:
        with self.assertRaises(QuestionWriterGenerationError) as caught:
            QuestionWriter().compress_hop(hop=_hop(0, "a", "b", "Object A", "Object B"))

        self.assertEqual(caught.exception.stage, "compress_hop_0")
        self.assertEqual(caught.exception.reason, "no_model_client")

    def test_image_entry_requires_llm_rewrite(self) -> None:
        image_hop = HopContext(
            hop_index=0,
            src_node_id="image-1",
            dst_node_id="text-1",
            src_modality="image",
            dst_modality="text",
            edge_id="edge-1",
            edge_type="image_depicts",
            relation="depicts",
            src_content={"node_id": "image-1", "node_type": "image", "caption": "A scene"},
            dst_content={"node_id": "text-1", "node_type": "text", "title": "Target"},
        )
        context = WriterContext(
            path_id="path-image-entry-no-model",
            trajectory={},
            hops=[image_hop],
            target_node={"node_id": "text-1", "node_type": "text", "title": "Target"},
        )

        with self.assertRaises(QuestionWriterGenerationError) as caught:
            QuestionWriter().build_entry_hop(
                path=SimpleNamespace(trajectory=SimpleNamespace(starts_with_image=True)),
                context=context,
                hop_summaries=[
                    {
                        "hop_index": 0,
                        "source": "A scene",
                        "target": "Target",
                        "statement": "A scene depicts Target.",
                    }
                ],
                target_ask={"ask_target": "What is the answer?"},
            )

        self.assertEqual(caught.exception.stage, "rewrite_image_first_hop")

    def test_compact_records_include_final_fallback_marker(self) -> None:
        sample = {
            "sample_id": "sample-fallback",
            "path": {"path_id": "path-fallback"},
            "draft": {
                "question": "A rule-composed question?",
                "answer": "answer",
                "metadata": {
                    "writer_fallback_used": True,
                    "writer_fallback_stages": ["compose_question"],
                    "writer_fallback_reason": "llm_compose_failed_or_invalid",
                },
            },
        }

        question_record = VqaBatchRunner._compact_question_record(sample, question_number=1)
        sample_record = VqaBatchRunner._compact_sample_record(sample)

        self.assertTrue(question_record["writer_fallback_used"])
        self.assertEqual(question_record["writer_fallback_stages"], ["compose_question"])
        self.assertTrue(sample_record["metadata"]["writer_fallback_used"])

    def test_filter_uses_sample_metadata_and_keeps_samples_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            vqa_dir = Path(directory)
            samples = [
                {
                    "sample_id": "sample-good",
                    "metadata": {"writer_fallback_used": False},
                },
                {
                    "sample_id": "sample-bad",
                    "metadata": {
                        "writer_fallback_used": True,
                        "writer_fallback_stages": ["compose_question"],
                    },
                },
            ]
            questions = [
                {"question_id": "q_000001", "sample_id": "sample-good"},
                {"question_id": "q_000002", "sample_id": "sample-bad"},
            ]
            for name, records in (("samples.jsonl", samples), ("questions.jsonl", questions)):
                (vqa_dir / name).write_text(
                    "".join(json.dumps(record) + "\n" for record in records),
                    encoding="utf-8",
                )

            result = filter_questions(vqa_dir=vqa_dir, in_place=True)

            self.assertEqual(result["removed_questions"], 1)
            self.assertEqual(result["kept_questions"], 1)
            remaining = [
                json.loads(line)
                for line in (vqa_dir / "questions.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual([item["sample_id"] for item in remaining], ["sample-good"])
            self.assertEqual(len((vqa_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
