from __future__ import annotations

import importlib.util
import unittest
import sys
from pathlib import Path


SCRIPT = Path(__file__).with_name("eval_image_uniqueness.py")
SPEC = importlib.util.spec_from_file_location("eval_image_uniqueness", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ExtractGoldExamplesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.examples = MODULE.load_gold_examples(MODULE.DEFAULT_DISCUSSION)
        cls.by_id = {item.image_id: item for item in cls.examples}

    def test_extracts_all_unique_image_records(self) -> None:
        self.assertEqual(len(self.examples), 61)
        self.assertEqual(len(self.by_id), 61)

    def test_later_expert_revisions_win(self) -> None:
        revised = {
            "image_cd6efdc2f9ae9e1c",
            "image_fdfe73a47598af8e",
            "image_dd43d38ad0816df5",
            "image_a4f8e11c04d717d1",
            "image_beee4ffd7cd55669",
        }
        for image_id in revised:
            with self.subTest(image_id=image_id):
                self.assertEqual(self.by_id[image_id].gold_label, "半唯一性")

    def test_known_non_unique_example(self) -> None:
        self.assertEqual(self.by_id["image_e879967e119c42e8"].gold_label, "不唯一性")

    def test_description_parser_handles_missing_space_before_text_id(self) -> None:
        description = self.by_id["image_f43468b041fe50bf"].description
        self.assertTrue(description.endswith("with the Yarra River flowing into it."))
        self.assertNotIn("text_", description)


class ModelResponseParserTest(unittest.TestCase):
    def test_strict_two_line_response(self) -> None:
        raw = (
            "分析：描述锁定一座可反复拍摄的建筑。\n"
            "image_deadbeef  分类：半唯一性  ｜  理由：不同时间和角度会产生不同照片。"
        )
        parsed = MODULE.parse_model_response(raw, expected_image_id="image_deadbeef")
        self.assertEqual(parsed["label"], "半唯一性")
        self.assertIsNone(parsed["parse_error"])
        self.assertIn("可反复拍摄", parsed["analysis"])

    def test_fallback_label_is_scored_but_warned(self) -> None:
        parsed = MODULE.parse_model_response("最终分类：Unique", expected_image_id="image_deadbeef")
        self.assertEqual(parsed["label"], "唯一性")
        self.assertEqual(parsed["parse_error"], "fallback_label_parse")

    def test_missing_label(self) -> None:
        parsed = MODULE.parse_model_response("只有分析，没有最终分类。", expected_image_id="image_deadbeef")
        self.assertIsNone(parsed["label"])
        self.assertEqual(parsed["parse_error"], "missing_final_label")


class MetricsTest(unittest.TestCase):
    @staticmethod
    def _result(image_id: str, gold: str, predicted: str | None) -> object:
        return MODULE.EvalResult(
            image_id=image_id,
            description="test",
            source_text_id="text_test",
            source_label="test",
            image_url="",
            gold_label=gold,
            expert_reason="",
            predicted_label=predicted,
            correct=predicted == gold,
            parse_error=None if predicted else "missing_final_label",
            analysis="",
            model_reason="",
            raw_response="",
            model_alias="test",
            served_model="test",
            usage=None,
            elapsed_seconds=0.0,
            attempt_count=1,
            error=None,
        )

    def test_unique_precision_and_recall_include_unparsed_false_negative(self) -> None:
        results = [
            self._result("image_00000001", "唯一性", "唯一性"),  # TP
            self._result("image_00000002", "唯一性", "半唯一性"),  # FN
            self._result("image_00000003", "唯一性", None),       # FN (unparsed)
            self._result("image_00000004", "半唯一性", "唯一性"), # FP
        ]
        metrics = MODULE.compute_metrics(results)
        self.assertAlmostEqual(metrics["unique_image_precision"], 0.5)
        self.assertAlmostEqual(metrics["unique_image_recall"], 1 / 3, places=6)
        self.assertEqual(metrics["unique_image_metrics"]["true_positive"], 1)
        self.assertEqual(metrics["unique_image_metrics"]["predicted_positive"], 2)
        self.assertEqual(metrics["unique_image_metrics"]["gold_positive"], 3)


if __name__ == "__main__":
    unittest.main()
