import unittest
from synthesis.post_process.fix.process_leakage import check_image_labels
from synthesis.post_process.fix.rewrite import checked_reasoning
from synthesis.post_process.fix.tests import test_pipeline
from synthesis.post_process.fix.validate import validate_structure


class ImageLabelTests(unittest.TestCase):
    def test_internal_labels_rejected_but_resource_ids_preserved(self):
        for label in ["IMAGE 0", "Image 1", "image #2", "IMAGE[3]", "ＩＭＡＧＥ ０"]:
            with self.subTest(label=label), self.assertRaises(ValueError):
                checked_reasoning({"reasoning": "Compare with " + label})
        for text in [
            "Read image_abc and img_1.",
            "Compare the original image with the candidate.",
        ]:
            check_image_labels(text)

    def test_final_gate_catches_unedited_original_prose(self):
        fixture = test_pipeline.PipelineTests()
        fixture.setUp()
        try:
            record, _, _, window, _, _, plan = fixture.plan("direct")
            record["conversations"][-1]["value"] = record["conversations"][-1][
                "value"
            ].replace("Supported.", "Supported by IMAGE 0.")
            updated, audit = fixture.assemble(record, window, plan)
            self.assertTrue(
                any(
                    "Internal image-label leakage" in e
                    for e in validate_structure(record, updated, window, plan, audit)
                )
            )
        finally:
            fixture.doCleanups()
