import unittest

from synthesis.sft import api_tools, tools
from synthesis.sft.sft_response_verifier import PROMPT_ORACLE_EVIDENCE_REPAIR


class ResourceIdPromptTests(unittest.TestCase):
    def test_agent_prompts_describe_compact_resource_ids(self) -> None:
        self.assertIn("source_page_id", api_tools.MANUAL_REACT_PROTOCOL)
        self.assertIn("image_id", api_tools.RESPONSES_TOOL_USE_TIPS)
        read_url_definition = next(
            item for item in tools.get_tool_definitions()
            if item["function"]["name"] == "read_url"
        )
        self.assertIn("resource_id", read_url_definition["function"]["parameters"]["properties"])

    def test_sft_auditor_examples_use_compact_ids(self) -> None:
        self.assertIn('"image_id": "image_a1b2c3d4"', PROMPT_ORACLE_EVIDENCE_REPAIR)
        self.assertNotIn('"image_url": "..."', PROMPT_ORACLE_EVIDENCE_REPAIR)
        self.assertNotIn('"link": "..."', PROMPT_ORACLE_EVIDENCE_REPAIR)


if __name__ == "__main__":
    unittest.main()
