import json
import unittest
from synthesis.post_process.fix.tool_schema import expose_image_argument


class ToolSchemaTests(unittest.TestCase):
    def block(self, properties=None):
        return (
            "<tools>"
            + json.dumps(
                [
                    {
                        "type": "function",
                        "function": {
                            "name": "i2i_search",
                            "parameters": {"properties": properties or {}},
                        },
                    }
                ]
            )
            + "</tools>"
        )

    def test_documentation_example_is_not_definition(self):
        prefix = "Available tools inside `<tools></tools>`.\n<tools>example</tools>\n"
        result = expose_image_argument(prefix + self.block() + "\nTail")
        self.assertTrue(result.startswith(prefix))
        self.assertTrue(result.endswith("\nTail"))
        self.assertIn('"image":', result)

    def test_missing_or_ambiguous_definition_rejected(self):
        for system in [
            "<tools></tools>",
            "<tools>{}</tools>",
            self.block() + self.block(),
        ]:
            with self.assertRaises(ValueError):
                expose_image_argument(system)

    def test_existing_image_schema_preserved_and_idempotent(self):
        result = expose_image_argument(
            self.block({"image": {"type": "string", "description": "Original"}})
        )
        self.assertIn("Original", result)
        self.assertEqual(expose_image_argument(result), result)
