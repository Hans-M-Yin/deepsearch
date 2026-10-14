import json
import unittest
from synthesis.post_process.fix2.quality import build_quality_reviewer_messages


class QualityImageTests(unittest.TestCase):
    def test_dedup_and_chronology(self):
        image = 'data:image/png;base64,YWJj'
        later = 'data:image/png;base64,ZGVm'
        record = {
            'question': 'Which object?', 'gold_answer': 'PRIVATE',
            'input_images': [image],
            'raw_messages': [
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': image}}]},
                {'role': 'tool', 'content': [{'type': 'text', 'text': 'image_id=img_1'}, {'type': 'image_url', 'image_url': {'url': later}}]},
                {'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': later}}]},
                {'role': 'tool', 'content': json.dumps({'results': [{'image_url': 'https://unopened.example/img.png'}]})},
            ],
        }
        content = build_quality_reviewer_messages(record)[1].content
        payload = json.loads(content[0]['text'])
        self.assertEqual(len([p for p in content if p['type'] == 'image_url']), 2)
        self.assertEqual([p['message_index'] for p in payload['image_evidence_map'][1]['occurrences']], [1, 2])
        self.assertNotIn('base64', payload['visible_trajectory'])
        self.assertNotIn('PRIVATE', json.dumps(content))
        self.assertIn('image_id=img_1', payload['visible_trajectory'])

    def test_text_only_question_with_tool_image(self):
        content = build_quality_reviewer_messages({'messages': [{'role': 'tool', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,YQ=='}}]}]})[1].content
        self.assertEqual(len([p for p in content if p['type'] == 'image_url']), 1)

    def test_no_images_allowed(self):
        self.assertEqual(len(build_quality_reviewer_messages({'question': 'Why?'})[1].content), 1)

    def test_missing_image_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            build_quality_reviewer_messages({'input_images': ['/nonexistent/quality_image.png']})
