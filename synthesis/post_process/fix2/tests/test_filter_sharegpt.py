import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from synthesis.post_process.fix2.filter_sharegpt import convert_record, main
from synthesis.post_process.fix2.quality import build_quality_reviewer_messages


class FilterShareGPTTests(unittest.TestCase):
    def record(self, index):
        return {"id": str(index), "images": [], "conversations": [
            {"from": "human", "value": "Question?"},
            {"from": "gpt", "value": "<thinking>Reason</thinking><answer>Answer</answer>"}]}

    def test_images_and_full_text(self):
        r = self.record(0)
        r['images'] = ['data:image/png;base64,YQ==']
        r['conversations'].insert(1, {'from': 'observation', 'value': 'x' * 65000 + '<image>END'})
        converted = convert_record(r, 0, Path('.'))
        self.assertEqual(converted['input_images'], [])
        payload = json.loads(build_quality_reviewer_messages(converted, full_transcript=True)[1].content[0]['text'])
        self.assertIn('END', payload['visible_trajectory'])
        self.assertNotIn('truncated', payload['visible_trajectory'])
        self.assertEqual(payload['image_evidence_map'][0]['occurrences'][0]['message_index'], 1)
        r['images'] = []
        with self.assertRaises(ValueError):
            convert_record(r, 0, Path('.'))

    def test_resume_export_and_config_guard(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'input.json'
            records = [self.record(i) for i in range(3)]
            source.write_text(json.dumps(records))
            args = ['--input', str(source), '--workdir', str(root / 'work'), '--workers', '2']
            calls = []
            def judge(record, index, *unused):
                calls.append(index)
                d = ['keep', 'filter', 'retry_pending'][index]
                return {'index': index, 'disposition': d, 'status': 'error' if index == 2 else 'ok', 'elapsed_seconds': 1}
            with patch('synthesis.post_process.fix2.filter_sharegpt.judge_record', side_effect=judge), contextlib.redirect_stdout(io.StringIO()):
                main(args + ['--limit', '2'])
                main(args + ['--resume', '--workers', '3'])
                self.assertCountEqual(calls, [0, 1, 2])
                main(args + ['--resume'])
                self.assertEqual(calls.count(2), 2)
                with self.assertRaises(ValueError):
                    main(args + ['--resume', '--model-alias', 'different'])
            self.assertEqual(json.loads((root / 'work/trajectories_sharegpt.json').read_text()), records[:1])
            self.assertEqual(json.loads(source.read_text()), records)
