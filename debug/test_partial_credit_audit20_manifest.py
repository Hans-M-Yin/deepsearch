import json
from pathlib import Path
import unittest


class Audit20ManifestTest(unittest.TestCase):
    def test_fixed_holdout_balance_and_provenance(self):
        root = Path(__file__).resolve().parent
        new = json.loads((root / 'partial_credit_audit20_manifest.json').read_text())
        old = json.loads((root / 'partial_credit_audit12_manifest.json').read_text())
        rows = new['samples']
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(r['correct'] for r in rows), 10)
        self.assertEqual(len({r['id'] for r in rows}), 20)
        self.assertFalse({r['id'] for r in rows} & {r['id'] for r in old['samples']})
        for row in rows:
            with self.subTest(data_id=row['id']):
                self.assertTrue(row['selection_reason'])
                self.assertTrue(row['statements'])
                self.assertEqual(len(row['rollout_sha256']), 64)
                self.assertEqual(row['raw'], row['historical_reward_components']['partial_raw_bonus'])
                self.assertFalse(row['historical_statement_verdicts_available'])


if __name__ == '__main__':
    unittest.main()
