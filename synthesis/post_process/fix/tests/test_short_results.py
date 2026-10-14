import unittest
from random import Random
from types import SimpleNamespace
from unittest.mock import Mock
from synthesis.post_process.fix.schema import Candidate, Config, Plan
from synthesis.post_process.fix.candidate_sets import build_plan, finish_recovery
from synthesis.post_process.fix.answer_consistency import check_answer


class ShortResultsTests(unittest.TestCase):
    def positive(self, i):
        return Candidate('old_image_' + str(i), {'image_id': 'image_'+str(i), 'title': 'Target', 'source':'site', 'source_page_id':'page_'+str(i)},
                         historical=True, image_paths=['image.jpg'], category='CONSISTENT', verdict='YES')

    def test_one_to_four_real_candidates(self):
        config = Config(branch_weights={'direct':1,'contrast':0,'convergent':0,'recovery':0})
        for count in range(1,5):
            cs = [self.positive(i) for i in range(count)]
            plan = build_plan(SimpleNamespace(protected_ids=[]), cs, [], {'anchor_image_id':'image_0'}, {}, config, Random(1))
            self.assertEqual(len(plan.first), count)

    def test_two_references_without_fillers(self):
        cs = [self.positive(i) for i in range(2)]
        plan = build_plan(SimpleNamespace(protected_ids=[]), cs, [],
                          {'anchor_image_ids':['image_0','image_1']}, {}, Config(), Random(1))
        self.assertEqual(len(plan.first),2)

    def test_retry_single_result(self):
        c = self.positive(0)
        plan = Plan('recovery', [], [])
        finish_recovery(plan, [c], c, Random(1))
        self.assertEqual(plan.second, [c])

    def test_answer_check_runs_only_for_same_entity(self):
        model = Mock()
        c = self.positive(0)
        plan = Plan('direct', [c], [c])
        record = {'conversations':[{'from':'gpt','value':'<answer>The photograph is an exact match.</answer>'}]}
        self.assertTrue(check_answer(record, plan, model, Config())['skipped'])
        model.ask.assert_not_called()
        c.category = 'SAME_ENTITY'
        model.ask.return_value = {'compatible':False,'reason':'Answer claims exact photo correspondence', 'conflicting_quotes':['exact match']}
        self.assertFalse(check_answer(record, plan, model, Config())['compatible'])
        self.assertEqual(record['conversations'][0]['value'], '<answer>The photograph is an exact match.</answer>')
