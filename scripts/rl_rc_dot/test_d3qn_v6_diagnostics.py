import unittest
import numpy as np
from .d3qn_v6_diagnostics import compare_matched_pair, policy_summary


class V6DiagnosticsTests(unittest.TestCase):
    def test_collapse_detection_and_pair_gap(self):
        q=np.zeros((10,25))
        q[:,9]=.1
        legal=np.ones((10,25),dtype=bool)
        out=policy_summary({},q,legal)
        self.assertEqual(out['dominant_action'],9)
        self.assertEqual(out['dominant_fraction'],1.)
        self.assertLess(out['pair_qgap_median'],0)

    def test_matched_prefix_and_rank_conflict(self):
        def case(a):
            return {'video':'train1','rate_kbps':120,'same_pre_action_state':True,
                    'q2_minus_q9':-.15, 'objective':.61 if a==2 else .60,
                    'frames':[{'poc':i,'action':2 if i < 3 else a if i==3 else 9,
                               'au_bits':100,'frame_f1':.8,'cumulative_bits':(i+1)*100}
                              for i in range(8)]}
        result=compare_matched_pair(case(2),case(9),action_step=3)
        self.assertTrue(result['ranking_conflict'])
        self.assertTrue(result['suffix_action_sequences_equal'])
        wrong=case(9)
        wrong['frames'][1]['au_bits']=99
        with self.assertRaisesRegex(ValueError,'Prefix mismatch'):
            compare_matched_pair(case(2),wrong,action_step=3)

    def test_rejects_missing_pre_action_equality(self):
        def case(a):
            return {'video':'x','rate_kbps':120,'same_pre_action_state':False,
                    'q2_minus_q9':-.1,'objective':.5,
                    'frames':[{'poc':0,'action':a,'au_bits':10,'frame_f1':1,'cumulative_bits':10}]}
        with self.assertRaisesRegex(ValueError,'Pre-action'):
            compare_matched_pair(case(2),case(9),action_step=0)
