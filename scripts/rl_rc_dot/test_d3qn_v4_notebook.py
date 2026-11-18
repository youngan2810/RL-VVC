"""Static Kaggle notebook protocol checks; no GPU, codec or dataset needed."""
import json
import re
import unittest
from pathlib import Path

NOTEBOOK=(Path(__file__).resolve().parents[2]/
          'notebooks/VVC_D3QN_V4_TerminalReplay_Mini16_Val4_Kaggle.ipynb')
PINNED_CODE_SHA='8b8feb8774119bcf39b22aba477dc733b99214db'


class V4NotebookContracts(unittest.TestCase):
    def test_pinned_protocol_runner_sampling_gate_and_safe(self):
        nb=json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        self.assertEqual(nb['nbformat'],4)
        self.assertEqual(len(nb['cells']),11)
        sources=[''.join(cell['source']) for cell in nb['cells']]
        setup=sources[1]
        self.assertIn('SOURCE_COMMIT="'+PINNED_CODE_SHA+'"',setup)
        self.assertIn("SOURCE_BRANCH='experiment/d3qn-terminal-replay-v4'",setup)
        self.assertIn("PROTOCOL='native-masked-d3qn-qp/v4-terminal-aware'",setup)
        self.assertIn("RESUME_SAFE_ZIP=None",setup)
        self.assertIn('D3QN_V4_TERMINAL_REPLAY_MINI16_VAL4_SAFE.zip',setup)
        self.assertIn("'scripts.rl_rc_dot.test_d3qn_native_v4'",sources[3])
        runner=sources[4]
        self.assertIn('scripts.rl_rc_dot.d3qn_native_runner_v4',runner)
        self.assertIn("'--n-step','3','--terminal-quota','4'",runner)
        self.assertIn("'d3qn-v4-terminal-aware-mini16-val4-learning-check/1'",runner)
        self.assertIn("RUN/'value_audit.json'",runner)
        self.assertIn("TARGET_RATE=120",setup)
        self.assertIn("MILESTONES=(1024,2048,3072,4096,5120)",setup)
        self.assertIn("CONTROLS=('qpa_anchor','global_minus2','roi_plus1_bg_plus2','roi_minus2_bg_zero')",setup)
        self.assertNotIn('scripts.rl_rc_dot.d3qn_native_runner_v3',runner)
        audit=sources[8]
        self.assertIn("CHECKPOINT",audit.upper())
        self.assertIn('D3QN_V4_VALUE_AUDIT_DONE',audit)
        self.assertIn("export_safe(",audit)
        self.assertIn("terminal_mean_target_minus_q",audit)
        self.assertIn("mean_behavior_mc_minus_q",audit)
        self.assertTrue(any('VVDEC_PARITY_PASS' in s for s in sources))

    def test_no_test_split_training_or_validation_path(self):
        sources=[''.join(c['source']) for c in json.loads(
            NOTEBOOK.read_text(encoding='utf-8'))['cells']]
        self.assertIn("catalog.check_available(('train','val'))",sources[1])
        self.assertIn("test_used':False",sources[4])
        self.assertNotIn("catalog.check_available(('train','val','test'))",sources[1])


if __name__=='__main__':
    unittest.main()
