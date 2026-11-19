"""Verify the V6 Kaggle training notebook's immutable provenance and gates."""
import json
import os
from pathlib import Path
import re
import subprocess
import unittest

NOTEBOOK = Path(os.environ.get('V6_HANDOFF_NOTEBOOK', str(Path(__file__).resolve().parents[2] / 'notebooks/VVC_D3QN_V6_ActionConditioned_Mini16_Val4_Kaggle.ipynb')))
OLD_V5_SOURCE = 'f10ec5de7c9a8b2bf3db561f6df46e9a8643eced'


class V6NotebookContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document=json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.code=[''.join(cell['source']) for cell in cls.document['cells'] if cell['cell_type']=='code']
        cls.all_code='\n'.join(cls.code)
        match=re.search(r"^TRAINING_SOURCE_COMMIT='([a-f0-9]{40})'$", cls.all_code, re.M)
        if not match:raise ValueError('Notebook has no immutable 40-character training source SHA')
        cls.pinned_source=match.group(1)

    def test_compiles_all_cells(self):
        self.assertGreaterEqual(len(self.code),7)
        for i,source in enumerate(self.code):compile(source,f'V6_cell_{i}','exec')

    def test_immutable_training_commit_is_not_historical_v5(self):
        self.assertTrue(re.fullmatch('[a-f0-9]{40}',self.pinned_source))
        self.assertNotEqual(self.pinned_source,OLD_V5_SOURCE)
        expected=os.environ.get('V6_EXPECTED_SOURCE_COMMIT')
        if expected:
            self.assertEqual(self.pinned_source,expected)
            checkout=Path(__file__).resolve().parents[2]
            actual=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
            self.assertEqual(actual,expected,'Kaggle tests must run on exact pinned Git checkout')
        self.assertIn("SOURCE_BRANCH='experiment/d3qn-v6-action-conditioned-reconstruction'",self.all_code)
        self.assertIn("'checkout','--detach',TRAINING_SOURCE_COMMIT",self.all_code)
        self.assertIn("git','rev-parse','HEAD'",self.all_code)
        self.assertNotIn('PIN_AFTER_SOURCE_COMMIT',self.all_code)

    def test_training_native_validation_preflight_and_safe(self):
        for expected in ('test_d3qn_v6_reconstruction','test_d3qn_v6_diagnostics',
                         'scripts.rl_rc_dot.d3qn_v6_native_runner',
                         "'--n-step',8", "'--action-mix',.45", "'--boundary-mix',.10",
                         "'REAL_VVC_STATE_ACTION_AU_DECODE_PASS'",
                         "'roi_minus2_bg_zero'", "'roi_minus1_bg_plus2'",
                         'D3QN_V6_ACTION_CONDITIONED_MINI16_VAL4_SAFE.zip',
                         'train_history.json','monitor_history.json','checkpoint.pt'):
            self.assertIn(expected,self.all_code)
        self.assertIn("'test_used':False",self.all_code)
        self.assertIn('RESUME_INPUT=None',self.all_code)


if __name__=='__main__':unittest.main()
