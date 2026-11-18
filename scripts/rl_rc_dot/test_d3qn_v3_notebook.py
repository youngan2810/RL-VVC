"""Static Kaggle learning-check contract checks without Kaggle/GPU/VVenC."""
import ast
import json
from pathlib import Path
import re
import unittest

ROOT=Path(__file__).resolve().parents[2]
NOTEBOOK=ROOT/'notebooks/VVC_D3QN_V3_Mini16_Val4_LearningCheck_Kaggle.ipynb'


class KaggleV3NotebookContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data=json.loads(NOTEBOOK.read_text(encoding='utf-8'))
        cls.code=[''.join(x['source']) for x in cls.data['cells']
                  if x['cell_type']=='code']
        cls.concatenated='\n'.join(cls.code)

    def test_all_python_cells_compile_syntactically(self):
        self.assertGreaterEqual(len(self.code),8)
        for index,source in enumerate(self.code):
            with self.subTest(cell=index):
                ast.parse(source,filename=f'kaggle_v3_cell_{index}.py')

    def test_checkpoint_and_recovery_contract(self):
        text=self.concatenated
        for term in ('MILESTONES=(1024,2048,3072,4096,5120)',
                     'train_history.json','baseline.json','monitor_history.json',
                     'monitor_partial.json','learning_manifest.json',
                     'latest_checkpoint.pt','best_checkpoint.pt',
                     'roi_plus1_bg_plus2','roi_minus2_bg_zero',
                     'Val4','SOURCE_COMMIT',
                     '--d3qn-only','--n-step','--target-tau',
                     '--epsilon-decay-steps','--session-start-utc',
                     'native_preflight32','test_d3qn_native_v3',
                     'RESUME_SAFE_ZIP','export_safe(',
                     "args_for(phase,folder,"):
            with self.subTest(term=term):
                self.assertIn(term,text)
        self.assertNotIn("'--phase','test'",text)

    def test_vvc_poc_export_is_coding_order_safe(self):
        text=self.concatenated
        # VVC POC is presentation order; native AU traces are encoding order.
        self.assertNotIn("assert row['poc']==n",text)
        self.assertIn('seen_pocs',text)
        self.assertIn("row['coding_step']=n",text)
        self.assertIn('reuse_completed_attempt',text)

    def test_train16_val4_not_test30(self):
        text=self.concatenated
        self.assertIn('len(TRAIN_IDS)==16',text)
        self.assertIn('len(VAL_IDS)==4',text)
        self.assertIn('catalog.check_available((\'train\',\'val\'))',text)
        self.assertIn("cat",text) if False else None
        self.assertIn('source(video)',text)
        self.assertIn("'test_used':False",text)
        self.assertIn("assert set(r['train_video'] for r in training[:16])==set(TRAIN_IDS)",text)
        self.assertIn("'trained_d3qn'",text)

    def test_source_commit_is_a_fixed_sha(self):
        text=self.concatenated
        match=re.search(r'SOURCE_COMMIT="([a-f0-9]{40})"',text)
        self.assertIsNotNone(match,'Source must be pinned to an exact 40-char SHA')
        self.assertIn('SOURCE_COMMIT',text)
        self.assertNotIn('git pull',text)


if __name__=='__main__':
    unittest.main()
