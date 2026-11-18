"""Static V5 Kaggle notebook contract checks; no GPU or BDD100K required."""
import json
import unittest
from pathlib import Path

NOTEBOOK=Path(__file__).resolve().parents[2]/"notebooks/VVC_D3QN_V5_CodingCycle_Mini16_Val4_Kaggle.ipynb"
PINNED_SOURCE="f4346f2d682a95eda513218d008a9ddfb806f1b8"

class V5NotebookContracts(unittest.TestCase):
    def source(self):
        n=json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        self.assertEqual(n["nbformat"],4)
        self.assertEqual(len(n["cells"]),11)
        return n["cells"],["".join(x["source"]) for x in n["cells"]]

    def test_all_executable_cells_compile(self):
        cells,sources=self.source()
        for i,(cell,s) in enumerate(zip(cells,sources)):
            if cell["cell_type"]=="code":
                compile(s,f"V5_Kaggle_cell_{i}.py","exec")

    def test_source_pin_state_schema_and_no_old_runner(self):
        _,x=self.source()
        self.assertIn('SOURCE_COMMIT="'+PINNED_SOURCE+'"',x[1])
        self.assertIn("SOURCE_BRANCH='experiment/d3qn-v5-coding-cycle-state'",x[1])
        self.assertIn("PROTOCOL='native-masked-d3qn-qp/v5-coding-cycle'",x[1])
        self.assertIn("RESUME_SAFE_ZIP=None",x[1])
        self.assertIn("D3QN_V5_CODING_CYCLE_MINI16_VAL4_SAFE.zip",x[1])
        self.assertIn("'scripts.rl_rc_dot.test_d3qn_native_v5'",x[3])
        self.assertIn("scripts.rl_rc_dot.d3qn_native_runner_v5",x[4])
        self.assertNotIn("scripts.rl_rc_dot.d3qn_native_runner_v4",x[4])
        self.assertIn("'--n-step','3','--terminal-quota','4'",x[4])
        self.assertIn("d3qn-v5-coding-cycle-mini16-val4-learning-check/1",x[4])
        self.assertIn("MILESTONES=(1024,2048,3072,4096,5120)",x[1])
        self.assertIn("TARGET_RATE=120",x[1])
        self.assertIn("CONTROLS=('qpa_anchor','global_minus2','roi_plus1_bg_plus2','roi_minus2_bg_zero')",x[1])
        self.assertIn("catalog.check_available(('train','val'))",x[1])
        self.assertNotIn("catalog.check_available(('train','val','test'))",x[1])
        self.assertIn("DuelingQNetworkV5()",x[8])
        self.assertIn("boundary_mean_abs_td",x[8])
        self.assertIn("ckpt['state_schema']=='v5-coding-cycle-global28'",x[8])
        self.assertIn("D3QN_V5_VALUE_AUDIT_DONE",x[8])
        self.assertTrue(any("ALL_PREFLIGHTS_PASS" in s for s in x))
        self.assertTrue(any("VVDEC_PARITY_PASS" in s for s in x))
        self.assertTrue(any("export_safe(" in s for s in x))
        self.assertTrue(any("test_used':False" in s for s in x))


if __name__=="__main__":
    unittest.main()
