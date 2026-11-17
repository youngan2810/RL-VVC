"""Live JSONL audit reader: complete current picture plus partial future row."""
import json,tempfile,unittest
from pathlib import Path
import numpy as np
from scripts.rl_rc_dot.action_audit import ActionAudit

class ActionTraceStream(unittest.TestCase):
    def test_partial_future_record_is_retried_without_loss_or_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'cus.jsonl';f.write_bytes(b'{"poc":0}\n{"poc":')
            audit=ActionAudit(p)
            self.assertEqual(list(audit.rows('cus.jsonl')),[{'poc':0}])
            self.assertEqual(audit.offsets['cus.jsonl'],10)
            self.assertEqual(list(audit.rows('cus.jsonl')),[])
            with f.open('ab') as out:out.write(b'1}\n')
            self.assertEqual(list(audit.rows('cus.jsonl')),[{'poc':1}])
            self.assertEqual(list(audit.rows('cus.jsonl')),[])

    def test_snapshot_excludes_concurrent_append(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'cus.jsonl';f.write_bytes(b'{"poc":0}\n')
            audit=ActionAudit(p);rows=audit.rows('cus.jsonl')
            self.assertEqual(next(rows),{'poc':0})
            with f.open('ab') as out:out.write(b'{"poc":1}\n')
            self.assertEqual(list(rows),[])
            self.assertEqual(list(audit.rows('cus.jsonl')),[{'poc':1}])

    def test_incomplete_current_picture_still_fails_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'qgs.jsonl').write_bytes(b'')
            (p/'cus.jsonl').write_bytes(b'{"poc":0')
            with self.assertRaisesRegex(ValueError,'Missing winning CU area'):
                ActionAudit(p).consume({'poc':0,'mean_qp':30},np.zeros((20,30)))

    def test_complete_current_picture_and_partial_future_picture(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);(p/'qgs.jsonl').write_bytes(b'')
            row=lambda poc:json.dumps({'poc':poc,'qp':30,'w':480,'h':320})+'\n'
            f=p/'cus.jsonl';future=row(1).encode();cut=9
            f.write_bytes(row(0).encode()+future[:cut]);audit=ActionAudit(p)
            first=audit.consume({'poc':0,'mean_qp':30},np.zeros((20,30)))
            self.assertEqual(first['final_cus'],1)
            with f.open('ab') as out:out.write(future[cut:])
            second=audit.consume({'poc':1,'mean_qp':30},np.zeros((20,30)))
            self.assertEqual(second['final_cus'],1)
            self.assertEqual(audit.summary()['frames'],2)

    def test_corrupt_complete_row_and_truncation_remain_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);f=p/'cus.jsonl';f.write_bytes(b'{bad}\n');audit=ActionAudit(p)
            with self.assertRaises(json.JSONDecodeError):list(audit.rows('cus.jsonl'))
            f.write_bytes(b'{"poc":0}\n');list(audit.rows('cus.jsonl'));f.write_bytes(b'')
            with self.assertRaisesRegex(ValueError,'truncated'):list(audit.rows('cus.jsonl'))

if __name__=='__main__':unittest.main()
