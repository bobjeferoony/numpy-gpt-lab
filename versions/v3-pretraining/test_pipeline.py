"""Integration checks for pretraining, transfer, early stopping and provenance."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from gpt import GPT


class PipelineTests(unittest.TestCase):
    def test_stages_and_best_restoration(self):
        script=Path(__file__).with_name('pipeline_train.py')
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            (root/'python-train.bin').write_bytes(b'def f(x):\n    return x+1\n'*20)
            (root/'python-validation.bin').write_bytes(b'def g(x):\n    return x-1\n'*20)
            pre=root/'pre.npz'
            result=subprocess.run([sys.executable,str(script),'--stage','pretrain','--data',str(root),
                '--out',str(pre),'--steps','3','--width','8','--layers','1','--context','64',
                '--lr','1e-30','--patience','1','--eval-every','1'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn('Early stopping',result.stdout)
            with np.load(root/'pre.best.npz') as best, np.load(root/'pre.last.npz') as last:
                self.assertEqual(int(best['step']),0)
                self.assertEqual(int(last['step']),1)
                for key,p in GPT.load(pre).p.items():
                    np.testing.assert_array_equal(p.data,best[key])
            meta=json.loads((root/'pre.run.json').read_text())
            self.assertEqual(len(meta['source_hashes']['python-train.bin']),64)
            for name,family in [('bugs-train.jsonl','a'),('bugs-validation.jsonl','b')]:
                (root/name).write_text(json.dumps(dict(buggy='def f():\n return 0',fixed='def f():\n return 1',family=family))+'\n')
            fine=root/'fine.npz'
            result=subprocess.run([sys.executable,str(script),'--stage','finetune','--data',str(root),
                '--init',str(pre),'--out',str(fine),'--steps','1','--eval-every','1'],capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(GPT.load(pre).config,GPT.load(fine).config)
            self.assertEqual(len((root/'fine.history.jsonl').read_text().splitlines()),2)
            result=subprocess.run([sys.executable,str(script),'--stage','finetune','--data',str(root),
                '--init',str(pre),'--out',str(fine),'--steps','1'],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('Output already exists',result.stderr)


if __name__=='__main__':
    unittest.main()
