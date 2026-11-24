import ast, hashlib, json, subprocess, sys, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ReleaseTests(unittest.TestCase):
    def test_python_syntax(self):
        for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf8'),filename=str(p))
    def test_sources(self):
        for x in json.loads((ROOT/'provenance/source_manifest.json').read_text()):
            self.assertEqual(hashlib.sha256((ROOT/x['file']).read_bytes()).hexdigest(),x['packaged_sha256'])
    def test_cli(self):
        out=subprocess.check_output([sys.executable,'-m','rase','list'],cwd=ROOT,text=True)
        self.assertIn('refiner',out)
        subprocess.check_call([sys.executable,'-m','rase','sacf','--config','configs/btcv.example.json','--dry-run'],cwd=ROOT,stdout=subprocess.DEVNULL)
    def test_no_binary_experiment_artifacts(self):
        self.assertFalse(any(p.suffix in {'.pt','.pth','.npz','.npy','.sqlite3'} for p in ROOT.rglob('*')))
