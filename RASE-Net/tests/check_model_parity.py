"""Run on a machine with torch installed: python tests/check_model_parity.py."""
import importlib.util,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from rase.models.egrc import EGRCCorrector
from rase.models.scar import SemanticContradictionField,FieldGuidedRefiner
torch.set_num_threads(1);torch.manual_seed(42)
root=Path(__file__).resolve().parents[1]/'experiments/internal_reference/reference'
for filename,name,cls in [('egrc.py','egrc',EGRCCorrector),('sacf_model.py','sacf',SemanticContradictionField)]:
    spec=importlib.util.spec_from_file_location(name,root/filename);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    ref=getattr(mod,cls.__name__)();new=cls();new.load_state_dict(ref.state_dict(),strict=True)
    assert all(torch.equal(v,new.state_dict()[k]) for k,v in ref.state_dict().items())
m=EGRCCorrector().eval();x=[torch.randn(1,768,32,32),torch.rand(1,1,512,512),torch.rand(1,1,512,512),torch.rand(1,4,1,256,256),torch.tensor([2])]
with torch.no_grad():
    y=m(*x);assert y.shape==(1,1,512,512) and torch.isfinite(y).all()
    assert torch.count_nonzero(y)==0
    m.decoder[-1].weight.normal_(std=.01);refmod=importlib.util.spec_from_file_location('reference_egrc',root/'egrc.py');ref=importlib.util.module_from_spec(refmod);refmod.loader.exec_module(ref)
    n=ref.EGRCCorrector().eval();n.load_state_dict(m.state_dict(),strict=True)
    assert torch.equal(m(*x),n(*x))
    x[3]=torch.zeros_like(x[3]);assert torch.count_nonzero(m(*x))==0
print('PASS: strict states, actual 512 EGRC forward, nonzero-residual reference parity, zero-exemplar gate')
