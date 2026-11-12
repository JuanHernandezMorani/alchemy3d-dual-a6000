"""Small deterministic unit tests; no real 4.8 GiB checkpoints are used."""
import importlib.util
from pathlib import Path
import torch

src=Path(__file__).resolve().parent.parent/'scripts/merge_texture.py'
spec=importlib.util.spec_from_file_location('merge_texture_test',src)
m=importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)

def test_no_delta():
    w0=torch.tensor([[1.,2.],[3.,4.]])
    for meth in m.METHODS:
        out=m.calculate(meth,w0,w0,None)
        assert torch.equal(out,w0),(meth,out)

def test_delta():
    w0=torch.tensor([[1.,1.],[1.,1.]])
    wi=w0+torch.tensor([[4.,-2.],[0.,3.]])
    ds=torch.tensor([[2.,1.],[0.,0.]])
    res=m.calculate('deltas',w0,wi,ds)
    assert torch.allclose(res,w0+0.25*(wi-w0+ds))
    capped=m.calculate('deltas_cap',w0,wi,ds)
    assert torch.linalg.vector_norm((capped-w0).double())<=0.100001*torch.linalg.vector_norm(w0.double())

def test_ties():
    w0=torch.zeros((1,10))
    wi=torch.tensor([[10.,9.,8.,7.,6.,5.,4.,3.,2.,1.]])
    ds=torch.tensor([[1.,10.,-9.,8.,7.,-6.,5.,4.,3.,2.]])
    result=m.calculate('ties',w0,wi,ds)
    assert torch.isfinite(result).all()
    assert torch.count_nonzero(result).item()>0
    assert torch.count_nonzero(m.trim_top(wi-w0)).item()==2
    assert torch.count_nonzero(m.trim_top(ds)).item()==2

def test_tie_order():
    t=torch.tensor([1.,1.,1.,1.,1.])
    out=m.trim_top(t,0.2)
    assert torch.equal(out,torch.tensor([1.,0.,0.,0.,0.]))

if __name__=='__main__':
    test_no_delta();test_delta();test_ties();test_tie_order()
    print('OK: cuatro tests sintéticos de merges; no prueban calidad ni ejecución 3D.')
