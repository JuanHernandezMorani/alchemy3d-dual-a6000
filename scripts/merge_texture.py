#!/usr/bin/env python3
"""Reproduce the *documented* Codex formulae in FP32 without modifying source checkpoints.

Original Codex merge.py was not attached. SHA256 comparisons are reported; exact
bitwise parity is NOT promised until its original source is compared.
"""
import argparse, hashlib, json, math, os, time
from pathlib import Path
import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE=Path(__file__).resolve().parent.parent
CFG=json.loads((HERE/'config/revisions.json').read_text('utf8'))
ROOT=Path(os.getenv('LAB_ROOT','/workspace'))
METHODS=('ties','deltas','trim_mean','deltas_cap')
STEM='tex_slat_denoiser'

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def lora_delta(key, lof, expected_shape):
    if not key.endswith('.weight') or len(expected_shape)!=2:
        return None
    stem=key[:-len('.weight')]
    ak=stem+'.lora_A.weight'; bk=stem+'.lora_B.weight'
    keys=lof.keys()
    if ak not in keys and bk not in keys: return None
    if ak not in keys or bk not in keys: raise ValueError('LoRA A/B incompleto: '+key)
    a=lof.get_tensor(ak).float()
    b=lof.get_tensor(bk).float()
    if a.ndim!=2 or b.ndim!=2 or a.shape[0]!=b.shape[1] or (b.shape[0],a.shape[1])!=tuple(expected_shape):
        raise ValueError(f'Forma LoRA incompatible {key} {tuple(a.shape)} {tuple(b.shape)} != {expected_shape}')
    # alpha=96, rank=96 -> scale 1.0 for Segment.
    return torch.mm(b,a)

def trim_top(t,fraction=0.2):
    flat=t.reshape(-1)
    n=flat.numel()
    k=max(1,int(math.floor(fraction*n)))
    if k==n:return t
    absflat=flat.abs()
    cut=torch.topk(absflat,k,largest=True,sorted=False).values.min()
    keep=absflat>cut
    need=k-int(keep.sum().item())
    if need>0:
        # preserve the smallest flattened indices if abs-magnitude ties.
        idx=torch.nonzero(absflat==cut,as_tuple=False).reshape(-1)[:need]
        keep[idx]=True
    return torch.where(keep.reshape(t.shape),t,torch.zeros_like(t))

def calculate(method,w0,wi,ds):
    di=wi-w0
    if ds is None:ds=torch.zeros_like(w0)
    if method=='deltas' or method=='deltas_cap':
        merged=w0 + 0.25*(di+ds)
        if method=='deltas_cap':
            d=merged-w0
            base_norm=torch.linalg.vector_norm(w0.flatten().double()).item()
            delta_norm=torch.linalg.vector_norm(d.flatten().double()).item()
            if delta_norm>0 and delta_norm>0.10*base_norm:
                c=min(1.0,0.10*base_norm/delta_norm)
                merged=w0+c*d
                real_norm=torch.linalg.vector_norm((merged-w0).flatten().double()).item()
                if real_norm>0.10*base_norm and real_norm>0:
                    merged=w0+((0.999999*0.10*base_norm/real_norm)*c)*d
        return merged.float().contiguous()
    a=trim_top(di)
    b=trim_top(ds)
    if method=='ties':
        chosen=torch.sign(a+b)
        ma=(a!=0)&(torch.sign(a)==chosen)
        mb=(b!=0)&(torch.sign(b)==chosen)
    elif method=='trim_mean':
        ma=a!=0;mb=b!=0
    else: raise KeyError(method)
    denom=ma.to(torch.int8)+mb.to(torch.int8)
    numerator=torch.where(ma,a,0)+torch.where(mb,b,0)
    update=torch.where(denom>0,numerator/denom.clamp(min=1),torch.zeros_like(numerator))
    return (w0+0.25*update).float().contiguous()

def run(method):
    base=ROOT/'weights/Base/ckpts'/f'{STEM}.safetensors'
    inst=ROOT/'weights/Instruct/ckpts'/f'{STEM}.safetensors'
    segment=ROOT/'weights/Segment/ckpts'/f'{STEM}_lora.safetensors'
    for p in (base,inst,segment):
        if not p.is_file():raise FileNotFoundError(f'{p} — ejecutar download_models.py antes')
    outdir=ROOT/'merges'/method
    outdir.mkdir(parents=True,exist_ok=True)
    outpath=outdir/f'{STEM}.safetensors'
    (outdir/f'{STEM}.json').write_bytes((ROOT/'weights/Base/ckpts'/f'{STEM}.json').read_bytes())
    records=[]; state={}; seen=set(); total=0
    with safe_open(str(base),framework='pt',device='cpu') as bf, safe_open(str(inst),framework='pt',device='cpu') as inf, safe_open(str(segment),framework='pt',device='cpu') as sf:
        if set(bf.keys())!=set(inf.keys()):raise ValueError('Los keys de Base e Instruct no coinciden')
        for i,key in enumerate(sorted(bf.keys())):
            w0=bf.get_tensor(key).float(); wi=inf.get_tensor(key).float()
            if w0.shape!=wi.shape:raise ValueError('Shape incompatible: '+key)
            ds=lora_delta(key,sf,w0.shape)
            if ds is not None:seen.add(key)
            w=calculate(method,w0,wi,ds)
            if not torch.isfinite(w).all():raise FloatingPointError('No finito: '+key)
            changed=int(torch.count_nonzero(w!=w0).item());total+=changed
            records.append({'key':key,'shape':list(w.shape),'changed':changed,'lora':ds is not None})
            state[key]=w
            if (i+1)%100==0:print(f'[{method}] {i+1}/{len(bf.keys())}',flush=True)
        # Strictly account for every Segment A/B target.
        lora_a={k[:-len('.lora_A.weight')] for k in sf.keys() if k.endswith('.lora_A.weight')}
        lora_b={k[:-len('.lora_B.weight')] for k in sf.keys() if k.endswith('.lora_B.weight')}
        if lora_a!=lora_b:raise ValueError('Pares LoRA no completos')
        if lora_a!={k[:-len('.weight')] for k in seen}:
            missing=lora_a-{k[:-len('.weight')] for k in seen}
            raise ValueError(f'LoRA sin aplicar: {sorted(missing)[:12]}')
    save_file(state,str(outpath))
    del state
    digest=sha256(outpath)
    expected=CFG['source_expected_sha256'].get(method)
    result={'method':method,'formula_version':'independent-v1','source_weights':{'base':sha256(base),'instruct':sha256(inst),'segment':sha256(segment)},
      'path':str(outpath),'sha256':digest,'codex_expected_sha256':expected,'bitwise_match_codex':digest==expected,
      'num_tensors':len(records),'lora_targets':len(seen),'changed_weights':total,
      'warning':'No afirmar paridad bitwise si no coincide: el código merge.py original de Codex no se proporcionó.'}
    (outdir/'receipt.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    print('[OK]',method,'sha256',digest,'coincide con Codex:',result['bitwise_match_codex'],flush=True)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS));args=p.parse_args()
    for method in args.methods:run(method)
if __name__=='__main__':main()
