#!/usr/bin/env python3
"""Fetch immutable HF revisions. Never pulls the 6.44 TB training dataset."""
import argparse, json, os, shutil, sys, time
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download, HfApi

HERE=Path(__file__).resolve().parent.parent
CFG=json.loads((HERE/'config/revisions.json').read_text('utf8'))
ROOT=Path(os.getenv('LAB_ROOT','/workspace'))

def space():
    du=shutil.disk_usage(ROOT)
    print(f'[STORAGE] disponible {du.free/1024**3:.2f} GiB en {ROOT}', flush=True)
    return du.free

def download(repo, revision, out, patterns=None):
    print(f'[HF] {repo}@{revision or "main"} -> {out}', flush=True)
    out.mkdir(parents=True,exist_ok=True)
    return snapshot_download(repo_id=repo, revision=revision, local_dir=str(out), allow_patterns=patterns, max_workers=3)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--profile',choices=['minimal','full'],default='minimal')
    ap.add_argument('--aux',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--min-free-gib',type=int,default=15)
    args=ap.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    if space()<args.min_free_gib*(1024**3): sys.exit('Insuficiente espacio de reserva.')
    manifest={'start_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()), 'profile':args.profile, 'downloaded':[]}
    for name,(repo,rev) in CFG['model_revisions'].items():
        if args.profile=='minimal' and name not in {'Base','Instruct','Segment'}:continue
        patterns=None
        if args.profile=='minimal' and name=='Base':
            patterns=['ckpts/tex_slat_denoiser.safetensors','ckpts/tex_slat_denoiser.json','pipeline.json']
        if args.profile=='minimal' and name=='Segment':
            patterns=['ckpts/tex_slat_denoiser_lora.safetensors','ckpts/tex_slat_denoiser_lora.json','pipeline.json']
        download(repo,rev,ROOT/'weights'/name,patterns)
        manifest['downloaded'].append(dict(name=name,repo=repo,revision=rev,patterns=patterns))
        space()
    if args.aux:
        # Pinned external TRELLIS.2 encoder/decoders required by Instruct, plus original TRELLIS SS VAE.
        required={
            'microsoft/TRELLIS.2-4B':[
                'ckpts/shape_enc_next_dc_f16c32_fp16.*', 'ckpts/shape_dec_next_dc_f16c32_fp16.*',
                'ckpts/tex_enc_next_dc_f16c32_fp16.*', 'ckpts/tex_dec_next_dc_f16c32_fp16.*',
                'pipeline.json',
            ],
            'microsoft/TRELLIS-image-large':[
                'ckpts/ss_enc_conv3d_16l8_fp16.*','ckpts/ss_dec_conv3d_16l8_fp16.*',
            ],
        }
        for repo,patterns in required.items():
            rev=CFG['external_revisions'].get(repo) or HfApi().model_info(repo).sha
            download(repo,rev,ROOT/'weights'/'external'/repo.replace('/','--'),patterns)
            manifest['downloaded'].append(dict(name='external',repo=repo,revision=rev,patterns=patterns))
        # Prefetch the text-and-vision model used by Alchemy3D-Instruct.
        repo='Qwen/Qwen3.5-2B'
        revision=CFG['external_revisions'][repo]
        # Transformers uses its shared hub cache, avoiding a second copy in weights/.
        local= snapshot_download(repo_id=repo,revision=revision,max_workers=3)
        manifest['downloaded'].append(dict(name='text_encoder',repo=repo,revision=revision,cache=local))
        space()
    manifest['end_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())
    (ROOT/'download_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf8')
    print('[OK] Descargas completas, manifest:',ROOT/'download_manifest.json')
if __name__=='__main__':main()
