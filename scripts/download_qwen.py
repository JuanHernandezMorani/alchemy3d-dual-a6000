#!/usr/bin/env python3
"""Download exact Qwen GGUF file once into the persisted workspace; no image bloat."""
import argparse
import os
from pathlib import Path
from huggingface_hub import hf_hub_download, HfApi

REPO = 'unsloth/Qwen3.8-27B-GGUF'
NAME = 'Qwen3.8-27B-UD-Q4_K_XL.gguf'

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo',default=REPO)
    p.add_argument('--filename',default=NAME)
    p.add_argument('--revision',default=None)
    p.add_argument('--destination',default='/workspace/qwen')
    args=p.parse_args()
    out=Path(args.destination)
    out.mkdir(parents=True,exist_ok=True)
    sha=args.revision or HfApi().model_info(args.repo).sha
    print('HF repo:',args.repo,'revision:',sha,'filename:',args.filename, flush=True)
    path=hf_hub_download(repo_id=args.repo,filename=args.filename,revision=sha,local_dir=str(out))
    print('READY:',path,'bytes:',os.stat(path).st_size,flush=True)
    print('Set QWEN_MODEL='+str(Path(path).resolve()),flush=True)

if __name__=='__main__':main()
