#!/usr/bin/env python3
"""Create an editable pipeline config pointing to ONE copy of external decoders.
Original HF snapshots are never modified."""
import json,os,shutil
from pathlib import Path
ROOT=Path(os.getenv('LAB_ROOT','/workspace'))
source=ROOT/'weights/Instruct'
target=ROOT/'overlays/Instruct'
config=json.loads((source/'pipeline.json').read_text('utf8'))
extra=config['args']['models']
for name,path in list(extra.items()):
    if not isinstance(path,str):continue
    for repo in ('microsoft/TRELLIS.2-4B','microsoft/TRELLIS-image-large'):
        if path.startswith(repo+'/'):
            rel=path[len(repo)+1:]
            p=ROOT/'weights/external'/repo.replace('/','--')/rel
            if not p.with_suffix('.safetensors').is_file() or not p.with_suffix('.json').is_file():
                raise FileNotFoundError(f'Faltan pesos auxiliares {name}: {p} (.safetensors + .json)')
            extra[name]=str(p)
if target.exists() and not target.is_dir():raise RuntimeError(f'Overlay no es directorio: {target}')
target.mkdir(parents=True,exist_ok=True)
link=target/'ckpts'
if not link.exists():link.symlink_to(source/'ckpts',target_is_directory=True)
# Exact pinned path to Qwen, not floating HF branch.
manifest=json.loads((ROOT/'download_manifest.json').read_text('utf8'))
qwen=next((x.get('cache') for x in manifest['downloaded'] if x.get('name')=='text_encoder'),None)
if not qwen or not Path(qwen).is_dir():raise RuntimeError('Falta Qwen cache predescargada')
config['args']['text_cond_model']['args']['text_model']=qwen
(target/'pipeline.json').write_text(json.dumps(config,indent=2),encoding='utf8')
print('[OK] Overlay creado con checkpoints externos compartidos:',target)
for n,v in extra.items():print(' ',n,v)
