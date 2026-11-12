#!/usr/bin/env python3
"""Nonsemantic reproducibility and mesh integrity checks. No 1-10 subjective scores."""
import argparse,csv,json,os
from pathlib import Path
from itertools import combinations
import numpy as np
import trimesh
root=Path(os.getenv('LAB_ROOT','/workspace'))
p=argparse.ArgumentParser();p.add_argument('--run',default='');args=p.parse_args()
run=Path(args.run) if args.run else max((root/'results/cuda_case04').glob('*'),key=lambda x:x.name)
records=json.loads((run/'results.json').read_text('utf8'))
rows=[]
for record in records:
 row={k:record.get(k,'') for k in ['method','prompt_name','status','elapsed_sec','peak_cuda_allocated_gib','checkpoint_sha256']}
 if record['status']!='ok':rows.append(row);continue
 path=Path(record['glb']['path'])
 try:
  scene=trimesh.load(str(path),force='scene',process=False)
  meshes=list(scene.geometry.values())
  row.update(meshes=len(meshes),vertices=sum(len(m.vertices) for m in meshes),faces=sum(len(m.faces) for m in meshes),
    watertight_all=all(m.is_watertight for m in meshes), is_empty=not bool(meshes))
 except Exception as e:row['parse_error']=str(e)
 rows.append(row)
fields=sorted(set().union(*(r.keys() for r in rows)))
with (run/'verification.csv').open('w',newline='',encoding='utf8') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print(json.dumps(rows,ensure_ascii=False,indent=2))
print('NOTA: La inspección automática no prueba obediencia semántica ni paridad CPU/CUDA.')
