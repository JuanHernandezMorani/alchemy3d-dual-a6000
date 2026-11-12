#!/usr/bin/env python3
import argparse,json,os,platform,shutil,sys,time
from pathlib import Path
import torch
root=Path(os.getenv('LAB_ROOT','/workspace'))
ap=argparse.ArgumentParser();ap.add_argument('--strict',action='store_true');args=ap.parse_args()
report={'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'hostname':platform.node(),
    'python':sys.version.split()[0],'torch':torch.__version__,'torch_cuda':torch.version.cuda,'cuda':torch.cuda.is_available()}
if torch.cuda.is_available():
    p=torch.cuda.get_device_properties(0)
    report['gpu']={'name':p.name,'vram_gib':round(p.total_memory/1024**3,2),'capability':torch.cuda.get_device_capability(0)}
report['disk_free_gib']=round(shutil.disk_usage(root).free/1024**3,2)
report['ram_gib']=None
try:
    import psutil;report['ram_gib']=round(psutil.virtual_memory().total/1024**3,2)
except Exception: pass
for item in ['o_voxel','flex_gemm','flash_attn','nvdiffrast','PIL','transformers']:
    try:
        __import__(item);report[item]='OK'
    except Exception as e:report[item]='ERROR: '+str(e)[:350]
(root/'results').mkdir(parents=True,exist_ok=True)
(root/'results/preflight.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf8')
print(json.dumps(report,indent=2,ensure_ascii=False))
if args.strict and (not report['cuda'] or any(report[x]!='OK' for x in ['o_voxel','flex_gemm','flash_attn'])):sys.exit(1)
