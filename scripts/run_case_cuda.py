#!/usr/bin/env python3
"""CUDA end-to-end Alchemy3D-Instruct test on the official source apple.

All methods share original Instruct SS + Shape stages and the same text encoder.
Only the texture denoiser weights change. This is NOT a geometry merge test.
"""
import argparse,gc,hashlib,json,os,time,traceback
from pathlib import Path
import torch
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

ROOT=Path(os.getenv('LAB_ROOT','/workspace'))
HERE=Path(__file__).resolve().parent.parent
METHODS=('instruct_control','base_texture_control','ties','deltas','trim_mean','deltas_cap')
PROMPTS={'green':'Change the stem color to a vibrant green.','blue':'Change the stem color to a vibrant blue.'}

def filehash(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def checkpoint(method):
 if method=='instruct_control':return ROOT/'weights/Instruct/ckpts/tex_slat_denoiser.safetensors'
 if method=='base_texture_control':return ROOT/'weights/Base/ckpts/tex_slat_denoiser.safetensors'
 return ROOT/'merges'/method/'tex_slat_denoiser.safetensors'

def load_pipe():
 from alchemy3d.pipelines import Alchemy3DInstructPipeline
 import o_voxel
 root=(ROOT/'overlays/Instruct') if (ROOT/'overlays/Instruct/pipeline.json').is_file() else (ROOT/'weights/Instruct')
 if not root.is_dir():raise RuntimeError('Falta weights/Instruct; usar download_models.py')
 pipe=Alchemy3DInstructPipeline.from_pretrained(str(root))
 # Normal original GPU path, no CPU substitutes.
 pipe.cuda()
 pipe.models['sparse_structure_flow_model'].eval()
 pipe.models['shape_slat_flow_model'].eval()
 pipe.models['tex_slat_flow_model'].eval()
 return pipe,o_voxel

def export(o_voxel,output,path,args):
 mesh=o_voxel.postprocess.to_glb(
     vertices=output.vertices,faces=output.faces,
     attr_volume=output.attrs,coords=output.coords,attr_layout=output.layout,
     voxel_size=output.voxel_size,aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
     decimation_target=args.decimation_target,texture_size=args.texture_size,
     remesh=not args.no_remesh,remesh_band=1,remesh_project=0,verbose=False)
 path.parent.mkdir(parents=True,exist_ok=True)
 mesh.export(str(path))
 return dict(path=str(path),sha256=filehash(path),bytes=path.stat().st_size)

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--methods',nargs='+',default=list(METHODS),choices=METHODS)
 p.add_argument('--prompts',nargs='+',default=['green','blue'],choices=list(PROMPTS))
 p.add_argument('--resolution',type=int,choices=[128,256,512],default=512)
 p.add_argument('--seed',type=int,default=42)
 p.add_argument('--texture-size',type=int,choices=[512,1024,2048,4096],default=1024)
 p.add_argument('--decimation-target',type=int,default=1000000)
 p.add_argument('--no-remesh',action='store_true')
 p.add_argument('--source',default='')
 args=p.parse_args()
 if not torch.cuda.is_available():raise SystemExit('CUDA no disponible')
 if args.source:source=Path(args.source)
 else:source=ROOT/'repos/Alchemy3D/assets/examples/edits/04/source.glb'
 if not source.is_file():raise FileNotFoundError(source)
 outdir=ROOT/'results'/'cuda_case04'/time.strftime('%Y%m%d-%H%M%S')
 outdir.mkdir(parents=True,exist_ok=True)
 records=[]
 pipe,o_voxel=load_pipe()
 denoiser=pipe.models['tex_slat_flow_model']
 for method in args.methods:
  ck=checkpoint(method)
  if not ck.is_file():raise FileNotFoundError(f'Falta checkpoint para {method}: {ck}')
  print(f'[CHECKPOINT] {method} {ck}',flush=True)
  # This stage must match strict=True exactly; never accept hidden missing keys.
  sd=load_file(str(ck),device='cpu')
  denoiser.load_state_dict(sd,strict=True)
  del sd
  denoiser.eval()
  for prompt_name in args.prompts:
   torch.cuda.reset_peak_memory_stats()
   start=time.perf_counter()
   rec={'method':method,'prompt_name':prompt_name,'prompt':PROMPTS[prompt_name],
       'source':str(source),'source_sha256':filehash(source), 'checkpoint_sha256':filehash(ck),
       'seed':args.seed,'resolution':args.resolution,'texture_size':args.texture_size,
       'decimation_target':args.decimation_target,'remesh':not args.no_remesh,
       'gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda_version':torch.version.cuda}
   try:
    with torch.inference_mode():
     output=pipe.run(source=str(source),instruction=PROMPTS[prompt_name],seed=args.seed,resolution=args.resolution,verbose=False)[0]
    torch.cuda.synchronize()
    glbpath=outdir/method/f'{prompt_name}.glb'
    rec['glb']=export(o_voxel,output,glbpath,args)
    rec['status']='ok'
   except Exception as e:
    rec['status']='error';rec['error']=str(e);rec['traceback']=traceback.format_exc()
    print('[ERROR]',method,prompt_name,str(e),flush=True)
   rec['elapsed_sec']=round(time.perf_counter()-start,2)
   rec['peak_cuda_allocated_gib']=round(torch.cuda.max_memory_allocated()/1024**3,3)
   rec['peak_cuda_reserved_gib']=round(torch.cuda.max_memory_reserved()/1024**3,3)
   records.append(rec)
   (outdir/'results.json').write_text(json.dumps(records,indent=2,ensure_ascii=False),encoding='utf8')
   print('[CASE]',method,prompt_name,rec['status'],rec['elapsed_sec'],'segundos',flush=True)
   gc.collect();torch.cuda.empty_cache()
 print('RESULTS:',outdir)
 if any(r['status']!='ok' for r in records):raise SystemExit(2)
if __name__=='__main__':main()
