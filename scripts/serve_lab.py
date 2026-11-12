#!/usr/bin/env python3
"""Authenticated, allowlisted job API. No shell, no arbitrary code execution."""
import argparse,json,os,secrets,subprocess,sys,threading,time,uuid
from pathlib import Path
from fastapi import FastAPI,Depends,Header,HTTPException
from pydantic import BaseModel,Field
from fastapi.responses import HTMLResponse
import uvicorn

ROOT=Path(os.getenv('LAB_ROOT','/workspace')).resolve()
HERE=Path(__file__).resolve().parent
TOKEN=os.getenv('LAB_TOKEN')
if not TOKEN or len(TOKEN)<24:raise SystemExit('LAB_TOKEN ausente/demasiado corto. Generar token aleatorio antes de iniciar.')
app=FastAPI(title='Alchemy3D A6000 Lab',docs_url=None,redoc_url=None)
lock=threading.Lock()
jobs={}
METHODS={'instruct_control','base_texture_control','ties','deltas','trim_mean','deltas_cap'}

def auth(authorization:str|None=Header(default=None)):
 if not authorization or not secrets.compare_digest(authorization, 'Bearer '+TOKEN):
  raise HTTPException(401,'No autorizado')

class JobRequest(BaseModel):
 methods:list[str]=Field(default=['instruct_control','base_texture_control','ties','deltas','trim_mean','deltas_cap'],min_length=1,max_length=6)
 resolution:int=Field(default=512)

@app.get('/', response_class=HTMLResponse)
def dashboard():
    return """<!doctype html><html lang="es"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Alchemy3D Lab — A6000</title>
<style>body{font:15px system-ui;background:#111827;color:#f9fafb;max-width:850px;margin:3rem auto;padding:1rem}h1{font-size:1.6rem}section{background:#1f2937;border-radius:12px;padding:1rem;margin:1rem 0}input,select,button{font:inherit;padding:.55rem;border-radius:6px}input{width:90%;max-width:500px}button{cursor:pointer;background:#2563eb;color:white;border:0;margin:.4rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#0f172a;padding:1rem;max-height:420px;overflow:auto}.muted{color:#9ca3af}</style>
<h1>Alchemy3D CUDA Lab</h1><p class="muted">Panel privado de pruebas allowlist. Sin acceso a terminal ni a ejecución arbitraria.</p>
<section><label>Token del laboratorio <input id="token" type="password" autocomplete="off" placeholder="LAB_TOKEN"></label><button onclick="showJobs()">Ver trabajos</button></section>
<section><h2>Iniciar experimento</h2><p>Instruct y Base sirven de controles. Se evalúa únicamente la fusión de textura.</p>
<label>Resolución <select id="res"><option>256</option><option selected>512</option><option>128</option></select></label><div id="checks"></div>
<button onclick="runTest()">Ejecutar prueba</button></section>
<section><h2>Estado / informes</h2><button onclick="showJobs()">Actualizar</button><button onclick="showResults()">Ver resultados</button><pre id="status">Sin conexión</pre></section>
<script>
const methods=['instruct_control','base_texture_control','ties','deltas','trim_mean','deltas_cap'];
document.getElementById('checks').innerHTML=methods.map(x=>`<label style="display:inline-block;margin:.5rem"><input type="checkbox" name="model" value="${x}" checked style="width:auto">${x}</label>`).join('');
async function api(path, options={}) {const token=document.getElementById('token').value;const r=await fetch(path,{...options,headers:{'Authorization':'Bearer '+token,'Content-Type':'application/json'}});const obj=await r.json();if(!r.ok)throw new Error(r.status+' '+JSON.stringify(obj));return obj}
function show(x){document.getElementById('status').textContent=JSON.stringify(x,null,2)}
async function showJobs(){try{show(await api('/api/jobs'))}catch(e){show({error:String(e)})}}
async function showResults(){try{show(await api('/api/results'))}catch(e){show({error:String(e)})}}
async function runTest(){try{const m=[...document.querySelectorAll('[name=model]:checked')].map(x=>x.value);show(await api('/api/test',{method:'POST',body:JSON.stringify({methods:m,resolution:Number(document.getElementById('res').value)})}))}catch(e){show({error:String(e)})}}
</script></html>"""

@app.get('/health')
def health():return {'status':'alive','gpu_test':'not implied'}

@app.get('/api/jobs',dependencies=[Depends(auth)])
def list_jobs():return jobs

@app.post('/api/test',dependencies=[Depends(auth)])
def run_test(req:JobRequest):
 if req.resolution not in (128,256,512) or any(m not in METHODS for m in req.methods):raise HTTPException(422,'Método o resolución inválidos')
 if not lock.acquire(blocking=False):raise HTTPException(409,'Ya hay una prueba GPU activa')
 jobid=uuid.uuid4().hex[:12]
 logfile=ROOT/'results'/f'job-{jobid}.log';logfile.parent.mkdir(parents=True,exist_ok=True)
 jobs[jobid]={'status':'running','methods':req.methods,'resolution':req.resolution,'log':str(logfile),'started':time.time()}
 def worker():
  try:
   cmd=[sys.executable,str(HERE/'run_case_cuda.py'),'--methods',*req.methods,'--resolution',str(req.resolution)]
   with logfile.open('w',encoding='utf8') as log:
    done=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,cwd=str(ROOT),check=False,env=dict(os.environ))
   jobs[jobid]['status']='ok' if done.returncode==0 else 'error'
   jobs[jobid]['returncode']=done.returncode
  except Exception as e:jobs[jobid]['status']='error';jobs[jobid]['error']=str(e)
  finally:jobs[jobid]['finished']=time.time();lock.release()
 threading.Thread(target=worker,daemon=True).start()
 return {'job_id':jobid,'status':'running'}

@app.get('/api/job/{jobid}',dependencies=[Depends(auth)])
def get_job(jobid:str):
 if jobid not in jobs:raise HTTPException(404)
 return jobs[jobid]

@app.get('/api/job/{jobid}/log',dependencies=[Depends(auth)])
def job_log(jobid:str):
 if jobid not in jobs:raise HTTPException(404)
 f=Path(jobs[jobid]['log'])
 if not f.is_file():return {'log':''}
 return {'log':f.read_text('utf8',errors='replace')[-60000:]}

@app.get('/api/results',dependencies=[Depends(auth)])
def results():
 base=ROOT/'results/cuda_case04'
 if not base.is_dir():return {'runs':[]}
 return {'runs':[x.name for x in sorted(base.iterdir()) if x.is_dir()]}

@app.get('/api/result/{runid}',dependencies=[Depends(auth)])
def read_result(runid:str):
 if not runid.isdigit() and not all(c.isdigit() or c=='-' for c in runid):raise HTTPException(400)
 p=(ROOT/'results/cuda_case04'/runid/'results.json').resolve()
 if not p.is_file() or not p.is_relative_to((ROOT/'results').resolve()):raise HTTPException(404)
 return json.loads(p.read_text('utf8'))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=8765)
 args=p.parse_args();uvicorn.run(app,host=args.host,port=args.port,access_log=False)
