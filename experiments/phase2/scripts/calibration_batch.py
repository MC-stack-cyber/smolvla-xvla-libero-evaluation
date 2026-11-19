"""Separately fixed80-step paired manipulation, after all prior model jobs."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'phase2';C=P/'clearance_calibration'
sys.path.insert(0,str(ROOT/'scripts'))
from thermal_guard import thermal_state
lock=(ROOT/'.batch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
plan=json.loads((C/'protocol.json').read_text())
(C / 'logs').mkdir(exist_ok=True)
for seed in plan['seeds']:
 for condition in plan['conditions']:
  folder=C/f'results/clearance_calibration/{condition}/smolvla/milk/chunk_default/index_0/seed_{seed}'
  if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text())['status']=='completed':continue
  start=time.monotonic();normal=None
  while True:
   if (P/'STOP').exists() or (ROOT/'STOP').exists():raise SystemExit(130)
   state=thermal_state();normal=(normal or time.monotonic()) if state==0 else None
   if normal and time.monotonic()-normal>=30:break
   if time.monotonic()-start>1800:raise RuntimeError('Cooldown timeout')
   time.sleep(2)
  cmd=[sys.executable,str(ROOT/'scripts/thermal_guard.py'),'--log',str(C/'logs/thermal.jsonl'),'--max-seconds','1800','--',sys.executable,'-u',str(P/'scripts/run_episode.py'),
       '--study','clearance_calibration','--intervention',condition,'--script-steps','80','--model','smolvla','--task','milk','--seed',str(seed),'--index','0','--steps','800']
  print('Calibration start',seed,condition,flush=True)
  with (C/f'logs/seed_{seed}_{condition}.log').open('a') as log:
   child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
   while child.poll() is None:
    if (P/'STOP').exists():child.send_signal(2)
    time.sleep(2)
  if child.returncode:raise SystemExit(child.returncode)
  print('Calibration completed',seed,condition,flush=True)
