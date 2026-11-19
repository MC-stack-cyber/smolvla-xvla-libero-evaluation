"""Run the separately frozen clearance follow-up after the main model queue."""
import fcntl,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'phase2'
sys.path.insert(0,str(ROOT/'scripts'))
from thermal_guard import thermal_state
lock=(ROOT/'.batch.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
plan=json.loads((P/'followup/protocol.json').read_text())
(P / 'followup/logs').mkdir(exist_ok=True)
for seed in plan['seeds']:
 folder=P/f'followup/results/geometry_followup/clearance/smolvla/milk/chunk_default/index_0/seed_{seed}'
 if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text())['status']=='completed':continue
 start=time.monotonic();normal=None
 while True:
  if (P/'STOP').exists() or (ROOT/'STOP').exists():raise SystemExit(130)
  state=thermal_state();normal=(normal or time.monotonic()) if state==0 else None
  if normal and time.monotonic()-normal>=30:break
  if time.monotonic()-start>1800:raise RuntimeError('Cooldown timeout')
  time.sleep(2)
 cmd=[sys.executable,str(ROOT/'scripts/thermal_guard.py'),'--log',str(P/'followup/logs/thermal.jsonl'),'--max-seconds','1800','--',sys.executable,'-u',str(P/'scripts/run_episode.py'),
      '--study','geometry_followup','--intervention','clearance','--model','smolvla','--task','milk','--seed',str(seed),'--index','0','--steps','800']
 print('Followup start',seed,flush=True)
 with (P/f'followup/logs/seed_{seed}.log').open('a') as log:
  child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
  while child.poll() is None:
   if (P/'STOP').exists():child.send_signal(2)
   time.sleep(2)
 if child.returncode:raise SystemExit(child.returncode)
 print('Followup completed',seed,flush=True)
