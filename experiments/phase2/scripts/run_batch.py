"""Single-model queue; all runs isolated and thermally guarded."""
import datetime,fcntl,json,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from thermal_guard import thermal_state
protocol=json.loads((ROOT/'phase2/protocol.json').read_text())
lock=(ROOT/'.batch.lock').open('a+')
fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(ROOT / 'phase2/logs').mkdir(exist_ok=True)
stopped=[]
signal.signal(signal.SIGTERM,lambda s,f:stopped.append(s))
def record(**r):
    r['time']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    with (ROOT/'phase2/logs/batch.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)
for i,q in enumerate(protocol['queue']):
    condition=q['intervention'] if q['intervention']!='none' else q['mask']
    label='default' if q['chunk'] is None else str(q['chunk'])
    folder=ROOT/'phase2/results'/q['study']/condition/q['model']/q['task']/f'chunk_{label}'/f"index_{q['index']}"/f"seed_{q['seed']}"
    if (folder/'result.json').exists() and json.loads((folder/'result.json').read_text())['status']=='completed':continue
    normal=None;start=time.monotonic()
    while True:
        if stopped or (ROOT/'STOP').exists() or (ROOT/'phase2/STOP').exists():sys.exit(130)
        state=thermal_state()
        normal=(normal or time.monotonic()) if state==0 else None
        if normal and time.monotonic()-normal>=30:break
        if time.monotonic()-start>1800:raise RuntimeError('Cooldown timed out')
        time.sleep(2)
    args=[sys.executable,str(ROOT/'scripts/thermal_guard.py'),'--log',str(ROOT/'phase2/logs/thermal.jsonl'),'--max-seconds','1800','--',sys.executable,'-u',str(ROOT/'phase2/scripts/run_episode.py')]
    for key in ['study','model','task','seed','index','mask','intervention']:args+=['--'+key,str(q[key])]
    args+=['--steps','800']
    if q['chunk'] is not None:args+=['--action-steps',str(q['chunk'])]
    record(event='start',number=i+1,**q)
    name=f"{i:03d}_{q['study']}_{condition}_{q['model']}_{q['task']}_{q['index']}_{q['seed']}.log"
    with (ROOT/'phase2/logs'/name).open('a') as log:
        child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            if stopped or (ROOT/'phase2/STOP').exists():child.send_signal(signal.SIGINT)
            time.sleep(2)
    record(event='exit',number=i+1,returncode=child.returncode)
    if child.returncode:sys.exit(child.returncode)
record(event='completed',episodes=len(protocol['queue']))
