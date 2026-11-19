"""One236-row index for all formal Mac episodes; does not alter raw results."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'phase2'
rows=[]
for phase,directory in [('baseline',ROOT/'results'),('main',P/'results'),('geometry40',P/'followup/results'),('calibration80',P/'clearance_calibration/results')]:
 for path in sorted(directory.rglob('result.json')):
  r=json.loads(path.read_text())
  if r['status']!='completed':raise RuntimeError(f'Incomplete {path}')
  rows.append({**r,'phase':phase,'study':r.get('study','baseline'),'intervention':r.get('intervention','none'),'mask':r.get('mask','none'),'success':int(r['success']),'path':path.relative_to(ROOT).as_posix()})
if len(rows)!=236 or len({r['path'] for r in rows})!=236:raise RuntimeError('Grid count mismatch')
fields=['phase','study','model','task','index','seed','n_action_steps','intervention','mask','success','steps','max_steps','script_steps_planned','intervention_trigger_step','intervention_steps','elapsed_seconds','path']
with (P/'all_mac_episodes.csv').open('w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
print('Exported236 completed episodes')
