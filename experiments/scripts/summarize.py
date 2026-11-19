"""Summarize completed rollouts; never count infrastructure stops as task failures."""
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
groups = defaultdict(list)
incomplete = []
for path in sorted((ROOT / 'results').rglob('result.json')):
    d = json.loads(path.read_text())
    if d.get('status') != 'completed':
        incomplete.append({'path': str(path), **d})
        continue
    groups[(d['model'], d['task'], d['n_action_steps'], d['index'])].append(d)
rows = []
for (model, task, action_steps, index), runs in sorted(groups.items()):
    n = len(runs)
    successes = sum(r['success'] for r in runs)
    p = successes / n
    z = 1.96
    center = (p + z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    rows.append(dict(model=model, task=task, n_action_steps=action_steps, index=index,
                     n=n, successes=successes, success_rate=p,
                     wilson_95_low=max(0, center-half), wilson_95_high=min(1, center+half),
                     average_seconds=sum(r['elapsed_seconds'] for r in runs)/n))
(ROOT / 'summary.json').write_text(json.dumps({'groups': rows, 'incomplete': incomplete}, indent=2))
if rows:
    with (ROOT / 'summary.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
print(json.dumps({'completed_episodes': sum(r['n'] for r in rows), 'incomplete': len(incomplete), 'groups': rows}, indent=2))
subprocess.run([sys.executable, str(ROOT / 'scripts/make_report.py')], check=True)
