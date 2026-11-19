"""Publication-style static plots, regenerated only from recorded results."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = json.loads((ROOT / 'summary.json').read_text())['groups']
lookup = {(r['model'], r['task'], r['n_action_steps']): r for r in data}
tasks = ['wine_bottle', 'black_bowl', 'milk']
labels = ['Bottle to cabinet', 'Bowl to stove', 'Milk to basket']
groups = [('smolvla', 50, 'SmolVLA / 50 steps', '#366a9a'),
          ('xvla', 30, 'X-VLA / 30 steps', '#d1843d'),
          ('smolvla', 10, 'SmolVLA / 10 steps', '#4a8a70')]
out = ROOT / 'figures'
out.mkdir(exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
x = np.arange(3)
width = .23
for gi, (model, chunk, label, color) in enumerate(groups):
    for i, task in enumerate(tasks):
        r = lookup.get((model, task, chunk))
        if r is None:
            continue
        pos = x[i] + (gi-1)*width
        ax.bar(pos, r['success_rate'], width, color=color, label=label if i == 0 else None)
        ax.errorbar(pos, r['success_rate'],
                    yerr=[[max(0, r['success_rate']-r['wilson_95_low'])],
                          [max(0, r['wilson_95_high']-r['success_rate'])]],
                    color='#333333', capsize=4, lw=1)
        ax.text(pos, r['wilson_95_high']+.025, f"{r['successes']}/{r['n']}", ha='center', fontsize=10)
ax.set(xticks=x, xticklabels=labels, ylim=(0,1.14), ylabel='Task success rate',
       title='Three-task evaluation (error bars: Wilson 95% intervals)')
ax.set_yticks(np.linspace(0,1,6), [f'{p:.0%}' for p in np.linspace(0,1,6)])
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=.18)
ax.set_axisbelow(True)
ax.legend(loc='upper center', bbox_to_anchor=(.5,-.12), ncol=3, frameon=False)
fig.savefig(out / 'success_rates.png', dpi=180)
plt.close(fig)
fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
for gi, (model, chunk, label, color) in enumerate(groups):
    values = [lookup.get((model, task, chunk), {}).get('average_seconds', np.nan) for task in tasks]
    ax.bar(x + (gi-1)*width, values, width, label=label, color=color)
ax.set(xticks=x, xticklabels=labels, ylabel='Mean seconds per episode',
       title='Episode time including model loading; cooldown excluded')
ax.spines[['top','right']].set_visible(False)
ax.grid(axis='y', alpha=.18)
ax.set_axisbelow(True)
ax.legend(frameon=False)
fig.savefig(out / 'episode_times.png', dpi=180)
plt.close(fig)
print(out)
