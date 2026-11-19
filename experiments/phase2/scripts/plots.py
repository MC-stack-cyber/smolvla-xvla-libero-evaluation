"""Static scientific figures from completed results only."""
import json,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];P=ROOT/'phase2'
summary=json.loads((P/'summary.json').read_text())
if summary['completed']!=116:raise RuntimeError('Wait for completed experiment')
new=summary['records'];base=[]
for p in (ROOT/'results').rglob('result.json'):
 r=json.loads(p.read_text())
 if r['status']=='completed':base.append(r)
def choose(rows,**k):return [r for r in rows if all(r.get(key)==value for key,value in k.items())]
def plot_count(ax,x,rows,width,color,label=None):
 n=len(rows);s=sum(r['success'] for r in rows);value=s/n
 ax.bar(x,value,width,color=color,label=label)
 ax.text(x,value+.035,f'{s}/{n}',ha='center',fontsize=9)
colors={'smolvla':'#3b719b','xvla':'#ca8039'}
tasks=['wine_bottle','black_bowl','milk'];names=['Bottle to cabinet','Bowl to stove','Milk to basket']
out=P/'figures';out.mkdir(exist_ok=True)
fig,axes=plt.subplots(1,3,figsize=(13,4.3),sharey=True,constrained_layout=True)
for ax,task,name in zip(axes,tasks,names):
 for i,index in enumerate([1,2,3]):
  for j,model in enumerate(['smolvla','xvla']):
   v=choose(new,study='generalization',task=task,index=index,model=model)
   plot_count(ax,i+(j-.5)*.33,v,.33,colors[model],model if i==0 else None)
 ax.set(xticks=[0,1,2],xticklabels=['Index 1','Index 2','Index 3'],ylim=(0,1.30),title=name)
 ax.set_yticks(np.linspace(0,1,6));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
axes[0].set_ylabel('Task success rate');axes[-1].legend(loc='upper right',frameon=False)
fig.suptitle('New initial states: 3 seeds per state; default execution lengths')
fig.savefig(out/'generalization.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5),constrained_layout=True)
settings=[('smolvla',50,'SmolVLA / 50','#3b719b'),('smolvla',10,'SmolVLA / 10','#4a8c70'),('xvla',30,'X-VLA / 30','#ca8039'),('xvla',10,'X-VLA / 10','#9c6798')]
for j,(model,chunk,label,color) in enumerate(settings):
 for i,task in enumerate(tasks):
  v=choose(new,study='matched_frequency',model=model,task=task) if (model,chunk)==('xvla',10) else choose(base,model=model,n_action_steps=chunk,task=task)
  plot_count(ax,i+(j-1.5)*.19,v,.19,color,label if i==0 else None)
ax.set(xticks=range(3),xticklabels=names,ylim=(0,1.2),ylabel='Task success rate',title='Execution-length control at Index 0 (10 seeds per condition)')
ax.set_yticks(np.linspace(0,1,6));ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
ax.legend(loc='upper center',bbox_to_anchor=(.5,-.1),ncol=4,frameon=False)
fig.savefig(out/'matched_frequency.png',dpi=180);plt.close(fig)
replay=json.loads((P/'analysis/baseline_metrics.json').read_text())
labels=[];values=[];annotations=[]
keys=['bilateral_contact_3_steps','lift_5cm_with_recent_contact']
for task,name in zip(tasks,names):
 for model,chunk in [('smolvla',50),('smolvla',10),('xvla',30)]:
  v=choose(replay,task=task,model=model,n_action_steps=chunk)
  counts=[sum(r['stage'][k] for r in v) for k in keys]+[sum(r['success'] for r in v)]
  values.append(np.array(counts)/len(v));annotations.append([f'{s}/{len(v)}' for s in counts]);labels.append(f'{name} | {model}/{chunk}')
fig,ax=plt.subplots(figsize=(10.5,6),constrained_layout=True)
im=ax.imshow(values,vmin=0,vmax=1,cmap='Blues',aspect='auto')
ax.set(xticks=range(3),xticklabels=['Bilateral contact','Lift >=5cm','Original Goal'],yticks=range(9),yticklabels=labels,title='Stage proxies from saved baseline states (recomputed contact)')
for i,row in enumerate(annotations):
 for j,value in enumerate(row):ax.text(j,i,value,ha='center',va='center',color='white' if values[i][j]>.6 else 'black')
fig.colorbar(im,ax=ax,label='Fraction of 10 seeds')
fig.savefig(out/'stages.png',dpi=180);plt.close(fig)
p=P/'analysis/intervention_deltas.json'
if p.exists():
 d=json.loads(p.read_text());fig,ax=plt.subplots(figsize=(8.5,4.5),constrained_layout=True)
 for seed in sorted({r['seed'] for r in d}):
  pair={r['condition']:r for r in d if r['seed']==seed}
  if len(pair)==2:
   vals=[pair[c]['max_object_rise_m']*100 for c in ['sham','lift']]
   ax.plot([0,1],vals,marker='o',alpha=.65,label=f'Seed {seed}')
 ax.set(xticks=[0,1],xticklabels=['Sham hold','Scripted lift'],ylabel='Maximum object-centre rise during intervention (cm)',title='Did the intervention physically lift the milk carton?')
 ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,ncol=2)
 fig.savefig(out/'intervention_actual_rise.png',dpi=180);plt.close(fig)
fa=json.loads((P/'followup/audit.json').read_text())
fixed=json.loads((P/'analysis/rim_clearance_check.json').read_text())
fig,ax=plt.subplots(figsize=(8.5,4.5),constrained_layout=True)
for r in fa['records']:
 if r['intervention_trigger_step'] is None:continue
 before=next(x['end_carton_bottom_above_rim_m'] for x in fixed if x['model']=='smolvla' and x['seed']==r['seed'])
 after=r['end_scripted_carton_bottom_above_rim_m']
 ax.plot([0,1],[before*100,after*100],marker='o',alpha=.7,label=f"Seed {r['seed']}")
ax.axhline(0,ls='--',color='#444444',lw=1)
ax.set(xticks=[0,1],xticklabels=['Fixed 12cm lift','Geometry-guided lift'],ylabel='Conservative carton bottom minus basket rim (cm)',title='Clearance at the end of the 40-step scripted segment')
ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,ncol=2)
fig.savefig(out/'clearance_followup.png',dpi=180);plt.close(fig)
ca=json.loads((P/'clearance_calibration/audit.json').read_text())
fig,ax=plt.subplots(figsize=(8.5,4.5),constrained_layout=True)
for seed in range(10):
 pair={r['intervention']:r for r in ca['records'] if r['seed']==seed and r['intervention_trigger_step'] is not None}
 if len(pair)==2:
  vals=[100*pair[c]['end_scripted_carton_bottom_above_rim_m'] for c in ['sham','clearance']]
  ax.plot([0,1],vals,marker='o',alpha=.7,label=f'Seed {seed}')
ax.axhline(0,ls='--',color='#444444',lw=1)
ax.set(xticks=[0,1],xticklabels=['Sham hold / 80 steps','Geometry lift / 80 steps'],ylabel='Conservative carton bottom minus rim at handoff (cm)',title='Physically verified 80-step manipulation (matched prefixes)')
ax.spines[['top','right']].set_visible(False);ax.legend(frameon=False,ncol=2)
fig.savefig(out/'clearance_calibration.png',dpi=180);plt.close(fig)
print(out)
