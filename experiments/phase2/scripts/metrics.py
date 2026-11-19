"""Geometric/contact metrics; thresholds fixed before phase-two outcomes."""
import numpy as np
TARGETS={'wine_bottle':'wine_bottle_1','black_bowl':'akita_black_bowl_1','milk':'milk_1'}
def measure(raw,task):
    raw=getattr(raw,"env",raw)
    sim=raw.sim
    name=TARGETS[task]
    obj=raw.objects_dict[name]
    pos=np.asarray(sim.data.body_xpos[raw.obj_body_id[name]]).copy()
    eef=np.asarray(sim.data.site_xpos[raw.robots[0].eef_site_id]).copy()
    grasp=bool(raw._check_grasp(raw.robots[0].gripper,obj))
    result={'object_xyz':pos.tolist(),'eef_xyz':eef.tolist(),'bilateral_contact':grasp,
            'eef_object_distance':float(np.linalg.norm(eef-pos))}
    if task=='milk':
        basket=np.asarray(sim.data.body_xpos[raw.obj_body_id['basket_1']]).copy()
        result['basket_xyz']=basket.tolist()
        result['object_basket_xy_distance']=float(np.linalg.norm(pos[:2]-basket[:2]))
    return result

def summarize(rows,initial):
    z0=initial['object_xyz'][2]
    contact=[r['bilateral_contact'] for r in rows]
    sustained=any(all(contact[i:i+3]) for i in range(max(0,len(contact)-2)))
    lift=[r['object_xyz'][2]-z0 for r in rows]
    # At least 5cm object-centre rise, with bilateral contact in current/recent 5 steps.
    carried=any(h>=.05 and any(contact[max(0,i-4):i+1]) for i,h in enumerate(lift))
    return {'approach_10cm':any(r['eef_object_distance']<.10 for r in rows),
            'bilateral_contact_3_steps':sustained,'lift_5cm_with_recent_contact':carried,
            'max_object_rise_m':max(lift,default=0),
            'max_object_rise_recent_contact_m':max([h for i,h in enumerate(lift) if any(contact[max(0,i-4):i+1])],default=0)}
