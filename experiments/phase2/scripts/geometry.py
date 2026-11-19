"""World-axis collision-geometry bounds; mesh bounds are conservative boxes."""
import itertools,numpy as np
_CACHE={}
def bounds(raw,name):
 raw=getattr(raw,'env',raw);sim=raw.sim
 cache_key=(id(raw),name)
 if cache_key not in _CACHE:
  shapes=[]
  for gname in raw.objects_dict[name].contact_geoms:
   gid=sim.model.geom_name2id(gname)
   if not (sim.model.geom_contype[gid] or sim.model.geom_conaffinity[gid]):continue
   kind=int(sim.model.geom_type[gid])
   if kind==6:
    half=np.array(sim.model.geom_size[gid]);lo=-half;hi=half
   elif kind==7:
    mesh=int(sim.model.geom_dataid[gid]);start=int(sim.model.mesh_vertadr[mesh]);count=int(sim.model.mesh_vertnum[mesh])
    vertices=np.array(sim.model.mesh_vert[start:start+count]);lo=vertices.min(axis=0);hi=vertices.max(axis=0)
   else:raise RuntimeError(f'Unsupported collision primitive {kind} for {gname}')
   corners=np.array(list(itertools.product(*zip(lo,hi))))
   shapes.append((gid,corners))
  if not shapes:raise RuntimeError('No collision geometry found')
  _CACHE[cache_key]=shapes
 xyz=[]
 for gid,corners in _CACHE[cache_key]:
  rot=np.asarray(sim.data.geom_xmat[gid]).reshape(3,3)
  xyz.append(corners@rot.T+sim.data.geom_xpos[gid])
 points=np.concatenate(xyz)
 return points.min(axis=0),points.max(axis=0)
