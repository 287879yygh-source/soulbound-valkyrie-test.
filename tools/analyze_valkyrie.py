import json,struct,numpy as np
from pathlib import Path
b=Path('valkyrie.glb').read_bytes()
assert b[:4]==b'glTF'
off=12; j=None; binary=b''
while off<len(b):
 n,t=struct.unpack_from('<I4s',b,off);off+=8
 chunk=b[off:off+n];off+=n
 if t==b'JSON':j=json.loads(chunk)
 if t==b'BIN\x00':binary=chunk
print('GLB bytes',len(b),'meshes',len(j.get('meshes',[])),'skins',len(j.get('skins',[])),'animations',len(j.get('animations',[])))
types={5120:np.int8,5121:np.uint8,5122:np.int16,5123:np.uint16,5125:np.uint32,5126:np.float32}
components={'SCALAR':1,'VEC2':2,'VEC3':3,'VEC4':4,'MAT4':16}
def read(i):
 a=j['accessors'][i];v=j['bufferViews'][a['bufferView']]
 dtype=np.dtype(types[a['componentType']]).newbyteorder('<');count=a['count'];cols=components[a['type']]
 offset=v.get('byteOffset',0)+a.get('byteOffset',0);stride=v.get('byteStride',cols*dtype.itemsize)
 return np.ndarray((count,cols),dtype=dtype,buffer=binary,offset=offset,strides=(stride,dtype.itemsize)).copy()
for mi,m in enumerate(j.get('meshes',[])):
 for pi,p in enumerate(m['primitives']):
  a=p['attributes'];pos=read(a['POSITION']);idx=read(p['indices']).ravel() if 'indices' in p else np.arange(len(pos))
  print('mesh',mi,'primitive',pi,'vertices',len(pos),'indices',len(idx),'range',pos.min(axis=0),pos.max(axis=0),'finite',np.isfinite(pos).all(),'out_of_range_indices',int(np.sum(idx>=len(pos))))
  if len(idx)%3==0 and idx.max()<len(pos):
   tri=pos[idx.reshape(-1,3)];area=np.linalg.norm(np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]),axis=1)/2
   print('triangles',len(area),'zero_area',int(np.sum(area<1e-10)),'median_area',float(np.median(area)),'p99_area',float(np.percentile(area,99)))
  for name in ('JOINTS_0','WEIGHTS_0','JOINTS_1','WEIGHTS_1'):
   if name in a:
    x=read(a[name]);print(name,'shape',x.shape,'min',x.min(),'max',x.max(),'nonzero',int(np.count_nonzero(x)))
  if 'WEIGHTS_0' in a:
   w=read(a['WEIGHTS_0']).astype(float)
   if 'WEIGHTS_1' in a:w=np.concatenate([w,read(a['WEIGHTS_1']).astype(float)],axis=1)
   print('weight sums min/max',w.sum(axis=1).min(),w.sum(axis=1).max(),'zero-weight vertices',int(np.sum(w.sum(axis=1)<.001)))
for si,skin in enumerate(j.get('skins',[])):
 print('skin',si,'joints',len(skin['joints']))
 if 'inverseBindMatrices' in skin:
  x=read(skin['inverseBindMatrices']);print('inverse bind matrices',x.shape,'finite',np.isfinite(x).all())

# Topology and skinning diagnostics
from collections import Counter
for mi,m in enumerate(j.get('meshes',[])):
 for pi,p in enumerate(m['primitives']):
  if 'indices' not in p:continue
  idx=read(p['indices']).ravel().astype(np.int64)
  if len(idx)%3:continue
  tri=idx.reshape(-1,3);edges=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1)
  unique,counts=np.unique(edges,axis=0,return_counts=True)
  print('TOPOLOGY mesh',mi,'primitive',pi,'boundary_edges',int(np.sum(counts==1)),'nonmanifold_edges',int(np.sum(counts>2)),'shared_edges',int(np.sum(counts==2)))
  pos=read(p['attributes']['POSITION'])
  lengths=np.linalg.norm(pos[edges[:,0]]-pos[edges[:,1]],axis=1)
  print('EDGE LENGTH median',float(np.median(lengths)),'p99',float(np.percentile(lengths,99)),'max',float(lengths.max()))
  if 'JOINTS_0' in p['attributes']:
   a=p['attributes'];joint=read(a['JOINTS_0']).astype(int);w=read(a['WEIGHTS_0']).astype(float)
   print('dominant joint histogram',dict(Counter(joint[np.arange(len(joint)),np.argmax(w,axis=1)].tolist())))
   print('skin joint index valid',bool(joint.max()<len(j['skins'][0]['joints'])))
   print('node transforms',[(n.get('name'),n.get('translation'),n.get('scale')) for n in j.get('nodes',[]) if 'mesh' in n or 'skin' in n])
   # detect mismatch between local geometry and inverse bind matrix translations
   if 'inverseBindMatrices' in j['skins'][0]:
    ibm=read(j['skins'][0]['inverseBindMatrices']).reshape(-1,4,4)
    print('inverse bind translation',ibm[:,3,:3].tolist())

# Geometric welding audit: duplicate vertex positions can explain index-level boundary edges.
for mi,m in enumerate(j.get('meshes',[])):
 for pi,p in enumerate(m['primitives']):
  if mi!=0 or 'indices' not in p:continue
  pos=read(p['attributes']['POSITION']).astype(np.float64)
  idx=read(p['indices']).ravel().astype(np.int64)
  print('WELD AUDIT vertices',len(pos),'used vertices',len(np.unique(idx)))
  for decimals in (7,6,5,4):
   rounded=np.round(pos,decimals)
   _,inverse=np.unique(rounded,axis=0,return_inverse=True)
   welded_tri=inverse[idx].reshape(-1,3)
   edge=np.sort(np.concatenate([welded_tri[:,[0,1]],welded_tri[:,[1,2]],welded_tri[:,[2,0]]]),axis=1)
   _,ct=np.unique(edge,axis=0,return_counts=True)
   degenerate=int(np.sum((welded_tri[:,0]==welded_tri[:,1])|(welded_tri[:,1]==welded_tri[:,2])|(welded_tri[:,2]==welded_tri[:,0])))
   print('WELD',decimals,'unique_positions',int(np.max(inverse)+1),'boundary',int(np.sum(ct==1)),'shared',int(np.sum(ct==2)),'nonmanifold',int(np.sum(ct>2)),'collapsed_triangles',degenerate)

# Quantify face islands and triangle-soup distribution before any destructive repair.
for mi,m in enumerate(j.get('meshes',[])):
 for pi,p in enumerate(m['primitives']):
  if mi!=0 or 'indices' not in p:continue
  faces=read(p['indices']).ravel().reshape(-1,3).astype(np.int64)
  n=len(faces);parent=np.arange(n,dtype=np.int64)
  def root(x):
   while parent[x]!=x:
    parent[x]=parent[parent[x]];x=parent[x]
   return x
  def union(x,y):
   x=root(x);y=root(y)
   if x!=y:parent[y]=x
  owners={}
  for fi,face in enumerate(faces):
   for v in face:
    v=int(v)
    if v in owners:union(fi,owners[v])
    else:owners[v]=fi
  groups=Counter(root(i) for i in range(n))
  print('FACE ISLANDS',len(groups),'isolated_triangles',sum(v==1 for v in groups.values()),'largest_island_faces',max(groups.values()),'island_sizes_top10',sorted(groups.values(),reverse=True)[:10])
  print('MESH ATTRIBUTES',sorted(p['attributes'].keys()),'primitive_mode',p.get('mode',4))
