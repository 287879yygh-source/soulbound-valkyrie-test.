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
