import json,struct,numpy as np
from pathlib import Path
import open3d as o3d
b=Path('valkyrie.glb').read_bytes();off=12
while off<len(b):
 n,t=struct.unpack_from('<I4s',b,off);off+=8
 c=b[off:off+n];off+=n
 if t==b'JSON':j=json.loads(c)
 if t==b'BIN\x00':raw=c
p=j['meshes'][0]['primitives'][0]
def read(i):
 a=j['accessors'][i];v=j['bufferViews'][a['bufferView']]
 dt=np.dtype('<f4');cols=3
 return np.ndarray((a['count'],cols),dtype=dt,buffer=raw,offset=v.get('byteOffset',0)+a.get('byteOffset',0),strides=(v.get('byteStride',12),4)).copy()
pos=read(p['attributes']['POSITION']);norm=read(p['attributes']['NORMAL'])
_,unique=np.unique(np.round(pos,6),axis=0,return_index=True)
pc=o3d.geometry.PointCloud()
pc.points=o3d.utility.Vector3dVector(pos[unique].astype(float))
pc.normals=o3d.utility.Vector3dVector(norm[unique].astype(float))
mesh,d=o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pc,depth=8,n_threads=4)
mesh.remove_vertices_by_mask(np.asarray(d)<np.quantile(d,.03))
mesh.remove_unreferenced_vertices()
o3d.io.write_triangle_mesh('valkyrie-candidate.ply',mesh)
report={'source_points':len(unique),'candidate_vertices':len(mesh.vertices),'candidate_triangles':len(mesh.triangles),'watertight':mesh.is_watertight(),'note':'Unrigged test candidate, original GLB unchanged'}
Path('reconstruction-report.json').write_text(json.dumps(report,indent=2))
print(report)
