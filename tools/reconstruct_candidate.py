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
# Keep original candidate and generate a separate largest-component-only candidate.
tri_clusters,tri_counts,_=mesh.cluster_connected_triangles()
tri_clusters=np.asarray(tri_clusters)
tri_counts=np.asarray(tri_counts)
if len(tri_counts):
 largest=int(np.argmax(tri_counts))
 cleaned=o3d.geometry.TriangleMesh(mesh)
 cleaned.remove_triangles_by_mask(tri_clusters!=largest)
 cleaned.remove_unreferenced_vertices()
 cleaned.compute_vertex_normals()
 o3d.io.write_triangle_mesh('valkyrie-main-surface.ply',cleaned)
 print('MAIN SURFACE',len(cleaned.vertices),'vertices',len(cleaned.triangles),'triangles','watertight',cleaned.is_watertight())
 # Save front/side/back diagnostic views from the actual reconstructed mesh.
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 from mpl_toolkits.mplot3d.art3d import Poly3DCollection
 verts=np.asarray(cleaned.vertices)
 tris=np.asarray(cleaned.triangles)
 rng=np.random.default_rng(17)
 selected=tris
 polygons=verts[selected][:,:,[0,2,1]]
 for label,azimuth in [('front',0),('side',90),('back',180)]:
  fig=plt.figure(figsize=(7,9),dpi=130,facecolor='#18202a')
  ax=fig.add_subplot(111,projection='3d')
  ax.add_collection3d(Poly3DCollection(polygons,facecolor='#3f8ea9',edgecolor='none',alpha=1))
  ax.set_facecolor('#18202a')
  rotated=verts[:,[0,2,1]]
  middle=(rotated.min(axis=0)+rotated.max(axis=0))/2
  radius=max((rotated.max(axis=0)-rotated.min(axis=0)))/2
  ax.set(xlim=(middle[0]-radius,middle[0]+radius),ylim=(middle[1]-radius,middle[1]+radius),zlim=(middle[2]-radius,middle[2]+radius))
  ax.view_init(elev=5,azim=azimuth-90)
  ax.set_proj_type('ortho')
  ax.set_box_aspect((1,1,1.8))
  ax.set_axis_off()
  fig.savefig('valkyrie-'+label+'-mesh-preview.png',bbox_inches='tight',facecolor='#18202a')
  plt.close(fig)
mesh.compute_vertex_normals()
components=mesh.cluster_connected_triangles()
cluster_ids=np.asarray(components[0]);counts=np.asarray(components[1]);
print('CONNECTED COMPONENTS',len(counts),'largest triangles',int(counts.max()) if len(counts) else 0,'small islands',int(np.sum(counts<100)))
report={'connected_components':len(counts),'largest_component_triangles':int(counts.max()) if len(counts) else 0,'small_components_under_100_triangles':int(np.sum(counts<100)),'source_points':len(unique),'candidate_vertices':len(mesh.vertices),'candidate_triangles':len(mesh.triangles),'watertight':mesh.is_watertight(),'note':'Unrigged test candidate, original GLB unchanged'}
Path('reconstruction-report.json').write_text(json.dumps(report,indent=2))
print(report)
