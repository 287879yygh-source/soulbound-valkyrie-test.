import bpy, math
from mathutils import Vector
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)
bpy.ops.import_scene.gltf(filepath='valkyrie.glb')
meshes=[o for o in bpy.context.scene.objects if o.type=='MESH']
armatures=[o for o in bpy.context.scene.objects if o.type=='ARMATURE']
print('BLENDER_IMPORT',[(o.name,len(o.data.vertices),len(o.data.polygons)) for o in meshes], 'ARMATURES',[(a.name,len(a.data.bones)) for a in armatures],flush=True)
if not meshes: raise RuntimeError('No mesh in original GLB')
scene=bpy.context.scene
scene.render.engine='CYCLES'
scene.cycles.samples=16
scene.render.resolution_x=850
scene.render.resolution_y=1100
scene.render.resolution_percentage=100
scene.render.image_settings.file_format='PNG'
scene.world.color=(.22,.22,.22)
for obj in meshes: obj.select_set(True)
bpy.context.view_layer.objects.active=meshes[0]
bpy.ops.object.select_all(action='DESELECT')
points=[]
for ob in meshes:
 for v in ob.bound_box: points.append(ob.matrix_world @ Vector(v))
low=Vector([min(p[i] for p in points) for i in range(3)])
high=Vector([max(p[i] for p in points) for i in range(3)])
center=(low+high)/2
size=max((high-low).length,1)
def aim(obj,target):
 obj.rotation_euler=(target-obj.location).to_track_quat('-Z','Y').to_euler()
bpy.ops.object.camera_add(location=center+Vector((size*.6,-size*1.7,size*.22)))
cam=bpy.context.object;aim(cam,center)
cam.data.type='ORTHO';cam.data.ortho_scale=size*1.12
scene.camera=cam
for xyz,power,scale in [((-.9,-1.0,1.1),1100,4),((1.1,.7,1.0),850,3)]:
 bpy.ops.object.light_add(type='AREA',location=center+Vector(xyz)*size)
 l=bpy.context.object;l.data.energy=power;l.data.shape='DISK';l.data.size=scale*size/3;aim(l,center)
scene.render.filepath='blender-original-valkyrie.png'
bpy.ops.render.render(write_still=True)
print('BLENDER_RENDER_DONE',flush=True)
