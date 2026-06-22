import numpy as np
import os
import trimesh
import nibabel as nib

segmented_valve_file = "/home/cyrilpillai36/Desktop/data/train/train_001-label.nii.gz"

#load segmented_valve as binary
def load_segmented_valve(segmented_valve):
    segmented_valve = nib.load(segmented_valve)
    segmented_valve = segmented_valve.get_fdata()
    segmented_valve = segmented_valve.astype(np.uint8)
    return segmented_valve

segmented_valve = load_segmented_valve(segmented_valve_file)

print(segmented_valve.shape)
print( np.unique( segmented_valve ) )

segmented_valve[ segmented_valve > 0 ] = 1
print( np.unique( segmented_valve ) )

# Create a mesh from the segmented valve
# `matrix_to_marching_cubes` may return a Trimesh or a tuple of (verts, faces, normals, values).
result = trimesh.voxel.ops.matrix_to_marching_cubes(segmented_valve, pitch=1.0)
if isinstance(result, trimesh.Trimesh):
    mesh = result
else:
    verts, faces, normals, values = result
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)

# Save the mesh to an STL file
mesh.export('/home/cyrilpillai36/Desktop/segmented_valve_mesh.stl')



