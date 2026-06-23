import dolfinx
import sys
print(sys.executable)
import ambit_fe
import pyvista
import meshio
import trimesh


import meshio
import pymeshfix
import pyvista as pv
import sys

import numpy as np
'''
# Highkey need to fix mesh

mesh_file = "/home/cyrilpillai36/Desktop/TEER/segmented_valve_mesh_smoothed.stl"

mesh = trimesh.load_mesh(mesh_file, process=True)  # removes degenerate/duplicate faces automatically [web:165][web:176]


# 2) Ensure correct dtypes for pymeshfix
v = np.asarray(mesh.vertices, dtype=np.float64)
f = np.asarray(mesh.faces,    dtype=np.int32)

# 3) Repair
vclean, fclean = pymeshfix.clean_from_arrays(v, f, verbose=True)

# 4) Back to trimesh and export
fixed = trimesh.Trimesh(vertices=vclean, faces=fclean, process=True)
print("watertight:", fixed.is_watertight)
fixed.export( "/home/cyrilpillai36/Desktop/TEER/segmented_valve_mesh_smoothed.stl" )
'''

# DOLFIN

from mpi4py import MPI
from dolfinx.io import gmsh as gmshio
from dolfinx import mesh as dmesh

mesh_data  = gmshio.read_from_msh("/home/cyrilpillai36/Desktop/TEER/mitral.msh", MPI.COMM_WORLD, 0)

mesh = mesh_data.mesh
cell_tags = mesh_data.cell_tags
facet_tags = mesh_data.facet_tags

tdim = mesh.topology.dim
mesh.topology.create_entities(tdim-1)


print("dim:", mesh.topology.dim)           # expect 3
print("cells:", mesh.topology.index_map(mesh.topology.dim).size_local)
print("facets:", mesh.topology.index_map(tdim-1).size_local)






