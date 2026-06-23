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

mesh, cell_tags, facet_tags = gmshio.read_from_msh("mitral.msh", MPI.COMM_WORLD, 0)

print("dim:", mesh.topology.dim)           # expect 3
print("cells:", mesh.topology.index_map(mesh.topology.dim).size_local)
print("facets:", mesh.topology.index_map(mesh.topology.dim-1).size_local)

print("cell tag ids:", cell_tags.values if cell_tags is not None else "None")
print("facet tag ids:", facet_tags.values if facet_tags is not None else "None")