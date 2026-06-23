from mpi4py import MPI
import numpy as np
import basix
from basix import ufl as b_ufl
import ufl
from petsc4py import PETSc
from dolfinx import fem
import dolfinx.fem.petsc as fem_petsc 
from dolfinx.io import gmsh as gmshio, XDMFFile

# ---------- 1. Load mesh ----------
mesh_data = gmshio.read_from_msh(
    "/home/cyrilpillai36/Desktop/TEER/mitral.msh",
    MPI.COMM_WORLD, 0, gdim=3
)
mesh = mesh_data.mesh

tdim = mesh.topology.dim
mesh.topology.create_entities(tdim - 1)

print("dim:", tdim)
print("cells:", mesh.topology.index_map(tdim).size_local)

# ---------- 2. Vector function space ----------
cell = mesh.basix_cell()
gdim = mesh.geometry.dim
k = 1

Ve = b_ufl.element("Lagrange", cell, k, shape=(gdim,))
V = fem.functionspace(mesh, Ve)

# ---------- 3. No-slip boundary on whole outer surface ----------
def boundary_all(x):
    eps = 1e-10
    coords = mesh.geometry.x
    xmin, ymin, zmin = coords.min(axis=0)
    xmax, ymax, zmax = coords.max(axis=0)
    return np.logical_or.reduce((
        np.isclose(x[0], xmin, atol=eps),
        np.isclose(x[0], xmax, atol=eps),
        np.isclose(x[1], ymin, atol=eps),
        np.isclose(x[1], ymax, atol=eps),
        np.isclose(x[2], zmin, atol=eps),
        np.isclose(x[2], zmax, atol=eps),
    ))

u_bc = fem.Function(V)
u_bc.x.array[:] = 0.0
dofs_u = fem.locate_dofs_geometrical(V, boundary_all)
bcu = fem.dirichletbc(u_bc, dofs_u)

# ---------- 4. Vector Poisson problem ----------
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

mu = fem.Constant(mesh, PETSc.ScalarType(1.0))

# Body force (drives "flow" roughly along +z)
f = fem.Constant(mesh, PETSc.ScalarType((0.0, 0.0, 1.0)))

a = mu * ufl.inner(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = ufl.inner(f, v) * ufl.dx

problem = fem_petsc.LinearProblem(
    a,
    L,
    bcs=[bcu],
    u=fem.Function(V),
    petsc_options_prefix="mitral_vec_",  # required on your version [web:377][web:387]
    petsc_options={"ksp_type": "cg", "pc_type": "ilu"}  # or leave empty dict
)
u_sol = problem.solve()
u_sol.name = "velocity_like"

# ---------- 5. Save to XDMF ----------
with XDMFFile(mesh.comm, "mitral_velocity_like.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_sol)

print("Wrote mitral_velocity_like.xdmf")

from dolfinx import io

with io.XDMFFile(mesh.comm, "mitral_velocity_like.pvd", "w",
                 encoding=io.XDMFFile.Encoding.ASCII) as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(u_sol)

