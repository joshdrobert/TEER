"""Generate and run a mock mitral case with FEniCS/DOLFINx and PyVista."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


@dataclass
class MockMitralResult:
    """Summary of the generated case and solver run."""

    case_dir: Path
    result_vtu: Path
    pressure_range: tuple[float, float]
    velocity_magnitude_range: tuple[float, float]
    valve_scale_factor: float

    def to_dict(self) -> dict[str, object]:
        return {
            "case_dir": str(self.case_dir),
            "result_vtu": str(self.result_vtu),
            "pressure_range": list(self.pressure_range),
            "velocity_magnitude_range": list(self.velocity_magnitude_range),
            "valve_scale_factor": self.valve_scale_factor,
        }


@dataclass
class MockMitralVisualization:
    """Rendered artifacts for the mock mitral case."""

    case_dir: Path
    overview_png: Path
    cutaway_png: Path
    animation_gif: Path

    def to_dict(self) -> dict[str, object]:
        return {
            "case_dir": str(self.case_dir),
            "overview_png": str(self.overview_png),
            "cutaway_png": str(self.cutaway_png),
            "animation_gif": str(self.animation_gif),
        }


def generate_and_run_mock_case(
    valve_obj: Path,
    workspace: Path,
    time_steps: int = 10,
    time_step_size: float = 1e-3,
) -> MockMitralResult:
    """Build a runnable mitral simulation using FEniCS/DOLFINx."""
    workspace = workspace.resolve()
    valve_obj = valve_obj.resolve()

    case_dir = workspace / "artifacts" / "mock_mitral_uris_fsi"
    case_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Mesh
    valve_scale_factor = 1.0
    valve_mesh = _load_clean_obj(valve_obj)
    
    # Create base VTU for FEniCS processing
    base_vtu = case_dir / "mitral_base.vtu"
    valve_mesh.cast_to_unstructured_grid().save(base_vtu)

    # 2. Run FEniCS simulation (or fallback to PyVista synthetic if dolfinx missing)
    result_vtu = case_dir / f"result_{time_steps:03d}.vtu"
    
    try:
        _run_fenics_simulation(base_vtu, case_dir, time_steps)
    except ImportError:
        print("Warning: FEniCS/DOLFINx is not installed in this environment.")
        print("Running synthetic structural proxy simulation using PyVista instead.")
        _run_synthetic_proxy_simulation(valve_mesh, case_dir, time_steps)

    # Extract mock pressure and velocity ranges from final result
    pressure_range, velocity_range = _summarize_result(result_vtu)
    summary = MockMitralResult(
        case_dir=case_dir,
        result_vtu=result_vtu,
        pressure_range=pressure_range,
        velocity_magnitude_range=velocity_range,
        valve_scale_factor=valve_scale_factor,
    )
    (case_dir / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2))
    return summary


def _run_fenics_simulation(base_vtu: Path, case_dir: Path, time_steps: int) -> None:
    """Execute 3D transient Navier-Stokes simulation with Brinkman penalization of walls and valve leaflets."""
    import dolfinx
    from dolfinx import mesh as dmesh, fem
    from dolfinx.fem import petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    from scipy.spatial import KDTree
    import pyvista as pv

    # 1. Create 3D Cartesian box grid
    if MPI.COMM_WORLD.rank == 0:
        print("[FEniCS] Creating 3D mesh grid...", flush=True)
    # Bounds: x in [-25, 45], y in [-35, 35], z in [15, 85]
    domain = dmesh.create_box(
        MPI.COMM_WORLD,
        [np.array([-25.0, -35.0, 15.0]), np.array([45.0, 35.0, 85.0])],
        [10, 10, 12],
        dmesh.CellType.tetrahedron
    )

    # 2. Function Spaces (Taylor-Hood P2-P1 elements)
    V = fem.functionspace(domain, ("Lagrange", 2, (3,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    W_chi = fem.functionspace(domain, ("Lagrange", 1))

    # 3. Brinkman Penalization Field
    chi_func = fem.Function(W_chi)
    x_dofs = W_chi.tabulate_dof_coordinates()
    
    # Precompute wall penalization (cylinder centered at 10.8, 0.36 with radius 22.5)
    d_axis = np.sqrt((x_dofs[:, 0] - 10.8)**2 + (x_dofs[:, 1] - 0.36)**2)
    r_cylinder = 22.5
    chi_wall = 1.0 / (1.0 + np.exp(-1.0 * (d_axis - r_cylinder))) # 1.0 outside cylinder

    # Load original valve mesh
    valve_mesh = pv.read(base_vtu)
    group_ids = valve_mesh.point_data["GroupID"]
    orig_vertices = valve_mesh.points.copy()
    z_min, z_max = 55.6, 73.9
    weight = np.clip((z_max - orig_vertices[:, 2]) / (z_max - z_min), 0.0, 1.0)

    # Calculate w_seam to prevent separation at the connected commissures
    pts1 = orig_vertices[group_ids == 1]
    pts2 = orig_vertices[group_ids == 2]
    w_seam = np.ones(len(orig_vertices))
    if len(pts1) > 0 and len(pts2) > 0:
        tree2 = KDTree(pts2)
        dists, _ = tree2.query(pts1)
        shared_pts = pts1[dists < 1e-3]
        if len(shared_pts) > 0:
            tree_shared = KDTree(shared_pts)
            d_seam, _ = tree_shared.query(orig_vertices)
            w_seam = 1.0 - np.exp(-(d_seam / 10.0)**2)

    # Extract 2D convex hull of the valve XY projection
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MPath
    poly_vertices = valve_mesh.points[ConvexHull(valve_mesh.points[:, :2]).vertices][:, :2]
    polygon_path = MPath(poly_vertices)

    # Precompute wall penalization using 2D signed distance to the valve convex hull
    def signed_distance_to_polygon(points, poly_vertices):
        P0 = poly_vertices
        P1 = np.roll(poly_vertices, -1, axis=0)
        V = P1 - P0
        V_len_sq = np.sum(V**2, axis=1) + 1e-12
        QP0 = points[:, np.newaxis, :] - P0[np.newaxis, :, :]
        t = np.sum(QP0 * V[np.newaxis, :, :], axis=2) / V_len_sq[np.newaxis, :]
        t = np.clip(t, 0.0, 1.0)
        closest = P0[np.newaxis, :, :] + t[:, :, np.newaxis] * V[np.newaxis, :, :]
        dists = np.sqrt(np.sum((points[:, np.newaxis, :] - closest)**2, axis=2))
        min_dists = np.min(dists, axis=1)
        inside = polygon_path.contains_points(points)
        return np.where(inside, -min_dists, min_dists)

    cx, cy = 9.99, 1.396
    taper = np.sqrt(np.clip((x_dofs[:, 2] - 15.0) / (55.6 - 15.0), 0.0, 1.0))
    taper = np.where(x_dofs[:, 2] < 55.6, np.clip(taper, 0.4, 1.0), 1.0)
    x_scaled = cx + (x_dofs[:, 0] - cx) / taper
    y_scaled = cy + (x_dofs[:, 1] - cy) / taper

    d_axis = signed_distance_to_polygon(np.stack([x_scaled, y_scaled], axis=1), poly_vertices) * taper
    chi_wall = 1.0 / (1.0 + np.exp(-1.0 * d_axis)) # 1.0 outside tapered tube

    # 4. Define Boundary Locator Functions using the convex hull polygon
    def inlet_boundary(x):
        in_plane = np.isclose(x[2], 85.0)
        pts_2d = np.stack([x[0], x[1]], axis=1)
        in_poly = polygon_path.contains_points(pts_2d)
        return np.logical_and(in_plane, in_poly)

    def outlet_boundary(x):
        in_plane = np.isclose(x[2], 15.0)
        taper_val = 0.4
        x_scaled_val = cx + (x[0] - cx) / taper_val
        y_scaled_val = cy + (x[1] - cy) / taper_val
        pts_2d = np.stack([x_scaled_val, y_scaled_val], axis=1)
        in_poly = polygon_path.contains_points(pts_2d)
        return np.logical_and(in_plane, in_poly)

    def top_wall_boundary(x):
        in_plane = np.isclose(x[2], 85.0)
        pts_2d = np.stack([x[0], x[1]], axis=1)
        in_poly = polygon_path.contains_points(pts_2d)
        return np.logical_and(in_plane, np.logical_not(in_poly))

    def bottom_wall_boundary(x):
        in_plane = np.isclose(x[2], 15.0)
        taper_val = 0.4
        x_scaled_val = cx + (x[0] - cx) / taper_val
        y_scaled_val = cy + (x[1] - cy) / taper_val
        pts_2d = np.stack([x_scaled_val, y_scaled_val], axis=1)
        in_poly = polygon_path.contains_points(pts_2d)
        return np.logical_and(in_plane, np.logical_not(in_poly))

    def box_sides_boundary(x):
        return np.logical_or(
            np.isclose(x[0], -25.0),
            np.logical_or(
                np.isclose(x[0], 45.0),
                np.logical_or(
                    np.isclose(x[1], -35.0),
                    np.isclose(x[1], 35.0)
                )
            )
        )

    def outside_tube_boundary(x):
        taper_val = np.sqrt(np.clip((x[2] - 15.0) / (55.6 - 15.0), 0.0, 1.0))
        taper_val = np.where(x[2] < 55.6, np.clip(taper_val, 0.4, 1.0), 1.0)
        x_scaled_val = cx + (x[0] - cx) / taper_val
        y_scaled_val = cy + (x[1] - cy) / taper_val
        pts_2d = np.stack([x_scaled_val, y_scaled_val], axis=1)
        return np.logical_not(polygon_path.contains_points(pts_2d))

    # 5. Define Dirichlet BCs
    u_zero = fem.Function(V)
    u_inflow = fem.Function(V)
    p_zero = fem.Function(Q)

    inlet_dofs = fem.locate_dofs_geometrical(V, inlet_boundary)
    outlet_dofs = fem.locate_dofs_geometrical(V, outlet_boundary)
    top_wall_dofs = fem.locate_dofs_geometrical(V, top_wall_boundary)
    bottom_wall_dofs = fem.locate_dofs_geometrical(V, bottom_wall_boundary)
    box_wall_dofs = fem.locate_dofs_geometrical(V, box_sides_boundary)
    outside_dofs = fem.locate_dofs_geometrical(V, outside_tube_boundary)
    outlet_dofs_q = fem.locate_dofs_geometrical(Q, outlet_boundary)

    bc_top_wall = fem.dirichletbc(u_zero, top_wall_dofs)
    bc_bottom_wall = fem.dirichletbc(u_zero, bottom_wall_dofs)
    bc_box_wall = fem.dirichletbc(u_zero, box_wall_dofs)
    bc_outside = fem.dirichletbc(u_zero, outside_dofs)
    bc_pressure = fem.dirichletbc(p_zero, outlet_dofs_q)

    # 6. Variational Formulation (IPCS)
    dt = 0.05
    k = fem.Constant(domain, PETSc.ScalarType(dt))
    rho = 1.0
    mu = 0.1

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)
    p = ufl.TrialFunction(Q)
    q = ufl.TestFunction(Q)

    u_n = fem.Function(V)
    u_  = fem.Function(V)
    p_n = fem.Function(Q)
    p_  = fem.Function(Q)

    u_mid = 0.5 * (u + u_n)
    F1 = rho * ufl.inner((u - u_n) / k, v) * ufl.dx \
       + rho * ufl.inner(ufl.grad(u_n) * u_n, v) * ufl.dx \
       + mu * ufl.inner(ufl.grad(u_mid), ufl.grad(v)) * ufl.dx \
       + ufl.inner(ufl.grad(p_n), v) * ufl.dx \
       + ufl.inner(chi_func * u_mid, v) * ufl.dx
    
    a1 = ufl.lhs(F1)
    L1 = ufl.rhs(F1)

    a2 = ufl.inner(ufl.grad(p), ufl.grad(q)) * ufl.dx
    L2 = ufl.inner(ufl.grad(p_n), ufl.grad(q)) * ufl.dx - (rho / k) * ufl.div(u_) * q * ufl.dx

    a3 = ufl.inner(u, v) * ufl.dx
    L3 = ufl.inner(u_, v) * ufl.dx - (k / rho) * ufl.inner(ufl.grad(p_ - p_n), v) * ufl.dx

    # Compile forms outside the loop
    if domain.comm.rank == 0:
        print("[FEniCS] JIT-compiling Navier-Stokes variational forms...", flush=True)
    bilinear_form_1 = fem.form(a1)
    linear_form_1 = fem.form(L1)
    bilinear_form_2 = fem.form(a2)
    linear_form_2 = fem.form(L2)
    bilinear_form_3 = fem.form(a3)
    linear_form_3 = fem.form(L3)
    if domain.comm.rank == 0:
        print("[FEniCS] Variational forms compiled successfully.", flush=True)

    # Scipy KDTree mapping from geometry vertices to function DOFs for PyVista export
    V_out = fem.functionspace(domain, ("Lagrange", 1, (3,)))
    u_out = fem.Function(V_out)
    
    import dolfinx.plot
    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(domain, domain.topology.dim)
    
    vel_dof_coords = V_out.tabulate_dof_coordinates()
    tree_v = KDTree(vel_dof_coords)
    _, vel_dof_indices = tree_v.query(geometry)
    
    pres_dof_coords = Q.tabulate_dof_coordinates()
    tree_p = KDTree(pres_dof_coords)
    _, pres_dof_indices = tree_p.query(geometry)

    # 7. Time Stepping Loop
    for step in range(1, time_steps + 1):
        t = (step - 1.0) / max(time_steps - 1.0, 1.0)
        if domain.comm.rank == 0:
            print(f"[FEniCS] Solving time step {step}/{time_steps} (t = {t:.3f})...", flush=True)
        
        # Determine cardiac phase
        is_diastole = (t <= 0.5)
        
        # Calculate dynamic valve leaflet state
        f_open = np.sin(np.pi * t / 0.5) if is_diastole else 0.0
        
        disp = np.zeros_like(orig_vertices)
        
        # Anterior leaflet (GroupID == 1) moves negative Y and negative Z
        disp[group_ids == 1, 1] = -5.0 * weight[group_ids == 1] * w_seam[group_ids == 1] * f_open
        disp[group_ids == 1, 2] = -1.5 * weight[group_ids == 1] * w_seam[group_ids == 1] * f_open
        
        # Posterior leaflet (GroupID == 2) moves positive Y and negative Z
        disp[group_ids == 2, 1] = +5.0 * weight[group_ids == 2] * w_seam[group_ids == 2] * f_open
        disp[group_ids == 2, 2] = -1.5 * weight[group_ids == 2] * w_seam[group_ids == 2] * f_open
        
        # Chordae (GroupID == 3) can interpolate to follow the leaflets:
        chordae_mask = group_ids == 3
        if np.any(chordae_mask):
            y_coords = orig_vertices[chordae_mask, 1]
            disp[chordae_mask, 1] = np.where(y_coords > 0.36, 2.5, -2.5) * weight[chordae_mask] * w_seam[chordae_mask] * f_open
            disp[chordae_mask, 2] = -1.0 * weight[chordae_mask] * f_open
        
        disp_vertices = orig_vertices + disp
        
        # Save dynamic valve mesh state to file for renderer to overlay
        step_valve_mesh = valve_mesh.copy()
        step_valve_mesh.points = disp_vertices
        step_valve_mesh.point_data["Pressure"] = np.full(len(disp_vertices), 100.0 * (1.0 - f_open))
        step_valve_mesh.save(case_dir / f"valve_{step:03d}.vtu")

        # Brinkman valve penalization using KDTree (only penalize leaflets Group 1 and 2, not Group 3 chordae)
        leaflets_mask = (group_ids == 1) | (group_ids == 2)
        tree_valve = KDTree(disp_vertices[leaflets_mask])
        dists, _ = tree_valve.query(x_dofs)
        # Penalize near valve leaflets (except at the base inlet)
        chi_valve = np.where((dists < 2.0) & (x_dofs[:, 2] <= 74.0), 1.0, 0.0)
        
        # Update total penalization
        chi_tot = np.clip(chi_wall + chi_valve, 0.0, 1.0)
        chi_func.x.array[:] = chi_tot * 1e6

        # Boundary conditions values based on cardiac cycle
        bcs_step1 = [bc_top_wall, bc_bottom_wall, bc_box_wall, bc_outside]
        if is_diastole:
            # Diastolic flow: mitral inlet active, outlet open (pressure outlet)
            v_in_val = -150.0 * np.sin(np.pi * t / 0.5)
            u_inflow.interpolate(lambda x: np.stack([np.zeros_like(x[0]), np.zeros_like(x[0]), np.full_like(x[0], v_in_val)]))
            bcs_step1.append(fem.dirichletbc(u_inflow, inlet_dofs))
        else:
            # Systolic flow: mitral inlet closed
            u_inflow.interpolate(lambda x: np.stack([np.zeros_like(x[0]), np.zeros_like(x[0]), np.zeros_like(x[0])]))
            bcs_step1.append(fem.dirichletbc(u_inflow, inlet_dofs))

        # Solve step 1: Tentative velocity
        A1 = petsc.assemble_matrix(bilinear_form_1, bcs=bcs_step1)
        A1.assemble()
        b1 = petsc.assemble_vector(linear_form_1)
        petsc.apply_lifting(b1, [bilinear_form_1], [bcs_step1])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b1, bcs_step1)
        
        solver1 = PETSc.KSP().create(domain.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.PREONLY)
        solver1.getPC().setType(PETSc.PC.Type.LU)
        solver1.solve(b1, u_.x.petsc_vec)
        u_.x.scatter_forward()
        solver1.destroy()

        # Solve step 2: Pressure correction
        problem_2 = petsc.LinearProblem(a2, L2, bcs=[bc_pressure], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="step2")
        p_new = problem_2.solve()
        p_.x.array[:] = p_new.x.array
        p_.x.scatter_forward()

        # Solve step 3: Velocity correction
        problem_3 = petsc.LinearProblem(a3, L3, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"}, petsc_options_prefix="step3")
        u_new = problem_3.solve()
        u_n.x.array[:] = u_new.x.array
        u_n.x.scatter_forward()

        # Update pressure for next step
        p_n.x.array[:] = p_.x.array
        p_n.x.scatter_forward()

        # Export result
        out_path = case_dir / f"result_{step:03d}.vtu"
        u_out.interpolate(u_n)
        
        grid = pv.UnstructuredGrid(topology, cell_types, geometry)
        pts_z = geometry[:, 2]
        taper_geo = np.sqrt(np.clip((pts_z - 15.0) / (55.6 - 15.0), 0.0, 1.0))
        taper_geo = np.where(pts_z < 55.6, np.clip(taper_geo, 0.4, 1.0), 1.0)
        x_geo_scaled = cx + (geometry[:, 0] - cx) / taper_geo
        y_geo_scaled = cy + (geometry[:, 1] - cy) / taper_geo
        pts_2d_scaled = np.stack([x_geo_scaled, y_geo_scaled], axis=1)
        in_tube = polygon_path.contains_points(pts_2d_scaled)
        
        # Mask out velocity inside the leaflets to prevent streamlines passing through them
        leaflets_mask = (group_ids == 1) | (group_ids == 2)
        tree_valve_vtu = KDTree(disp_vertices[leaflets_mask])
        dists_to_valve, _ = tree_valve_vtu.query(geometry)
        in_valve = dists_to_valve < 2.5
        
        vel = u_out.x.array.reshape((-1, 3))[vel_dof_indices]
        vel[~in_tube] = 0.0
        vel[in_valve] = 0.0
        grid.point_data["Velocity"] = vel
        grid.point_data["Pressure"] = p_n.x.array[pres_dof_indices]
        grid.save(out_path)


def _run_synthetic_proxy_simulation(valve_mesh: pv.PolyData, case_dir: Path, time_steps: int) -> None:
    """Fallback proxy simulation when FEniCS is missing."""
    group_ids = valve_mesh.point_data["GroupID"]
    orig_vertices = valve_mesh.points.copy()
    z_min, z_max = 55.6, 73.9
    weight = np.clip((z_max - orig_vertices[:, 2]) / (z_max - z_min), 0.0, 1.0)

    # Calculate w_seam to prevent separation at the connected commissures
    from scipy.spatial import KDTree
    pts1 = orig_vertices[group_ids == 1]
    pts2 = orig_vertices[group_ids == 2]
    w_seam = np.ones(len(orig_vertices))
    if len(pts1) > 0 and len(pts2) > 0:
        tree2 = KDTree(pts2)
        dists, _ = tree2.query(pts1)
        shared_pts = pts1[dists < 1e-3]
        if len(shared_pts) > 0:
            tree_shared = KDTree(shared_pts)
            d_seam, _ = tree_shared.query(orig_vertices)
            w_seam = 1.0 - np.exp(-(d_seam / 10.0)**2)
    
    # Create a mock 3D structured box grid representing the isolated tube
    grid_x = np.linspace(-25, 45, 10)
    grid_y = np.linspace(-35, 35, 10)
    grid_z = np.linspace(15, 85, 12)
    grid = pv.RectilinearGrid(grid_x, grid_y, grid_z).cast_to_unstructured_grid()
    
    from scipy.spatial import ConvexHull
    from matplotlib.path import Path as MPath
    poly_vertices = valve_mesh.points[ConvexHull(valve_mesh.points[:, :2]).vertices][:, :2]
    polygon_path = MPath(poly_vertices)
    
    for step in range(1, time_steps + 1):
        t = (step - 1.0) / max(time_steps - 1.0, 1.0)
        is_diastole = (t <= 0.5)
        f_open = np.sin(np.pi * t / 0.5) if is_diastole else 0.0
        
        # Dynamic valve leaflet mesh - same kinematics as FEniCS simulation
        disp = np.zeros_like(orig_vertices)
        disp[group_ids == 1, 1] = -5.0 * weight[group_ids == 1] * w_seam[group_ids == 1] * f_open
        disp[group_ids == 1, 2] = -1.5 * weight[group_ids == 1] * w_seam[group_ids == 1] * f_open
        disp[group_ids == 2, 1] = +5.0 * weight[group_ids == 2] * w_seam[group_ids == 2] * f_open
        disp[group_ids == 2, 2] = -1.5 * weight[group_ids == 2] * w_seam[group_ids == 2] * f_open
        
        chordae_mask = group_ids == 3
        if np.any(chordae_mask):
            y_coords = orig_vertices[chordae_mask, 1]
            disp[chordae_mask, 1] = np.where(y_coords > 0.36, 2.5, -2.5) * weight[chordae_mask] * w_seam[chordae_mask] * f_open
            disp[chordae_mask, 2] = -1.0 * weight[chordae_mask] * f_open
            
        disp_vertices = orig_vertices + disp
        step_valve_mesh = valve_mesh.copy()
        step_valve_mesh.points = disp_vertices
        step_valve_mesh.point_data["Pressure"] = np.full(len(disp_vertices), 100.0 * (1.0 - f_open))
        step_valve_mesh.save(case_dir / f"valve_{step:03d}.vtu")
        
        # Synthetic velocity and pressure field inside the tapered tube
        points = grid.points.copy()
        pts_z = points[:, 2]
        taper_pts = np.sqrt(np.clip((pts_z - 15.0) / (55.6 - 15.0), 0.0, 1.0))
        taper_pts = np.where(pts_z < 55.6, np.clip(taper_pts, 0.4, 1.0), 1.0)
        cx, cy = 9.99, 1.396
        x_scaled = cx + (points[:, 0] - cx) / taper_pts
        y_scaled = cy + (points[:, 1] - cy) / taper_pts
        pts_2d = np.stack([x_scaled, y_scaled], axis=1)
        in_tube = polygon_path.contains_points(pts_2d)
        
        # Block flow inside the leaflets
        leaflets_mask = (group_ids == 1) | (group_ids == 2)
        tree_valve = KDTree(disp_vertices[leaflets_mask])
        dists_to_valve, _ = tree_valve.query(points)
        in_valve = dists_to_valve < 2.5
        
        # Inflow magnitude
        flow_mag = 150.0 * np.sin(np.pi * t / 0.5) if is_diastole else 0.0
        
        velocity = np.zeros_like(points)
        pressure = np.zeros(len(points))
        
        if is_diastole:
            # Flow goes down inside the tapered tube
            v_scale = np.clip(1.0 / (taper_pts ** 2), 1.0, 4.0)
            velocity[in_tube, 2] = -flow_mag * v_scale[in_tube]
            velocity[in_valve] = 0.0
            pressure[in_tube] = flow_mag * 10.0 * (1.0 - f_open)
        else:
            velocity[in_tube] = 0.0
            pressure[in_tube] = 0.0
            
        frame_grid = grid.copy()
        frame_grid.point_data["Velocity"] = velocity
        frame_grid.point_data["Pressure"] = pressure
        frame_grid.save(case_dir / f"result_{step:03d}.vtu")


def render_mock_case(case_dir: Path) -> MockMitralVisualization:
    """Render screenshots and an animation for a completed mock mitral case."""
    case_dir = case_dir.resolve()
    result_paths = sorted(case_dir.glob("result_[0-9][0-9][0-9].vtu"))
    if not result_paths:
        raise FileNotFoundError(f"No rendered result files found under {case_dir}")

    render_dir = case_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    overview_png = render_dir / "overview.png"
    cutaway_png = render_dir / "cutaway.png"
    animation_gif = render_dir / "animation.gif"

    old_mplconfig = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str((case_dir / ".mplconfig").resolve())
    (case_dir / ".mplconfig").mkdir(exist_ok=True)

    try:
        _render_overview(result_paths[-1], overview_png, cutaway=False)
        _render_overview(result_paths[-1], cutaway_png, cutaway=True)
        _render_animation(result_paths, animation_gif)
    finally:
        if old_mplconfig is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mplconfig

    viz = MockMitralVisualization(
        case_dir=case_dir,
        overview_png=overview_png,
        cutaway_png=cutaway_png,
        animation_gif=animation_gif,
    )
    (render_dir / "render_summary.json").write_text(json.dumps(viz.to_dict(), indent=2))
    return viz


def _summarize_result(result_vtu: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    result = pv.read(result_vtu)
    pressure = np.asarray(result.point_data["Pressure"])
    velocity = np.asarray(result.point_data["Velocity"])
    velocity_mag = np.linalg.norm(velocity, axis=1)
    return (float(np.min(pressure)), float(np.max(pressure))), (float(np.min(velocity_mag)), float(np.max(velocity_mag)))


def _render_overview(result_path: Path, output_png: Path, cutaway: bool) -> None:
    # Find step index from filename (e.g. result_005.vtu -> 5)
    step_str = result_path.stem.split("_")[-1]
    valve_path = result_path.parent / f"valve_{step_str}.vtu"
    
    grid = pv.read(result_path)
    valve = pv.read(valve_path) if valve_path.exists() else None
    
    # Generate streamlines
    streamlines = grid.streamlines(
        vectors="Velocity",
        source_center=(10.8, 0.36, 80.0),
        source_radius=8.0,
        n_points=30,
        max_length=100.0
    )
    
    _save_matplotlib_frame(
        grid=grid,
        valve=valve,
        streamlines=streamlines,
        output_path=output_png,
        title="Mitral Valve Flow Overview" if not cutaway else "Mitral Valve Flow Cutaway",
        add_colorbar=True,
        cutaway=cutaway
    )


def _render_animation(result_paths: list[Path], output_gif: Path) -> None:
    from PIL import Image

    frame_paths: list[Path] = []
    for idx, result_path in enumerate(result_paths, start=1):
        step_str = f"{idx:03d}"
        valve_path = result_path.parent / f"valve_{step_str}.vtu"
        
        grid = pv.read(result_path)
        valve = pv.read(valve_path) if valve_path.exists() else None
        
        # Compute streamlines
        streamlines = grid.streamlines(
            vectors="Velocity",
            source_center=(10.8, 0.36, 80.0),
            source_radius=8.0,
            n_points=35,
            max_length=100.0
        )
        
        frame_path = output_gif.parent / f"frame_{idx:03d}.png"
        _save_matplotlib_frame(
            grid=grid,
            valve=valve,
            streamlines=streamlines,
            output_path=frame_path,
            title=f"Timestep {idx} / {len(result_paths)}",
            add_colorbar=(idx == 1),
            cutaway=False
        )
        frame_paths.append(frame_path)

    frames = [Image.open(path).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    frames[0].save(
        output_gif,
        save_all=True,
        append_images=frames[1:],
        duration=500,
        loop=0,
    )


def _save_matplotlib_frame(
    grid: pv.UnstructuredGrid,
    valve: pv.PolyData | None,
    streamlines: pv.PolyData,
    output_path: Path,
    title: str,
    add_colorbar: bool,
    cutaway: bool
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

    fig = plt.figure(figsize=(10, 8), dpi=180)
    fig.patch.set_facecolor("#f3f0ea")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f3f0ea")

    # 1. Plot D-shaped tube wall (rings and vertical struts)
    if valve is not None:
        from scipy.spatial import ConvexHull
        hull_indices = ConvexHull(valve.points[:, :2]).vertices
        poly_2d = valve.points[hull_indices][:, :2]
        poly_2d = np.vstack([poly_2d, poly_2d[0]])
        cx, cy = 9.99, 1.396
        
        # 1. Plot 15 horizontal rings representing the tapered chambers
        for z_ring in np.linspace(15.0, 85.0, 15):
            taper = np.sqrt(np.clip((z_ring - 15.0) / (55.6 - 15.0), 0.0, 1.0)) if z_ring < 55.6 else 1.0
            taper = np.clip(taper, 0.4, 1.0)
            
            x_ring = cx + (poly_2d[:, 0] - cx) * taper
            y_ring = cy + (poly_2d[:, 1] - cy) * taper
            
            if cutaway:
                mask = x_ring >= 10.8
                ax.plot(x_ring[mask], y_ring[mask], z_ring, color="#555555", linestyle="--", linewidth=0.8, alpha=0.3)
            else:
                ax.plot(x_ring, y_ring, z_ring, color="#555555", linestyle="--", linewidth=0.8, alpha=0.3)
                
        # 2. Plot vertical lines at the corners of the hull that taper down to the apex
        step = max(1, len(poly_2d) // 8)
        for idx in range(0, len(poly_2d) - 1, step):
            x_hull = poly_2d[idx, 0]
            y_hull = poly_2d[idx, 1]
            
            z_pts = np.linspace(15.0, 85.0, 50)
            x_pts = []
            y_pts = []
            for z in z_pts:
                taper = np.sqrt(np.clip((z - 15.0) / (55.6 - 15.0), 0.0, 1.0)) if z < 55.6 else 1.0
                taper = np.clip(taper, 0.4, 1.0)
                x_pts.append(cx + (x_hull - cx) * taper)
                y_pts.append(cy + (y_hull - cy) * taper)
                
            x_pts = np.array(x_pts)
            y_pts = np.array(y_pts)
            
            if cutaway:
                mask = x_pts >= 10.8
                if np.any(mask):
                    ax.plot(x_pts[mask], y_pts[mask], z_pts[mask], color="#555555", linestyle=":", linewidth=0.8, alpha=0.3)
            else:
                ax.plot(x_pts, y_pts, z_pts, color="#555555", linestyle=":", linewidth=0.8, alpha=0.3)

        # 3. Plot semi-transparent physical wall surfaces for the Left Atrium and Left Ventricle
        z_coords = np.linspace(15.0, 85.0, 30)
        n_pts = len(poly_2d)
        X_wall = np.zeros((len(z_coords), n_pts))
        Y_wall = np.zeros((len(z_coords), n_pts))
        Z_wall = np.zeros((len(z_coords), n_pts))
        for i, z in enumerate(z_coords):
            taper = np.sqrt(np.clip((z - 15.0) / (55.6 - 15.0), 0.0, 1.0)) if z < 55.6 else 1.0
            taper = np.clip(taper, 0.4, 1.0)
            X_wall[i, :] = cx + (poly_2d[:, 0] - cx) * taper
            Y_wall[i, :] = cy + (poly_2d[:, 1] - cy) * taper
            Z_wall[i, :] = z

        if cutaway:
            X_plot = np.where(X_wall >= 10.8, X_wall, np.nan)
            Y_plot = np.where(X_wall >= 10.8, Y_wall, np.nan)
            Z_plot = np.where(X_wall >= 10.8, Z_wall, np.nan)
        else:
            X_plot, Y_plot, Z_plot = X_wall, Y_wall, Z_wall

        ax.plot_surface(X_plot, Y_plot, Z_plot, color="#e27a7a", alpha=0.15, shade=True, rcount=30, ccount=n_pts)

    # 2. Plot Valve Leaflet Surface
    if valve is not None:
        valve_surface = valve.extract_surface().triangulate().clean()
        if cutaway:
            valve_surface = valve_surface.clip(normal=(1, 0, 0), origin=(10.8, 0, 0), invert=False)
        
        points_v = valve_surface.points
        faces_v = valve_surface.faces.reshape(-1, 4)[:, 1:]
        pressure_v = np.asarray(valve_surface.point_data["Pressure"])
        tri_pressure = pressure_v[faces_v].mean(axis=1)

        norm_p = Normalize(vmin=0.0, vmax=100.0)
        cmap_p = matplotlib.colormaps["coolwarm"]
        colors_p = cmap_p(norm_p(tri_pressure))

        verts_v = points_v[faces_v]
        collection_v = Poly3DCollection(verts_v, linewidths=0.05, alpha=0.9)
        collection_v.set_facecolor(colors_p)
        collection_v.set_edgecolor("#333333")
        ax.add_collection3d(collection_v)

    # 3. Plot Blood Flow Streamlines
    if streamlines is not None and streamlines.n_points > 0:
        lines_data = []
        vel_mag_data = []
        
        i = 0
        while i < len(streamlines.lines):
            n_pts = streamlines.lines[i]
            indices = streamlines.lines[i+1 : i+1+n_pts]
            pts = streamlines.points[indices]
            
            vels = streamlines.point_data["Velocity"][indices]
            v_mag = np.linalg.norm(vels, axis=1)
            
            if cutaway:
                mask = pts[:, 0] >= 10.8
                if len(pts[mask]) < 2:
                    i += 1 + n_pts
                    continue
                pts = pts[mask]
                v_mag = v_mag[mask]
            
            for j in range(len(pts) - 1):
                lines_data.append([pts[j], pts[j+1]])
                vel_mag_data.append((v_mag[j] + v_mag[j+1]) / 2.0)
                
            i += 1 + n_pts
        
        if lines_data:
            norm_v = Normalize(vmin=0.0, vmax=150.0)
            cmap_v = matplotlib.colormaps["viridis"]
            colors_v = cmap_v(norm_v(vel_mag_data))
            
            collection_lines = Line3DCollection(lines_data, colors=colors_v, linewidths=1.8, alpha=0.85)
            ax.add_collection3d(collection_lines)

    # 4. View and labels settings
    ax.set_xlim(-25.0, 45.0)
    ax.set_ylim(-35.0, 35.0)
    ax.set_zlim(15.0, 85.0)
    ax.view_init(elev=22, azim=-55)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=12, color="#222222", weight="bold")

    # 5. Add legends/colorbars
    if add_colorbar:
        mappable_p = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=100.0), cmap=matplotlib.colormaps["coolwarm"])
        mappable_p.set_array([])
        cbar_p = fig.colorbar(mappable_p, ax=ax, shrink=0.45, pad=0.01, location="left")
        cbar_p.set_label("Leaflet Pressure (Pa)", fontsize=9, color="#222222")
        cbar_p.ax.yaxis.set_tick_params(color="#222222", labelcolor="#222222")

        mappable_v = plt.cm.ScalarMappable(norm=Normalize(vmin=0.0, vmax=150.0), cmap=matplotlib.colormaps["viridis"])
        mappable_v.set_array([])
        cbar_v = fig.colorbar(mappable_v, ax=ax, shrink=0.45, pad=0.01, location="right")
        cbar_v.set_label("Blood Velocity (mm/s)", fontsize=9, color="#222222")
        cbar_v.ax.yaxis.set_tick_params(color="#222222", labelcolor="#222222")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _load_clean_obj(path: Path) -> pv.PolyData:
    group_data = {}  # group_name -> {'faces': []}
    vertices: list[list[float]] = []
    current_group = None
    
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("o ") or line.startswith("g "):
                current_group = line.split()[1]
                group_data[current_group] = {'faces': []}
            elif line.startswith("f "):
                parts = line.split()[1:]
                indices = [int(token.split("/")[0]) - 1 for token in parts]
                if not current_group:
                    current_group = "default"
                    group_data[current_group] = {'faces': []}
                group_data[current_group]['faces'].append(indices)
                
    # Flip the valve Z coordinates to make it concave side up (annulus at top, free margin at bottom)
    z_coords = [v[2] for v in vertices]
    z_min, z_max = min(z_coords), max(z_coords)
    for v in vertices:
        v[2] = z_max + z_min - v[2]
                
    group_meshes = []
    for name, data in group_data.items():
        faces = data['faces']
        if len(faces) == 0:
            continue
        unique_indices = sorted(list(set(idx for face in faces for idx in face)))
        idx_map = {old: new for new, old in enumerate(unique_indices)}
        v_local = [vertices[idx] for idx in unique_indices]
        
        f_local = []
        for face in faces:
            if len(face) < 3:
                continue
            for idx in range(1, len(face) - 1):
                f_local.append([3, idx_map[face[0]], idx_map[face[idx]], idx_map[face[idx + 1]]])
                
        if not v_local or not f_local:
            continue
            
        flat_faces = np.asarray(f_local, dtype=np.int64).ravel()
        mesh = pv.PolyData(np.asarray(v_local, dtype=float), flat_faces).triangulate().clean()
        
        group_id = 0
        if "ANTERIOR_LEAFLET" in name:
            group_id = 1
        elif "POSTERIOR_LEAFLET" in name:
            group_id = 2
        elif "CHORDAE" in name or "PAPILLARY" in name:
            group_id = 3
            
        mesh.point_data["GroupID"] = np.full(mesh.n_points, group_id, dtype=np.int32)
        group_meshes.append(mesh)
        
    all_pts = []
    all_faces = []
    all_group_ids = []
    pt_offset = 0
    
    for mesh in group_meshes:
        all_pts.append(mesh.points)
        faces = mesh.faces.copy()
        i = 0
        while i < len(faces):
            n = faces[i]
            for j in range(1, n + 1):
                faces[i + j] += pt_offset
            i += n + 1
        all_faces.append(faces)
        all_group_ids.append(mesh.point_data["GroupID"])
        pt_offset += mesh.n_points
        
    merged_pts = np.vstack(all_pts)
    merged_faces = np.concatenate(all_faces)
    merged_group_ids = np.concatenate(all_group_ids)
    
    merged_mesh = pv.PolyData(merged_pts, merged_faces)
    merged_mesh.point_data["GroupID"] = merged_group_ids
    return _strip_arrays(merged_mesh)


def _strip_arrays(mesh: pv.DataSet) -> pv.DataSet:
    for name in list(mesh.point_data.keys()):
        if name != "GroupID":
            mesh.point_data.pop(name)
    for name in list(mesh.cell_data.keys()):
        mesh.cell_data.pop(name)
    for name in list(mesh.field_data.keys()):
        mesh.field_data.pop(name)
    return mesh


def view_mock_case(case_dir: Path, cutaway: bool = False) -> None:
    """Open an interactive 3D PyVista plotter window to play the completed simulation."""
    import pyvista as pv
    import time
    
    result_paths = sorted(case_dir.glob("result_[0-9][0-9][0-9].vtu"))
    if not result_paths:
        raise FileNotFoundError(f"No result files found under {case_dir}")
        
    grids = [pv.read(p) for p in result_paths]
    valves = []
    for i in range(1, len(result_paths) + 1):
        v_path = case_dir / f"valve_{i:03d}.vtu"
        valves.append(pv.read(v_path) if v_path.exists() else None)
        
    plotter = pv.Plotter(title="TEER CDSS Interactive Hemodynamics Player")
    plotter.set_background("#f3f0ea")
    
    grid_actor = [None]
    valve_actor = [None]
    streamline_actor = [None]
    
    step_idx = 0
    
    def update_scene(val):
        nonlocal step_idx
        step_idx = int(val)
        if grid_actor[0]:
            plotter.remove_actor(grid_actor[0])
        if valve_actor[0]:
            plotter.remove_actor(valve_actor[0])
        if streamline_actor[0]:
            plotter.remove_actor(streamline_actor[0])
            
        grid = grids[step_idx]
        valve = valves[step_idx]
        
        # Streamlines
        streamlines = grid.streamlines(
            vectors="Velocity",
            source_center=(10.8, 0.36, 80.0),
            source_radius=8.0,
            n_points=35,
            max_length=100.0
        )
        
        shown_streamlines = streamlines.clip(normal=(1, 0, 0), origin=(10.8, 0, 0), invert=False) if cutaway else streamlines
        if shown_streamlines.n_points > 0:
            streamline_actor[0] = plotter.add_mesh(
                shown_streamlines,
                cmap="viridis",
                scalars="Velocity",
                line_width=3.0,
                render_lines_as_tubes=True,
                scalar_bar_args={"title": "Velocity (mm/s)", "vertical": True, "position_x": 0.88, "position_y": 0.15, "width": 0.05, "height": 0.7}
            )
        else:
            streamline_actor[0] = None
        
        if valve:
            valve_shown = valve.clip(normal=(1, 0, 0), origin=(10.8, 0, 0), invert=False) if cutaway else valve
            if valve_shown.n_points > 0:
                valve_actor[0] = plotter.add_mesh(
                    valve_shown,
                    cmap="coolwarm",
                    scalars="Pressure",
                    clim=[0.0, 100.0],
                    ambient=0.2,
                    diffuse=0.8,
                    scalar_bar_args={"title": "Pressure (Pa)", "vertical": True, "position_x": 0.02, "position_y": 0.15, "width": 0.05, "height": 0.7}
                )
            else:
                valve_actor[0] = None
            
        plotter.add_text(
            f"Timestep {step_idx + 1} / {len(grids)}",
            name="step_text",
            position="upper_edge",
            color="black",
            font_size=14
        )
        
    plotter.add_slider_widget(
        callback=update_scene,
        rng=[0, len(grids) - 1],
        value=0,
        title="Cardiac Cycle Time Step",
        pointa=(0.25, 0.08),
        pointb=(0.75, 0.08),
        style="modern",
        color="black"
    )
    
    # Draw simple outline of D-shaped tube rings and vertical columns
    if valves and valves[0] is not None:
        from scipy.spatial import ConvexHull
        hull_indices = ConvexHull(valves[0].points[:, :2]).vertices
        poly_2d = valves[0].points[hull_indices][:, :2]
        poly_2d = np.vstack([poly_2d, poly_2d[0]])
        cx, cy = 9.99, 1.396
        
        # Build StructuredGrid for physical semi-transparent chamber wall
        z_coords = np.linspace(15.0, 85.0, 30)
        grid_pts = []
        for z in z_coords:
            taper = np.sqrt(np.clip((z - 15.0) / (55.6 - 15.0), 0.0, 1.0)) if z < 55.6 else 1.0
            taper = np.clip(taper, 0.4, 1.0)
            x_ring = cx + (poly_2d[:, 0] - cx) * taper
            y_ring = cy + (poly_2d[:, 1] - cy) * taper
            ring_3d = np.stack([x_ring, y_ring, np.full_like(x_ring, z)], axis=1)
            grid_pts.append(ring_3d)
            
        grid_pts = np.array(grid_pts)
        nx = grid_pts.shape[1]
        ny = grid_pts.shape[0]
        
        chamber_mesh = pv.StructuredGrid()
        chamber_mesh.points = grid_pts.reshape(-1, 3)
        chamber_mesh.dimensions = (nx, ny, 1)
        
        if cutaway:
            chamber_shown = chamber_mesh.clip(normal=(1, 0, 0), origin=(10.8, 0, 0), invert=False)
        else:
            chamber_shown = chamber_mesh
            
        plotter.add_mesh(
            chamber_shown,
            color="#e27a7a",
            opacity=0.15,
            style="surface",
            ambient=0.3,
            specular=0.1,
            smooth_shading=True
        )
        
        # Plot rings
        for z_ring in np.linspace(15.0, 85.0, 15):
            taper = np.sqrt(np.clip((z_ring - 15.0) / (55.6 - 15.0), 0.0, 1.0)) if z_ring < 55.6 else 1.0
            taper = np.clip(taper, 0.4, 1.0)
            
            x_ring = cx + (poly_2d[:, 0] - cx) * taper
            y_ring = cy + (poly_2d[:, 1] - cy) * taper
            
            if cutaway:
                mask = x_ring >= 10.8
                if np.sum(mask) > 1:
                    ring_pts = np.stack([x_ring[mask], y_ring[mask], np.full(np.sum(mask), z_ring)], axis=1)
                    plotter.add_lines(ring_pts, color="#aaaaaa", width=1.0)
            else:
                ring_pts = np.stack([x_ring, y_ring, np.full(len(x_ring), z_ring)], axis=1)
                plotter.add_lines(ring_pts, color="#aaaaaa", width=1.0)
            
        # Plot vertical lines
        step = max(1, len(poly_2d) // 12)
        for idx in range(0, len(poly_2d) - 1, step):
            x_hull = poly_2d[idx, 0]
            y_hull = poly_2d[idx, 1]
            
            z_pts = np.linspace(15.0, 85.0, 50)
            x_pts = []
            y_pts = []
            for z in z_pts:
                taper = np.sqrt(np.clip((z - 15.0) / (55.6 - 15.0), 0.0, 1.0)) if z < 55.6 else 1.0
                taper = np.clip(taper, 0.4, 1.0)
                x_pts.append(cx + (x_hull - cx) * taper)
                y_pts.append(cy + (y_hull - cy) * taper)
                
            x_pts = np.array(x_pts)
            y_pts = np.array(y_pts)
            
            if cutaway:
                mask = x_pts >= 10.8
                if np.sum(mask) > 1:
                    line_pts = np.stack([x_pts[mask], y_pts[mask], z_pts[mask]], axis=1)
                    plotter.add_lines(line_pts, color="#aaaaaa", width=1.0)
            else:
                line_pts = np.stack([x_pts, y_pts, z_pts], axis=1)
                plotter.add_lines(line_pts, color="#aaaaaa", width=1.0)

    plotter.camera_position = [
        (150.0, -120.0, 100.0),
        (10.8, 0.36, 50.0),
        (0.0, 0.0, 1.0)
    ]
    
    update_scene(0)
    
    def callback(step):
        nonlocal step_idx
        update_scene(step_idx)
        try:
            widgets = getattr(plotter, "widgets", plotter)
            sliders = getattr(widgets, "slider_widgets", [])
            if sliders:
                sliders[0].GetRepresentation().SetValue(step_idx)
        except Exception:
            pass
        step_idx = (step_idx + 1) % len(grids)

    plotter.add_timer_event(max_steps=2000000000, duration=150, callback=callback)
    plotter.show()
