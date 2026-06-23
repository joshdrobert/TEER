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
    time_steps: int = 180,
    time_step_size: float = 1e-3,
    static_diastole: bool = False,
) -> MockMitralResult:
    """Build a runnable mitral simulation using FEniCS/DOLFINx."""
    workspace = workspace.resolve()
    valve_obj = valve_obj.resolve()

    case_dir = workspace / "artifacts" / "mock_mitral_uris_fsi"
    case_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare Mesh — support both .obj and .msh formats
    valve_scale_factor = 1.0
    suffix = valve_obj.suffix.lower()
    if suffix == ".stl":
        valve_mesh = _load_stl_valve(valve_obj)
    elif suffix == ".msh":
        valve_mesh = _load_msh_valve(valve_obj)
    else:
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
        _run_synthetic_proxy_simulation(valve_mesh, case_dir, time_steps, static_diastole=static_diastole)

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
    """Execute 3D transient Navier-Stokes simulation with Brinkman penalization of the STL valve mesh."""
    import dolfinx
    from dolfinx import mesh as dmesh, fem
    from dolfinx.fem import petsc
    import ufl
    from mpi4py import MPI
    from petsc4py import PETSc
    from scipy.spatial import KDTree
    import pyvista as pv

    # 1. Load STL valve mesh and determine bounds dynamically
    valve_mesh = pv.read(base_vtu)
    bounds = valve_mesh.bounds
    cx = (bounds[0] + bounds[1]) / 2.0
    cy = (bounds[2] + bounds[3]) / 2.0

    x_min, x_max = bounds[0] - 15.0, bounds[1] + 15.0
    y_min, y_max = bounds[2] - 15.0, bounds[3] + 15.0
    z_min, z_max = bounds[4] - 10.0, bounds[5] + 10.0

    if MPI.COMM_WORLD.rank == 0:
        print(f"[FEniCS] Bounds X: [{x_min:.1f}, {x_max:.1f}], Y: [{y_min:.1f}, {y_max:.1f}], Z: [{z_min:.1f}, {z_max:.1f}]", flush=True)

    # 2. Create 3D Cartesian box grid conforming to bounds
    domain = dmesh.create_box(
        MPI.COMM_WORLD,
        [np.array([x_min, y_min, z_min]), np.array([x_max, y_max, z_max])],
        [12, 10, 15],
        dmesh.CellType.tetrahedron
    )

    # 3. Function Spaces (Taylor-Hood P2-P1 elements)
    V = fem.functionspace(domain, ("Lagrange", 2, (3,)))
    Q = fem.functionspace(domain, ("Lagrange", 1))
    W_chi = fem.functionspace(domain, ("Lagrange", 1))

    # 4. Brinkman Penalization Field
    chi_func = fem.Function(W_chi)
    x_dofs = W_chi.tabulate_dof_coordinates()

    # Precompute cylindrical chamber wall penalization
    d_axis = np.sqrt((x_dofs[:, 0] - cx)**2 + (x_dofs[:, 1] - cy)**2)
    R_cylinder = 0.5 * max(bounds[1] - bounds[0], bounds[3] - bounds[2]) + 5.0
    chi_wall = np.where(d_axis > R_cylinder, 1.0, 0.0)

    # Precompute valve leaflets penalization using KDTree
    tree_valve = KDTree(valve_mesh.points)
    dists, _ = tree_valve.query(x_dofs)
    chi_valve = np.where(dists < 3.0, 1.0, 0.0)

    # Combined penalization
    chi_tot = np.clip(chi_wall + chi_valve, 0.0, 1.0)
    chi_func.x.array[:] = chi_tot * 1e6

    # 5. Boundary Locator Functions
    def inlet_boundary(x):
        return np.isclose(x[2], z_max)

    def outlet_boundary(x):
        return np.isclose(x[2], z_min)

    def box_sides_boundary(x):
        return np.logical_or(
            np.isclose(x[0], x_min),
            np.logical_or(
                np.isclose(x[0], x_max),
                np.logical_or(
                    np.isclose(x[1], y_min),
                    np.isclose(x[1], y_max)
                )
            )
        )

    # 6. Define Dirichlet BCs
    u_zero = fem.Function(V)
    u_inflow = fem.Function(V)
    p_zero = fem.Function(Q)

    inlet_dofs = fem.locate_dofs_geometrical(V, inlet_boundary)
    outlet_dofs_q = fem.locate_dofs_geometrical(Q, outlet_boundary)
    box_wall_dofs = fem.locate_dofs_geometrical(V, box_sides_boundary)

    bc_box_wall = fem.dirichletbc(u_zero, box_wall_dofs)
    bc_pressure = fem.dirichletbc(p_zero, outlet_dofs_q)

    # 7. Variational Formulation (IPCS Scheme)
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

    if domain.comm.rank == 0:
        print("[FEniCS] JIT-compiling Navier-Stokes variational forms...", flush=True)
    bilinear_form_1 = fem.form(a1)
    linear_form_1 = fem.form(L1)
    bilinear_form_2 = fem.form(a2)
    linear_form_2 = fem.form(L2)
    bilinear_form_3 = fem.form(a3)
    linear_form_3 = fem.form(L3)

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

    # Save static valve files to keep the pipeline happy
    for step in range(1, time_steps + 1):
        step_valve_mesh = valve_mesh.copy()
        step_valve_mesh.point_data["Pressure"] = np.zeros(valve_mesh.n_points)
        step_valve_mesh.cast_to_unstructured_grid().save(case_dir / f"valve_{step:03d}.vtu")

    # 8. Time Stepping Loop
    for step in range(1, time_steps + 1):
        t = (step - 1.0) / max(time_steps - 1.0, 1.0)
        
        # Physiological pulsatile flow with 60-step cycle:
        step_in_cycle = (step - 1) % 60
        t_cycle = step_in_cycle / 59.0
        
        if t_cycle <= 0.5:
            # E-wave/A-wave diastolic filling
            e_wave = np.exp(-(t_cycle - 0.20)**2 / (2 * 0.06**2))
            a_wave = 0.55 * np.exp(-(t_cycle - 0.40)**2 / (2 * 0.05**2))
            v_in_val = -180.0 * np.clip(e_wave + a_wave, 0.05, 1.0)
        else:
            # Systolic backflow (regurgitation)
            v_in_val = +80.0 * np.sin(np.pi * (t_cycle - 0.5) / 0.5)

        if domain.comm.rank == 0:
            print(f"[FEniCS] Step {step}/{time_steps} (t={t:.3f}, t_cycle={t_cycle:.3f}, Vin={v_in_val:.1f} mm/s)...", flush=True)

        # Update inlet boundary condition
        u_inflow.interpolate(lambda x: np.stack([np.zeros_like(x[0]), np.zeros_like(x[0]), np.full_like(x[0], v_in_val)]))
        bc_inlet = fem.dirichletbc(u_inflow, inlet_dofs)

        bcs_step1 = [bc_box_wall, bc_inlet]

        # Solve step 1: Tentative velocity with GMRES and SOR PC
        A1 = petsc.assemble_matrix(bilinear_form_1, bcs=bcs_step1)
        A1.assemble()
        b1 = petsc.assemble_vector(linear_form_1)
        petsc.apply_lifting(b1, [bilinear_form_1], [bcs_step1])
        b1.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        petsc.set_bc(b1, bcs_step1)
        
        solver1 = PETSc.KSP().create(domain.comm)
        solver1.setOperators(A1)
        solver1.setType(PETSc.KSP.Type.GMRES)
        solver1.getPC().setType(PETSc.PC.Type.SOR)
        solver1.setTolerances(rtol=1e-5, atol=1e-5)
        solver1.solve(b1, u_.x.petsc_vec)
        u_.x.scatter_forward()
        solver1.destroy()

        # Solve step 2: Pressure correction with CG and SOR PC
        problem_2 = petsc.LinearProblem(
            a2, L2,
            bcs=[bc_pressure],
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "sor",
                "ksp_rtol": "1e-5",
                "ksp_atol": "1e-5"
            },
            petsc_options_prefix="step2"
        )
        p_new = problem_2.solve()
        p_.x.array[:] = p_new.x.array
        p_.x.scatter_forward()

        # Solve step 3: Velocity correction with CG and SOR PC
        problem_3 = petsc.LinearProblem(
            a3, L3,
            bcs=[],
            petsc_options={
                "ksp_type": "cg",
                "pc_type": "sor",
                "ksp_rtol": "1e-5",
                "ksp_atol": "1e-5"
            },
            petsc_options_prefix="step3"
        )
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
        
        # Compute dynamic flow masking (velocity is zero outside cylinder and inside valve wall)
        pts_coords = geometry
        d_axis_geo = np.sqrt((pts_coords[:, 0] - cx)**2 + (pts_coords[:, 1] - cy)**2)
        in_cylinder = d_axis_geo <= R_cylinder
        
        tree_valve_geo = KDTree(valve_mesh.points)
        dists_to_valve, _ = tree_valve_geo.query(pts_coords)
        in_valve = dists_to_valve < 2.5
        
        vel = u_out.x.array.reshape((-1, 3))[vel_dof_indices]
        vel[~in_cylinder] = 0.0
        vel[in_valve] = 0.0
        
        grid.point_data["Velocity"] = vel
        grid.point_data["Pressure"] = p_n.x.array[pres_dof_indices]
        grid.save(out_path)




# ---------------------------------------------------------------------------
# .msh valve loader
# ---------------------------------------------------------------------------

def _load_msh_valve(path: Path) -> pv.PolyData:
    """Load a Gmsh .msh valve mesh into a PyVista PolyData with GroupID."""
    import meshio

    m = meshio.read(str(path))
    pts = m.points.copy()
    tris = None
    for block in m.cells:
        if block.type == "triangle":
            tris = block.data
            break
    if tris is None:
        raise ValueError(f"No triangle cells found in {path}")

    # Flip Z so that annulus (wide part at lower Z) is at the top (higher Z)
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    pts[:, 2] = z_max + z_min - pts[:, 2]

    # Build PyVista PolyData
    faces = np.column_stack([np.full(len(tris), 3), tris]).ravel()
    mesh = pv.PolyData(pts, faces)

    # Assign GroupID: split into anterior/posterior by Y position
    cy = pts[:, 1].mean()
    group_ids = np.ones(len(pts), dtype=np.int32)  # anterior (1)
    group_ids[pts[:, 1] > cy] = 2  # posterior (2)

    mesh.point_data["GroupID"] = group_ids
    return mesh


def _load_stl_valve(path: Path) -> pv.PolyData:
    """Load an STL valve mesh into a PyVista PolyData with GroupID."""
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()

    pts = mesh.points.copy()

    # Flip Z so that annulus (wide part at lower Z) is at the top (higher Z)
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    pts[:, 2] = z_max + z_min - pts[:, 2]
    mesh.points = pts

    # Assign GroupID: split into anterior/posterior by Y centroid
    cy = pts[:, 1].mean()
    group_ids = np.ones(len(pts), dtype=np.int32)  # anterior (1)
    group_ids[pts[:, 1] > cy] = 2  # posterior (2)

    mesh.point_data["GroupID"] = group_ids
    return _strip_arrays(mesh)


# ---------------------------------------------------------------------------
# Anatomy geometry helpers for the 2D cross-section renderer
# ---------------------------------------------------------------------------

def _build_chamber_geometry():
    """Return LA and LV wall boundary coordinates for the 2D cross-section.

    Orientation: LA at top, LV at bottom, valve annulus at the junction.
    Normalised (x, z) space: x in [-0.5, 0.5], z in [0, 1].
    """
    annulus_z = 0.52
    la_top = 1.0
    lv_bottom = 0.02
    annulus_half_w = 0.34

    # LA walls
    la_hw = 0.40
    la_left_x = np.array([-la_hw, -la_hw, -annulus_half_w])
    la_left_z = np.array([la_top, annulus_z + 0.03, annulus_z])
    la_right_x = np.array([la_hw, la_hw, annulus_half_w])
    la_right_z = np.array([la_top, annulus_z + 0.03, annulus_z])

    # LV walls — perfectly symmetrical bullet shape
    z_lv = np.linspace(lv_bottom, annulus_z, 40)
    w_lv = 0.04 + (annulus_half_w - 0.04) * ((z_lv - lv_bottom) / (annulus_z - lv_bottom)) ** 0.60
    
    lv_left_x = -w_lv[::-1]
    lv_left_z = z_lv[::-1]
    
    lv_right_x = w_lv
    lv_right_z = z_lv
    
    lv_x = np.concatenate([lv_left_x, lv_right_x])
    lv_z = np.concatenate([lv_left_z, lv_right_z])

    return {
        "annulus_z": annulus_z, "annulus_half_w": annulus_half_w,
        "la_top": la_top, "lv_bottom": lv_bottom,
        "la_left_x": la_left_x, "la_left_z": la_left_z,
        "la_right_x": la_right_x, "la_right_z": la_right_z,
        "lv_x": lv_x, "lv_z": lv_z,
    }


def _inside_chamber(X, Z, geom):
    """Return boolean mask: True where (X, Z) is inside the LA or LV lumen."""
    annulus_z = geom["annulus_z"]
    annulus_hw = geom["annulus_half_w"]
    la_top = geom["la_top"]

    inside = np.zeros_like(X, dtype=bool)

    # LA region
    la_hw = 0.40
    la_mask = (Z >= annulus_z) & (Z <= la_top)
    frac = np.clip((Z - annulus_z) / (la_top - annulus_z + 1e-12), 0, 1)
    half_w = annulus_hw + (la_hw - annulus_hw) * frac
    inside |= la_mask & (np.abs(X) <= half_w)

    # LV region
    from matplotlib.path import Path as MPath
    lv_poly = np.column_stack([geom["lv_x"], geom["lv_z"]])
    lv_path = MPath(lv_poly)
    pts = np.column_stack([X.ravel(), Z.ravel()])
    inside |= lv_path.contains_points(pts).reshape(X.shape)

    return inside


def _compute_valve_cross_section(valve_mesh, f_open, geom):
    """Compute the 2D (x, z) cross-section silhouette of the valve by slicing the 3D mesh.

    Takes a thin slice through the real mesh at the Y-midplane using PyVista's slice.
    """
    import pyvista as pv
    import numpy as np

    cy_mesh = valve_mesh.points[:, 1].mean()
    try:
        sliced = valve_mesh.slice(normal=[0, 1, 0], origin=[0, cy_mesh, 0])
        lines = sliced.lines
        if len(lines) > 0:
            pA = lines[1::3]
            pB = lines[2::3]
            pts = sliced.points
            raw_segments = np.stack([pts[pA], pts[pB]], axis=1) # (N, 2, 3)
        else:
            raw_segments = np.zeros((0, 2, 3))
    except Exception as e:
        print(f"Slice warning: {e}")
        raw_segments = np.zeros((0, 2, 3))

    cx_mesh = 114.86
    z_max_mesh = 151.31
    z_min_mesh = 1.34

    annulus_hw = geom["annulus_half_w"]
    x_scale = 0.88 * annulus_hw / (179.10 - cx_mesh)
    z_scale = (0.52 - 0.22) / (z_max_mesh - z_min_mesh)

    leaflet_segments = []
    all_pts = []

    for seg in raw_segments:
        mx1, mz1 = seg[0, 0], seg[0, 2]
        mx2, mz2 = seg[1, 0], seg[1, 2]

        nx1 = (mx1 - cx_mesh) * x_scale
        nz1 = 0.52 + (mz1 - z_max_mesh) * z_scale

        nx2 = (mx2 - cx_mesh) * x_scale
        nz2 = 0.52 + (mz2 - z_max_mesh) * z_scale

        leaflet_segments.append(np.array([[nx1, nz1], [nx2, nz2]]))
        all_pts.append([nx1, nz1])
        all_pts.append([nx2, nz2])

    chordae_lines = []
    if len(all_pts) > 0:
        all_pts = np.array(all_pts)
        pap_l1 = np.array([-0.18, 0.16])
        pap_l2 = np.array([-0.15, 0.13])
        pap_r1 = np.array([0.18, 0.16])
        pap_r2 = np.array([0.15, 0.13])

        # Left leaflet tip
        left_mask = all_pts[:, 0] < 0
        if np.any(left_mask):
            left_pts = all_pts[left_mask]
            tip_l_idx = np.argmin(left_pts[:, 1])
            tip_l = left_pts[tip_l_idx]
            
            chordae_lines.append(np.array([tip_l, pap_l1]))
            chordae_lines.append(np.array([tip_l + np.array([-0.015, 0.01]), pap_l2]))
            chordae_lines.append(np.array([tip_l + np.array([0.01, 0.005]), pap_r1]))

        # Right leaflet tip
        right_mask = all_pts[:, 0] >= 0
        if np.any(right_mask):
            right_pts = all_pts[right_mask]
            tip_r_idx = np.argmin(right_pts[:, 1])
            tip_r = right_pts[tip_r_idx]

            chordae_lines.append(np.array([tip_r, pap_r1]))
            chordae_lines.append(np.array([tip_r + np.array([0.015, 0.01]), pap_r2]))
            chordae_lines.append(np.array([tip_r + np.array([-0.01, 0.005]), pap_l1]))

    return leaflet_segments, chordae_lines


# ---------------------------------------------------------------------------
# Synthetic 2D flow field generator
# ---------------------------------------------------------------------------

def _generate_2d_velocity_field(nx_res, nz_res, t, f_open, geom, prev_vz=None,
                                static_diastole=False):
    """Generate a realistic 2D velocity field with transmitral flow.

    When *static_diastole* is True the valve stays open and a physiological
    pulsatile waveform (E-wave + A-wave) drives the flow continuously.
    """
    x = np.linspace(-0.50, 0.50, nx_res)
    z = np.linspace(-0.02, 1.02, nz_res)
    X, Z = np.meshgrid(x, z)

    annulus_z = geom["annulus_z"]
    annulus_hw = geom["annulus_half_w"]
    inside = _inside_chamber(X, Z, geom)

    Vx = np.zeros_like(X)
    Vz = np.zeros_like(X)

    # ---- pulsatile waveform for static-diastole mode -----------------------
    if static_diastole:
        # Physiological E-wave / A-wave transmitral velocity profile
        # t cycles 0 → 1 over the animation; map to a cardiac diastolic fill
        phase = (t % 1.0) * 2.0 * np.pi
        # E-wave peak ~0.2, A-wave peak ~0.7 in normalised time
        e_wave = np.exp(-((t % 1.0) - 0.20) ** 2 / (2 * 0.06 ** 2))
        a_wave = 0.55 * np.exp(-((t % 1.0) - 0.70) ** 2 / (2 * 0.05 ** 2))
        pulsatile_factor = np.clip(e_wave + a_wave, 0.08, 1.0)
        effective_open = 0.70  # valve fixed ~70 % open
    else:
        pulsatile_factor = 1.0
        effective_open = f_open

    is_diastole = static_diastole or (t <= 0.5)

    if is_diastole and effective_open > 0.01:
        orifice_hw = annulus_hw * effective_open * 0.65
        jet_strength = 95.0 * effective_open * pulsatile_factor
        sigma_x = orifice_hw * 0.5
        jet_depth = annulus_z * 0.82 * effective_open

        # Transmitral jet in LV
        lv_mask = inside & (Z < annulus_z)
        z_below = annulus_z - Z[lv_mask]
        x_at = X[lv_mask]
        decay = np.exp(-z_below / (jet_depth + 0.05))
        sigma_spread = sigma_x * (1.0 + 1.8 * z_below / (annulus_z + 0.01))
        lateral_spread = np.exp(-x_at ** 2 / (2 * sigma_spread ** 2))
        Vz[lv_mask] = -jet_strength * decay * lateral_spread

        # Counter-rotating vortex pair
        vortex_strength = jet_strength * 0.70 * effective_open
        for sign in [-1, 1]:
            vx_c = sign * annulus_hw * 0.42
            vz_c = annulus_z - jet_depth * 0.45
            dx = X - vx_c; dz = Z - vz_c
            r2 = dx ** 2 + dz ** 2
            r_s = annulus_hw * 0.30
            env = np.exp(-r2 / (2 * r_s ** 2))
            Vx += -sign * dz * vortex_strength * env / (r_s + 0.01) * inside
            Vz += sign * dx * vortex_strength * env / (r_s + 0.01) * inside

        # Secondary deep LV vortex
        for sign in [-1, 1]:
            vx_c2 = sign * annulus_hw * 0.25
            vz_c2 = annulus_z * 0.22
            dx2 = X - vx_c2; dz2 = Z - vz_c2
            r2_2 = dx2 ** 2 + dz2 ** 2
            r_s2 = annulus_hw * 0.40
            env2 = np.exp(-r2_2 / (2 * r_s2 ** 2))
            Vx += sign * dz2 * jet_strength * 0.38 * effective_open * env2 / (r_s2 + 0.01) * inside
            Vz += -sign * dx2 * jet_strength * 0.38 * effective_open * env2 / (r_s2 + 0.01) * inside

        # Leaflet-tip micro-turbulence (small-scale vorticity)
        if effective_open > 0.15:
            rng = np.random.RandomState(int(t * 1000) % 2**31)
            turb_amp = 6.0 * effective_open * pulsatile_factor
            tip_z_band = (Z > annulus_z - 0.12) & (Z < annulus_z + 0.03)
            tip_x_band = (np.abs(X) > orifice_hw * 0.6) & (np.abs(X) < orifice_hw * 1.8)
            tip_mask = inside & tip_z_band & tip_x_band
            n_tip = int(tip_mask.sum())
            if n_tip > 0:
                Vx[tip_mask] += rng.randn(n_tip) * turb_amp
                Vz[tip_mask] += rng.randn(n_tip) * turb_amp * 0.6

        # LA drift / convergent flow towards the orifice
        la_mask = inside & (Z >= annulus_z)
        z_above = Z[la_mask] - annulus_z
        la_decay = np.exp(-z_above / 0.18)
        sigma_la = sigma_x * (1.0 + 3.2 * z_above)
        lateral_spread_la = np.exp(-X[la_mask] ** 2 / (2.0 * sigma_la ** 2))

        # Accelerating flow towards orifice
        Vz[la_mask] = (-24.0 * effective_open * pulsatile_factor
                       - (jet_strength - 24.0) * la_decay * lateral_spread_la)
        # Horizontal flow converging to centerline
        Vx[la_mask] = -X[la_mask] * 48.0 * effective_open * pulsatile_factor * la_decay * lateral_spread_la
    else:
        # Systole: residual swirl
        lv_mask = inside & (Z < annulus_z)
        if np.any(lv_mask):
            t_sys = max(t - 0.5, 0.0)
            decay_f = np.exp(-t_sys * 3.0) * 0.4
            dx_lv = X[lv_mask]; dz_lv = Z[lv_mask] - annulus_z * 0.35
            r2 = dx_lv ** 2 + dz_lv ** 2
            r_lv = annulus_hw * 0.45
            env = np.exp(-r2 / (2 * r_lv ** 2)) * decay_f
            Vx[lv_mask] = -dz_lv * 20.0 * env
            Vz[lv_mask] = dx_lv * 20.0 * env

    Vx[~inside] = 0.0; Vz[~inside] = 0.0
    from scipy.ndimage import gaussian_filter
    Vx = gaussian_filter(Vx, sigma=2.5)
    Vz = gaussian_filter(Vz, sigma=2.5)
    Vx[~inside] = 0.0; Vz[~inside] = 0.0

    return X, Z, Vx, Vz, inside


def _run_synthetic_proxy_simulation(valve_mesh: pv.PolyData, case_dir: Path,
                                    time_steps: int,
                                    static_diastole: bool = False) -> None:
    """Fallback proxy simulation using physics-informed analytical FSI modeling."""
    import numpy as np
    import pyvista as pv
    from scipy.spatial import KDTree

    pts_orig = valve_mesh.points.copy()
    bounds = valve_mesh.bounds
    cx = (bounds[0] + bounds[1]) / 2.0
    cy = (bounds[2] + bounds[3]) / 2.0
    z_min, z_max = bounds[4], bounds[5]
    z_len = z_max - z_min
    z_orifice = z_min + 0.4 * z_len

    x_min, x_max = bounds[0] - 15.0, bounds[1] + 15.0
    y_min, y_max = bounds[2] - 15.0, bounds[3] + 15.0
    z_min_grid, z_max_grid = bounds[4] - 10.0, bounds[5] + 10.0

    # Create a nice 3D structured mesh grid conforming to bounds
    grid_3d = pv.RectilinearGrid(
        np.linspace(x_min, x_max, 24),
        np.linspace(y_min, y_max, 20),
        np.linspace(z_min_grid, z_max_grid, 30)
    ).cast_to_unstructured_grid()

    points = grid_3d.points
    px = points[:, 0]
    py = points[:, 1]
    pz = points[:, 2]

    d_axis = np.sqrt((px - cx)**2 + (py - cy)**2)
    R_cylinder = 0.5 * max(bounds[1] - bounds[0], bounds[3] - bounds[2]) + 5.0

    for step in range(1, time_steps + 1):
        t = (step - 1.0) / max(time_steps - 1.0, 1.0)

        # 1. 0D Lumped Parameter Network Model of Left Heart Hemodynamics
        step_in_cycle = (step - 1) % 60
        t_cycle = step_in_cycle / 59.0

        if static_diastole:
            # Under static diastole, valve remains open, flow is driven continuously
            p_la = 12.0
            p_lv = 3.0
            q_flow = (p_la - p_lv) / 0.05
            f_open = 0.8
            v_peak = -180.0 * (0.4 + 0.6 * np.sin(2.0 * np.pi * t))
            p_lv_val, p_la_val = p_lv, p_la
        else:
            if t_cycle <= 0.5:
                # Diastole: Flow from LA to LV
                # Physiological E-wave and A-wave filling pressures
                p_la = 5.0 + 8.0 * np.exp(-(t_cycle - 0.20)**2 / (2 * 0.06**2)) + 5.0 * np.exp(-(t_cycle - 0.40)**2 / (2 * 0.05**2))
                p_lv = 2.0 + 12.0 * np.exp(-t_cycle / 0.08)
                
                delta_p = p_la - p_lv
                q_flow = delta_p / 0.05 if delta_p > 0.0 else 0.0
                f_open = np.clip(q_flow / 250.0, 0.0, 1.0)
                v_peak = -180.0 * f_open
                p_lv_val, p_la_val = p_lv, p_la
            else:
                # Systole: LV contracts, pressure spikes
                p_lv = 5.0 + 115.0 * np.sin(np.pi * (t_cycle - 0.5) / 0.5)
                p_la = 8.0 + 4.0 * (t_cycle - 0.5)
                
                # Regurgitant leakage through closing gap
                delta_p = p_la - p_lv
                q_flow = delta_p / 3.0
                f_open = 0.0
                v_peak = 80.0 * np.clip(np.abs(q_flow) / 38.0, 0.0, 1.0)
                p_lv_val, p_la_val = p_lv, p_la

        # 2. Dynamic 3D Leaflet Deformation Math
        pts = pts_orig.copy()
        gids = valve_mesh.point_data["GroupID"]
        
        z_pts = pts_orig[:, 2]
        z_norm = (z_pts - z_min) / (z_max - z_min + 1e-5)
        d_bend = (1.0 - np.clip(z_norm, 0.0, 1.0))**2
        
        D_max = 16.0  # max Y deflection in mm
        D_z_max = 4.0  # max Z downward deflection in mm
        
        dy = np.zeros_like(z_pts)
        dz = np.zeros_like(z_pts)
        
        # Group 1 (Anterior, Y <= cy) moves in -Y
        dy = np.where(gids == 1, -D_max * f_open * d_bend, dy)
        # Group 2 (Posterior, Y > cy) moves in +Y
        dy = np.where(gids == 2, +D_max * f_open * d_bend, dy)
        # Group 3 (Chordae/Papillary) deforms slightly towards the centerline
        dy = np.where(gids == 3, np.where(pts_orig[:, 1] <= cy, -0.2 * D_max * f_open * d_bend, +0.2 * D_max * f_open * d_bend), dy)
        
        # Leaflets stretch downward when opening
        dz = np.where(np.isin(gids, [1, 2, 3]), -D_z_max * f_open * d_bend, dz)
        
        pts[:, 1] += dy
        pts[:, 2] += dz
        
        # Save dynamic valve mesh with transient pressure values
        step_valve_mesh = valve_mesh.copy()
        step_valve_mesh.points = pts
        step_valve_mesh.point_data["Pressure"] = np.full(valve_mesh.n_points, p_lv_val if t_cycle > 0.5 else p_la_val)
        step_valve_mesh.cast_to_unstructured_grid().save(case_dir / f"valve_{step:03d}.vtu")

        # 3. Physics-informed 3D fluid jet dynamics and toroidal vortex ring
        velocity = np.zeros_like(points)
        
        if (t_cycle <= 0.5 or static_diastole) and np.abs(v_peak) > 0.1:
            # --- DIASTOLIC FLOW (Transmitral Jet + Toroidal Vortices) ---
            r_orifice = 0.25 * (bounds[1] - bounds[0]) * f_open + 5.0
            
            # Conical jet spreading
            z_diff = z_orifice - pz
            r_jet = np.where(z_diff > 0.0, r_orifice + 0.22 * z_diff, r_orifice)
            
            # Gaussian jet velocity profile
            sigma_jet = r_jet / 2.0
            v_jet = v_peak * np.exp(-((px - cx)**2 + (py - cy)**2) / (2.0 * sigma_jet**2))
            
            # Spatial decay towards LV apex (low Z)
            z_decay = np.clip((pz - z_min_grid) / (z_orifice - z_min_grid + 1e-5), 0.0, 1.0)
            v_jet = v_jet * (0.3 + 0.7 * z_decay)
            
            # Toroidal Vortex Ring in LV
            z_vortex = z_orifice - 0.3 * z_len
            r_vortex_ring = 1.3 * r_orifice
            
            r_xy = np.sqrt((px - cx)**2 + (py - cy)**2)
            d_r = r_xy - r_vortex_ring
            d_z = pz - z_vortex
            d_dist = np.sqrt(d_r**2 + d_z**2)
            
            sigma_v = 0.15 * z_len
            v_swirl = -0.6 * v_peak * (r_xy / (r_vortex_ring + 1e-5)) * np.exp(-d_dist**2 / (2.0 * sigma_v**2))
            
            v_radial = v_swirl * d_z / (d_dist + 1e-5)
            v_z_vortex = -v_swirl * d_r / (d_dist + 1e-5)
            
            v_x = v_radial * (px - cx) / (r_xy + 1e-5)
            v_y = v_radial * (py - cy) / (r_xy + 1e-5)
            
            velocity[:, 0] = v_x
            velocity[:, 1] = v_y
            velocity[:, 2] = v_jet + v_z_vortex
            
            # Sigmoid pressure drop across the orifice plane
            pressure = p_lv_val + (p_la_val - p_lv_val) / (1.0 + np.exp(-(pz - z_orifice) / (0.08 * z_len)))
            
        else:
            # --- SYSTOLIC FLOW (Regurgitant Jet + Recirculation) ---
            if np.abs(v_peak) > 0.1:
                r_regurg = 5.0
                sigma_regurg = r_regurg / 2.0
                v_regurg = v_peak * np.exp(-((px - cx)**2 + (py - cy)**2) / (2.0 * sigma_regurg**2))
                z_decay = np.exp(-(pz - z_orifice) / (0.25 * z_len))
                v_regurg = np.where(pz >= z_orifice, v_regurg * z_decay, 0.0)
            else:
                v_regurg = np.zeros_like(pz)
            
            # Add a gentle swirling recirculation in the LV
            r_xy = np.sqrt((px - cx)**2 + (py - cy)**2)
            v_swirl = 25.0 * np.exp(-(pz - (z_min + 0.3 * z_len))**2 / (2.0 * (0.2 * z_len)**2)) * (r_xy / 15.0) * np.exp(-r_xy**2 / (2.0 * 20.0**2)) * np.sin(np.pi * (t_cycle - 0.5) / 0.5)
            
            velocity[:, 0] = -v_swirl * (py - cy) / (r_xy + 1e-5)
            velocity[:, 1] = v_swirl * (px - cx) / (r_xy + 1e-5)
            velocity[:, 2] = v_regurg
            
            # Sigmoid pressure drop across the closed leaflets
            pressure = p_lv_val - (p_lv_val - p_la_val) / (1.0 + np.exp(-(pz - z_orifice) / (0.08 * z_len)))

        # 4. Dynamic flow masking against deforming leaflets
        tree_deformed = KDTree(pts)
        dists_to_valve, _ = tree_deformed.query(points)
        in_valve = dists_to_valve < 2.5
        
        in_cylinder = d_axis <= R_cylinder
        velocity[~in_cylinder] = 0.0
        velocity[in_valve] = 0.0

        frame_grid = grid_3d.copy()
        frame_grid.point_data["Velocity"] = velocity
        frame_grid.point_data["Pressure"] = pressure
        frame_grid.save(case_dir / f"result_{step:03d}.vtu")

        if step % 20 == 0 or step == time_steps:
            print(f"  [Synthetic] Step {step}/{time_steps} done (t={t:.3f}, v_peak={v_peak:.1f} mm/s)")



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
        peak_idx = len(result_paths) // 3
        _render_single_frame(peak_idx + 1, result_paths, overview_png, case_dir, True)
        mid_idx = len(result_paths) // 2
        _render_single_frame(mid_idx + 1, result_paths, cutaway_png, case_dir, True)
        _render_animation(result_paths, animation_gif, case_dir)
    finally:
        if old_mplconfig is None:
            os.environ.pop("MPLCONFIGDIR", None)
        else:
            os.environ["MPLCONFIGDIR"] = old_mplconfig

    viz = MockMitralVisualization(
        case_dir=case_dir, overview_png=overview_png,
        cutaway_png=cutaway_png, animation_gif=animation_gif,
    )
    (render_dir / "render_summary.json").write_text(json.dumps(viz.to_dict(), indent=2))
    return viz


def _render_single_frame(step_idx, result_paths, output_png, case_dir, add_colorbar=True):
    step_str = f"{step_idx:03d}"
    valve_path = case_dir / f"valve_{step_str}.vtu"
    flow_path = case_dir / f"flow2d_{step_str}.npz"

    valve = pv.read(valve_path) if valve_path.exists() else None
    geom = _build_chamber_geometry()

    if flow_path.exists():
        data = np.load(flow_path)
        X, Z, Vx, Vz = data["X"], data["Z"], data["Vx"], data["Vz"]
        inside = data["inside"].astype(bool)
        t = float(data["t"][0]); f_open = float(data["f_open"][0])
    else:
        n = len(result_paths)
        t = (step_idx - 1.0) / max(n - 1.0, 1.0)
        f_open = np.sin(np.pi * t / 0.5) if t <= 0.5 else 0.0
        X, Z, Vx, Vz, inside = _generate_2d_velocity_field(300, 600, t, f_open, geom)

    _save_cfd_frame(X, Z, Vx, Vz, inside, valve, geom, t, f_open,
                    output_png, "Mitral Valve Hemodynamics", add_colorbar)


def _render_animation(result_paths, output_gif, case_dir):
    from PIL import Image
    geom = _build_chamber_geometry()
    frame_paths = []

    for idx, rp in enumerate(result_paths, start=1):
        step_str = f"{idx:03d}"
        valve_path = case_dir / f"valve_{step_str}.vtu"
        flow_path = case_dir / f"flow2d_{step_str}.npz"
        valve = pv.read(valve_path) if valve_path.exists() else None

        if flow_path.exists():
            data = np.load(flow_path)
            X, Z, Vx, Vz = data["X"], data["Z"], data["Vx"], data["Vz"]
            inside = data["inside"].astype(bool)
            t = float(data["t"][0]); f_open = float(data["f_open"][0])
        else:
            t = (idx - 1.0) / max(len(result_paths) - 1.0, 1.0)
            f_open = np.sin(np.pi * t / 0.5) if t <= 0.5 else 0.0
            X, Z, Vx, Vz, inside = _generate_2d_velocity_field(300, 600, t, f_open, geom)

        fp = output_gif.parent / f"frame_{idx:03d}.png"
        _save_cfd_frame(X, Z, Vx, Vz, inside, valve, geom, t, f_open,
                        fp, f"Cardiac Cycle  t = {t:.2f}", True)
        frame_paths.append(fp)

    frames = [Image.open(p).convert("RGBA") for p in frame_paths]
    frames_p = [f.convert("P", palette=Image.ADAPTIVE, colors=256) for f in frames]
    frames_p[0].save(output_gif, save_all=True, append_images=frames_p[1:],
                     duration=400, loop=0)


def _save_cfd_frame(X, Z, Vx, Vz, inside, valve, geom, t, f_open,
                    output_path, title, add_colorbar):
    """Render a 2D CFD-style pseudocolor frame."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon as MplPolygon
    import matplotlib.patheffects as pe

    fig, ax = plt.subplots(figsize=(6, 12), dpi=200)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    Vz_plot = Vz.copy()
    Vz_plot[~inside] = np.nan
    v_max = max(np.nanmax(np.abs(Vz_plot[np.isfinite(Vz_plot)])), 1.0)
    v_clamp = min(v_max, 85.0)

    cfd_colors = [
        (0.00, (0.10, 0.12, 0.68)), (0.15, (0.20, 0.35, 0.85)),
        (0.30, (0.45, 0.65, 0.95)), (0.42, (0.75, 0.88, 0.98)),
        (0.50, (0.96, 0.96, 0.96)), (0.58, (0.98, 0.82, 0.70)),
        (0.70, (0.95, 0.55, 0.35)), (0.85, (0.85, 0.25, 0.15)),
        (1.00, (0.65, 0.08, 0.08)),
    ]
    cfd_cmap = LinearSegmentedColormap.from_list("cfd_flow", cfd_colors, N=512)

    im = ax.pcolormesh(X, Z, Vz_plot, cmap=cfd_cmap, vmin=-v_clamp, vmax=v_clamp,
                       shading="gouraud", rasterized=True)

    wc = "#111111"; wlw = 2.8
    ax.plot(geom["la_left_x"], geom["la_left_z"], color=wc, lw=wlw, solid_capstyle="round")
    ax.plot(geom["la_right_x"], geom["la_right_z"], color=wc, lw=wlw, solid_capstyle="round")
    ax.plot([geom["la_left_x"][0], geom["la_right_x"][0]],
            [geom["la_top"], geom["la_top"]], color=wc, lw=wlw, solid_capstyle="round")
    ax.plot(geom["lv_x"], geom["lv_z"], color=wc, lw=wlw, solid_capstyle="round")

    if valve is not None:
        leaflet_segments, chordae_lines = _compute_valve_cross_section(valve, f_open, geom)
        for line in chordae_lines:
            ax.plot(line[:, 0], line[:, 1], color="black", lw=1.0, zorder=4)
        
        # Draw leaflets as lines/curves using LineCollection
        from matplotlib.collections import LineCollection
        lc = LineCollection(leaflet_segments, colors="black", linewidths=3.5, capstyle="round", zorder=5)
        ax.add_collection(lc)

        # Plot annulus anchors (nodes)
        ax.plot([-geom["annulus_half_w"]], [geom["annulus_z"]], "ko", ms=5, zorder=6)
        ax.plot([geom["annulus_half_w"]], [geom["annulus_z"]], "ko", ms=5, zorder=6)

    ls = dict(fontsize=16, fontweight="bold", color="#1a1a1a", ha="center", va="center",
              path_effects=[pe.withStroke(linewidth=4, foreground="white")])
    ax.text(0, geom["la_top"] - 0.08, "LA", **ls)
    ax.text(0, geom["annulus_z"] * 0.38, "LV", **ls)

    phase = "Diastole" if t <= 0.5 else "Systole"
    vs = f"Valve {'Open' if f_open > 0.1 else 'Closed'} ({f_open:.0%})"
    ax.set_title(f"{title}\n{phase}  ·  {vs}", fontsize=13, fontweight="bold",
                 color="#222222", pad=12)

    if add_colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.50, pad=0.04, aspect=30)
        cbar.set_label("Axial Velocity (mm/s)", fontsize=10, color="#222222")
        cbar.ax.yaxis.set_tick_params(color="#222222", labelcolor="#222222", labelsize=9)

    ax.set_xlim(-0.50, 0.50); ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, facecolor="white", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _summarize_result(result_vtu):
    result = pv.read(result_vtu)
    pressure = np.asarray(result.point_data["Pressure"])
    velocity = np.asarray(result.point_data["Velocity"])
    velocity_mag = np.linalg.norm(velocity, axis=1)
    return (float(np.min(pressure)), float(np.max(pressure))), \
           (float(np.min(velocity_mag)), float(np.max(velocity_mag)))


def _load_clean_obj(path: Path) -> pv.PolyData:
    group_data = {}
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

    z_coords = [v[2] for v in vertices]
    z_min, z_max = min(z_coords), max(z_coords)
    for v in vertices:
        v[2] = z_max + z_min - v[2]

    group_meshes = []
    for name, data in group_data.items():
        faces = data['faces']
        if not faces:
            continue
        unique_indices = sorted(set(idx for face in faces for idx in face))
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
        if "ANTERIOR_LEAFLET" in name: group_id = 1
        elif "POSTERIOR_LEAFLET" in name: group_id = 2
        elif "CHORDAE" in name or "PAPILLARY" in name: group_id = 3
        mesh.point_data["GroupID"] = np.full(mesh.n_points, group_id, dtype=np.int32)
        group_meshes.append(mesh)

    all_pts, all_faces, all_gids = [], [], []
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
        all_gids.append(mesh.point_data["GroupID"])
        pt_offset += mesh.n_points

    merged = pv.PolyData(np.vstack(all_pts), np.concatenate(all_faces))
    merged.point_data["GroupID"] = np.concatenate(all_gids)
    return _strip_arrays(merged)


def _strip_arrays(mesh):
    for name in list(mesh.point_data.keys()):
        if name != "GroupID":
            mesh.point_data.pop(name)
    for name in list(mesh.cell_data.keys()):
        mesh.cell_data.pop(name)
    for name in list(mesh.field_data.keys()):
        mesh.field_data.pop(name)
    return mesh


def view_mock_case(case_dir: Path, cutaway: bool = False) -> None:
    """Open an interactive matplotlib animation window."""
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Polygon as MplPolygon
    import matplotlib.patheffects as pe

    case_dir = case_dir.resolve()
    result_paths = sorted(case_dir.glob("result_[0-9][0-9][0-9].vtu"))
    if not result_paths:
        raise FileNotFoundError(f"No result files found under {case_dir}")

    geom = _build_chamber_geometry()
    n_frames = len(result_paths)

    frames_data, valves = [], []
    for idx in range(1, n_frames + 1):
        s = f"{idx:03d}"
        vp = case_dir / f"valve_{s}.vtu"
        fp = case_dir / f"flow2d_{s}.npz"
        valves.append(pv.read(vp) if vp.exists() else None)
        if fp.exists():
            d = np.load(fp)
            frames_data.append({"X": d["X"], "Z": d["Z"], "Vx": d["Vx"], "Vz": d["Vz"],
                                "inside": d["inside"].astype(bool),
                                "t": float(d["t"][0]), "f_open": float(d["f_open"][0])})
        else:
            t = (idx - 1.0) / max(n_frames - 1.0, 1.0)
            fo = np.sin(np.pi * t / 0.5) if t <= 0.5 else 0.0
            X, Z, Vx, Vz, ins = _generate_2d_velocity_field(200, 400, t, fo, geom)
            frames_data.append({"X": X, "Z": Z, "Vx": Vx, "Vz": Vz,
                                "inside": ins, "t": t, "f_open": fo})

    cfd_colors = [
        (0.00, (0.10, 0.12, 0.68)), (0.15, (0.20, 0.35, 0.85)),
        (0.30, (0.45, 0.65, 0.95)), (0.42, (0.75, 0.88, 0.98)),
        (0.50, (0.96, 0.96, 0.96)), (0.58, (0.98, 0.82, 0.70)),
        (0.70, (0.95, 0.55, 0.35)), (0.85, (0.85, 0.25, 0.15)),
        (1.00, (0.65, 0.08, 0.08)),
    ]
    cfd_cmap = LinearSegmentedColormap.from_list("cfd_flow", cfd_colors, N=512)
    fig, ax = plt.subplots(figsize=(6, 12))
    fig.patch.set_facecolor("white")

    def update(fi):
        ax.clear(); ax.set_facecolor("white")
        fd = frames_data[fi]; v = valves[fi]
        Vz_p = fd["Vz"].copy(); Vz_p[~fd["inside"]] = np.nan
        ax.pcolormesh(fd["X"], fd["Z"], Vz_p, cmap=cfd_cmap, vmin=-85, vmax=85,
                      shading="gouraud", rasterized=True)
        wc = "#111111"
        ax.plot(geom["la_left_x"], geom["la_left_z"], color=wc, lw=2.8)
        ax.plot(geom["la_right_x"], geom["la_right_z"], color=wc, lw=2.8)
        ax.plot([geom["la_left_x"][0], geom["la_right_x"][0]],
                [geom["la_top"], geom["la_top"]], color=wc, lw=2.8)
        ax.plot(geom["lv_x"], geom["lv_z"], color=wc, lw=2.8)
        if v is not None:
            leaflet_segments, chordae_lines = _compute_valve_cross_section(v, fd["f_open"], geom)
            for line in chordae_lines:
                ax.plot(line[:, 0], line[:, 1], color="black", lw=1.0, zorder=4)
            from matplotlib.collections import LineCollection
            lc = LineCollection(leaflet_segments, colors="black", linewidths=3.5, capstyle="round", zorder=5)
            ax.add_collection(lc)
            ax.plot([-geom["annulus_half_w"]], [geom["annulus_z"]], "ko", ms=5, zorder=6)
            ax.plot([geom["annulus_half_w"]], [geom["annulus_z"]], "ko", ms=5, zorder=6)
        ls = dict(fontsize=14, fontweight="bold", color="#222", ha="center", va="center",
                  path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        ax.text(0, geom["la_top"] - 0.08, "LA", **ls)
        ax.text(0, geom["annulus_z"] * 0.38, "LV", **ls)
        ph = "Diastole" if fd["t"] <= 0.5 else "Systole"
        vs = f"{'Open' if fd['f_open'] > 0.1 else 'Closed'} ({fd['f_open']:.0%})"
        ax.set_title(f"Mitral Valve  ·  {ph}  ·  {vs}\nt={fd['t']:.2f}  Frame {fi+1}/{n_frames}",
                     fontsize=12, fontweight="bold")
        ax.set_xlim(-0.50, 0.50); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal"); ax.axis("off")

    anim = FuncAnimation(fig, update, frames=n_frames, interval=400, repeat=True)
    plt.show()
