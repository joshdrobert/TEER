"""Generate and run a mock mitral URIS-FSI case with svMultiPhysics."""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv
from scipy.spatial import cKDTree


PIPE_URIS_CASE = Path("external/svMultiPhysics/tests/cases/uris/pipe_uris_fsi")
DEFAULT_SOLVER = Path("external/svMultiPhysics/build/svMultiPhysics-build/bin/svmultiphysics")


@dataclass
class MockMitralResult:
    """Summary of the generated case and solver run."""

    case_dir: Path
    solver_xml: Path
    solver_binary: Path
    result_vtu: Path
    pressure_range: tuple[float, float]
    velocity_magnitude_range: tuple[float, float]
    valve_scale_factor: float

    def to_dict(self) -> dict[str, object]:
        return {
            "case_dir": str(self.case_dir),
            "solver_xml": str(self.solver_xml),
            "solver_binary": str(self.solver_binary),
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
    solver_binary: Path | None = None,
    time_steps: int = 5,
    time_step_size: float = 1e-3,
) -> MockMitralResult:
    """Build a runnable LV-like URIS-FSI mock case and execute svMultiPhysics."""
    workspace = workspace.resolve()
    valve_obj = valve_obj.resolve()
    solver_binary = (solver_binary or DEFAULT_SOLVER).resolve()

    case_dir = workspace / "artifacts" / "mock_mitral_uris_fsi"
    mesh_dir = case_dir / "meshes"
    fluid_dir = mesh_dir / "lv_domain_fluid"
    wall_dir = mesh_dir / "lv_domain_wall"
    for path in (fluid_dir / "mesh-surfaces", wall_dir / "mesh-surfaces"):
        path.mkdir(parents=True, exist_ok=True)

    _copy_and_morph_reference_meshes(fluid_dir, wall_dir)
    valve_scale_factor = _write_mitral_valve_assets(valve_obj, mesh_dir)
    solver_xml = _write_solver_xml(case_dir, time_steps=time_steps, time_step_size=time_step_size)

    completed = subprocess.run(
        [str(solver_binary), str(solver_xml)],
        cwd=case_dir,
        check=True,
        capture_output=True,
        text=True,
    )
    (case_dir / "solver.stdout.log").write_text(completed.stdout)
    (case_dir / "solver.stderr.log").write_text(completed.stderr)

    result_vtu = _resolve_result_path(case_dir, time_steps)
    pressure_range, velocity_range = _summarize_result(result_vtu)
    summary = MockMitralResult(
        case_dir=case_dir,
        solver_xml=solver_xml,
        solver_binary=solver_binary,
        result_vtu=result_vtu,
        pressure_range=pressure_range,
        velocity_magnitude_range=velocity_range,
        valve_scale_factor=valve_scale_factor,
    )
    (case_dir / "summary.json").write_text(json.dumps(summary.to_dict(), indent=2))
    return summary


def render_mock_case(case_dir: Path) -> MockMitralVisualization:
    """Render screenshots and an animation for a completed mock mitral case."""
    case_dir = case_dir.resolve()
    result_dir = case_dir / "1-procs" if (case_dir / "1-procs").exists() else case_dir
    result_paths = sorted(result_dir.glob("result_[0-9][0-9][0-9].vtu"))
    valve_paths = sorted(result_dir.glob("result_uris_MitralValve_[0-9][0-9][0-9].vtu"))
    if not result_paths or not valve_paths:
        raise FileNotFoundError(f"No rendered result files found under {result_dir}")

    render_dir = case_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)
    overview_png = render_dir / "overview.png"
    cutaway_png = render_dir / "cutaway.png"
    animation_gif = render_dir / "animation.gif"

    old_mplconfig = os.environ.get("MPLCONFIGDIR")
    os.environ["MPLCONFIGDIR"] = str((case_dir / ".mplconfig").resolve())
    (case_dir / ".mplconfig").mkdir(exist_ok=True)

    try:
        _render_overview(result_paths[-1], valve_paths[-1], overview_png, cutaway=False)
        _render_overview(result_paths[-1], valve_paths[-1], cutaway_png, cutaway=True)
        _render_animation(result_paths, valve_paths, animation_gif)
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


def _copy_and_morph_reference_meshes(fluid_dir: Path, wall_dir: Path) -> None:
    reference_root = PIPE_URIS_CASE / "meshes"
    fluid_ref = reference_root / "cylinder-mesh-complete_domain-1"
    wall_ref = reference_root / "cylinder-mesh-complete_domain-2"

    _morph_dataset(fluid_ref / "mesh-complete.mesh.vtu", fluid_dir / "mesh-complete.mesh.vtu")
    _morph_dataset(fluid_ref / "mesh-complete.exterior.vtp", fluid_dir / "mesh-complete.exterior.vtp")
    _morph_dataset(fluid_ref / "mesh-surfaces" / "inlet.vtp", fluid_dir / "mesh-surfaces" / "inlet.vtp")
    _morph_dataset(fluid_ref / "mesh-surfaces" / "outlet.vtp", fluid_dir / "mesh-surfaces" / "outlet.vtp")
    _morph_dataset(fluid_ref / "mesh-surfaces" / "wall.vtp", fluid_dir / "mesh-surfaces" / "wall.vtp")

    _morph_dataset(wall_ref / "mesh-complete.mesh.vtu", wall_dir / "mesh-complete.mesh.vtu")
    _morph_dataset(wall_ref / "mesh-complete.exterior.vtp", wall_dir / "mesh-complete.exterior.vtp")
    _morph_dataset(wall_ref / "mesh-surfaces" / "inlet.vtp", wall_dir / "mesh-surfaces" / "inlet.vtp")
    _morph_dataset(wall_ref / "mesh-surfaces" / "outlet.vtp", wall_dir / "mesh-surfaces" / "outlet.vtp")
    _morph_dataset(wall_ref / "mesh-surfaces" / "wall.vtp", wall_dir / "mesh-surfaces" / "wall.vtp")
    _morph_dataset(wall_ref / "walls_combined_connected_region_0.vtp", wall_dir / "walls_combined_connected_region_0.vtp")
    _morph_dataset(wall_ref / "walls_combined_connected_region_1.vtp", wall_dir / "walls_combined_connected_region_1.vtp")


def _morph_dataset(src: Path, dst: Path) -> None:
    mesh = pv.read(src)
    mesh = mesh.copy(deep=True)
    mesh.points = _lv_like_transform(mesh.points)
    mesh.save(dst)


def _lv_like_transform(points: np.ndarray) -> np.ndarray:
    transformed = np.array(points, dtype=float, copy=True)
    x = transformed[:, 0] + 0.145
    y = transformed[:, 1] + 0.072
    z_lv = -transformed[:, 2]

    scale = 0.72 + 0.90 * np.exp(-((z_lv - 0.3) / 2.6) ** 2)
    x = 0.92 * x * scale
    y = 1.05 * y * scale
    z = 1.12 * z_lv

    x += 0.08 * np.sin(0.45 * (z_lv + 1.5))
    y += 0.05 * np.cos(0.35 * (z_lv - 0.5))
    return np.column_stack([x, y, z])


def _write_mitral_valve_assets(valve_obj: Path, mesh_dir: Path) -> float:
    valve = _load_clean_obj(valve_obj)
    annulus_points, chordae_points = _extract_boundary_loops(valve)
    annulus_centroid = annulus_points.mean(axis=0)

    annulus_basis, normal = _annulus_frame(annulus_points)
    if np.dot(chordae_points.mean(axis=0) - annulus_centroid, normal) > 0.0:
        normal = -normal
    basis_y = np.cross(normal, annulus_basis)
    basis_y /= np.linalg.norm(basis_y)
    basis_x = np.cross(basis_y, normal)
    basis_x /= np.linalg.norm(basis_x)
    rotation = np.column_stack([basis_x, basis_y, normal])

    local_points = (valve.points - annulus_centroid) @ rotation
    annulus_local = (annulus_points - annulus_centroid) @ rotation

    annulus_radius = np.mean(np.linalg.norm(annulus_local[:, :2], axis=1))
    target_radius = 1.05
    scale = target_radius / annulus_radius
    local_points *= scale
    annulus_local *= scale

    local_points[:, 2] *= 0.22
    target_center = np.array([0.0, 0.0, 2.55])
    local_points += target_center

    valve_local = pv.PolyData(local_points, valve.faces).triangulate().clean()
    valve_local = _strip_arrays(valve_local)
    valve_local.cast_to_unstructured_grid().save(mesh_dir / "mitral_valve.vtu")

    _write_motion_file(mesh_dir / "mitral_motion_open.dat", _generate_motion_series(valve_local.points, annulus_local + target_center, opening=True))
    _write_motion_file(mesh_dir / "mitral_motion_close.dat", _generate_motion_series(valve_local.points, annulus_local + target_center, opening=False))
    (mesh_dir / "normal.dat").write_text("0.0 0.0 -1.0\n")
    return float(scale)


def _load_clean_obj(path: Path) -> pv.PolyData:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("f "):
                indices = [int(token.split("/")[0]) - 1 for token in line.split()[1:]]
                if len(indices) < 3:
                    continue
                for idx in range(1, len(indices) - 1):
                    faces.append([3, indices[0], indices[idx], indices[idx + 1]])
    if not vertices or not faces:
        raise RuntimeError(f"Could not parse vertices/faces from {path}")
    flat_faces = np.asarray(faces, dtype=np.int64).ravel()
    valve = pv.PolyData(np.asarray(vertices, dtype=float), flat_faces).triangulate().clean()
    return _strip_arrays(valve)


def _strip_arrays(mesh: pv.DataSet) -> pv.DataSet:
    for name in list(mesh.point_data.keys()):
        mesh.point_data.pop(name)
    for name in list(mesh.cell_data.keys()):
        mesh.cell_data.pop(name)
    for name in list(mesh.field_data.keys()):
        mesh.field_data.pop(name)
    return mesh


def _extract_boundary_loops(valve: pv.PolyData) -> tuple[np.ndarray, np.ndarray]:
    boundary = valve.extract_feature_edges(
        boundary_edges=True,
        non_manifold_edges=False,
        feature_edges=False,
        manifold_edges=False,
    )
    boundary = boundary.connectivity()
    if "RegionId" in boundary.cell_data:
        region_ids = np.asarray(boundary.cell_data["RegionId"])
        selector = "cell"
    elif "RegionId" in boundary.point_data:
        region_ids = np.asarray(boundary.point_data["RegionId"])
        selector = "point"
    else:
        raise RuntimeError("Boundary connectivity did not expose RegionId labels.")
    loops: list[np.ndarray] = []
    for region_id in sorted(set(region_ids.tolist())):
        ids = np.where(region_ids == region_id)[0]
        loop = boundary.extract_cells(ids).clean() if selector == "cell" else boundary.extract_points(ids).clean()
        loops.append(loop.points)
    loops.sort(key=lambda pts: pts.shape[0], reverse=True)
    if len(loops) < 2:
        raise RuntimeError("Expected annulus and distal chordal boundary loops.")
    return loops[0], loops[1]


def _annulus_frame(annulus_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centered = annulus_points - annulus_points.mean(axis=0)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    basis_x = vh[0]
    normal = vh[2]
    basis_x /= np.linalg.norm(basis_x)
    normal /= np.linalg.norm(normal)
    return basis_x, normal


def _generate_motion_series(points: np.ndarray, annulus_points: np.ndarray, opening: bool) -> np.ndarray:
    tree = cKDTree(annulus_points)
    distances, _ = tree.query(points)
    normalized = np.clip(distances / (np.percentile(distances, 90) + 1e-8), 0.0, 1.0)
    weights = normalized**1.5

    radial = points[:, :2].copy()
    radial_norm = np.linalg.norm(radial, axis=1, keepdims=True)
    radial_norm[radial_norm == 0.0] = 1.0
    radial /= radial_norm

    z_direction = np.array([0.0, 0.0, -1.0 if opening else 1.0])
    radial_gain = 0.18 if opening else -0.16
    axial_gain = 0.42 if opening else 0.18

    frames: list[np.ndarray] = []
    for alpha in np.linspace(0.0, 1.0, 6):
        smooth = 0.5 - 0.5 * np.cos(np.pi * alpha)
        displacement = np.zeros_like(points)
        displacement[:, :2] = radial_gain * smooth * weights[:, None] * radial
        displacement += axial_gain * smooth * weights[:, None] * z_direction
        if not opening:
            displacement[:, :2] += -0.05 * smooth * weights[:, None] * radial
        frames.append(points + displacement)
    return np.stack(frames, axis=0)


def _write_motion_file(path: Path, frames: np.ndarray) -> None:
    n_steps, n_points, _ = frames.shape
    with path.open("w") as handle:
        handle.write(f"{n_steps} {n_points}\n")
        for frame in frames:
            for point in frame:
                handle.write(f"{point[0]:.16e} {point[1]:.16e} {point[2]:.16e}\n")


def _write_solver_xml(case_dir: Path, time_steps: int, time_step_size: float) -> Path:
    solver_xml = case_dir / "solver.xml"
    solver_xml.write_text(
        f"""<?xml version="1.0" encoding="UTF-8" ?>
<svMultiPhysicsFile version="0.1">

<GeneralSimulationParameters>
  <Continue_previous_simulation> false </Continue_previous_simulation>
  <Number_of_spatial_dimensions> 3 </Number_of_spatial_dimensions>
  <Number_of_time_steps> {time_steps} </Number_of_time_steps>
  <Time_step_size> {time_step_size:.6f} </Time_step_size>
  <Spectral_radius_of_infinite_time_step> 0.5 </Spectral_radius_of_infinite_time_step>
  <Searched_file_name_to_trigger_stop> STOP_SIM </Searched_file_name_to_trigger_stop>
  <Save_results_to_VTK_format> 1 </Save_results_to_VTK_format>
  <Name_prefix_of_saved_VTK_files> result </Name_prefix_of_saved_VTK_files>
  <Increment_in_saving_VTK_files> 1 </Increment_in_saving_VTK_files>
  <Start_saving_after_time_step> 1 </Start_saving_after_time_step>
  <Increment_in_saving_restart_files> 1 </Increment_in_saving_restart_files>
  <Convert_BIN_to_VTK_format> 0 </Convert_BIN_to_VTK_format>
  <Verbose> 1 </Verbose>
  <Warning> 1 </Warning>
  <Debug> 0 </Debug>
</GeneralSimulationParameters>

<Add_mesh name="lv_fluid">
  <Mesh_file_path> meshes/lv_domain_fluid/mesh-complete.mesh.vtu </Mesh_file_path>
  <Add_face name="lv_wall">
      <Face_file_path> meshes/lv_domain_fluid/mesh-surfaces/wall.vtp </Face_file_path>
  </Add_face>
  <Add_face name="la_inlet">
      <Face_file_path> meshes/lv_domain_fluid/mesh-surfaces/inlet.vtp </Face_file_path>
  </Add_face>
  <Add_face name="apex_outlet">
      <Face_file_path> meshes/lv_domain_fluid/mesh-surfaces/outlet.vtp </Face_file_path>
  </Add_face>
  <Domain> 0 </Domain>
</Add_mesh>

<Add_mesh name="lv_wall_struct">
  <Mesh_file_path> meshes/lv_domain_wall/mesh-complete.mesh.vtu </Mesh_file_path>
  <Add_face name="wall_inner">
      <Face_file_path> meshes/lv_domain_wall/walls_combined_connected_region_0.vtp </Face_file_path>
  </Add_face>
  <Add_face name="wall_outer">
      <Face_file_path> meshes/lv_domain_wall/walls_combined_connected_region_1.vtp </Face_file_path>
  </Add_face>
  <Add_face name="wall_inlet_ring">
      <Face_file_path> meshes/lv_domain_wall/mesh-surfaces/inlet.vtp </Face_file_path>
  </Add_face>
  <Add_face name="wall_apex_ring">
      <Face_file_path> meshes/lv_domain_wall/mesh-surfaces/outlet.vtp </Face_file_path>
  </Add_face>
  <Domain> 1 </Domain>
</Add_mesh>

<Add_projection name="wall_inner">
   <Project_from_face> lv_wall </Project_from_face>
</Add_projection>

<Add_URIS_mesh name="MitralValve">
  <Add_URIS_face name="mitral_leaflets">
    <Face_file_path> meshes/mitral_valve.vtu </Face_file_path>
    <Open_motion_file_path> meshes/mitral_motion_open.dat </Open_motion_file_path>
    <Close_motion_file_path> meshes/mitral_motion_close.dat </Close_motion_file_path>
  </Add_URIS_face>
  <Mesh_scale_factor> 1.0 </Mesh_scale_factor>
  <Thickness> 0.12 </Thickness>
  <Closed_thickness> 0.18 </Closed_thickness>
  <Resistance> 2.5e5 </Resistance>
  <Invert_normal> false </Invert_normal>
  <Positive_flow_normal_file_path> meshes/normal.dat </Positive_flow_normal_file_path>
</Add_URIS_mesh>

<Add_equation type="FSI">
   <Coupled> true </Coupled>
   <Min_iterations> 1 </Min_iterations>
   <Max_iterations> 10 </Max_iterations>
   <Tolerance> 1e-10 </Tolerance>
   <Explicit_geometric_coupling> true </Explicit_geometric_coupling>

   <Domain id="0">
      <Equation> fluid </Equation>
      <Density> 1.06 </Density>
      <Viscosity model="Constant">
         <Value> 0.04 </Value>
      </Viscosity>
      <Backflow_stabilization_coefficient> 0.2 </Backflow_stabilization_coefficient>
   </Domain>

   <Domain id="1">
      <Equation> struct </Equation>
      <Constitutive_model type="neoHookean"> </Constitutive_model>
      <Dilational_penalty_model> M94 </Dilational_penalty_model>
      <Density> 1.0 </Density>
      <Elasticity_modulus> 5.0e7 </Elasticity_modulus>
      <Poisson_ratio> 0.3 </Poisson_ratio>
   </Domain>

   <LS type="GMRES">
      <Linear_algebra type="fsils">
         <Preconditioner> fsils </Preconditioner>
      </Linear_algebra>
      <Tolerance> 1e-10 </Tolerance>
      <Max_iterations> 500 </Max_iterations>
      <Krylov_space_dimension> 200 </Krylov_space_dimension>
   </LS>

   <Output type="Spatial">
     <Displacement> true </Displacement>
     <Velocity> true </Velocity>
     <Pressure> true </Pressure>
   </Output>

   <Output type="Alias">
       <Displacement> FS_Displacement </Displacement>
   </Output>

   <Output type="B_INT">
     <Pressure> true </Pressure>
     <Velocity> true </Velocity>
   </Output>

   <Output type="V_INT">
     <Pressure> true </Pressure>
   </Output>

   <Add_BC name="la_inlet">
      <Type> Neu </Type>
      <Value> 5.0e4 </Value>
   </Add_BC>

   <Add_BC name="apex_outlet">
      <Type> Neu </Type>
      <Value> 0.0 </Value>
   </Add_BC>

   <Add_BC name="wall_inlet_ring">
      <Type> Dir </Type>
      <Value> 0.0 </Value>
      <Impose_on_state_variable_integral> true </Impose_on_state_variable_integral>
      <Zero_out_perimeter> false </Zero_out_perimeter>
      <Effective_direction> (0, 0, 1) </Effective_direction>
   </Add_BC>

   <Add_BC name="wall_apex_ring">
      <Type> Dir </Type>
      <Value> 0.0 </Value>
      <Impose_on_state_variable_integral> true </Impose_on_state_variable_integral>
      <Zero_out_perimeter> false </Zero_out_perimeter>
      <Effective_direction> (0, 0, 1) </Effective_direction>
   </Add_BC>
</Add_equation>

<Add_equation type="mesh">
   <Coupled> true </Coupled>
   <Min_iterations> 1 </Min_iterations>
   <Max_iterations> 10 </Max_iterations>
   <Tolerance> 1e-10 </Tolerance>
   <Poisson_ratio> 0.3 </Poisson_ratio>

   <LS type="CG">
      <Linear_algebra type="fsils">
         <Preconditioner> fsils </Preconditioner>
      </Linear_algebra>
      <Tolerance> 1e-10 </Tolerance>
      <Max_iterations> 5000 </Max_iterations>
      <Krylov_space_dimension> 300 </Krylov_space_dimension>
   </LS>

   <Output type="Spatial">
     <Displacement> true </Displacement>
   </Output>

   <Add_BC name="la_inlet">
      <Type> Dir </Type>
      <Value> 0.0 </Value>
   </Add_BC>

   <Add_BC name="apex_outlet">
      <Type> Dir </Type>
      <Value> 0.0 </Value>
   </Add_BC>
</Add_equation>

</svMultiPhysicsFile>
""",
    )
    return solver_xml


def _summarize_result(result_vtu: Path) -> tuple[tuple[float, float], tuple[float, float]]:
    result = pv.read(result_vtu)
    pressure = np.asarray(result.point_data["Pressure"])
    velocity = np.asarray(result.point_data["Velocity"])
    velocity_mag = np.linalg.norm(velocity, axis=1)
    return (float(np.min(pressure)), float(np.max(pressure))), (float(np.min(velocity_mag)), float(np.max(velocity_mag)))


def _resolve_result_path(case_dir: Path, time_steps: int) -> Path:
    direct = case_dir / f"result_{time_steps:03d}.vtu"
    if direct.exists():
        return direct
    mpi_dir = case_dir / "1-procs" / f"result_{time_steps:03d}.vtu"
    if mpi_dir.exists():
        return mpi_dir
    raise FileNotFoundError(f"Could not locate result_{time_steps:03d}.vtu under {case_dir}")


def _render_overview(result_path: Path, valve_path: Path, output_png: Path, cutaway: bool) -> None:
    fluid = pv.read(result_path)
    valve = pv.read(valve_path).extract_surface().triangulate().clean()
    pressure = np.asarray(fluid.point_data["Pressure"])
    fluid["Pressure_kPa"] = pressure / 1000.0
    fluid_surface = fluid.extract_surface().triangulate().clean()
    if cutaway:
        fluid_surface = fluid_surface.clip(normal=(1, 0, 0), origin=(0.0, 0.0, 2.4), invert=False)
    _save_matplotlib_frame(fluid_surface, valve, output_png, title="Mitral URIS-FSI Mock", add_colorbar=True)


def _render_animation(result_paths: list[Path], valve_paths: list[Path], output_gif: Path) -> None:
    from PIL import Image

    frame_paths: list[Path] = []
    for idx, (fluid_path, valve_path) in enumerate(zip(result_paths, valve_paths, strict=True), start=1):
        fluid = pv.read(fluid_path)
        fluid["Pressure_kPa"] = np.asarray(fluid.point_data["Pressure"]) / 1000.0
        fluid_surface = fluid.extract_surface().triangulate().clean()
        valve = pv.read(valve_path).extract_surface().triangulate().clean()
        frame_path = output_gif.parent / f"frame_{idx:03d}.png"
        _save_matplotlib_frame(fluid_surface, valve, frame_path, title=f"Timestep {idx}", add_colorbar=(idx == 1))
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
    fluid_surface: pv.PolyData,
    valve_surface: pv.PolyData,
    output_path: Path,
    title: str,
    add_colorbar: bool,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure(figsize=(10, 8), dpi=180)
    fig.patch.set_facecolor("#f3f0ea")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#f3f0ea")

    fluid_points = fluid_surface.points
    fluid_faces = fluid_surface.faces.reshape(-1, 4)[:, 1:]
    fluid_pressure = np.asarray(fluid_surface.point_data["Pressure_kPa"])
    tri_pressure = fluid_pressure[fluid_faces].mean(axis=1)
    norm = Normalize(vmin=float(tri_pressure.min()), vmax=float(tri_pressure.max()))
    cmap = matplotlib.colormaps["coolwarm"]
    colors = cmap(norm(tri_pressure))

    fluid_verts = fluid_points[fluid_faces]
    fluid_collection = Poly3DCollection(fluid_verts, linewidths=0.0, alpha=0.42)
    fluid_collection.set_facecolor(colors)
    fluid_collection.set_edgecolor("none")
    ax.add_collection3d(fluid_collection)

    valve_points = valve_surface.points
    valve_faces = valve_surface.faces.reshape(-1, 4)[:, 1:]
    valve_verts = valve_points[valve_faces]
    valve_collection = Poly3DCollection(valve_verts, linewidths=0.0, alpha=0.96)
    valve_collection.set_facecolor("#d97706")
    valve_collection.set_edgecolor("none")
    ax.add_collection3d(valve_collection)

    all_points = np.vstack([fluid_points, valve_points])
    mins = all_points.min(axis=0)
    maxs = all_points.max(axis=0)
    spans = np.maximum(maxs - mins, 1e-6)
    center = (mins + maxs) / 2.0
    radius = spans.max() / 2.0
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.view_init(elev=24, azim=-58)
    ax.set_axis_off()
    ax.set_title(title, fontsize=12, pad=12)

    if add_colorbar:
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, shrink=0.68, pad=0.02)
        cbar.set_label("Pressure (kPa)")

    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
