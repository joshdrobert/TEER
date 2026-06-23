"""Render a professional 3D MP4 of blood flow through the mitral valve.

Produces a publication-quality 3D medical visual of the STL mesh,
and a volumetric liquid blood flow jet with streamlines.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyvista as pv


@dataclass
class BloodFlowMP4Result:
    mp4_path: Path
    n_frames: int
    duration_seconds: float

    def to_dict(self) -> dict:
        return {
            "mp4_path": str(self.mp4_path),
            "n_frames": self.n_frames,
            "duration_seconds": round(self.duration_seconds, 2),
        }


def render_blood_flow_mp4(
    case_dir: Path,
    output_path: Path | None = None,
    fps: int = 30,
    dpi: int = 150,
) -> BloodFlowMP4Result:
    """Render 3D blood-flow MP4 from computed result_*.vtu and valve_*.vtu files."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    case_dir = case_dir.resolve()
    result_files = sorted(case_dir.glob("result_[0-9][0-9][0-9].vtu"))
    if not result_files:
        raise FileNotFoundError(f"No result_*.vtu files under {case_dir}")

    n_frames = len(result_files)
    if output_path is None:
        output_path = case_dir / "renders" / "blood_flow_simulation.mp4"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load first valve file to get reference mesh, center and bounds
    valve_path = case_dir / "valve_001.vtu"
    if not valve_path.exists():
        raise FileNotFoundError(f"Missing base valve file: {valve_path}")
    valve_mesh = pv.read(valve_path)
    bounds = valve_mesh.bounds
    cx = (bounds[0] + bounds[1]) / 2.0
    cy = (bounds[2] + bounds[3]) / 2.0
    R_cylinder = 0.5 * max(bounds[1] - bounds[0], bounds[3] - bounds[2]) + 5.0

    # 1. Custom High-Contrast Blue-White-Red Diverging Colormap
    cfd_colors = [
        (0.00, "#00caff"),  # bright cyan
        (0.30, "#003bff"),  # royal blue
        (0.50, "#1d2333"),  # deep background matching color
        (0.70, "#ff4b00"),  # bright orange
        (1.00, "#ff003c"),  # neon red
    ]
    cfd_cmap = LinearSegmentedColormap.from_list("cfd_diverging", cfd_colors, N=256)

    # 2. Seed points for 3D streamlines (subset to 40 for guidelines)
    rng = np.random.RandomState(42)
    seed_x = rng.uniform(cx - 35.0, cx + 35.0, 220)
    seed_y = rng.uniform(cy - 35.0, cy + 35.0, 220)
    seed_z = rng.uniform(bounds[4] - 5.0, bounds[5] + 5.0, 220)
    dist_from_center = np.sqrt((seed_x - cx)**2 + (seed_y - cy)**2)
    valid = dist_from_center <= R_cylinder * 0.8
    seed_pts = np.column_stack([seed_x[valid], seed_y[valid], seed_z[valid]])
    seed_pts_bg = seed_pts[:min(40, len(seed_pts))]
    seed_poly_bg = pv.PolyData(seed_pts_bg)

    # 3. Refinement grid for volume rendering and smooth background streamlines
    x_min, x_max = bounds[0] - 15.0, bounds[1] + 15.0
    y_min, y_max = bounds[2] - 15.0, bounds[3] + 15.0
    z_min, z_max = bounds[4] - 10.0, bounds[5] + 10.0
    
    nx, ny, nz = 50, 50, 75
    dx = (x_max - x_min) / (nx - 1)
    dy = (y_max - y_min) / (ny - 1)
    dz = (z_max - z_min) / (nz - 1)
    
    try:
        ref_grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(dx, dy, dz),
            origin=(x_min, y_min, z_min)
        )
    except AttributeError:
        ref_grid = pv.UniformGrid(
            dimensions=(nx, ny, nz),
            spacing=(dx, dy, dz),
            origin=(x_min, y_min, z_min)
        )

    # 4. Setup temporary folder for frame PNGs
    tmp_dir = case_dir / "tmp_frames"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    print(f"[MP4] Off-screen rendering {n_frames} frames to {tmp_dir}...")

    # Determine peak flow value for normalization
    v_max = 180.0

    # Custom opacity mapping to show high-velocity flow while making static regions transparent
    x_opacity = np.linspace(-1.0, 1.0, 256)
    abs_x = np.abs(x_opacity)
    # Smooth step function: transparent below 25.0 mm/s, ramps to 0.85 opacity
    opacity_array = np.clip((abs_x - 0.14) / 0.10, 0.0, 1.0) * 0.85

    # Render 3D scene using PyVista off-screen plotter
    plotter = pv.Plotter(off_screen=True, window_size=[1000, 1000])
    plotter.set_background("#0a0e1a")

    camera_pos = [cx + 190.0, cy - 190.0, bounds[5] + 130.0]
    focal_point = [cx, cy, (bounds[4] + bounds[5]) / 2]

    # Frame generation loop
    for fi in range(n_frames):
        step = fi + 1
        t = fi / max(n_frames - 1, 1)

        # Clear previous frame's actors
        plotter.clear()

        # Get peak boundary velocity to match current flow direction in repeating 60-step cycle
        step_in_cycle = fi % 60
        t_cycle = step_in_cycle / 59.0

        if t_cycle <= 0.5:
            e_wave = np.exp(-(t_cycle - 0.20)**2 / (2 * 0.06**2))
            a_wave = 0.55 * np.exp(-(t_cycle - 0.40)**2 / (2 * 0.05**2))
            v_peak = -180.0 * np.clip(e_wave + a_wave, 0.05, 1.0)
        else:
            v_peak = +80.0 * np.sin(np.pi * (t_cycle - 0.5) / 0.5)

        # Read simulation flow field
        flow_grid = pv.read(result_files[fi])
        flow_grid.set_active_vectors("Velocity")

        # Interpolate onto refined image grid to eliminate blockiness and enable volume rendering
        smooth_flow = ref_grid.sample(flow_grid)
        smooth_flow.set_active_vectors("Velocity")

        # Mask the Z-velocity and velocity magnitude to keep the jet focused in the lumen
        pts_coords = smooth_flow.points
        d_axis_geo = np.sqrt((pts_coords[:, 0] - cx)**2 + (pts_coords[:, 1] - cy)**2)
        in_lumen = d_axis_geo <= R_cylinder * 0.82

        from scipy.spatial import KDTree
        tree_valve_geo = KDTree(valve_mesh.points)
        dists_to_valve, _ = tree_valve_geo.query(pts_coords)
        outside_valve = dists_to_valve >= 1.5

        valid_flow = in_lumen & outside_valve
        
        # Velocity magnitude masking
        vel_data = smooth_flow.point_data["Velocity"]
        v_mag = np.linalg.norm(vel_data, axis=1)
        v_mag_masked = np.where(valid_flow, v_mag, 0.0)
        smooth_flow.point_data["Vmag"] = v_mag_masked

        # Extract dynamic liquid flow jet (3D isosurface)
        v_thresh = max(15.0, 0.15 * np.abs(v_peak))
        try:
            contour = smooth_flow.contour(scalars="Vmag", isosurfaces=[v_thresh])
            if contour is not None and contour.n_points > 0:
                vz_contour = contour.point_data["Velocity"][:, 2]
                contour.point_data["Z_Velocity"] = vz_contour
            else:
                contour = None
        except Exception:
            contour = None

        # Tracing thin background streamlines as faint guidelines
        try:
            streamlines = smooth_flow.streamlines(
                vectors="Velocity",
                source=seed_poly_bg,
                max_time=1.5,
                terminal_speed=1e-2,
                integration_direction="both"
            )
            if streamlines is not None and streamlines.n_points > 0:
                vz_stream = streamlines.point_data["Velocity"][:, 2]
                streamlines.point_data["Z_Velocity"] = vz_stream
        except Exception:
            streamlines = None

        # Add STL valve mesh (semi-transparent white)
        plotter.add_mesh(
            valve_mesh,
            color="#e2e8f0",
            opacity=0.25,
            specular=0.5,
            ambient=0.3,
            smooth_shading=True,
            show_scalar_bar=False
        )

        # Add dynamic liquid flow jet (colored by local axial velocity)
        if contour is not None:
            plotter.add_mesh(
                contour,
                scalars="Z_Velocity",
                cmap=cfd_cmap,
                clim=[-v_max, v_max],
                opacity=0.85,
                specular=0.9,
                specular_power=40.0,
                ambient=0.4,
                smooth_shading=True,
                show_scalar_bar=False
            )

        # Add thin background streamlines as guidelines
        if streamlines is not None and streamlines.n_points > 0:
            plotter.add_mesh(
                streamlines,
                scalars="Z_Velocity",
                cmap=cfd_cmap,
                clim=[-v_max, v_max],
                line_width=1.0,
                render_lines_as_tubes=False,
                opacity=0.12,
                show_scalar_bar=False
            )

        # Add cylinder outline representing flow boundaries
        plotter.add_mesh(
            pv.Cylinder(
                center=(cx, cy, (bounds[4] + bounds[5]) / 2),
                radius=R_cylinder,
                height=(bounds[5] - bounds[4] + 15.0),
                direction=(0, 0, 1)
            ),
            color="#1d2d44",
            style="wireframe",
            opacity=0.15,
            line_width=1.0,
            show_scalar_bar=False
        )

        # Symmetrical Isometric camera view
        plotter.camera_position = [camera_pos, focal_point, [0, 0, 1]]
        plotter.enable_eye_dome_lighting()

        # Capture frame image directly using PyVista
        frame_path = tmp_dir / f"frame_{fi:03d}.png"
        plotter.screenshot(frame_path)

        if step % 20 == 0 or step == n_frames:
            print(f"  [MP4] Rendered frame {step}/{n_frames}")

    plotter.close()

    # Compile frames using FFmpeg
    print(f"[MP4] Compiling frames with FFmpeg to {output_path}...")
    cmd = [
        "ffmpeg", "-y",
        "-r", str(fps),
        "-i", str(tmp_dir / "frame_%03d.png"),
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        "-preset", "slow",
        str(output_path)
    ]
    
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Clean up temporary frames
    shutil.rmtree(tmp_dir)

    duration = n_frames / fps
    print(f"[MP4] Done rendering 3D blood flow animation: {output_path}")
    return BloodFlowMP4Result(
        mp4_path=output_path,
        n_frames=n_frames,
        duration_seconds=duration
    )
