"""Convert a triangular STL surface into a tetrahedral volume mesh.

Usage:
  python segmentation/mesh_functions/mesh_reconstruction.py \
    --input /path/to/surface.stl --output-dir ./artifacts --alpha 1.0 --decimate 0.5

The script will attempt to repair the surface (using pymeshfix if available),
optionally decimate it, run `pyvista.PolyData.delaunay_3d` to produce a volume,
save a `.vtu` and also export a `.msh` (via meshio) for downstream tools.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

try:
    import pyvista as pv
except Exception as exc:  # pragma: no cover - environment dependent
    print("pyvista is required: pip install pyvista vtk", file=sys.stderr)
    raise

try:
    import trimesh
except Exception:  # pragma: no cover - optional
    trimesh = None

try:
    import pymeshfix
except Exception:  # pragma: no cover - optional
    pymeshfix = None

try:
    import meshio
except Exception:
    meshio = None


def load_surface(path: Path) -> pv.PolyData:
    path = Path(path)
    if trimesh is not None:
        tm = trimesh.load_mesh(str(path), process=True)
        if isinstance(tm, trimesh.Trimesh):
            verts = np.asarray(tm.vertices, dtype=np.float64)
            faces = np.asarray(tm.faces, dtype=np.int64)
            # pyvista expects faces as [n, i0, i1, i2, n, ...]
            faces_flat = np.hstack([np.full((faces.shape[0], 1), 3, dtype=np.int64), faces]).ravel()
            return pv.PolyData(verts, faces_flat)
    # fallback to pyvista reader
    return pv.read(str(path))


def repair_surface(pv_surface: pv.PolyData) -> pv.PolyData:
    if pymeshfix is None or trimesh is None:
        return pv_surface.clean()

    # Convert to trimesh, run pymeshfix, and return repaired pyvista surface
    verts = pv_surface.points.copy()
    faces = pv_surface.faces.reshape((-1, 4))[:, 1:]
    vclean, fclean = pymeshfix.clean_from_arrays(verts, faces, verbose=False)
    faces_flat = np.hstack([np.full((fclean.shape[0], 1), 3, dtype=np.int64), fclean]).ravel()
    return pv.PolyData(vclean, faces_flat).clean()


def build_volume(surface: pv.PolyData, alpha: float) -> pv.UnstructuredGrid:
    # Ensure surface is triangulated and cleaned
    s = surface.triangulate().clean()
    vol = s.delaunay_3d(alpha=float(alpha))
    return vol


def export_volume(volume: pv.UnstructuredGrid, out_dir: Path, base_name: str) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    vtu_path = out_dir / f"{base_name}_volume.vtu"
    msh_path = out_dir / f"{base_name}_volume.msh"
    volume.save(str(vtu_path))
    result = {"vtu": vtu_path}
    if meshio is not None:
        try:
            mesh = meshio.read(str(vtu_path))
            meshio.write(str(msh_path), mesh, file_format="gmsh22")
            result["msh"] = msh_path
        except Exception:
            pass
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="STL -> tet volume mesh converter")
    p.add_argument("--input", required=True, help="Input surface STL/VTK/OBJ path")
    p.add_argument("--output-dir", default=Path("artifacts"), help="Output directory")
    p.add_argument("--alpha", type=float, default=1.0, help="Alpha for delaunay_3d (bigger = coarser)")
    p.add_argument("--decimate", type=float, default=0.0, help="Surface decimation fraction (0-1)")
    p.add_argument("--fill-holes", type=float, default=0.0, help="Fill hole size (0 = none)")
    args = p.parse_args(argv)

    inp = Path(args.input)
    out = Path(args.output_dir)
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        return 2

    print(f"Loading surface: {inp}")
    surf = load_surface(inp)
    print(f"Surface points: {surf.n_points}, faces: {int(surf.n_faces)}")

    if args.fill_holes and args.fill_holes > 0.0:
        try:
            surf = surf.fill_holes(args.fill_holes)
        except Exception:
            pass

    if pymeshfix is not None:
        print("Attempting mesh repair with pymeshfix (if available)")
        surf = repair_surface(surf)

    if args.decimate and args.decimate > 0.0:
        print(f"Decimating surface by fraction {args.decimate}")
        try:
            surf = surf.decimate(args.decimate)
        except Exception:
            print("Decimation failed, continuing with cleaned surface")

    surf = surf.clean()
    print(f"Post-clean points: {surf.n_points}, faces: {int(surf.n_faces)}")

    print(f"Running delaunay_3d with alpha={args.alpha} (this may take a while) ...")
    volume = build_volume(surf, args.alpha)
    print(f"Volume cells: {int(volume.n_cells) if hasattr(volume, 'n_cells') else 'unknown'}")

    base = inp.stem
    exported = export_volume(volume, out, base)
    print("Exported:")
    for k, v in exported.items():
        print(f" - {k}: {v}")

    if hasattr(volume, "n_cells") and volume.n_cells > 2_000_000:
        print("WARNING: Generated very large volume mesh (over 2M cells). Consider increasing --alpha or decimating more.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

import gmsh
import os
import scipy

mesh_path = "/home/cyrilpillai36/Desktop/TEER/segmented_valve_mesh_smoothed.stl"
out_folder = "/home/cyrilpillai36/Desktop/TEER/"

import os
import trimesh
import gmsh
import meshio

os.makedirs(out_folder, exist_ok=True)
out_path = os.path.join(out_folder, "leaflet_volume.msh")

# --- Step 1: Repair & verify surface mesh ---
mesh = trimesh.load(mesh_path, process=True)
mesh.fill_holes()
mesh.fix_normals()
mesh.export("fixed_mesh.stl")
print("Is watertight:", mesh.is_watertight)  # must be True

gmsh.initialize()
gmsh.option.setNumber("General.Verbosity", 5)  # see error messages

gmsh.merge("fixed_mesh.stl")

# Classify + create geometry
gmsh.model.mesh.classifySurfaces(gmsh.pi, True, True, gmsh.pi)
gmsh.model.mesh.createGeometry()

# --- KEY FIX: explicitly create a volume from all surfaces ---
surfaces = gmsh.model.getEntities(dim=2)
surface_tags = [s[1] for s in surfaces]
surface_loop = gmsh.model.geo.addSurfaceLoop(surface_tags)
gmsh.model.geo.addVolume([surface_loop])
gmsh.model.geo.synchronize()  # sync before meshing

gmsh.option.setNumber("Mesh.Algorithm3D", 4)
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)

gmsh.model.mesh.generate(3)
gmsh.write(out_path)
gmsh.finalize()

# Verify
tet_mesh = meshio.read(out_path, file_format="gmsh")
tet_cells = [c for c in tet_mesh.cells if c.type == "tetra"]
print("Tet elements:", sum(len(c.data) for c in tet_cells))

# --- Step 2: Generate volume mesh ---
gmsh.initialize()
gmsh.merge("fixed_mesh.stl")
gmsh.model.mesh.classifySurfaces(gmsh.pi, True, True, gmsh.pi)
gmsh.model.mesh.createGeometry()

gmsh.option.setNumber("Mesh.Algorithm3D", 4)       # Frontal-Delaunay
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 5.0)  # tune to your scale

gmsh.model.mesh.generate(3)
gmsh.write(out_path)

# --- Step 3: Debug — check what was generated ---
for dim, tag in gmsh.model.getEntities():
    print(f"Dim: {dim}, Tag: {tag}")

gmsh.finalize()

# --- Step 4: Verify with meshio ---
tet_mesh = meshio.read(out_path, file_format="gmsh")
tet_cells = [c for c in tet_mesh.cells if c.type == "tetra"]
print("Tet elements:", sum(len(c.data) for c in tet_cells))  # must be > 0
print("Points:", tet_mesh.points.shape)