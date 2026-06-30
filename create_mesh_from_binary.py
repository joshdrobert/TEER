import os
import numpy as np
import nibabel as nib
import pyvista as pv
import pymeshfix
from skimage import measure
import tetgen

in_directory = "/home/cyrilpillai36/Desktop/data/train/"
out_directory = "/home/cyrilpillai36/Desktop/data/train/meshes/"
os.makedirs(out_directory, exist_ok=True)

SMOOTH_ITERATIONS = 50
SMOOTH_RELAXATION = 0.1
DECIMATE_REDUCTION = 0.0

for filename in os.listdir(in_directory):
    if not filename.endswith("label.nii.gz"):
        continue

    print(f"\n{'=' * 60}")
    print(f"Processing: {filename}")

    mask_img = nib.load(os.path.join(in_directory, filename))
    mask_data = (mask_img.get_fdata() > 0).astype(np.uint8)
    spacing = mask_img.header.get_zooms()[:3]

    print(f"  Voxel spacing (mm): {spacing}")
    print(f"  Mask shape: {mask_data.shape}, nonzero voxels: {mask_data.sum()}")

    verts, faces, normals, _ = measure.marching_cubes(
        mask_data,
        level=0.5,
        spacing=spacing
    )

    faces_pv = np.hstack([
        np.full((faces.shape[0], 1), 3, dtype=np.int64),
        faces
    ]).ravel()

    mesh = pv.PolyData(verts.astype(np.float64), faces_pv)
    print(f"  [1/6] Marching cubes -> {mesh.n_points} pts, {mesh.n_cells} faces")

    mf = pymeshfix.MeshFix(mesh.points, mesh.faces.reshape(-1, 4)[:, 1:])
    mf.repair()

    repaired_verts = mf.points
    repaired_faces = mf.faces
    faces_rep = np.hstack([
        np.full((repaired_faces.shape[0], 1), 3, dtype=np.int64),
        repaired_faces
    ]).ravel()

    mesh = pv.PolyData(repaired_verts.astype(np.float64), faces_rep)
    print(f"  [2/6] Repaired       -> {mesh.n_points} pts, {mesh.n_cells} faces")

    mesh = mesh.fill_holes(hole_size=1000)
    mesh = mesh.clean()
    mesh = mesh.smooth(
        n_iter=SMOOTH_ITERATIONS,
        relaxation_factor=SMOOTH_RELAXATION,
        boundary_smoothing=True,
        edge_angle=150,
        feature_smoothing=False,
    )

    if DECIMATE_REDUCTION > 0:
        mesh = mesh.decimate(DECIMATE_REDUCTION)

    mesh = mesh.compute_normals(
        auto_orient_normals=True,
        consistent_normals=True
    )

    print(f"  [3/6] Smoothed       -> {mesh.n_points} pts, {mesh.n_cells} faces")

    stem = filename.replace(".nii.gz", "")
    surface_path = os.path.join(out_directory, stem + "_surface.stl")
    mesh.save(surface_path)
    print(f"  [4/6] Surface saved  -> {surface_path}")

    try:
        tet = tetgen.TetGen(mesh)
        nodes, elems, _, _ = tet.tetrahedralize(order=1, mindihedral=20, minratio=1.5)
        print(f"  [5/6] TetGen         -> {nodes.shape[0]} nodes, {elems.shape[0]} tetrahedra")

        grid = tet.grid
        tet_path = os.path.join(out_directory, stem + "_tet.vtu")
        grid.save(tet_path)
        print(f"  [6/6] Tet mesh saved -> {tet_path}")

        np.savetxt(os.path.join(out_directory, stem + "_tet_nodes.txt"), nodes, fmt="%.6f")
        np.savetxt(os.path.join(out_directory, stem + "_tet_elements.txt"), elems, fmt="%d")

    except Exception as e:
        print(f"  [5/6] Tetrahedralization failed: {e}")

print("\nAll done.")