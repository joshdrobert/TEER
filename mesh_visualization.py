import os
import random
import numpy as np
import nibabel as nib
import pyvista as pv

nii_folder = "/home/cyrilpillai36/Desktop/data/train"
mesh_folder = "/home/cyrilpillai36/Desktop/data/train/meshes"

n_samples = 10          # 10 cases -> 20 panels total
label_value = None      # None = any nonzero voxel, or set e.g. 1, 2, 3
random.seed(42)         # remove for a different random sample each run

# ------------------------------------------------------------
# Build matching file pairs by base name
# Example:
#   case001.nii.gz  <->  case001.stl
# ------------------------------------------------------------
nii_map = {
    f.replace("label.nii.gz", ""): os.path.join(nii_folder, f)
    for f in os.listdir(nii_folder)
    if f.lower().endswith("label.nii.gz")
}

stl_map = {
    f.replace("label_surface.stl", ""): os.path.join(mesh_folder, f)
    for f in os.listdir(mesh_folder)
    if f.lower().endswith("label_surface.stl")
}

common_keys = sorted(set(nii_map) & set(stl_map))

if len(common_keys) == 0:
    raise ValueError("No matching .nii.gz / .stl pairs found by filename.")

selected_keys = random.sample(common_keys, min(n_samples, len(common_keys)))

rows = len(selected_keys)
cols = 2  # left = volume, right = mesh

plotter = pv.Plotter(
    shape=(rows, cols),
    border=True,
    window_size=(900, 350 * rows)
)

for i, key in enumerate(selected_keys):
    nii_path = nii_map[key]
    stl_path = stl_map[key]

    # --------------------------------------------------------
    # Load segmentation volume
    # --------------------------------------------------------
    img = nib.load(nii_path)
    data = img.get_fdata()
    spacing = img.header.get_zooms()[:3]

    if label_value is None:
        binary = (data > 0).astype(np.uint8)
        label_text = "nonzero"
    else:
        binary = (data == label_value).astype(np.uint8)
        label_text = f"label={label_value}"

    # Build ImageData for volume display
    grid = pv.ImageData()
    grid.dimensions = np.array(binary.shape) + 1
    grid.spacing = spacing
    grid.origin = (0, 0, 0)
    grid.cell_data["seg"] = binary.flatten(order="F")

    # --------------------------------------------------------
    # Load precomputed STL mesh
    # --------------------------------------------------------
    mesh = pv.read(stl_path)

    # --------------------------------------------------------
    # Left panel: segmentation volume
    # --------------------------------------------------------
    plotter.subplot(i, 0)
    if binary.sum() > 0:
        plotter.add_volume(
            grid,
            scalars="seg",
            cmap="viridis",
            opacity=[0.0, 0.15, 0.9],
            shade=True,
            show_scalar_bar=False
        )
    else:
        plotter.add_text("Empty segmentation", font_size=10)

    plotter.add_text(f"{key}\nVolume ({label_text})", font_size=10)
    plotter.view_isometric()

    # --------------------------------------------------------
    # Right panel: STL mesh
    # --------------------------------------------------------
    plotter.subplot(i, 1)
    plotter.add_mesh(
        mesh,
        color="lightcoral",
        smooth_shading=True,
        show_edges=False
    )
    plotter.add_text(f"{key}\nMesh (.stl)", font_size=10)
    plotter.view_isometric()

plotter.enable_trackball_style()
plotter.link_views(False)   # optional: makes each panel independent
plotter.view_isometric()
plotter.camera.zoom(1.3)
plotter.show()
