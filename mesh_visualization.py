import os
import math
import random
import numpy as np
import pyvista as pv

mesh_folder = "../data/train/meshes"
out_file = "./all_meshes_publication.png"
n_samples = 10
random.seed(42)

stl_files = sorted(
    os.path.join(mesh_folder, f)
    for f in os.listdir(mesh_folder)
    if f.lower().endswith("label_surface.stl")
)

if not stl_files:
    raise ValueError("No STL files found.")

selected_files = random.sample(stl_files, min(n_samples, len(stl_files)))

colors = [
    "#d73027", "#4575b4", "#1a9850", "#984ea3", "#ff8c00",
    "#00acc1", "#8c564b", "#e7298a", "#66a61e", "#4c4c4c"
]

pv.set_plot_theme("document")


def largest_boundary_loop_points(mesh):
    edges = mesh.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    if edges.n_points == 0:
        return None

    conn = edges.connectivity()
    region_ids = conn["RegionId"]
    regions = np.unique(region_ids)

    best_pts = None
    best_n = -1

    for rid in regions:
        pts = conn.points[region_ids == rid]
        if pts.shape[0] > best_n:
            best_n = pts.shape[0]
            best_pts = pts

    return best_pts


def camera_for_opening(mesh):
    pts = largest_boundary_loop_points(mesh)

    center = mesh.center
    bounds = np.array(mesh.bounds).reshape(3, 2)
    size = np.max(bounds[:, 1] - bounds[:, 0])

    if pts is None or len(pts) < 3:
        return None, center, size

    loop_center = pts.mean(axis=0)
    X = pts - loop_center

    # PCA on boundary loop to get best-fit plane
    _, _, vh = np.linalg.svd(X, full_matrices=False)
    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    # Make the normal point outward from the object center toward the opening
    if np.dot(normal, loop_center - np.array(center)) < 0:
        normal = -normal

    # Camera position: back off along opening normal
    cam_pos = loop_center + normal * (2.5 * size)

    return cam_pos, loop_center, size


n = len(selected_files)
cols = 5 if n >= 5 else n
rows = math.ceil(n / cols)

plotter = pv.Plotter(
    shape=(rows, cols),
    off_screen=True,
    border=True,
    window_size=(cols * 900, rows * 900),
)
plotter.set_background("white")

for i, stl_path in enumerate(selected_files):
    r, c = divmod(i, cols)
    plotter.subplot(r, c)
    plotter.set_background("white")

    key = os.path.basename(stl_path).replace("label_surface.stl", "")
    mesh = pv.read(stl_path)
    mesh = mesh.triangulate().clean()
    mesh = mesh.compute_normals(
        cell_normals=True,
        point_normals=False,
        auto_orient_normals=True,
        consistent_normals=True,
        split_vertices=True,
    )

    plotter.remove_all_lights()
    plotter.add_light(pv.Light(position=(1, 1, 1), intensity=0.8, light_type="camera light"))
    plotter.add_light(pv.Light(position=(-1, -1, 1), intensity=0.4, light_type="camera light"))

    plotter.add_mesh(
        mesh,
        color=colors[i % len(colors)],
        smooth_shading=False,
        show_edges=True,
        edge_color="black",
        line_width=0.6,
        ambient=0.8,
        diffuse=0.2,
        specular=0.0,
    )

    plotter.add_text(key, font_size=16, color="black")

    cam_pos, focal_point, size = camera_for_opening(mesh)

    if cam_pos is None:
        plotter.view_isometric()
        plotter.reset_camera()
    else:
        plotter.camera_position = [
            tuple(cam_pos),
            tuple(focal_point),
            (0, 0, 1),
        ]

    plotter.camera.zoom(1.15)

for j in range(n, rows * cols):
    r, c = divmod(j, cols)
    plotter.subplot(r, c)
    plotter.set_background("white")

plotter.screenshot(out_file, scale=2, transparent_background=False)
plotter.close()

print(f"Saved: {out_file}")
