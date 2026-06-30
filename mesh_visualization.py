import os
import math
import random
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

    # Make mesh rendering more robust
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
        smooth_shading=False,   # important
        show_edges=True,        # important for thin meshes
        edge_color="black",
        line_width=0.6,
        ambient=0.8,            # stronger base visibility
        diffuse=0.2,
        specular=0.0,           # no white glare
    )

    plotter.add_text(key, font_size=16, color="black")
    plotter.view_isometric()
    plotter.reset_camera()
    plotter.camera.zoom(1.2)

for j in range(n, rows * cols):
    r, c = divmod(j, cols)
    plotter.subplot(r, c)
    plotter.set_background("white")

plotter.screenshot(out_file, scale=2, transparent_background=False)
plotter.close()

print(f"Saved: {out_file}")
