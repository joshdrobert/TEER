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

# Strong, distinct colors that stay visible on white
colors = [
    "#D81B60",  # magenta
    "#1E88E5",  # blue
    "#43A047",  # green
    "#8E24AA",  # purple
    "#FB8C00",  # orange
    "#00897B",  # teal
    "#6D4C41",  # brown
    "#3949AB",  # indigo
    "#C0CA33",  # olive
    "#E53935",  # red
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

for i, stl_path in enumerate(selected_files):
    r, c = divmod(i, cols)
    plotter.subplot(r, c)
    plotter.set_background("white")

    key = os.path.basename(stl_path).replace("label_surface.stl", "")
    mesh = pv.read(stl_path)

    # Clean mesh normals so lighting behaves better
    mesh = mesh.compute_normals(
        cell_normals=False,
        point_normals=True,
        split_vertices=False,
        consistent_normals=True,
        auto_orient_normals=True,
    )

    plotter.add_mesh(
        mesh,
        color=colors[i % len(colors)],
        smooth_shading=False,   # more robust than True for thin/odd meshes
        show_edges=False,
        ambient=0.45,           # brighter base color
        diffuse=0.55,
        specular=0.05,          # reduce white glare
        specular_power=8,
    )

    plotter.add_text(key, font_size=16, color="black")

    # Softer, more even lighting
    plotter.remove_all_lights()
    plotter.add_light(pv.Light(position=(1, 1, 1), intensity=0.6, light_type="camera light"))
    plotter.add_light(pv.Light(position=(-1, -1, 0.5), intensity=0.4, light_type="camera light"))
    plotter.add_light(pv.Light(position=(0, 0, 1), intensity=0.3, light_type="camera light"))

    plotter.view_isometric()
    plotter.camera.zoom(1.35)

# Hide unused panels
for j in range(n, rows * cols):
    r, c = divmod(j, cols)
    plotter.subplot(r, c)
    plotter.set_background("white")

plotter.screenshot(out_file, scale=2, transparent_background=False)
plotter.close()

print(f"Saved: {out_file}")
