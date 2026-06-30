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

# Distinct, visible colors
colors = [
    "#e41a1c",  # red
    "#377eb8",  # blue
    "#4daf4a",  # green
    "#984ea3",  # purple
    "#ff7f00",  # orange
    "#a65628",  # brown
    "#f781bf",  # pink
    "#17becf",  # cyan
    "#bcbd22",  # olive
    "#636363",  # dark gray
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

    key = os.path.basename(stl_path).replace("label_surface.stl", "")
    mesh = pv.read(stl_path)

    plotter.add_mesh(
        mesh,
        color=colors[i % len(colors)],
        smooth_shading=True,
        show_edges=False,
        ambient=0.25,
        diffuse=0.75,
        specular=0.35,
        specular_power=25,
    )

    plotter.add_text(key, font_size=16, color="black")
    plotter.add_light(pv.Light(position=(1, 1, 1), intensity=0.9, light_type="camera light"))
    plotter.add_light(pv.Light(position=(-1, -0.5, 0.5), intensity=0.45, light_type="camera light"))
    plotter.view_isometric()
    plotter.camera.zoom(1.35)

# Hide any unused panels
for j in range(n, rows * cols):
    r, c = divmod(j, cols)
    plotter.subplot(r, c)
    plotter.set_background("white")

plotter.screenshot(out_file, scale=2, transparent_background=False)
plotter.close()

print(f"Saved: {out_file}")
