import os
import random
import pyvista as pv

mesh_folder = "../data/train/meshes"
out_dir = "."
n_samples = 10
random.seed(42)

stl_files = [
    os.path.join(mesh_folder, f)
    for f in os.listdir(mesh_folder)
    if f.lower().endswith("label_surface.stl")
]

if not stl_files:
    raise ValueError("No STL files found.")

selected_files = random.sample(stl_files, min(n_samples, len(stl_files)))

pv.global_theme.background = "white"
pv.global_theme.font.color = "black"
pv.global_theme.smooth_shading = True

for stl_path in selected_files:
    key = os.path.basename(stl_path).replace("label_surface.stl", "")
    mesh = pv.read(stl_path)

    p = pv.Plotter(off_screen=True, window_size=(3000, 3000))
    p.set_background("white")

    p.add_mesh(
        mesh,
        color="whitesmoke",
        smooth_shading=True,
        show_edges=False,
        ambient=0.25,
        diffuse=0.7,
        specular=0.35,
        specular_power=20,
    )

    p.add_light(pv.Light(position=(1, 1, 1), intensity=0.8, light_type="camera light"))
    p.add_light(pv.Light(position=(-1, -0.5, 0.5), intensity=0.4, light_type="camera light"))

    p.view_isometric()
    p.camera.zoom(1.4)

    p.screenshot(os.path.join(out_dir, f"{key}_mesh_pub.png"), transparent_background=False)
    p.close()
