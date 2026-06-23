import pyvista as pv

grid = pv.read("/home/cyrilpillai36/Desktop/TEER/mitral_velocity_like.pvd")
print(grid.point_data.keys())  # confirm the array name, likely "velocity_like"

grid.set_active_vectors("velocity_like")

pl = pv.Plotter()
pl.add_mesh(grid, opacity=0.25, show_edges=False)

glyphs = grid.glyph(orient="velocity_like", scale=False, factor=0.3)
pl.add_mesh(glyphs)

pl.show()