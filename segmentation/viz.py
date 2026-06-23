import pyvista as pv

grid = pv.read("/home/cyrilpillai36/Desktop/TEER/mitral.msh")
pl = pv.Plotter()
pl.add_mesh(grid, show_edges=True, color="lightgray")
pl.show()


