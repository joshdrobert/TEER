"""TEER backend: real geometry loading, a physics-based reduced-order
hemodynamic model, placement optimization, and a FastAPI service.

The reduced-order model is a fast, physiologically-grounded surrogate for the
full FEniCSx fluid-structure solve (see ``fsi_adapter.py``). It runs anywhere;
the full solver requires DOLFINx and is intended for the backend GPU service.
"""

__all__ = ["hemodynamics", "cases", "optimization"]
