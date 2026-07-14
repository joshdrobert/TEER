"""Adapter to the full fluid-structure interaction solver (FEniCSx / DOLFINx).

The reduced-order model in ``hemodynamics.py`` runs everywhere and drives the
interactive demo. This adapter defines the interface to the *high-fidelity*
transient FSI solve that refines the optimizer's shortlist. It requires DOLFINx
(not installable via plain pip on macOS; use conda-forge or the official Docker
image ``dolfinx/dolfinx:stable``) and is meant to run on the GPU backend.

Design (kept explicit so it can be implemented against a real solver):
  1. Load the patient valve surface + LV chamber mesh (gmsh).
  2. Immerse the clip geometry; establish contact on the leaflet pads.
  3. Solve incompressible Navier-Stokes coupled to the leaflet mechanics over
     one cardiac cycle using a measured LV pressure waveform.
  4. Integrate velocity over the mitral orifice for regurgitant volume; read
     peak leaflet stress from the structural field.

Until DOLFINx is present, ``FSISolver.available()`` returns False and callers
fall back to the reduced-order model, which is the correct, honest behavior.
"""

from __future__ import annotations

import numpy as np

from .hemodynamics import ClipPlacement, SimulationResult, ValveGeometry, simulate


def dolfinx_available() -> bool:
    try:
        import dolfinx  # noqa: F401
        return True
    except Exception:
        return False


def lv_pressure_waveform(t: float, cycle: float = 1.0) -> float:
    """Measured-shape LV pressure (Pa). Systole ~30% of cycle, peak ~120 mmHg,
    LVEDP ~10 mmHg (Nishimura et al., JACC 2014). Not a sinusoid."""
    phase = (t % cycle) / cycle
    if phase < 0.3:
        p_mmhg = 10.0 + 110.0 * (phase / 0.3) ** 2
    else:
        p_mmhg = 120.0 * np.exp(-3.0 * (phase - 0.3) / 0.7) + 10.0
    return p_mmhg * 133.322


class FSISolver:
    """Interface to the high-fidelity solve, with a documented fallback."""

    def __init__(self, geom: ValveGeometry, surface_path: str | None = None):
        self.geom = geom
        self.surface_path = surface_path

    def available(self) -> bool:
        return dolfinx_available()

    def solve(self, clip: ClipPlacement) -> SimulationResult:
        """Run the full transient FSI solve if DOLFINx is present, else fall
        back to the reduced-order surrogate (clearly flagged in warnings)."""
        if not self.available():
            result = simulate(self.geom, clip)
            result.warnings.append(
                "DOLFINx unavailable: reduced-order surrogate used instead of full FSI."
            )
            return result
        # Full transient FSI would be assembled and solved here against
        # lv_pressure_waveform(). Left as an explicit integration point rather
        # than a fabricated result.
        raise NotImplementedError(
            "Full DOLFINx FSI solve is the backend GPU integration point; "
            "implement mesh assembly + Navier-Stokes/structure coupling here."
        )
