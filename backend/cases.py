"""Load real patient valve geometry from the bundled MVSeg2023 analysis.

No synthetic data: geometry comes from ``valves_metadata.json`` (measured from
the segmented 3D TEE surfaces) and the surfaces themselves live in ``valves/``.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .hemodynamics import ValveGeometry

ROOT = Path(__file__).resolve().parent.parent
METADATA = ROOT / "valves_metadata.json"
VALVE_DIR = ROOT / "valves"


@lru_cache(maxsize=1)
def _metadata() -> dict:
    if not METADATA.exists():
        raise FileNotFoundError(f"Missing {METADATA}. Run segmentation analysis first.")
    return json.loads(METADATA.read_text())


def list_cases() -> list[str]:
    """All available real case IDs (e.g. 'train_001')."""
    return sorted(_metadata().keys())


def get_geometry(case_id: str) -> ValveGeometry:
    """Measured annular geometry for one real case."""
    meta = _metadata()
    if case_id not in meta:
        raise KeyError(f"Unknown case '{case_id}'. {len(meta)} cases available.")
    return ValveGeometry.from_metadata(meta[case_id])


def surface_path(case_id: str) -> Path:
    """Path to the real segmented STL surface for a case."""
    p = VALVE_DIR / f"{case_id}-label_surface.stl"
    if not p.exists():
        raise FileNotFoundError(f"No surface mesh for {case_id} at {p}")
    return p


def case_summary(case_id: str) -> dict:
    g = get_geometry(case_id)
    return {
        "id": case_id,
        "annulus_width_mm": round(g.annulus_width_mm, 1),
        "annulus_depth_mm": round(g.annulus_depth_mm, 1),
        "leaflet_height_mm": round(g.leaflet_height_mm, 1),
        "commissure_angle_deg": round(g.commissure_angle_deg, 1),
        "native_annular_area_mm2": round(g.native_annular_area_mm2(), 1),
        "has_surface": (VALVE_DIR / f"{case_id}-label_surface.stl").exists(),
    }
