"""Clip placement optimization over the reduced-order model.

Returns the top-k ranked configurations for a real valve. This is the fast
pass that shortlists candidates; in the full system the top-k are then
re-evaluated with the FEniCSx FSI solver (see ``fsi_adapter.py``).
"""

from __future__ import annotations

from dataclasses import asdict

from .hemodynamics import ClipPlacement, ValveGeometry, objective, simulate


def _grid():
    for clips in (1, 2):
        for i in range(9):                 # position 0.30 .. 0.70
            pos = 0.30 + i * 0.05
            for grasp in (3.0, 4.0, 5.0, 6.0):
                for angle in (-20.0, -10.0, 0.0, 10.0, 20.0):
                    yield ClipPlacement(pos, grasp, angle, clips)


def optimize(geom: ValveGeometry, top_k: int = 3) -> list[dict]:
    """Grid-search placements; return ranked results (best first)."""
    scored = []
    for clip in _grid():
        result = simulate(geom, clip)
        scored.append((objective(result), clip, result))
    scored.sort(key=lambda t: t[0])

    ranked = []
    for rank, (score, clip, result) in enumerate(scored[:top_k], start=1):
        ranked.append(
            {
                "rank": rank,
                "objective": round(score, 3),
                "placement": asdict(clip),
                "predicted": result.to_dict(),
                "recommendation": "Recommended"
                if result.within_physiologic_range and result.regurgitation_ml <= 1.0
                else "Review",
            }
        )
    return ranked
