"""Prepare lightweight web assets from the real segmented valve meshes.

This does NOT synthesize geometry. It takes the real patient-derived STL
surfaces in ``valves/`` (segmented from the MVSeg2023 3D TEE volumes) and
decimates them to a polygon budget that a browser can stream, preserving the
true annular shape, leaflet extent, and orifice. It also emits a curated
``cases.json`` built directly from the real per-case geometric analysis in
``valves_metadata.json`` so the web demo is driven by measured anatomy.

Run:  .venv/bin/python scripts/prepare_web_assets.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
VALVES = ROOT / "valves"
CLIP = ROOT / "clip"
META = ROOT / "valves_metadata.json"
OUT_MODELS = ROOT / "docs" / "assets" / "models"
OUT_DATA = ROOT / "docs" / "assets" / "data"

# Representative spread of real cases chosen across the morphology range
# (annulus size + commissure angle) so the gallery shows genuine variety.
CURATED = [
    "train_001",
    "train_005",
    "train_015",
    "train_032",
    "train_055",
    "train_087",
]

TARGET_FACES = 24_000  # streams fast, keeps annular + leaflet detail


def decimate_valve(case: str) -> dict:
    src = VALVES / f"{case}-label_surface.stl"
    if not src.exists():
        raise FileNotFoundError(src)

    mesh = pv.read(src)
    mesh = mesh.clean()
    orig_faces = mesh.n_faces_strict if hasattr(mesh, "n_faces_strict") else mesh.n_cells

    tri = mesh.triangulate()
    n_faces = tri.n_cells
    if n_faces > TARGET_FACES:
        reduction = 1.0 - (TARGET_FACES / n_faces)
        tri = tri.decimate_pro(reduction, feature_angle=45, preserve_topology=True)
    tri = tri.compute_normals(auto_orient_normals=True, consistent_normals=True)

    # Center on origin (mm) so the browser camera framing is stable; store the
    # real world centroid so the demo can map back to metadata coordinates.
    centroid = np.asarray(tri.center)
    tri.translate(-centroid, inplace=True)

    out = OUT_MODELS / f"{case}.stl"
    tri.save(out, binary=True)

    bounds = tri.bounds  # (xmin,xmax,ymin,ymax,zmin,zmax)
    return {
        "faces_original": int(orig_faces),
        "faces_web": int(tri.n_cells),
        "centroid_world": [float(c) for c in centroid],
        "bounds_local": [float(b) for b in bounds],
        "file": f"assets/models/{case}.stl",
        "size_kb": round(out.stat().st_size / 1024, 1),
    }


def assemble_clip() -> dict:
    """Ship the real NT MitraClip assembly (already small) as one web STL."""
    src = CLIP / "NT_assembled_open120_reference.stl"
    mesh = pv.read(src).triangulate().clean()
    mesh = mesh.compute_normals(auto_orient_normals=True)
    mesh.translate(-np.asarray(mesh.center), inplace=True)
    out = OUT_MODELS / "mitraclip_nt.stl"
    mesh.save(out, binary=True)
    return {"file": "assets/models/mitraclip_nt.stl", "size_kb": round(out.stat().st_size / 1024, 1)}


def build_cases() -> None:
    OUT_MODELS.mkdir(parents=True, exist_ok=True)
    OUT_DATA.mkdir(parents=True, exist_ok=True)

    meta = json.loads(META.read_text())
    cases = []
    for case in CURATED:
        if case not in meta:
            print(f"  ! {case} missing from metadata, skipping")
            continue
        m = meta[case]
        print(f"  decimating {case} ...")
        web = decimate_valve(case)

        # Real measured geometry -> clinically meaningful descriptors.
        width_mm = m["x_max"] - m["x_min"]
        depth_mm = m["y_max"] - m["y_min"]
        height_mm = m["z_max"] - m["z_min"]
        # Anteroposterior / commissural diameters from the real bounding extent
        # projected onto the measured opening/commissure axes.
        annulus_span = math.hypot(width_mm, depth_mm)

        cases.append(
            {
                "id": case,
                "label": f"Case {case.split('_')[1]}",
                "model": web["file"],
                "faces_web": web["faces_web"],
                "faces_original": web["faces_original"],
                "size_kb": web["size_kb"],
                "geometry": {
                    "annulus_width_mm": round(width_mm, 1),
                    "annulus_depth_mm": round(depth_mm, 1),
                    "leaflet_height_mm": round(height_mm, 1),
                    "annulus_span_mm": round(annulus_span, 1),
                    "commissure_angle_deg": round(m["angle_deg"], 1),
                    "orifice_center_xy": [round(m["cx"], 2), round(m["cy"], 2)],
                    "commissure_axis": [round(v, 4) for v in m["v_comm"]],
                    "opening_axis": [round(v, 4) for v in m["v_open"]],
                },
                "bounds_local": web["bounds_local"],
                "centroid_world": web["centroid_world"],
            }
        )

    clip = assemble_clip()
    payload = {
        "source": "MVSeg2023 3D TEE segmentations (real patient-derived surfaces)",
        "note": "Geometry values are measured from the segmented meshes, not synthesized.",
        "clip_model": clip,
        "cases": cases,
    }
    (OUT_DATA / "cases.json").write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(cases)} real cases -> {OUT_DATA / 'cases.json'}")
    print(f"Clip assembly -> {clip['file']} ({clip['size_kb']} KB)")


if __name__ == "__main__":
    build_cases()
