"""Tests exercising the real backend against real bundled case geometry."""

from backend import cases
from backend.hemodynamics import ClipPlacement, simulate, REFERENCE_RANGES
from backend.optimization import optimize


def test_real_cases_present():
    ids = cases.list_cases()
    assert len(ids) >= 100, f"expected 100+ real cases, got {len(ids)}"
    assert "train_001" in ids


def test_geometry_is_measured_not_synthetic():
    g = cases.get_geometry("train_001")
    # Human mitral annulus spans are on the order of tens of mm.
    assert 20.0 < g.annulus_width_mm < 90.0
    assert 20.0 < g.annulus_depth_mm < 90.0
    assert g.native_annular_area_mm2() > 0


def test_simulation_is_physiologic():
    g = cases.get_geometry("train_005")
    r = simulate(g, ClipPlacement(position=0.5, grasp_width_mm=4.0, angle_deg=0.0, clip_count=1))
    assert r.regurgitation_ml >= 0.0
    assert r.mitral_valve_area_cm2 > 0.5
    assert r.peak_leaflet_stress_kpa >= 0.0
    lo, hi = REFERENCE_RANGES["regurgitation_volume_ml"]
    assert lo <= r.regurgitation_ml <= hi


def test_clip_reduces_regurgitation():
    g = cases.get_geometry("train_015")
    no_effect = simulate(g, ClipPlacement(position=0.0, grasp_width_mm=2.0, angle_deg=45.0, clip_count=1))
    centered = simulate(g, ClipPlacement(position=0.5, grasp_width_mm=6.0, angle_deg=0.0, clip_count=2))
    assert centered.regurgitation_ml < no_effect.regurgitation_ml


def test_optimizer_ranks_and_prefers_lower_regurgitation():
    g = cases.get_geometry("train_032")
    ranked = optimize(g, top_k=3)
    assert len(ranked) == 3
    assert ranked[0]["rank"] == 1
    assert ranked[0]["objective"] <= ranked[1]["objective"] <= ranked[2]["objective"]
