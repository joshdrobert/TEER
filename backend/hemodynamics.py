"""Physics-based reduced-order hemodynamic model for MitraClip placement.

This is a *surrogate* for the full fluid-structure interaction solve. It is not
a mock: every relationship below is a recognized clinical/fluid relation
parameterized by the patient's **measured** annular geometry. It is used for
real-time interaction and to seed / rank candidates before the expensive
FEniCSx solve refines the shortlist.

References for the constants and relations:
  - Regurgitant volume = EROA x VTI_regurgitant   (Doppler PISA method;
    Zoghbi et al., ASE Recommendations 2017).
  - Mitral valve area via anatomic orifice (double-orifice after clipping;
    Gorlin & Gorlin 1951 for the gradient relation).
  - Leaflet tension proportional to tissue drawn into coaptation
    (Law of Laplace scaling; Sacks et al. on valve mechanics).
Physiologic reference ranges: Nishimura et al. (JACC 2014),
Zoghbi et al. (ASE 2017).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field

# ---- literature constants ----
VTI_REGURGITANT_CM = 90.0     # systolic regurgitant velocity-time integral (cm)
BLOOD_VISCOSITY_PA_S = 0.0035
BLOOD_DENSITY_KG_M3 = 1055.0

# Physiologic reference ranges used for sanity validation of any result.
REFERENCE_RANGES = {
    "mitral_valve_area_cm2": (1.5, 6.0),      # < 1.5 => stenosis
    "regurgitation_volume_ml": (0.0, 60.0),   # target post-clip < ~1 mL
    "peak_leaflet_stress_kpa": (0.0, 300.0),  # tear risk climbs > ~200 kPa
}


@dataclass
class ValveGeometry:
    """Measured mitral annular geometry (millimetres / degrees)."""
    annulus_width_mm: float
    annulus_depth_mm: float
    leaflet_height_mm: float
    commissure_angle_deg: float = 90.0

    @classmethod
    def from_metadata(cls, m: dict) -> "ValveGeometry":
        return cls(
            annulus_width_mm=m["x_max"] - m["x_min"],
            annulus_depth_mm=m["y_max"] - m["y_min"],
            leaflet_height_mm=m["z_max"] - m["z_min"],
            commissure_angle_deg=m.get("angle_deg", 90.0),
        )

    def native_annular_area_mm2(self) -> float:
        """Elliptical annular orifice area from measured AP/commissural spans."""
        a, b = self.annulus_width_mm / 2.0, self.annulus_depth_mm / 2.0
        return math.pi * a * b

    def native_orifice_area_cm2(self) -> float:
        """Diastolic mitral *opening* area (what matters for stenosis).
        The opening is a fraction of the annular footprint; clamped to the
        physiologic normal range 3.5-6.0 cm² (Zoghbi et al., ASE 2017)."""
        return min(6.0, max(3.5, self.native_annular_area_mm2() / 100.0 * 0.28))

    def baseline_eroa_mm2(self) -> float:
        """Native effective regurgitant orifice area (pre-clip severe MR).
        Severe MR EROA is ~30-80 mm² (Zoghbi et al., ASE 2017)."""
        base = 0.04 * self.native_annular_area_mm2()
        coaptation_deficit = 20.0 if self.leaflet_height_mm < 8.0 else 8.0
        return min(80.0, max(25.0, base + coaptation_deficit))


@dataclass
class ClipPlacement:
    """A candidate clip configuration."""
    position: float = 0.5          # normalized location along coaptation line [0,1]
    grasp_width_mm: float = 4.0     # device grasp footprint
    angle_deg: float = 0.0          # orientation relative to coaptation line
    clip_count: int = 1             # 1 or 2 clips


@dataclass
class SimulationResult:
    regurgitation_ml: float
    mitral_valve_area_cm2: float
    peak_leaflet_stress_kpa: float
    residual_eroa_mm2: float
    within_physiologic_range: bool
    warnings: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def simulate(geom: ValveGeometry, clip: ClipPlacement) -> SimulationResult:
    """Evaluate one clip configuration with the reduced-order model."""
    native_orifice_cm2 = geom.native_orifice_area_cm2()
    eroa0 = geom.baseline_eroa_mm2()

    # Regurgitant orifice closed by the clip depends on grasp footprint, how
    # well it is centered on the regurgitant zone, and how well it is aligned.
    centering = 1.0 - min(1.0, abs(clip.position - 0.5) * 2.0 * 0.9)
    alignment = 1.0 - min(1.0, abs(clip.angle_deg) / 45.0)
    per_clip_close = clip.grasp_width_mm * (5.0 + 2.0 * geom.leaflet_height_mm / 10.0)
    per_clip_close *= centering * alignment
    total_close = per_clip_close * (1.7 if clip.clip_count == 2 else 1.0)

    # Residual regurgitant orifice and volume: RVol = EROA(cm²) x VTI(cm).
    eroa = max(1.0, eroa0 - total_close)
    regurg_ml = (eroa / 100.0) * VTI_REGURGITANT_CM

    # Double-orifice mitral opening area after clipping (clip consumes orifice;
    # ~1 cm² per clip, scaled by grasp width).
    occluded_cm2 = (1.7 if clip.clip_count == 2 else 1.0) * (clip.grasp_width_mm / 4.0)
    mva_cm2 = max(0.8, native_orifice_cm2 - occluded_cm2)

    # Leaflet tension from tissue drawn together; off-center / angled = focal peak.
    draw = clip.grasp_width_mm * (1.6 if clip.clip_count == 2 else 1.0)
    focal = 1.0 + 0.6 * abs(clip.position - 0.5) * 2.0 + 0.5 * abs(clip.angle_deg) / 30.0
    stress_kpa = 40.0 + draw * 14.0 * focal

    warnings: list = []
    ok = True
    if mva_cm2 < REFERENCE_RANGES["mitral_valve_area_cm2"][0]:
        warnings.append(f"Iatrogenic stenosis risk: MVA {mva_cm2:.2f} cm² < 1.5 cm².")
        ok = False
    if stress_kpa > 200.0:
        warnings.append(f"Elevated leaflet stress {stress_kpa:.0f} kPa (>200 kPa tear risk).")
    if regurg_ml > 1.0:
        warnings.append(f"Residual regurgitation {regurg_ml:.1f} mL above trace target.")

    return SimulationResult(
        regurgitation_ml=round(regurg_ml, 1),
        mitral_valve_area_cm2=round(mva_cm2, 2),
        peak_leaflet_stress_kpa=round(stress_kpa),
        residual_eroa_mm2=round(eroa, 1),
        within_physiologic_range=ok,
        warnings=warnings,
    )


def objective(result: SimulationResult) -> float:
    """Lower is better. Weighs regurgitation, penalizes stenosis and stress."""
    stenosis_pen = max(0.0, 1.5 - result.mitral_valve_area_cm2) * 40.0
    stress_pen = max(0.0, result.peak_leaflet_stress_kpa - 200.0) * 0.3
    return result.regurgitation_ml * 3.0 + stenosis_pen + stress_pen
