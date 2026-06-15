"""Physician-facing export schemas and serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List

from .exceptions import ExportSerializationError
from .schemas import CandidateOutcome, ExportOverlayPoint, ExportPayload


class PhysicianExportService:
    """Serialize ranked clip recommendations for fusion-imaging systems."""

    def build_payloads(self, subject_id: str, candidates: List[CandidateOutcome]) -> List[ExportPayload]:
        """Map candidate placements to external overlay payloads."""
        payloads: List[ExportPayload] = []
        for rank, candidate in enumerate(candidates, start=1):
            overlay_points = [
                ExportOverlayPoint(
                    label=f"clip_{index + 1}",
                    coordinates_mm=(placement.x_mm, placement.y_mm, placement.z_mm),
                    orientation_deg=placement.theta_deg,
                )
                for index, placement in enumerate(candidate.placements)
            ]
            payloads.append(
                ExportPayload(
                    subject_id=subject_id,
                    rank=rank,
                    clip_count=candidate.clip_count,
                    objective_value=candidate.objective_value,
                    regurgitant_volume_ml=candidate.regurgitant_volume_ml,
                    max_leaflet_stress_kpa=candidate.max_leaflet_stress_kpa,
                    overlay_points=overlay_points,
                    metadata={
                        "simulation_artifact_dir": str(candidate.simulation_artifact_dir) if candidate.simulation_artifact_dir else None,
                        "stress_map_path": str(candidate.stress_map_path) if candidate.stress_map_path else None,
                        "fusion_space": "TEE_probe_patient_frame",
                    },
                )
            )
        return payloads

    def export_json(self, payloads: List[ExportPayload], output_path: Path) -> Path:
        """Write payloads as a JSON array for downstream clinical systems."""
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps([payload.to_dict() for payload in payloads], indent=2))
        except OSError as exc:
            raise ExportSerializationError(f"Failed to write export payloads to {output_path}") from exc
        return output_path
