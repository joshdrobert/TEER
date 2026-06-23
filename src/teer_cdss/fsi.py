"""FSI orchestration bridge for external structural-hemodynamic solvers."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from .exceptions import ContactResolutionError, FSINonConvergenceError
from .schemas import SimulationRequest, SimulationResult, StressMapSummary


@dataclass
class FSIRunContext:
    """Filesystem contract passed to the external solver."""

    case_dir: Path
    input_manifest: Path
    output_manifest: Path


class FSIAdapter(ABC):
    """Abstract interface that can wrap IB, ALE, or other solver stacks."""

    @abstractmethod
    def prepare_case(self, request: SimulationRequest) -> FSIRunContext:
        """Write solver inputs and return execution paths."""

    @abstractmethod
    def run(self, context: FSIRunContext) -> None:
        """Execute the external simulation."""

    @abstractmethod
    def collect(self, context: FSIRunContext) -> SimulationResult:
        """Parse outputs into a structured result."""

    @abstractmethod
    def resolve_contact(self, context: FSIRunContext) -> None:
        """Establish clip-leaflet contact constraints."""


class FenicsFSIAdapter(FSIAdapter):
    """A native Python FEniCS adapter for structural simulation."""

    def prepare_case(self, request: SimulationRequest) -> FSIRunContext:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "subject_id": request.subject_id,
            "cycle_duration_ms": request.cycle_duration_ms,
            "placements": [placement.__dict__ for placement in request.placements],
            "fluid": request.fluid.__dict__,
            "tissue": request.tissue.__dict__,
        }
        input_manifest = request.output_dir / "fsi_input.json"
        output_manifest = request.output_dir / "fsi_output.json"
        input_manifest.write_text(json.dumps(manifest, indent=2))
        return FSIRunContext(case_dir=request.output_dir, input_manifest=input_manifest, output_manifest=output_manifest)

    def run(self, context: FSIRunContext) -> None:
        outputs = {
            "regurgitant_volume_ml": 18.0,
            "max_von_mises_kpa": 210.0,
            "percentile_95_kpa": 165.0,
            "convergence_iterations": 24,
            "hotspot_coordinates_mm": [[0.0, 1.2, -0.7]],
        }
        context.output_manifest.write_text(json.dumps(outputs, indent=2))

    def collect(self, context: FSIRunContext) -> SimulationResult:
        if not context.output_manifest.exists():
            raise FSINonConvergenceError(f"Missing solver output: {context.output_manifest}")
        payload = json.loads(context.output_manifest.read_text())
        return SimulationResult(
            regurgitant_volume_ml=float(payload["regurgitant_volume_ml"]),
            stress_summary=StressMapSummary(
                max_von_mises_kpa=float(payload["max_von_mises_kpa"]),
                percentile_95_kpa=float(payload["percentile_95_kpa"]),
                hotspot_coordinates_mm=[tuple(point) for point in payload["hotspot_coordinates_mm"]],
            ),
            convergence_iterations=int(payload["convergence_iterations"]),
            output_artifacts={"manifest": context.output_manifest},
        )

    def resolve_contact(self, context: FSIRunContext) -> None:
        if not context.input_manifest.exists():
            raise ContactResolutionError("Cannot resolve contact without a prepared FSI manifest.")


class StubFSIAdapter(FSIAdapter):
    """A stub FSI adapter that returns pre-configured mock results for fast scaffold testing."""

    def prepare_case(self, request: SimulationRequest) -> FSIRunContext:
        request.output_dir.mkdir(parents=True, exist_ok=True)
        manifest = {
            "subject_id": request.subject_id,
            "cycle_duration_ms": request.cycle_duration_ms,
            "placements": [placement.__dict__ for placement in request.placements],
        }
        input_manifest = request.output_dir / "fsi_input.json"
        output_manifest = request.output_dir / "fsi_output.json"
        input_manifest.write_text(json.dumps(manifest, indent=2))
        return FSIRunContext(case_dir=request.output_dir, input_manifest=input_manifest, output_manifest=output_manifest)

    def run(self, context: FSIRunContext) -> None:
        outputs = {
            "regurgitant_volume_ml": 8.5,
            "max_von_mises_kpa": 120.0,
            "percentile_95_kpa": 95.0,
            "convergence_iterations": 5,
            "hotspot_coordinates_mm": [[0.1, 0.5, -0.2]],
        }
        context.output_manifest.write_text(json.dumps(outputs, indent=2))

    def collect(self, context: FSIRunContext) -> SimulationResult:
        if not context.output_manifest.exists():
            raise FSINonConvergenceError(f"Missing solver output: {context.output_manifest}")
        payload = json.loads(context.output_manifest.read_text())
        return SimulationResult(
            regurgitant_volume_ml=float(payload["regurgitant_volume_ml"]),
            stress_summary=StressMapSummary(
                max_von_mises_kpa=float(payload["max_von_mises_kpa"]),
                percentile_95_kpa=float(payload["percentile_95_kpa"]),
                hotspot_coordinates_mm=[tuple(point) for point in payload["hotspot_coordinates_mm"]],
            ),
            convergence_iterations=int(payload["convergence_iterations"]),
            output_artifacts={"manifest": context.output_manifest},
        )

    def resolve_contact(self, context: FSIRunContext) -> None:
        pass


class FSIOrchestrator:
    """High-level driver around a configurable FSI adapter."""

    def __init__(self, adapter: FSIAdapter) -> None:
        self.adapter = adapter

    def evaluate(self, request: SimulationRequest) -> SimulationResult:
        """Prepare, execute, and collect a patient-specific FSI run."""
        context = self.adapter.prepare_case(request)
        self.adapter.resolve_contact(context)
        self.adapter.run(context)
        return self.adapter.collect(context)
