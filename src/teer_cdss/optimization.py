"""Global search over clip count and placement candidates."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Sequence

import numpy as np

from .exceptions import OptimizationSearchError
from .fsi import FSIOrchestrator
from .schemas import (
    CandidateOutcome,
    ClipPlacement,
    OptimizationWeights,
    SimulationRequest,
    TissueProperties,
    FluidProperties,
)


@dataclass
class SearchSpace:
    """Continuous bounds for candidate placement generation."""

    x_bounds_mm: tuple[float, float]
    y_bounds_mm: tuple[float, float]
    z_bounds_mm: tuple[float, float]
    theta_bounds_deg: tuple[float, float]


class CandidateGenerator:
    """Generate clip placements using a simple Latin-hypercube style sampler."""

    def __init__(self, search_space: SearchSpace, random_seed: int = 7) -> None:
        self.search_space = search_space
        self.rng = np.random.default_rng(random_seed)

    def sample(self, clip_count: int, samples: int) -> List[List[ClipPlacement]]:
        """Generate candidate placement lists for a given clip count."""
        candidates: List[List[ClipPlacement]] = []
        for _ in range(samples):
            placements: List[ClipPlacement] = []
            for _clip_idx in range(clip_count):
                placements.append(
                    ClipPlacement(
                        x_mm=float(self.rng.uniform(*self.search_space.x_bounds_mm)),
                        y_mm=float(self.rng.uniform(*self.search_space.y_bounds_mm)),
                        z_mm=float(self.rng.uniform(*self.search_space.z_bounds_mm)),
                        theta_deg=float(self.rng.uniform(*self.search_space.theta_bounds_deg)),
                    )
                )
            candidates.append(placements)
        return candidates


class TEERObjective:
    """Objective function implementing the requested clinical tradeoff."""

    def __init__(self, weights: OptimizationWeights) -> None:
        self.weights = weights

    def __call__(self, clip_count: int, regurgitant_volume_ml: float, max_leaflet_stress_kpa: float) -> float:
        return (
            self.weights.alpha * regurgitant_volume_ml
            + self.weights.beta * max_leaflet_stress_kpa
            + self.weights.gamma * float(clip_count)
        )


class ClipOptimizationEngine:
    """Parallel search engine that retains top-k clinically reviewable options."""

    def __init__(
        self,
        fsi: FSIOrchestrator,
        objective: TEERObjective,
        candidate_generator: CandidateGenerator,
        fluid: FluidProperties,
        tissue: TissueProperties,
    ) -> None:
        self.fsi = fsi
        self.objective = objective
        self.candidate_generator = candidate_generator
        self.fluid = fluid
        self.tissue = tissue

    def search(
        self,
        subject_id: str,
        cycle_duration_ms: float,
        output_root: Path,
        top_k: int,
        samples_per_clip_count: int = 4,
    ) -> List[CandidateOutcome]:
        """Evaluate candidate clip strategies across 1 to 3 clips."""
        futures = []
        outcomes: List[CandidateOutcome] = []
        with ThreadPoolExecutor(max_workers=4) as executor:
            for clip_count in (1, 2, 3):
                for candidate_idx, placements in enumerate(self.candidate_generator.sample(clip_count, samples_per_clip_count)):
                    candidate_dir = output_root / f"candidate_{clip_count}_{candidate_idx}"
                    request = SimulationRequest(
                        subject_id=subject_id,
                        placements=placements,
                        fluid=self.fluid,
                        tissue=self.tissue,
                        cycle_duration_ms=cycle_duration_ms,
                        output_dir=candidate_dir,
                    )
                    futures.append(executor.submit(self._evaluate_candidate, request))
            for future in as_completed(futures):
                outcomes.append(future.result())
        if not outcomes:
            raise OptimizationSearchError("No candidate outcomes were produced.")
        return sorted(outcomes, key=lambda outcome: outcome.objective_value)[:top_k]

    def _evaluate_candidate(self, request: SimulationRequest) -> CandidateOutcome:
        result = self.fsi.evaluate(request)
        objective_value = self.objective(
            len(request.placements),
            result.regurgitant_volume_ml,
            result.stress_summary.max_von_mises_kpa,
        )
        return CandidateOutcome(
            clip_count=len(request.placements),
            placements=request.placements,
            objective_value=objective_value,
            regurgitant_volume_ml=result.regurgitant_volume_ml,
            max_leaflet_stress_kpa=result.stress_summary.max_von_mises_kpa,
            stress_map_path=result.output_artifacts.get("manifest"),
            simulation_artifact_dir=request.output_dir,
        )
