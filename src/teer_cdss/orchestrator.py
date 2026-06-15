"""Top-level orchestration for the TEER decision-support pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .clinical import ClinicalPreprocessor, DICOMIngestor, HIPAAAnonymizer
from .export import PhysicianExportService
from .fsi import FSIOrchestrator, StubFSIAdapter
from .mesh import MeshReconstructor
from .optimization import CandidateGenerator, ClipOptimizationEngine, SearchSpace, TEERObjective
from .schemas import PipelineRunSummary, TEERPipelineConfig
from .segmentation import MitralSegmentationEngine


@dataclass
class PipelineComponents:
    """Concrete services wired into the orchestrator."""

    dicom_ingestor: DICOMIngestor
    anonymizer: HIPAAAnonymizer
    preprocessor: ClinicalPreprocessor
    segmenter: MitralSegmentationEngine
    mesher: MeshReconstructor
    optimizer: ClipOptimizationEngine
    exporter: PhysicianExportService


class TEERPipelineOrchestrator:
    """Coordinate ingestion, segmentation, simulation, optimization, and export."""

    def __init__(self, config: TEERPipelineConfig, workspace: Path, components: Optional[PipelineComponents] = None) -> None:
        self.config = config
        self.workspace = workspace
        self.components = components or self._build_components(config)

    def run(self, dicom_paths: Sequence[Path], operator: str = "system") -> PipelineRunSummary:
        """Execute the scaffolded TEER planning workflow for one study."""
        datasets, _frames = self.components.dicom_ingestor.read_series(dicom_paths)
        anonymization = self.components.anonymizer.anonymize(datasets[0], operator=operator)
        clinical_volume = self.components.preprocessor.preprocess(datasets, anonymization)

        volume = clinical_volume.voxel_data
        if volume.ndim == 4:
            reference_frame = volume[volume.shape[0] // 2]
        else:
            reference_frame = volume

        segmentation = self._placeholder_segmentation(reference_frame)
        mesh_artifacts = self.components.mesher.build(segmentation)
        mesh_paths = self.components.mesher.save(mesh_artifacts, self.workspace / "artifacts" / anonymization.subject_hash / "mesh")

        candidates = self.components.optimizer.search(
            subject_id=anonymization.subject_hash,
            cycle_duration_ms=800.0,
            output_root=self.workspace / "artifacts" / anonymization.subject_hash / "optimization",
            top_k=self.config.top_k_recommendations,
        )
        payloads = self.components.exporter.build_payloads(anonymization.subject_hash, candidates)
        export_path = self.components.exporter.export_json(
            payloads,
            self.workspace / "artifacts" / anonymization.subject_hash / "exports" / "recommendations.json",
        )
        artifacts: Dict[str, Path] = {**mesh_paths, "export_payload": export_path}
        return PipelineRunSummary(
            subject_id=anonymization.subject_hash,
            candidate_rankings=candidates,
            export_payloads=payloads,
            artifacts=artifacts,
        )

    def _build_components(self, config: TEERPipelineConfig) -> PipelineComponents:
        fsi = FSIOrchestrator(StubFSIAdapter())
        optimizer = ClipOptimizationEngine(
            fsi=fsi,
            objective=TEERObjective(config.objective),
            candidate_generator=CandidateGenerator(
                SearchSpace(
                    x_bounds_mm=(-6.0, 6.0),
                    y_bounds_mm=(-3.0, 3.0),
                    z_bounds_mm=(-2.5, 2.5),
                    theta_bounds_deg=(-35.0, 35.0),
                )
            ),
            fluid=config.fluid,
            tissue=config.tissue,
        )
        return PipelineComponents(
            dicom_ingestor=DICOMIngestor(),
            anonymizer=HIPAAAnonymizer(),
            preprocessor=ClinicalPreprocessor(config.preprocessing),
            segmenter=MitralSegmentationEngine(config.segmentation),
            mesher=MeshReconstructor(config.mesh, config.clip_spec),
            optimizer=optimizer,
            exporter=PhysicianExportService(),
        )

    @staticmethod
    def _placeholder_segmentation(volume: np.ndarray) -> np.ndarray:
        """Create a coarse anatomical proxy until trained inference is integrated."""
        threshold = np.percentile(volume, 85.0)
        segmentation = np.zeros_like(volume, dtype=np.uint8)
        segmentation[volume >= threshold] = 2
        segmentation[(volume >= np.percentile(volume, 70.0)) & (volume < threshold)] = 3
        return segmentation
