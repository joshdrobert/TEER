"""Surface and volumetric mesh generation for mitral valve modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pyvista as pv

from .exceptions import MeshGenerationError
from .schemas import MeshConfig, MitraClipSpecification


@dataclass(slots=True)
class MeshArtifacts:
    """Output meshes used for mechanics and hemodynamics."""

    surface_mesh: pv.PolyData
    volumetric_mesh: pv.UnstructuredGrid
    clip_mesh: pv.PolyData


class MeshReconstructor:
    """Convert dense label maps into solver-ready meshes."""

    def __init__(self, config: MeshConfig, clip_spec: MitraClipSpecification) -> None:
        self.config = config
        self.clip_spec = clip_spec

    def build(self, segmentation: np.ndarray) -> MeshArtifacts:
        """Generate surface, tetrahedral volume, and a parametric clip surrogate."""
        try:
            surface = self._surface_from_label(segmentation)
            volume = surface.delaunay_3d(alpha=self.config.tetrahedral_edge_length_mm)
            clip_mesh = self._build_clip_geometry()
            return MeshArtifacts(surface_mesh=surface, volumetric_mesh=volume, clip_mesh=clip_mesh)
        except Exception as exc:  # noqa: BLE001
            raise MeshGenerationError("Failed to build patient-specific meshes.") from exc

    def _surface_from_label(self, segmentation: np.ndarray) -> pv.PolyData:
        grid = pv.ImageData(dimensions=np.array(segmentation.shape) + 1)
        grid.cell_data["label"] = segmentation.flatten(order="F")
        surface = grid.contour([self.config.marching_cubes_isovalue], scalars="label")
        surface = surface.smooth(n_iter=self.config.surface_smoothing_iterations)
        if self.config.surface_decimation > 0.0:
            surface = surface.decimate(self.config.surface_decimation)
        if surface.n_points == 0:
            raise MeshGenerationError("Surface extraction produced an empty mesh.")
        return surface.clean()

    def _build_clip_geometry(self) -> pv.PolyData:
        spec = self.clip_spec
        arm_a = pv.Box(bounds=(-spec.grasp_length_mm / 2.0, spec.grasp_length_mm / 2.0, -0.5, 0.5, -0.5, 0.5))
        arm_b = arm_a.copy().translate((0.0, spec.arm_width_mm, 0.0))
        bridge = pv.Cylinder(
            center=(0.0, spec.arm_width_mm / 2.0, 0.0),
            direction=(1.0, 0.0, 0.0),
            radius=spec.arm_thickness_mm / 2.0,
            height=spec.grasp_length_mm / 3.0,
        )
        return arm_a.merge(arm_b).merge(bridge).triangulate().clean()

    def save(self, artifacts: MeshArtifacts, output_dir: Path) -> Dict[str, Path]:
        """Persist meshes for downstream solver ingestion."""
        output_dir.mkdir(parents=True, exist_ok=True)
        surface_path = output_dir / "leaflet_surface.vtp"
        volume_path = output_dir / "leaflet_volume.vtu"
        clip_path = output_dir / "mitraclip.vtp"
        artifacts.surface_mesh.save(surface_path)
        artifacts.volumetric_mesh.save(volume_path)
        artifacts.clip_mesh.save(clip_path)
        return {"surface": surface_path, "volume": volume_path, "clip": clip_path}
