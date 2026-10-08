"""Surface and volumetric mesh generation for mitral valve modeling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import json

import numpy as np
import pyvista as pv

from .exceptions import MeshGenerationError
from .schemas import MeshConfig, MitraClipSpecification


@dataclass
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

    def assess_repeatability(self, segmentation: np.ndarray, reference_mesh: pv.PolyData = None) -> Dict[str, float]:
        """
        Assess bounded alignment/feature invariance on a mesh.
        
        This method computes metrics for:
        - Alignment invariance (how consistently mesh aligns under transformations)
        - Feature invariance (how consistently key geometric features are preserved)
        
        Args:
            segmentation: Input segmentation array
            reference_mesh: Optional reference mesh for comparison
            
        Returns:
            Dictionary of repeatability metrics
        """
        # Generate the mesh from segmentation
        artifacts = self.build(segmentation)
        surface_mesh = artifacts.surface_mesh
        
        # Initialize result dictionary with proper physical units
        # Note: PyVista returns raw coordinates, not physical units
        # Physical conversion requires explicit spacing information
        metrics = {
            "alignment_invariance": None,
            "feature_invariance": None,
            "mesh_volume_raw": surface_mesh.volume,
            "mesh_surface_area_raw": surface_mesh.area
        }
        
        # If we have a reference mesh, compute alignment and feature invariance
        if reference_mesh is not None:
            # Compute Hausdorff distance for alignment invariance
            try:
                hd = surface_mesh.compute_hausdorff_distance(reference_mesh)
                # Check for finite values to avoid infinity
                if np.isfinite(hd):
                    metrics["alignment_invariance"] = float(hd)
                else:
                    metrics["alignment_invariance"] = None
            except Exception:
                # If Hausdorff computation fails, set to None (unavailable)
                metrics["alignment_invariance"] = None
            
            # Compute feature invariance using key geometric measurements
            try:
                # Compare surface area and volume between meshes
                reference_volume = reference_mesh.volume
                reference_area = reference_mesh.area
                
                if reference_volume > 0:
                    volume_ratio = abs(surface_mesh.volume - reference_volume) / reference_volume
                    # Check for finite values to avoid infinity
                    if np.isfinite(volume_ratio):
                        metrics["feature_invariance"] = volume_ratio
                    else:
                        metrics["feature_invariance"] = None
                else:
                    # If reference has zero volume, use surface area ratio
                    if reference_area > 0:
                        area_ratio = abs(surface_mesh.area - reference_area) / reference_area
                        # Check for finite values to avoid infinity
                        if np.isfinite(area_ratio):
                            metrics["feature_invariance"] = area_ratio
                        else:
                            metrics["feature_invariance"] = None
                    else:
                        metrics["feature_invariance"] = None
                    
            except Exception:
                # If feature computation fails, set to None (unavailable)
                metrics["feature_invariance"] = None
        
        return metrics

    def record_assessment(self, mesh_name: str, segmentation: np.ndarray, 
                         reference_mesh: pv.PolyData = None) -> Dict[str, object]:
        """
        Record assessment results to research/geometry_repeatability.json
        
        Args:
            mesh_name: Name of the mesh being assessed
            segmentation: Input segmentation array  
            reference_mesh: Optional reference mesh for comparison
            
        Returns:
            Dictionary with assessment data and updated file content
        """
        # Get the assessment metrics
        metrics = self.assess_repeatability(segmentation, reference_mesh)
        
        # Load existing data or create new structure
        try:
            with open("research/geometry_repeatability.json", "r") as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            import datetime
            data = {
                "version": "1.0",
                "assessment_date": datetime.datetime.utcnow().isoformat() + "Z",
                "meshes_assessed": [],
                "metrics": {
                    "alignment_invariance": {},
                    "feature_invariance": {}
                }
            }
        
        # Add mesh to the assessment list
        import datetime
        mesh_entry = {
            "name": mesh_name,
            "metrics": metrics,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z"
        }
        
        data["meshes_assessed"].append(mesh_entry)
        
        # Save back to file
        with open("research/geometry_repeatability.json", "w") as f:
            json.dump(data, f, indent=2)
        
        return data
