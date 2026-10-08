"""Test cases for mesh transform repeatability assessment."""

import pytest
import numpy as np
import pyvista as pv
from src.teer_cdss.mesh import MeshReconstructor
from src.teer_cdss.schemas import MeshConfig, MitraClipSpecification


def test_mesh_repeatability_assessment():
    """Test the mesh repeatability assessment functionality."""
    # Create a simple segmentation (2D array for testing)
    segmentation = np.zeros((10, 10, 10))
    segmentation[3:7, 3:7, 3:7] = 1  # Create a cube
    
    # Create mesh configuration
    config = MeshConfig(
        marching_cubes_isovalue=0.5,
        surface_smoothing_iterations=10,
        surface_decimation=0.0,
        tetrahedral_edge_length_mm=1.0
    )
    
    # Create clip specification
    clip_spec = MitraClipSpecification(
        grasp_length_mm=20.0,
        arm_width_mm=5.0,
        arm_thickness_mm=1.0
    )
    
    # Create mesh reconstructor
    reconstructor = MeshReconstructor(config, clip_spec)
    
    # Test basic assessment without reference mesh (should return None for invariance metrics)
    metrics = reconstructor.assess_repeatability(segmentation)
    
    assert "alignment_invariance" in metrics
    assert "feature_invariance" in metrics
    assert "mesh_volume_raw" in metrics
    assert "mesh_surface_area_raw" in metrics
    
    # With no reference mesh, invariance should be None
    assert metrics["alignment_invariance"] is None
    assert metrics["feature_invariance"] is None
    
    # Test with a simple reference mesh (a sphere)
    reference_mesh = pv.Sphere()
    
    # Test assessment with reference mesh
    metrics_with_ref = reconstructor.assess_repeatability(segmentation, reference_mesh)
    
    # Should have the same keys
    assert "alignment_invariance" in metrics_with_ref
    assert "feature_invariance" in metrics_with_ref
    assert "mesh_volume_raw" in metrics_with_ref
    assert "mesh_surface_area_raw" in metrics_with_ref
    
    # Should not be None for volume and area (they should have actual values)
    assert metrics_with_ref["mesh_volume_raw"] is not None
    assert metrics_with_ref["mesh_surface_area_raw"] is not None


def test_record_assessment():
    """Test recording assessment results to JSON file."""
    # Create a simple segmentation 
    segmentation = np.zeros((10, 10, 10))
    segmentation[3:7, 3:7, 3:7] = 1
    
    # Create mesh configuration
    config = MeshConfig(
        marching_cubes_isovalue=0.5,
        surface_smoothing_iterations=10,
        surface_decimation=0.0,
        tetrahedral_edge_length_mm=1.0
    )
    
    # Create clip specification
    clip_spec = MitraClipSpecification(
        grasp_length_mm=20.0,
        arm_width_mm=5.0,
        arm_thickness_mm=1.0
    )
    
    # Create mesh reconstructor
    reconstructor = MeshReconstructor(config, clip_spec)
    
    # Test recording assessment (should not fail even if file doesn't exist yet)
    result = reconstructor.record_assessment("test_mesh_001", segmentation)
    
    assert "version" in result
    assert "assessment_date" in result
    assert "meshes_assessed" in result
    assert len(result["meshes_assessed"]) == 1
    
    # Check that the recorded mesh has correct structure
    recorded_mesh = result["meshes_assessed"][0]
    assert recorded_mesh["name"] == "test_mesh_001"
    assert "metrics" in recorded_mesh


def test_edge_cases():
    """Test edge cases for repeatability assessment."""
    # Create a simple segmentation 
    segmentation = np.zeros((5, 5, 5))
    
    # Create mesh configuration
    config = MeshConfig(
        marching_cubes_isovalue=0.5,
        surface_smoothing_iterations=10,
        surface_decimation=0.0,
        tetrahedral_edge_length_mm=1.0
    )
    
    # Create clip specification
    clip_spec = MitraClipSpecification(
        grasp_length_mm=20.0,
        arm_width_mm=5.0,
        arm_thickness_mm=1.0
    )
    
    # Create mesh reconstructor
    reconstructor = MeshReconstructor(config, clip_spec)
    
    # Test with empty segmentation (should not crash)
    metrics = reconstructor.assess_repeatability(segmentation)
    
    assert isinstance(metrics, dict)
    assert "alignment_invariance" in metrics
    assert "feature_invariance" in metrics
    assert "mesh_volume_raw" in metrics
    assert "mesh_surface_area_raw" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v"])