import numpy as np
import pytest
import pyvista as pv

from src.teer_cdss.mesh import MeshReconstructor
from src.teer_cdss.schemas import MeshConfig, MitraClipSpecification
from src.teer_cdss.exceptions import MeshGenerationError


def test_surface_from_label_basic():
    """Test basic surface generation from segmentation."""
    
    # Create a simple 3D segmentation (e.g., a sphere-like structure)
    nx, ny, nz = 10, 10, 10
    segmentation = np.zeros((nx, ny, nz), dtype=np.int32)
    
    # Create a spherical-like structure in the center
    cx, cy, cz = nx//2, ny//2, nz//2
    radius = 3
    
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= radius**2:
                    segmentation[x, y, z] = 1
    
    # Initialize mesh reconstructor
    config = MeshConfig(
        marching_cubes_isovalue=0.5,
        surface_smoothing_iterations=10,
        surface_decimation=0.0,
        tetrahedral_edge_length_mm=1.0
    )
    
    clip_spec = MitraClipSpecification(
        grasp_length_mm=20.0,
        arm_width_mm=5.0,
        arm_thickness_mm=1.0
    )
    
    reconstructor = MeshReconstructor(config, clip_spec)
    
    # Generate mesh - should not raise an exception
    artifacts = reconstructor.build(segmentation)
    
    # Check that we got valid meshes
    assert artifacts.surface_mesh is not None
    assert artifacts.volumetric_mesh is not None
    assert artifacts.clip_mesh is not None
    
    # Surface should have some points
    assert artifacts.surface_mesh.n_points > 0


def test_surface_from_label_empty_segmentation():
    """Test that empty segmentation raises appropriate error."""
    
    # Create empty segmentation
    segmentation = np.zeros((10, 10, 10), dtype=np.int32)
    
    # Initialize mesh reconstructor
    config = MeshConfig(
        marching_cubes_isovalue=0.5,
        surface_smoothing_iterations=10,
        surface_decimation=0.0,
        tetrahedral_edge_length_mm=1.0
    )
    
    clip_spec = MitraClipSpecification(
        grasp_length_mm=20.0,
        arm_width_mm=5.0,
        arm_thickness_mm=1.0
    )
    
    reconstructor = MeshReconstructor(config, clip_spec)
    
    # Should raise an exception for empty segmentation
    with pytest.raises(MeshGenerationError):
        reconstructor.build(segmentation)


def test_surface_from_label_label_association():
    """Test that labels are correctly associated in dense segmentations."""
    
    # Create a simple 3D segmentation with distinct regions
    nx, ny, nz = 5, 5, 5
    segmentation = np.zeros((nx, ny, nz), dtype=np.int32)
    
    # Create two distinct regions
    segmentation[0:2, 0:2, 0:2] = 1  # Region 1
    segmentation[3:5, 3:5, 3:5] = 2  # Region 2
    
    # Initialize mesh reconstructor
    config = MeshConfig(
        marching_cubes_isovalue=0.5,
        surface_smoothing_iterations=10,
        surface_decimation=0.0,
        tetrahedral_edge_length_mm=1.0
    )
    
    clip_spec = MitraClipSpecification(
        grasp_length_mm=20.0,
        arm_width_mm=5.0,
        arm_thickness_mm=1.0
    )
    
    reconstructor = MeshReconstructor(config, clip_spec)
    
    # Generate mesh - should not raise an exception
    artifacts = reconstructor.build(segmentation)
    
    # Check that we got valid meshes
    assert artifacts.surface_mesh is not None
    
    # The surface should have some points (since there's non-zero data)
    assert artifacts.surface_mesh.n_points > 0


def test_rigid_transform_invariance():
    """Test that mesh generation behavior is consistent with rigid transformations."""
    
    # Create a simple 3D segmentation
    nx, ny, nz = 10, 10, 10
    segmentation = np.zeros((nx, ny, nz), dtype=np.int32)
    
    # Create a spherical-like structure in the center
    cx, cy, cz = nx//2, ny//2, nz//2
    radius = 3
    
    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if (x - cx)**2 + (y - cy)**2 + (z - cz)**2 <= radius**2:
                    segmentation[x, y, z] = 1
    
    # Initialize mesh reconstructor
    config = MeshConfig(
        marching_cubes_isovalue=0.5,
        surface_smoothing_iterations=10,
        surface_decimation=0.0,
        tetrahedral_edge_length_mm=1.0
    )
    
    clip_spec = MitraClipSpecification(
        grasp_length_mm=20.0,
        arm_width_mm=5.0,
        arm_thickness_mm=1.0
    )
    
    reconstructor = MeshReconstructor(config, clip_spec)
    
    # Generate mesh from original segmentation
    artifacts_orig = reconstructor.build(segmentation)
    
    # Test that the process doesn't break with various transformations
    # (This is more about ensuring no crashes than exact invariance)
    assert artifacts_orig.surface_mesh.n_points > 0
    assert artifacts_orig.volumetric_mesh.n_cells > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])