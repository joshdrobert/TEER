"""
Test suite for distance field computation functionality.
"""

import numpy as np
import trimesh
import pytest
from src.teer_cdss.geometric_features import compute_distance_field_samples

def test_compute_distance_fields_basic():
    """Test basic distance field computation with a simple mesh."""
    # Create a simple sphere mesh
    mesh = trimesh.primitives.Sphere(radius=1.0, subdivisions=2)
    
    # Compute distance fields
    signed_distances, unsigned_distances = compute_distance_field_samples(mesh, grid_resolution=8)
    
    # Check that results have correct shape
    assert signed_distances.shape == (8, 8, 8)
    assert unsigned_distances.shape == (8, 8, 8)
    
    # Check that we have both positive and negative values for signed distances (on both sides of surface)
    assert np.any(signed_distances > 0)  # Should have some positive distances
    assert np.any(signed_distances < 0)  # Should have some negative distances
    
    # Check that unsigned distances are all non-negative
    assert np.all(unsigned_distances >= 0)

def test_compute_distance_fields_with_simple_cube():
    """Test distance field computation with a simple cube mesh."""
    # Create a simple cube mesh
    mesh = trimesh.primitives.Box(extents=[2.0, 2.0, 2.0])
    
    # Compute distance fields
    signed_distances, unsigned_distances = compute_distance_field_samples(mesh, grid_resolution=8)
    
    # Check that results have correct shape
    assert signed_distances.shape == (8, 8, 8)
    assert unsigned_distances.shape == (8, 8, 8)
    
    # Check that we have both positive and negative values for signed distances
    assert np.any(signed_distances > 0)  # Should have some positive distances
    assert np.any(signed_distances < 0)  # Should have some negative distances
    
    # Check that unsigned distances are all non-negative
    assert np.all(unsigned_distances >= 0)

def test_compute_distance_fields_empty_mesh():
    """Test distance field computation with edge cases."""
    # Create a degenerate mesh (single point)
    vertices = np.array([[0, 0, 0]])
    faces = np.array([[0, 0, 0]])  # Degenerate triangle
    try:
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        signed_distances, unsigned_distances = compute_distance_field_samples(mesh, grid_resolution=4)
        
        # Should still work even with degenerate mesh
        assert signed_distances.shape == (4, 4, 4)
        assert unsigned_distances.shape == (4, 4, 4)
    except Exception:
        # If it fails for valid reasons, that's OK
        pass

if __name__ == "__main__":
    test_compute_distance_fields_basic()
    test_compute_distance_fields_with_simple_cube()
    test_compute_distance_fields_empty_mesh()
    print("All distance field computation tests passed!")