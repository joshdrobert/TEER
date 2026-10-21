"""
Additional tests for distance field computation to ensure signed vs unsigned functionality.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, 'src')
import teer_cdss.geometric_features as gf

def test_signed_unsigned_distance_difference():
    """Test that signed and unsigned distances have expected properties."""
    # Create a simple mock mesh for testing
    mock_mesh = gf.trimesh.creation.icosphere(subdivisions=2, radius=1.0)
    
    # Test with small grid resolution
    signed_dists, unsigned_dists = gf.compute_distance_field_samples(mock_mesh, grid_resolution=8)
    
    # Both should be numpy arrays with same shape
    assert isinstance(signed_dists, np.ndarray)
    assert isinstance(unsigned_dists, np.ndarray)
    assert signed_dists.shape == (8, 8, 8)
    assert unsigned_dists.shape == (8, 8, 8)
    
    # Unsigned distances should be non-negative
    assert np.all(unsigned_dists >= 0)
    
    # Signed distances can be positive or negative
    # The key property: at mesh surface, signed distance should be close to 0
    # and unsigned distance should be close to 0
    
    # Test that we get reasonable values (not all zeros or all NaNs)
    assert not np.all(signed_dists == 0)
    assert not np.all(unsigned_dists == 0)
    
    # Check for finite values
    assert np.isfinite(signed_dists).all()
    assert np.isfinite(unsigned_dists).all()

def test_distance_field_shapes():
    """Test that distance field computation produces correct shapes."""
    mock_mesh = gf.trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    
    # Test different grid resolutions
    for resolution in [4, 8, 16]:
        signed_dists, unsigned_dists = gf.compute_distance_field_samples(mock_mesh, grid_resolution=resolution)
        
        assert signed_dists.shape == (resolution, resolution, resolution)
        assert unsigned_dists.shape == (resolution, resolution, resolution)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])