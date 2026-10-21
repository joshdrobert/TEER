import numpy as np
import pytest
import os
from unittest.mock import Mock, patch

# Add the src directory to the path so we can import our module
import sys
sys.path.insert(0, 'src')

from teer_cdss.geometric_features import (
    compute_distance_field_samples,
    compute_laplace_beltrami_spectrum,
    extract_geometric_features,
    validate_geometric_features
)

def test_compute_distance_field_samples():
    """Test distance field computation with a simple mesh"""
    # Create a simple mock mesh
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    
    # Test with default resolution
    signed_dists, unsigned_dists = compute_distance_field_samples(mesh, grid_resolution=8)
    
    # Check that results are numpy arrays with correct shape
    assert isinstance(signed_dists, np.ndarray)
    assert isinstance(unsigned_dists, np.ndarray)
    assert signed_dists.shape == (8, 8, 8)
    assert unsigned_dists.shape == (8, 8, 8)
    
    # Check that values are finite
    assert np.all(np.isfinite(signed_dists))
    assert np.all(np.isfinite(unsigned_dists))

def test_compute_laplace_beltrami_spectrum():
    """Test Laplace-Beltrami spectrum computation"""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    
    # Test with default parameters
    spectrum = compute_laplace_beltrami_spectrum(mesh, num_eigenvalues=10)
    
    # Check that result is numpy array with correct shape
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.shape == (10,)
    
    # Check that values are finite
    assert np.all(np.isfinite(spectrum))

def test_extract_geometric_features():
    """Test main extraction function"""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    
    # Test with a mock mesh (since we can't load real STL in test)
    with patch('trimesh.load') as mock_load:
        mock_load.return_value = mesh
        
        features = extract_geometric_features("dummy_path.stl", grid_resolution=8)
        
        # Check that all required keys are present
        assert 'signed_distances' in features
        assert 'unsigned_distances' in features
        assert 'laplace_beltrami_spectrum' in features
        
        # Check that shapes are correct
        assert features['signed_distances'].shape == (8, 8, 8)
        assert features['unsigned_distances'].shape == (8, 8, 8)
        assert features['laplace_beltrami_spectrum'].shape == (10,)

def test_validate_geometric_features():
    """Test feature validation function"""
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=1, radius=1.0)
    
    with patch('trimesh.load') as mock_load:
        mock_load.return_value = mesh
        
        features = extract_geometric_features("dummy_path.stl", grid_resolution=8)
        
        # Test valid features
        assert validate_geometric_features(features) == True
        
        # Test invalid features (missing key)
        invalid_features = features.copy()
        del invalid_features['signed_distances']
        assert validate_geometric_features(invalid_features) == False
        
        # Test invalid spectrum shape
        invalid_features = features.copy()
        invalid_features['laplace_beltrami_spectrum'] = np.array([1, 2, 3])
        assert validate_geometric_features(invalid_features) == False

if __name__ == "__main__":
    pytest.main([__file__, "-v"])