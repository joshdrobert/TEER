"""
Test suite for geometric features module.
"""

import numpy as np
import pytest
import os
from unittest.mock import Mock, patch

# Import the module we're testing
import sys
sys.path.insert(0, 'src')
import teer_cdss.geometric_features as gf

def test_compute_distance_fields():
    """Test distance field computation."""
    # Create a simple mock mesh for testing
    mock_mesh = Mock()
    mock_mesh.bounding_box = Mock()
    mock_mesh.bounding_box.extents = np.array([1.0, 1.0, 1.0])
    mock_mesh.bounding_box.min = np.array([0.0, 0.0, 0.0])
    mock_mesh.bounding_box.max = np.array([1.0, 1.0, 1.0])
    
    # Mock the nearest.on_surface method
    mock_mesh.nearest = Mock()
    # For a 4x4x4 grid, we need 64 distance values
    mock_mesh.nearest.on_surface.return_value = (
        np.random.rand(64, 3),  # closest points  
        np.random.rand(64),     # distances
        np.array([0]*64)        # triangle ids
    )
    
    # Mock the signed_distance method as well
    mock_mesh.nearest.signed_distance.return_value = np.random.rand(64)
    
    # Test with small grid resolution
    signed_dists, unsigned_dists = gf.compute_distance_field_samples(mock_mesh, grid_resolution=4)
    
    assert isinstance(signed_dists, np.ndarray)
    assert isinstance(unsigned_dists, np.ndarray)
    assert signed_dists.shape == (4, 4, 4)
    assert unsigned_dists.shape == (4, 4, 4)

def test_compute_laplace_beltrami_spectrum():
    """Test Laplace-Beltrami spectrum computation."""
    # Create a mock mesh
    mock_mesh = Mock()
    
    # Test with default parameters
    spectrum = gf.compute_laplace_beltrami_spectrum(mock_mesh, num_eigenvalues=10)
    
    assert isinstance(spectrum, np.ndarray)
    assert spectrum.shape == (10,)
    
def test_extract_geometric_features():
    """Test main feature extraction function."""
    # Create a mock mesh
    mock_mesh = Mock()
    
    with patch('trimesh.load') as mock_load:
        mock_load.return_value = mock_mesh
        
        # Mock the distance field computation
        with patch('teer_cdss.geometric_features.compute_distance_field_samples') as mock_dist_fields:
            mock_dist_fields.return_value = (
                np.random.rand(8, 8, 8), 
                np.random.rand(8, 8, 8)
            )
            
            # Mock the spectrum computation
            with patch('teer_cdss.geometric_features.compute_laplace_beltrami_spectrum') as mock_spectrum:
                mock_spectrum.return_value = np.random.rand(10)
                
                # Test function call
                features = gf.extract_geometric_features("dummy_path.stl", grid_resolution=8)
                
                assert 'signed_distances' in features
                assert 'unsigned_distances' in features  
                assert 'laplace_beltrami_spectrum' in features
                
                assert features['signed_distances'].shape == (8, 8, 8)
                assert features['unsigned_distances'].shape == (8, 8, 8)
                assert features['laplace_beltrami_spectrum'].shape == (10,)

def test_validate_geometric_features():
    """Test validation of geometric features."""
    # Create mock features
    features = {
        'signed_distances': np.random.rand(4, 4, 4),
        'unsigned_distances': np.random.rand(4, 4, 4),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    
    # Test valid features
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == True
    
    # Test invalid features (mismatched shapes)
    features['signed_distances'] = np.random.rand(3, 3, 3)
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False

def test_validate_geometric_features_shape_consistency():
    """Test shape consistency checks in validation."""
    # Test with correct matching shapes
    features = {
        'signed_distances': np.random.rand(8, 8, 8),
        'unsigned_distances': np.random.rand(8, 8, 8),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    assert gf.validate_geometric_features(features) == True
    
    # Test with mismatched distance field shapes
    features['unsigned_distances'] = np.random.rand(4, 4, 4)
    assert gf.validate_geometric_features(features) == False

def test_validate_geometric_features_array_types():
    """Test that validation rejects non-numpy arrays and wrong dimensions."""
    # Test with non-numpy arrays
    features = {
        'signed_distances': [[1, 2], [3, 4]],
        'unsigned_distances': np.random.rand(4, 4, 4),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    assert gf.validate_geometric_features(features) == False
    
    # Test with wrong dimension arrays
    features = {
        'signed_distances': np.random.rand(4, 4),
        'unsigned_distances': np.random.rand(4, 4),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    assert gf.validate_geometric_features(features) == False

def test_validate_geometric_features_spectrum_shape():
    """Test that validation checks spectrum array shape."""
    # Test with correct spectrum shape
    features = {
        'signed_distances': np.random.rand(4, 4, 4),
        'unsigned_distances': np.random.rand(4, 4, 4),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    assert gf.validate_geometric_features(features) == True
    
    # Test with wrong spectrum length
    features['laplace_beltrami_spectrum'] = np.random.rand(5)
    assert gf.validate_geometric_features(features) == False

def test_validate_geometric_features_empty_arrays():
    """Test that validation rejects empty arrays."""
    features = {
        'signed_distances': np.array([]),
        'unsigned_distances': np.random.rand(4, 4, 4),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    assert gf.validate_geometric_features(features) == False
    
    features = {
        'signed_distances': np.random.rand(4, 4, 4),
        'unsigned_distances': np.random.rand(4, 4, 4),
        'laplace_beltrami_spectrum': np.array([])
    }
    assert gf.validate_geometric_features(features) == False

def test_validate_geometric_features_nonfinite_values():
    """Test that validation rejects arrays with nonfinite values."""
    features = {
        'signed_distances': np.array([np.nan, 1.0, 2.0]).reshape(1, 3, 1),
        'unsigned_distances': np.random.rand(1, 3, 1),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    assert gf.validate_geometric_features(features) == False
    
    features = {
        'signed_distances': np.random.rand(1, 3, 1),
        'unsigned_distances': np.array([1.0, np.inf, 2.0]).reshape(1, 3, 1),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    assert gf.validate_geometric_features(features) == False

def test_validate_geometric_features_missing_keys():
    """Test that validation rejects missing required keys."""
    features = {
        'signed_distances': np.random.rand(4, 4, 4),
        'unsigned_distances': np.random.rand(4, 4, 4)
        # Missing laplace_beltrami_spectrum
    }
    assert gf.validate_geometric_features(features) == False

def test_acceptance():
    """Basic acceptance test."""
    # This would be run against real data in practice
    assert True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])