"""
Boundary and edge case tests for geometric features validation.
"""

import numpy as np
import pytest
import sys
sys.path.insert(0, 'src')
import teer_cdss.geometric_features as gf

def test_validate_empty_arrays():
    """Test that empty arrays are rejected."""
    # Test with empty distance arrays
    features = {
        'signed_distances': np.array([]),
        'unsigned_distances': np.array([]),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False
    
    # Test with empty spectrum
    features['signed_distances'] = np.random.rand(4, 4, 4)
    features['unsigned_distances'] = np.random.rand(4, 4, 4)
    features['laplace_beltrami_spectrum'] = np.array([])
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False

def test_validate_nonfinite_arrays():
    """Test that arrays with NaN or Inf values are rejected."""
    # Test with NaN in distance arrays
    features = {
        'signed_distances': np.array([np.nan]),
        'unsigned_distances': np.random.rand(4, 4, 4),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False
    
    # Test with Inf in distance arrays
    features['signed_distances'] = np.array([np.inf])
    features['unsigned_distances'] = np.random.rand(4, 4, 4)
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False
    
    # Test with NaN in spectrum
    features['signed_distances'] = np.random.rand(4, 4, 4)
    features['unsigned_distances'] = np.random.rand(4, 4, 4)
    features['laplace_beltrami_spectrum'] = np.array([np.nan])
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False

def test_validate_correct_shapes():
    """Test that valid three-dimensional arrays are accepted."""
    # Test with correct shapes
    features = {
        'signed_distances': np.random.rand(8, 8, 8),
        'unsigned_distances': np.random.rand(8, 8, 8),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == True

def test_validate_wrong_dimensions():
    """Test that wrong dimensional arrays are rejected."""
    # Test with 2D distance arrays
    features = {
        'signed_distances': np.random.rand(8, 8),
        'unsigned_distances': np.random.rand(8, 8),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False
    
    # Test with 1D spectrum
    features['signed_distances'] = np.random.rand(8, 8, 8)
    features['unsigned_distances'] = np.random.rand(8, 8, 8)
    features['laplace_beltrami_spectrum'] = np.random.rand(5)  # Wrong length
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False

def test_validate_missing_keys():
    """Test that missing keys are rejected."""
    # Test with missing key
    features = {
        'signed_distances': np.random.rand(8, 8, 8),
        'unsigned_distances': np.random.rand(8, 8, 8)
        # Missing laplace_beltrami_spectrum
    }
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False

def test_validate_non_numpy_arrays():
    """Test that non-numpy arrays are rejected."""
    # Test with list instead of numpy array
    features = {
        'signed_distances': [1, 2, 3],
        'unsigned_distances': np.random.rand(8, 8, 8),
        'laplace_beltrami_spectrum': np.random.rand(10)
    }
    
    is_valid = gf.validate_geometric_features(features)
    assert is_valid == False