"""
Boundary tests for mesh transformation validation including:
- NaN/Inf values
- Singular transforms  
- Malformed bottom row
- Valid rotation/translation
- Compatible nonsingular scaling
"""

import numpy as np
import pytest
from src.teer_cdss.unit_validator import (
    UnitValidator,
    UnitConsistencyError
)


def test_mesh_transformation_consistency_nan_inf_values():
    """Test that NaN and Inf values in transformation matrix are rejected."""
    validator = UnitValidator()
    
    # Create valid meshes
    original_mesh = {
        'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        'faces': [[0, 1]]
    }
    
    transformed_mesh = {
        'points': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        'faces': [[0, 1]]
    }
    
    # Test with NaN in transformation matrix
    transform_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0], 
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, np.nan]  # Invalid bottom row
    ])
    
    with pytest.raises(UnitConsistencyError):
        validator.validate_mesh_transformation_consistency(
            original_mesh, transformed_mesh, transform_matrix
        )
        
    # Test with Inf in transformation matrix
    transform_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, np.inf]  # Invalid bottom row
    ])
    
    with pytest.raises(UnitConsistencyError):
        validator.validate_mesh_transformation_consistency(
            original_mesh, transformed_mesh, transform_matrix
        )


def test_mesh_transformation_consistency_singular_matrix():
    """Test that singular (non-invertible) transformation matrices are rejected."""
    validator = UnitValidator()
    
    # Create valid meshes
    original_mesh = {
        'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        'faces': [[0, 1]]
    }
    
    transformed_mesh = {
        'points': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        'faces': [[0, 1]]
    }
    
    # Create a singular transformation matrix (determinant = 0)
    transform_matrix = np.array([
        [0.0, 0.0, 0.0, 0.0],  # Singular row
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])
    
    with pytest.raises(UnitConsistencyError):
        validator.validate_mesh_transformation_consistency(
            original_mesh, transformed_mesh, transform_matrix
        )


def test_mesh_transformation_consistency_malformed_bottom_row():
    """Test that malformed bottom rows in transformation matrices are rejected."""
    validator = UnitValidator()
    
    # Create valid meshes
    original_mesh = {
        'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        'faces': [[0, 1]]
    }
    
    transformed_mesh = {
        'points': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        'faces': [[0, 1]]
    }
    
    # Test with malformed bottom row (not [0,0,0,1])
    transform_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 1.0, 1.0]  # Should be [0,0,0,1] for affine transform
    ])
    
    with pytest.raises(UnitConsistencyError):
        validator.validate_mesh_transformation_consistency(
            original_mesh, transformed_mesh, transform_matrix
        )


def test_mesh_transformation_consistency_valid_rotation_translation():
    """Test that valid rotation and translation transforms are accepted."""
    validator = UnitValidator()
    
    # Create valid meshes
    original_mesh = {
        'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        'faces': [[0, 1]]
    }
    
    transformed_mesh = {
        'points': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        'faces': [[0, 1]]
    }
    
    # Create a valid rotation/translation matrix
    transform_matrix = np.array([
        [1.0, 0.0, 0.0, 1.0],  # Translation
        [0.0, 1.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 1.0],
        [0.0, 0.0, 0.0, 1.0]   # Proper bottom row for affine transform
    ])
    
    assert validator.validate_mesh_transformation_consistency(
        original_mesh, transformed_mesh, transform_matrix
    ) is True


def test_mesh_transformation_consistency_valid_scaling():
    """Test that valid compatible scaling transforms are accepted."""
    validator = UnitValidator()
    
    # Create valid meshes
    original_mesh = {
        'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        'faces': [[0, 1]]
    }
    
    transformed_mesh = {
        'points': [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]],
        'faces': [[0, 1]]
    }
    
    # Create a valid scaling matrix (uniform scale)
    transform_matrix = np.array([
        [2.0, 0.0, 0.0, 0.0],  # Scale by 2
        [0.0, 2.0, 0.0, 0.0],
        [0.0, 0.0, 2.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]   # Proper bottom row for affine transform
    ])
    
    assert validator.validate_mesh_transformation_consistency(
        original_mesh, transformed_mesh, transform_matrix
    ) is True


def test_mesh_transformation_consistency_non_affine():
    """Test that non-affine transformation matrices are rejected."""
    validator = UnitValidator()
    
    # Create valid meshes
    original_mesh = {
        'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        'faces': [[0, 1]]
    }
    
    transformed_mesh = {
        'points': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        'faces': [[0, 1]]
    }
    
    # Create a non-affine matrix (bottom row not [0,0,0,1])
    transform_matrix = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0]   # Invalid - should be [0,0,0,1]
    ])
    
    with pytest.raises(UnitConsistencyError):
        validator.validate_mesh_transformation_consistency(
            original_mesh, transformed_mesh, transform_matrix
        )