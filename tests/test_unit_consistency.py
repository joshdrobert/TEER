"""
Unit consistency validation tests for coordinate frames, mesh transformations, and feature computations.
"""

import numpy as np
import pytest
from unittest.mock import Mock

from src.teer_cdss.unit_validator import (
    UnitValidator,
    UnitConsistencyError,
    UnitSystem,
    Unit,
    validate_coordinate_frame_consistency,
    validate_mesh_transformation_consistency,
    validate_feature_computation_consistency
)


class TestUnitValidator:
    """Test cases for the UnitValidator class."""
    
    def test_coordinate_frame_consistency_valid(self):
        """Test valid coordinate frame consistency validation."""
        validator = UnitValidator()
        
        # Valid 3D points
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        
        assert validator.validate_coordinate_frame_consistency(
            points, "world", "local", UnitSystem.SI
        ) is True
        
    def test_coordinate_frame_consistency_invalid_shape(self):
        """Test coordinate frame consistency validation with invalid shape."""
        validator = UnitValidator()
        
        # Invalid shape - not 2D with 3 columns
        points = np.array([[0.0, 0.0], [1.0, 1.0]])  # 2x2 array
        
        with pytest.raises(UnitConsistencyError):
            validator.validate_coordinate_frame_consistency(
                points, "world", "local", UnitSystem.SI
            )
    
    def test_coordinate_frame_consistency_non_finite_values(self):
        """Test coordinate frame consistency validation with non-finite values."""
        validator = UnitValidator()
        
        # Points with NaN values
        points = np.array([[0.0, 0.0, 0.0], [np.nan, 1.0, 1.0], [2.0, 2.0, 2.0]])
        
        with pytest.raises(UnitConsistencyError):
            validator.validate_coordinate_frame_consistency(
                points, "world", "local", UnitSystem.SI
            )
    
    def test_mesh_transformation_consistency_valid(self):
        """Test valid mesh transformation consistency validation."""
        validator = UnitValidator()
        
        # Create mock meshes
        original_mesh = {
            'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            'faces': [[0, 1, 2]]
        }
        
        transformed_mesh = {
            'points': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],
            'faces': [[0, 1, 2]]
        }
        
        # Identity transformation matrix
        transform_matrix = np.eye(4)
        
        assert validator.validate_mesh_transformation_consistency(
            original_mesh, transformed_mesh, transform_matrix
        ) is True
    
    def test_mesh_transformation_consistency_invalid_points(self):
        """Test mesh transformation consistency with mismatched point counts."""
        validator = UnitValidator()
        
        # Create meshes with different numbers of points
        original_mesh = {
            'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],  # 2 points
            'faces': [[0, 1]]
        }
        
        transformed_mesh = {
            'points': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 3.0]],  # 3 points
            'faces': [[0, 1, 2]]
        }
        
        transform_matrix = np.eye(4)
        
        with pytest.raises(UnitConsistencyError):
            validator.validate_mesh_transformation_consistency(
                original_mesh, transformed_mesh, transform_matrix
            )
    
    def test_mesh_transformation_consistency_invalid_matrix(self):
        """Test mesh transformation consistency with invalid matrix shape."""
        validator = UnitValidator()
        
        original_mesh = {
            'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            'faces': [[0, 1]]
        }
        
        transformed_mesh = {
            'points': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            'faces': [[0, 1]]
        }
        
        # Invalid transformation matrix (3x3 instead of 4x4)
        transform_matrix = np.eye(3)
        
        with pytest.raises(UnitConsistencyError):
            validator.validate_mesh_transformation_consistency(
                original_mesh, transformed_mesh, transform_matrix
            )
    
    def test_feature_computation_consistency_valid(self):
        """Test valid feature computation consistency validation."""
        validator = UnitValidator()
        
        # Valid scalar feature value
        feature_value = 5.0
        
        expected_units = Unit(length=1.0)  # meters
        actual_units = Unit(length=1.0)    # meters
        
        assert validator.validate_feature_computation_consistency(
            "test_feature", feature_value, expected_units, actual_units
        ) is True
    
    def test_feature_computation_consistency_invalid_units(self):
        """Test feature computation consistency with inconsistent units."""
        validator = UnitValidator()
        
        feature_value = 5.0
        
        expected_units = Unit(length=1.0)   # meters
        actual_units = Unit(mass=1.0)       # kilograms (inconsistent)
        
        with pytest.raises(UnitConsistencyError):
            validator.validate_feature_computation_consistency(
                "test_feature", feature_value, expected_units, actual_units
            )
    
    def test_feature_computation_consistency_invalid_values(self):
        """Test feature computation consistency with invalid values."""
        validator = UnitValidator()
        
        # Feature value with NaN
        feature_value = np.nan
        
        expected_units = Unit(length=1.0)
        actual_units = Unit(length=1.0)
        
        with pytest.raises(UnitConsistencyError):
            validator.validate_feature_computation_consistency(
                "test_feature", feature_value, expected_units, actual_units
            )
    
    def test_cross_frame_operations_valid(self):
        """Test cross-frame operations validation with valid operations."""
        validator = UnitValidator()
        
        operations = {
            'transform_mesh': {
                'input_units': Unit(length=1.0),
                'output_units': Unit(length=1.0)
            },
            'compute_volume': {
                'input_units': Unit(length=1.0),
                'output_units': Unit(length=3.0)  # volume has length^3
            }
        }
        
        assert validator.validate_cross_frame_operations(operations) is True
    
    def test_cross_frame_operations_invalid(self):
        """Test cross-frame operations validation with invalid operations."""
        validator = UnitValidator()
        
        operations = {
            'transform_mesh': {
                'input_units': Unit(length=1.0),
                'output_units': Unit(mass=1.0)  # Inconsistent units
            }
        }
        
        with pytest.raises(UnitConsistencyError):
            validator.validate_cross_frame_operations(operations)


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_validate_coordinate_frame_consistency(self):
        """Test the convenience function for coordinate frame consistency."""
        points = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        
        assert validate_coordinate_frame_consistency(
            points, "world", "local", UnitSystem.SI
        ) is True
    
    def test_validate_mesh_transformation_consistency(self):
        """Test the convenience function for mesh transformation consistency."""
        original_mesh = {
            'points': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
            'faces': [[0, 1]]
        }
        
        transformed_mesh = {
            'points': [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
            'faces': [[0, 1]]
        }
        
        transform_matrix = np.eye(4)
        
        assert validate_mesh_transformation_consistency(
            original_mesh, transformed_mesh, transform_matrix
        ) is True
    
    def test_validate_feature_computation_consistency(self):
        """Test the convenience function for feature computation consistency."""
        feature_value = 5.0
        
        expected_units = Unit(length=1.0)
        actual_units = Unit(length=1.0)
        
        assert validate_feature_computation_consistency(
            "test_feature", feature_value, expected_units, actual_units
        ) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])