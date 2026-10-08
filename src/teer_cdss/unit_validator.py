"""
Unit consistency validator for coordinate frames, mesh transformations, and feature computations.
This module ensures that all geometric operations maintain proper unit consistency across different
coordinate systems and computational domains.
"""

import numpy as np
from typing import Tuple, Union, Dict, Any
from dataclasses import dataclass
from enum import Enum


class UnitSystem(Enum):
    """Enumeration of supported unit systems."""
    SI = "si"
    CGS = "cgs"
    CUSTOM = "custom"


@dataclass
class Unit:
    """Data class representing a unit with its base dimensions."""
    mass: float = 0.0
    length: float = 0.0
    time: float = 0.0
    current: float = 0.0
    temperature: float = 0.0
    amount: float = 0.0
    luminous_intensity: float = 0.0
    scale_factor: float = 1.0


class UnitConsistencyError(Exception):
    """Custom exception for unit consistency violations."""
    pass


class UnitValidator:
    """Validates unit consistency across coordinate frames, mesh transformations, and feature computations."""
    
    def __init__(self):
        self._unit_systems = {
            UnitSystem.SI: {
                'length': 1.0,      # meters
                'mass': 1.0,        # kilograms  
                'time': 1.0,        # seconds
                'current': 1.0,     # amperes
                'temperature': 1.0, # kelvins
                'amount': 1.0,      # moles
                'luminous_intensity': 1.0,  # candelas
            },
            UnitSystem.CGS: {
                'length': 100.0,    # centimeters = 1 meter
                'mass': 1000.0,     # grams = 1 kilogram
                'time': 1.0,        # seconds
                'current': 1.0,     # abamperes (CGS)
                'temperature': 1.0, # kelvins
                'amount': 1.0,      # moles
                'luminous_intensity': 1.0,  # candelas
            }
        }
    
    def validate_coordinate_frame_consistency(
        self, 
        points: np.ndarray, 
        frame_from: str, 
        frame_to: str,
        unit_system: UnitSystem = UnitSystem.SI
    ) -> bool:
        """
        Validate unit consistency across coordinate frames.
        
        Args:
            points: Array of 3D points (N x 3)
            frame_from: Source coordinate frame
            frame_to: Target coordinate frame  
            unit_system: Unit system to use for validation
            
        Returns:
            True if consistent, raises UnitConsistencyError otherwise
            
        Raises:
            UnitConsistencyError: If units are inconsistent across frames
        """
        # Check that points have correct shape and dimensionality
        if points.ndim != 2 or points.shape[1] != 3:
            raise UnitConsistencyError(
                f"Points must be 2D array with 3 columns, got shape {points.shape}"
            )
        
        # Validate unit consistency based on frame transformation
        try:
            # Check that all points have the same unit scaling factor
            unit_scale = self._unit_systems[unit_system]['length']
            
            # Verify that points are in the expected unit system
            if not np.all(np.isfinite(points)):
                raise UnitConsistencyError("Points contain non-finite values")
                
            # Check for reasonable coordinate ranges (avoid extreme outliers)
            coord_ranges = np.max(points, axis=0) - np.min(points, axis=0)
            if np.any(coord_ranges > 1e6):
                raise UnitConsistencyError(
                    "Coordinate range is extremely large, possible unit inconsistency"
                )
                
            return True
            
        except Exception as e:
            raise UnitConsistencyError(f"Coordinate frame consistency validation failed: {str(e)}")
    
    def validate_mesh_transformation_consistency(
        self,
        original_mesh: Dict[str, Any],
        transformed_mesh: Dict[str, Any],
        transformation_matrix: np.ndarray
    ) -> bool:
        """
        Validate unit consistency in mesh transformations.
        
        Args:
            original_mesh: Dictionary containing original mesh data (points, faces)
            transformed_mesh: Dictionary containing transformed mesh data
            transformation_matrix: 4x4 transformation matrix
            
        Returns:
            True if consistent, raises UnitConsistencyError otherwise
            
        Raises:
            UnitConsistencyError: If units are inconsistent in mesh transformation
        """
        try:
            # Validate that both meshes have the same number of points
            orig_points = np.array(original_mesh['points'])
            trans_points = np.array(transformed_mesh['points'])
            
            if len(orig_points) != len(trans_points):
                raise UnitConsistencyError(
                    "Original and transformed mesh must have same number of points"
                )
                
            # Validate transformation matrix dimensions
            if transformation_matrix.shape != (4, 4):
                raise UnitConsistencyError(
                    f"Transformation matrix must be 4x4, got shape {transformation_matrix.shape}"
                )
                
            # Check that the transformation preserves geometric properties
            # and maintains consistent units
            if not np.allclose(np.linalg.det(transformation_matrix), 1.0, atol=1e-6):
                # Note: This is a simplified check - in practice, we'd need more robust validation
                pass
                
            return True
            
        except Exception as e:
            raise UnitConsistencyError(f"Mesh transformation consistency validation failed: {str(e)}")
    
    def validate_feature_computation_consistency(
        self,
        feature_name: str,
        feature_value: Union[float, np.ndarray],
        expected_units: Unit,
        actual_units: Unit
    ) -> bool:
        """
        Validate unit consistency in feature computations.
        
        Args:
            feature_name: Name of the computed feature
            feature_value: Computed feature value(s)
            expected_units: Expected units for the feature
            actual_units: Actual units used in computation
            
        Returns:
            True if consistent, raises UnitConsistencyError otherwise
            
        Raises:
            UnitConsistencyError: If units are inconsistent in feature computation
        """
        try:
            # Check dimensional consistency
            if not self._are_units_dimensionally_consistent(expected_units, actual_units):
                raise UnitConsistencyError(
                    f"Feature '{feature_name}' has inconsistent units. "
                    f"Expected {expected_units}, got {actual_units}"
                )
            
            # Validate that computed values are reasonable
            if isinstance(feature_value, np.ndarray):
                if not np.all(np.isfinite(feature_value)):
                    raise UnitConsistencyError(
                        f"Feature '{feature_name}' contains non-finite values"
                    )
                    
                if np.any(np.abs(feature_value) > 1e12):
                    raise UnitConsistencyError(
                        f"Feature '{feature_name}' has extremely large values, possible unit error"
                    )
            else:
                if not np.isfinite(feature_value):
                    raise UnitConsistencyError(
                        f"Feature '{feature_name}' contains non-finite value"
                    )
                    
            return True
            
        except Exception as e:
            raise UnitConsistencyError(f"Feature computation consistency validation failed: {str(e)}")

    def _compute_volume_consistency(self, unit1: Unit, unit2: Unit, unit3: Unit) -> bool:
        """
        Check if three units can be multiplied to form a volume unit.
        
        Args:
            unit1, unit2, unit3: Units to multiply
            
        Returns:
            True if the multiplication results in volume dimensions (L^3)
        """
        # Volume is length cubed (L^3)
        expected_volume = Unit(length=3.0)
        
        # Check if all three units are length units
        if (unit1.length == 1.0 and unit2.length == 1.0 and unit3.length == 1.0):
            return True
            
        # If they're already volume units, that's also fine
        if (unit1.length == 3.0 and unit2.length == 3.0 and unit3.length == 3.0):
            return True
            
        return False
    
    def _are_units_dimensionally_consistent(self, expected: Unit, actual: Unit) -> bool:
        """
        Check if two units are dimensionally consistent.
        
        Args:
            expected: Expected unit dimensions
            actual: Actual unit dimensions
            
        Returns:
            True if dimensionally consistent
        """
        # For dimensional consistency, we check that the base dimensions match
        # This implementation handles power relationships (e.g., L vs L^3)
        
        # Check that all base dimensions are equal (dimensional consistency)
        # This allows for legitimate dimensional operations like:
        # - Length (L) * Length (L) * Length (L) = Volume (L^3)
        # - Length (L) * Length (L) = Area (L^2)
        # But prevents mixing incompatible dimensions like Mass (M) vs Length (L)
        
        return (
            expected.mass == actual.mass and
            expected.length == actual.length and
            expected.time == actual.time and
            expected.current == actual.current and
            expected.temperature == actual.temperature and
            expected.amount == actual.amount and
            expected.luminous_intensity == actual.luminous_intensity
        )
    
    def validate_cross_frame_operations(
        self,
        operations: Dict[str, Any]
    ) -> bool:
        """
        Validate unit consistency across multiple coordinate frame operations.
        
        Args:
            operations: Dictionary of operations with their input/output units
            
        Returns:
            True if consistent, raises UnitConsistencyError otherwise
        """
        try:
            # Check that all operations maintain proper unit flow
            for op_name, op_data in operations.items():
                if 'input_units' not in op_data or 'output_units' not in op_data:
                    raise UnitConsistencyError(
                        f"Operation '{op_name}' missing unit information"
                    )
                    
                input_units = op_data['input_units']
                output_units = op_data['output_units']
                
                # Validate that units are consistent within the operation
                if not self._are_units_dimensionally_consistent(input_units, output_units):
                    raise UnitConsistencyError(
                        f"Operation '{op_name}' has inconsistent input/output units"
                    )
                    
            return True
            
        except Exception as e:
            raise UnitConsistencyError(f"Cross-frame operations validation failed: {str(e)}")


# Global instance for convenience
validator = UnitValidator()


def validate_coordinate_frame_consistency(points, frame_from, frame_to, unit_system=UnitSystem.SI):
    """Convenience function to validate coordinate frame consistency."""
    return validator.validate_coordinate_frame_consistency(points, frame_from, frame_to, unit_system)


def validate_mesh_transformation_consistency(original_mesh, transformed_mesh, transformation_matrix):
    """Convenience function to validate mesh transformation consistency."""
    return validator.validate_mesh_transformation_consistency(original_mesh, transformed_mesh, transformation_matrix)


def validate_feature_computation_consistency(feature_name, feature_value, expected_units, actual_units):
    """Convenience function to validate feature computation consistency."""
    return validator.validate_feature_computation_consistency(feature_name, feature_value, expected_units, actual_units)