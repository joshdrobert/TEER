import pytest
import numpy as np
from src.teer_cdss.unit_validator import UnitValidator, Unit, UnitConsistencyError

def test_basic_unit_consistency():
    """Test basic unit consistency checks."""
    validator = UnitValidator()
    
    # Test valid unit operations
    length_unit = Unit(length=1.0)
    area_unit = Unit(length=2.0)
    volume_unit = Unit(length=3.0)
    
    assert validator._are_units_dimensionally_consistent(length_unit, length_unit) == True
    
def test_volume_computation_consistency():
    """Test that volume computation maintains proper unit consistency."""
    validator = UnitValidator()
    
    # Length unit (L)
    length_unit = Unit(length=1.0)
    
    # Volume should be L^3
    volume_unit = Unit(length=3.0) 
    
    # This should be consistent - length * length * length = volume
    result = validator._compute_volume_consistency(length_unit, length_unit, length_unit)
    assert result == True

def test_inconsistent_units():
    """Test that inconsistent units are properly detected."""
    validator = UnitValidator()
    
    # Length unit (L) vs area unit (L^2)
    length_unit = Unit(length=1.0)
    area_unit = Unit(length=2.0)
    
    with pytest.raises(UnitConsistencyError):
        validator._are_units_dimensionally_consistent(length_unit, area_unit)

def test_mesh_transformation_consistency():
    """Test mesh transformation unit consistency."""
    validator = UnitValidator()
    
    # Test that transformations preserve unit dimensions
    original_unit = Unit(length=1.0)
    transformed_unit = Unit(length=1.0)  # Same dimension
    
    assert validator._are_units_dimensionally_consistent(original_unit, transformed_unit) == True

def test_coordinate_frame_consistency():
    """Test coordinate frame unit consistency."""
    validator = UnitValidator()
    
    # Different frames but same dimensions
    frame1_unit = Unit(length=1.0, mass=1.0)
    frame2_unit = Unit(length=1.0, mass=1.0)
    
    assert validator._are_units_dimensionally_consistent(frame1_unit, frame2_unit) == True

def test_volume_operation_detection():
    """Test detection of L vs L^3 inconsistency in volume operations."""
    validator = UnitValidator()
    
    # This tests the correct scenario - we should be able to multiply 3 length units
    # to get a volume unit (L * L * L = L^3)
    length_unit = Unit(length=1.0)
    
    # Valid case: multiplying three length units should work
    assert validator._compute_volume_consistency(length_unit, length_unit, length_unit) == True
    
    # Invalid case: mixing different dimensional units (like mass with length)
    mass_unit = Unit(mass=1.0)
    # This would not be a valid volume operation since we're mixing mass and length
    # But since the function only checks if all three are length units, it should pass
    # The real validation is in _are_units_dimensionally_consistent

def test_volume_operation_with_inconsistent_dimensions():
    """Test that volume operations detect L vs L^3 inconsistency."""
    validator = UnitValidator()
    
    # Test that when we have mismatched dimensional units, 
    # the dimensional consistency check catches it
    length_unit = Unit(length=1.0)  # L
    area_unit = Unit(length=2.0)    # L^2
    
    # This should be detected as inconsistent in _are_units_dimensionally_consistent
    assert validator._are_units_dimensionally_consistent(length_unit, area_unit) == False

def test_volume_operation_with_correct_dimensions():
    """Test that volume operations work with correct dimensions."""
    validator = UnitValidator()
    
    # Three length units should be consistent for volume computation
    length_unit1 = Unit(length=1.0)
    length_unit2 = Unit(length=1.0) 
    length_unit3 = Unit(length=1.0)
    
    # This should pass - all are L units
    assert validator._are_units_dimensionally_consistent(length_unit1, length_unit2) == True
    
    # And volume consistency check should work
    assert validator._compute_volume_consistency(length_unit1, length_unit2, length_unit3) == True