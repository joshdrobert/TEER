#!/usr/bin/env python3

from src.teer_cdss.unit_validator import UnitValidator, Unit

# Test current behavior
validator = UnitValidator()

# Create test units as per the failing test
length_unit = Unit(length=1.0)
area_unit = Unit(length=2.0)

print("Length unit:", length_unit)
print("Area unit:", area_unit)

result = validator._are_units_dimensionally_consistent(length_unit, area_unit)
print("Consistency result:", result)

# What should happen:
# - Length unit: L (length=1.0)  
# - Area unit: L^2 (length=2.0)
# These should be inconsistent because they represent different dimensional powers

# But actually looking at the Unit class, I think the design is that:
# Each field represents the power of that dimension
# So length=1.0 means L^1 = L 
# And length=2.0 means L^2 = L^2

# This would be consistent with the test expectation that they should be detected as inconsistent.
# The problem is in my implementation - let me check what's happening:
print("\nDebug:")
print("length_unit.length =", length_unit.length)
print("area_unit.length =", area_unit.length)
print("Are they equal?", length_unit.length == area_unit.length)