import pytest
from src.teer_cdss.unit_validator import Unit, UnitValidator, UnitConsistencyError

def test_equal_base_names_do_not_allow_different_exponents():
    validator=UnitValidator()
    assert not validator._are_units_dimensionally_consistent(Unit(length=1),Unit(length=2))
    with pytest.raises(UnitConsistencyError):
        validator.validate_cross_frame_operations({'transform_mesh':{'input_units':Unit(length=1),'output_units':Unit(length=3)}})

def test_volume_preserves_all_dimensional_exponents():
    validator=UnitValidator()
    assert validator._compute_volume_consistency(Unit(length=1),Unit(length=1),Unit(length=1))
    assert not validator._compute_volume_consistency(Unit(length=3),Unit(length=3),Unit(length=3))
    assert not validator._compute_volume_consistency(Unit(length=1,mass=1),Unit(length=1),Unit(length=1))
