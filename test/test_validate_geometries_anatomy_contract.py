import os
import sys
import pytest
from unittest.mock import Mock, patch

# Add the scripts directory to path to import validate_mesh
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from validate_geometries import validate_mesh

def test_anatomy_contract_metadata_validation():
    """Test that validate_mesh function properly handles anatomy contract metadata verification"""
    
    # Mock a valid STL file path
    mock_file_path = "/fake/path/test.stl"
    
    # Test with a basic mesh validation
    result = validate_mesh(mock_file_path)
    
    # Verify the basic structure is maintained
    assert "filename" in result
    assert "parse_success" in result
    assert "file_size" in result
    assert "status" in result
    
    # Check that new anatomy contract fields are included
    expected_fields = [
        "anatomy_contract_compliant",
        "coordinate_system_type",
        "coordinate_system_units",
        "observable_anatomy_found",
        "landmarks_found"
    ]
    
    for field in expected_fields:
        assert field in result, f"Missing anatomy contract field: {field}"
        
def test_anatomy_contract_validation_with_metadata():
    """Test that validate_mesh function validates metadata fields correctly"""
    
    # This would be a more comprehensive test if we had actual metadata to validate against
    # For now, just check that the function doesn't crash and returns expected structure
    
    mock_file_path = "/fake/path/test.stl"
    result = validate_mesh(mock_file_path)
    
    # Check default values for anatomy contract fields
    assert result["anatomy_contract_compliant"] is False  # Default to not compliant until validated
    assert result["coordinate_system_type"] is None
    assert result["coordinate_system_units"] is None
    assert result["observable_anatomy_found"] == []
    assert result["landmarks_found"] == []

if __name__ == "__main__":
    pytest.main([__file__, "-v"])