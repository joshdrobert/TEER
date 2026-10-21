import json
import pytest
from pathlib import Path

# Test to validate anatomy contract against valves_metadata.json geometry metadata fields and coordinate system definitions

def test_coordinate_system_definition():
    """Test that the coordinate system definition in anatomy contract matches expected right-handed system"""
    # Read the anatomy contract
    contract_path = Path("research/anatomy_contract.yaml")
    assert contract_path.exists(), "Anatomy contract file not found"
    
    # Basic validation of coordinate system definition
    with open(contract_path, 'r') as f:
        content = f.read()
        
    # Check that it contains the expected right-handed coordinate system definition
    assert "right-handed" in content
    assert "millimeters (mm)" in content
    assert "center of valve structure" in content

def test_mesh_provenance():
    """Test mesh provenance information in anatomy contract"""
    contract_path = Path("research/anatomy_contract.yaml")
    with open(contract_path, 'r') as f:
        content = f.read()
        
    # Check mesh provenance details
    assert "triangular mesh" in content
    assert "high-resolution" in content
    assert "medical imaging data" in content
    assert "noise reduction" in content
    assert "manifold, watertight" in content

def test_observable_anatomy_fields():
    """Test that observable anatomy fields match metadata structure"""
    contract_path = Path("research/anatomy_contract.yaml")
    with open(contract_path, 'r') as f:
        content = f.read()
        
    # Check that the expected anatomy structures are defined
    assert "valve_comm" in content
    assert "valve_open" in content
    assert "commisure" in content
    assert "opening point" in content

def test_metadata_fields_match_contract():
    """Test that metadata fields match contract definitions"""
    # Read the anatomy contract
    contract_path = Path("research/anatomy_contract.yaml")
    with open(contract_path, 'r') as f:
        contract_content = f.read()
        
    # Read the metadata file
    metadata_path = Path("valves_metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
        
    # Get first sample to test against
    sample_id = list(metadata.keys())[0]
    sample_data = metadata[sample_id]
    
    # Validate expected fields are present in metadata
    expected_fields = ['cx', 'cy', 'angle_rad', 'angle_deg', 'v_comm', 'v_open']
    for field in expected_fields:
        assert field in sample_data, f"Missing field {field} in metadata"
        
    # Validate coordinate system information
    assert "center_x" in contract_content or "center_y" in contract_content
    assert "millimeters" in contract_content

def test_coordinate_system_consistency():
    """Test that the coordinate system definitions are consistent"""
    # Read the anatomy contract
    contract_path = Path("research/anatomy_contract.yaml")
    with open(contract_path, 'r') as f:
        content = f.read()
        
    # Validate axis definitions
    assert "horizontal axis pointing from left to right" in content
    assert "vertical axis pointing from bottom to top" in content
    assert "depth axis pointing from front to back" in content

# Cache cleanup hook for clean pytest runs
def pytest_sessionstart(session):
    """Clean up Python cache files before pytest run to avoid import conflicts"""
    import shutil
    import sys
    import os
    
    # Find and remove all __pycache__ directories
    for root, dirs, files in os.walk('.'):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                cache_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(cache_path)
                    print(f"Cleaned cache: {cache_path}")
                except Exception as e:
                    print(f"Warning: Could not clean {cache_path}: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])