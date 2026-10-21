import yaml
import os
from pathlib import Path

def test_anatomy_contract_exists():
    """Test that anatomy contract file exists"""
    contract_path = Path("research/anatomy_contract.yaml")
    assert contract_path.exists(), "anatomy_contract.yaml file must exist"

def test_anatomy_contract_structure():
    """Test that anatomy contract has required structure"""
    contract_path = Path("research/anatomy_contract.yaml")
    with open(contract_path, 'r') as f:
        contract = yaml.safe_load(f)
    
    # Check required sections exist
    assert 'coordinate_system' in contract
    assert 'frame' in contract  
    assert 'mesh' in contract
    assert 'observable_anatomy' in contract
    assert 'landmarks' in contract
    assert 'data_source' in contract
    
    # Check coordinate system has required fields
    coord_sys = contract['coordinate_system']
    assert 'type' in coord_sys
    assert 'units' in coord_sys
    assert 'origin' in coord_sys
    
    # Check data source has required fields
    data_src = contract['data_source']
    assert 'metadata_file' in data_src
    assert 'sample_count' in data_src
    assert 'sample_ids' in data_src

def test_anatomy_contract_observable_anatomy():
    """Test that observable anatomy sections are properly defined"""
    contract_path = Path("research/anatomy_contract.yaml")
    with open(contract_path, 'r') as f:
        contract = yaml.safe_load(f)
    
    observable = contract['observable_anatomy']
    assert len(observable) >= 2, "Should have at least 2 observable anatomy items"
    
    # Check commisure and opening points are defined
    commisure_found = False
    opening_found = False
    
    for item in observable:
        if item['name'] == 'valve_comm':
            commisure_found = True
        elif item['name'] == 'valve_open':
            opening_found = True
    
    assert commisure_found, "valve_comm observable anatomy must be defined"
    assert opening_found, "valve_open observable anatomy must be defined"

def test_anatomy_contract_landmarks():
    """Test that landmarks are properly defined"""
    contract_path = Path("research/anatomy_contract.yaml")
    with open(contract_path, 'r') as f:
        contract = yaml.safe_load(f)
    
    landmarks = contract['landmarks']
    assert len(landmarks) >= 4, "Should have at least 4 landmark definitions"
    
    # Check required landmarks exist
    landmark_names = [lm['name'] for lm in landmarks]
    required_landmarks = ['center_x', 'center_y', 'angle_rad', 'angle_deg']
    
    for req_lm in required_landmarks:
        assert req_lm in landmark_names, f"Required landmark {req_lm} not found"

if __name__ == "__main__":
    test_anatomy_contract_exists()
    test_anatomy_contract_structure()
    test_anatomy_contract_observable_anatomy()
    test_anatomy_contract_landmarks()
    print("All tests passed!")