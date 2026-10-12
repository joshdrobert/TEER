import pytest
import yaml
from pathlib import Path

def test_anatomy_contract_no_forbidden_terms():
    """Test that anatomy contract doesn't contain forbidden scientific terms"""
    
    contract_path = Path("research/anatomy_contract.yaml")
    assert contract_path.exists(), "Anatomy contract file should exist"
    
    with open(contract_path, 'r') as f:
        contract = yaml.safe_load(f)
    
    # Forbidden terms that shouldn't appear in source-backed contract
    forbidden_terms = [
        "clinical validation",
        "validated against clinical standards",
        "manually verified for accuracy", 
        "consistent across all samples",
        "uniform leaflet thickness",
        "rigid valve structure",
        "symmetric leaflet geometry",
        "fixed commisure positions",
        "force distribution modeling",
        "dynamic opening/closing behavior",
        "true anatomical variation",
        "accuracy of structural analysis"
    ]
    
    contract_text = str(contract)
    
    # Check that no forbidden terms appear in the contract
    for term in forbidden_terms:
        assert term not in contract_text, f"Forbidden term '{term}' found in anatomy contract"
        
    print("✓ No forbidden terms found in anatomy contract")

def test_anatomy_contract_source_backed():
    """Test that anatomy contract only uses source-backed information"""
    
    contract_path = Path("research/anatomy_contract.yaml")
    assert contract_path.exists(), "Anatomy contract file should exist"
    
    with open(contract_path, 'r') as f:
        contract = yaml.safe_load(f)
    
    # Check that unsupported elements are properly marked
    assert "unsupported" in str(contract.get('coordinate_system', {})), "Coordinate system should be marked as unsupported"
    assert "unsupported" in str(contract.get('frame', {})), "Frame should be marked as unsupported"
    assert "unsupported" in str(contract.get('mesh', {}).get('topology', '')), "Mesh topology should be marked as unsupported"
    
    print("✓ All unsupported elements properly marked")

if __name__ == "__main__":
    test_anatomy_contract_no_forbidden_terms()
    test_anatomy_contract_source_backed()
    print("All validation tests passed!")