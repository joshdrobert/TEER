#!/usr/bin/env python3

import yaml
from pathlib import Path

# Read the anatomy contract
contract_path = Path("research/anatomy_contract.yaml")
with open(contract_path, 'r') as f:
    contract_content = f.read()

print("=== ANALYSIS OF FORBIDDEN TERMS ===")

# Check for potential forbidden terms that might not be obvious from the text
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
    "accuracy of structural analysis",
    "clinical endpoints",
    "performance metrics",
    "annulus",
    "leaflet identity",
    "expert annotation",
    "uncertainty bounds",
    "watertight",
    "manifold"
]

print("Checking for forbidden terms:")
found_terms = []
for term in forbidden_terms:
    if term in contract_content.lower():
        print(f"  ❌ FOUND: '{term}'")
        found_terms.append(term)
    else:
        print(f"  ✅ NOT FOUND: '{term}'")

if not found_terms:
    print("\n✅ No forbidden terms found - this should pass the validation!")
else:
    print(f"\n❌ Found {len(found_terms)} forbidden terms that need to be removed")
    
# Also check for scientific language
scientific_indicators = [
    "clinical",
    "validation",
    "expert",
    "certified",
    "standard",
    "reference",
    "benchmark",
    "accuracy",
    "precision",
    "performance",
    "endpoint"
]

print("\nChecking for scientific indicators:")
scientific_found = []
for indicator in scientific_indicators:
    if indicator in contract_content.lower():
        print(f"  ⚠️  POTENTIAL: '{indicator}' found")
        scientific_found.append(indicator)

if not scientific_found:
    print("  ✅ No scientific language found")

print("\n=== CONTRACT SUMMARY ===")
lines = contract_content.split('\n')
print(f"Total lines: {len(lines)}")
print(f"Total characters: {len(contract_content)}")