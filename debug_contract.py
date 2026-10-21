#!/usr/bin/env python3

import yaml
from pathlib import Path

# Read the anatomy contract
contract_path = Path("research/anatomy_contract.yaml")
with open(contract_path, 'r') as f:
    contract_content = f.read()

print("=== ANATOMY CONTRACT CONTENT ===")
print(contract_content)
print("\n=== ANALYSIS ===")

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

print("\nChecking for forbidden terms:")
for term in forbidden_terms:
    if term in contract_content.lower():
        print(f"  ❌ FOUND: '{term}'")
    else:
        print(f"  ✅ NOT FOUND: '{term}'")

# Also check for any scientific claims or clinical validation language
scientific_indicators = [
    "clinical",
    "validation",
    "expert",
    "certified",
    "standard",
    "reference",
    "benchmark",
    "accuracy",
    "precision"
]

print("\nChecking for scientific indicators:")
for indicator in scientific_indicators:
    if indicator in contract_content.lower():
        print(f"  ⚠️  POTENTIAL: '{indicator}' found")