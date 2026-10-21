#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tests.test_transform_repeatability import test_mesh_repeatability_assessment, test_record_assessment, test_edge_cases

if __name__ == "__main__":
    print("Running repeatability assessment tests...")
    
    try:
        test_mesh_repeatability_assessment()
        print("✓ Basic assessment test passed")
    except Exception as e:
        print(f"✗ Basic assessment test failed: {e}")
        
    try:
        test_record_assessment()
        print("✓ Record assessment test passed")
    except Exception as e:
        print(f"✗ Record assessment test failed: {e}")
        
    try:
        test_edge_cases()
        print("✓ Edge cases test passed")
    except Exception as e:
        print(f"✗ Edge cases test failed: {e}")
        
    print("Test run completed.")