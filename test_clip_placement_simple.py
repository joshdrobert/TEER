#!/usr/bin/env python3
"""
Simple test for clip placement module
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from src.teer_cdss.clip_placement import (
        compute_clip_placement_recommendations,
        validate_clip_placement_recommendations,
        get_clip_placement_advice,
        example_clip_placement_workflow
    )
    
    # Test basic functionality
    print("Testing clip placement module...")
    
    # Test with sample geometric features
    geometric_features = {
        'surface_area': 5000.0,
        'volume': 2000.0,
        'mean_curvature': -0.005,
        'boundary_length': 150.0
    }
    
    # Test recommendation computation
    recommendations = compute_clip_placement_recommendations(geometric_features)
    print(f"Recommendations: {recommendations}")
    
    # Test validation
    is_valid = validate_clip_placement_recommendations(recommendations)
    print(f"Validation result: {is_valid}")
    
    # Test advice generation
    advice = get_clip_placement_advice(geometric_features)
    print(f"Advice: {advice}")
    
    # Test complete workflow
    result = example_clip_placement_workflow(geometric_features)
    print(f"Workflow result keys: {list(result.keys())}")
    
    print("Clip placement module test completed successfully!")
    
except Exception as e:
    print(f"Error running test: {e}")
    import traceback
    traceback.print_exc()