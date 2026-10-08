"""
Test suite for clip placement decision support module.
"""

import pytest
import numpy as np
from src.teer_cdss.clip_placement import (
    compute_clip_placement_recommendations,
    validate_clip_placement_recommendations,
    get_clip_placement_advice,
    example_clip_placement_workflow
)


def test_compute_clip_placement_recommendations_standard():
    """Test standard case for clip placement recommendations."""
    # Test with typical geometric features
    geometric_features = {
        'surface_area': 5000.0,
        'volume': 2000.0,
        'mean_curvature': -0.005,
        'boundary_length': 150.0
    }
    
    recommendations = compute_clip_placement_recommendations(geometric_features)
    
    assert 'placement_strategy' in recommendations
    assert 'confidence_score' in recommendations
    assert 'recommendation_details' in recommendations
    
    # Should be standard strategy for typical features
    assert recommendations['placement_strategy'] == 'standard'
    assert 0.0 <= recommendations['confidence_score'] <= 1.0


def test_compute_clip_placement_recommendations_advanced():
    """Test advanced case for clip placement recommendations."""
    # Test with large anatomical features
    geometric_features = {
        'surface_area': 15000.0,
        'volume': 8000.0,
        'mean_curvature': -0.002,
        'boundary_length': 200.0
    }
    
    recommendations = compute_clip_placement_recommendations(geometric_features)
    
    assert recommendations['placement_strategy'] == 'advanced'


def test_compute_clip_placement_recommendations_reinforced():
    """Test reinforced case for clip placement recommendations."""
    # Test with low curvature features
    geometric_features = {
        'surface_area': 3000.0,
        'volume': 1000.0,
        'mean_curvature': -0.05,
        'boundary_length': 120.0
    }
    
    recommendations = compute_clip_placement_recommendations(geometric_features)
    
    assert recommendations['placement_strategy'] == 'reinforced'


def test_validate_clip_placement_recommendations():
    """Test validation of clip placement recommendations."""
    # Valid recommendations
    valid_recommendations = {
        'placement_strategy': 'standard',
        'confidence_score': 0.75,
        'recommendation_details': {}
    }
    
    assert validate_clip_placement_recommendations(valid_recommendations) == True
    
    # Invalid recommendations
    invalid_recommendations = {
        'placement_strategy': 'invalid_strategy',
        'confidence_score': 0.3,
        'recommendation_details': {}
    }
    
    assert validate_clip_placement_recommendations(invalid_recommendations) == False
    
    # Recommendations without confidence score
    no_confidence_recommendations = {
        'placement_strategy': 'standard',
        'recommendation_details': {}
    }
    
    assert validate_clip_placement_recommendations(no_confidence_recommendations) == False


def test_get_clip_placement_advice():
    """Test human-readable advice generation."""
    geometric_features = {
        'surface_area': 5000.0,
        'volume': 2000.0,
        'mean_curvature': -0.005,
        'boundary_length': 150.0
    }
    
    advice = get_clip_placement_advice(geometric_features)
    
    assert isinstance(advice, str)
    assert len(advice) > 0
    assert "confidence" in advice.lower()


def test_example_workflow():
    """Test the complete example workflow."""
    geometric_features = {
        'surface_area': 5000.0,
        'volume': 2000.0,
        'mean_curvature': -0.005,
        'boundary_length': 150.0
    }
    
    result = example_clip_placement_workflow(geometric_features)
    
    assert 'recommendations' in result
    assert 'is_valid' in result
    assert 'advice' in result
    
    assert isinstance(result['is_valid'], bool)
    assert isinstance(result['advice'], str)


def test_edge_cases():
    """Test edge cases for clip placement recommendations."""
    # Empty features
    empty_features = {}
    recommendations = compute_clip_placement_recommendations(empty_features)
    
    assert 'placement_strategy' in recommendations
    assert 'confidence_score' in recommendations
    
    # Very large features
    large_features = {
        'surface_area': 100000.0,
        'volume': 50000.0,
        'mean_curvature': -0.1,
        'boundary_length': 1000.0
    }
    
    recommendations = compute_clip_placement_recommendations(large_features)
    assert 0.0 <= recommendations['confidence_score'] <= 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])