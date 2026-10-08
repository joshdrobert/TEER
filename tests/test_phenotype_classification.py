import pytest
import numpy as np
from src.teer_cdss.phenotype_classifier import PhenotypeClassifier, MorphologicalPhenotype, PlacementParameters

def test_morphological_phenotype_dataclass():
    """Test that MorphologicalPhenotype dataclass works correctly."""
    phenotype = MorphologicalPhenotype(
        name="test_phenotype",
        description="A test phenotype",
        parameters={"param1": 1.0, "param2": 2.0},
        uncertainties={"param1": 0.1, "param2": 0.2},
        confidence=0.8
    )
    
    assert phenotype.name == "test_phenotype"
    assert phenotype.description == "A test phenotype"
    assert phenotype.parameters["param1"] == 1.0
    assert phenotype.confidence == 0.8

def test_morphological_phenotype_validation():
    """Test that MorphologicalPhenotype validates confidence correctly."""
    with pytest.raises(ValueError):
        MorphologicalPhenotype(
            name="test_phenotype",
            description="A test phenotype",
            parameters={"param1": 1.0},
            uncertainties={"param1": 0.1},
            confidence=1.5  # Invalid confidence
        )

def test_placement_parameters_dataclass():
    """Test that PlacementParameters dataclass works correctly."""
    placement = PlacementParameters(
        clip_position=(1.0, 2.0, 3.0),
        clip_orientation=(0.1, 0.2, 0.3),
        clip_size=(10.0, 20.0),
        uncertainties={
            "position": {"x": 0.1, "y": 0.2, "z": 0.3},
            "orientation": {"roll": 0.05, "pitch": 0.05, "yaw": 0.05}
        }
    )
    
    assert placement.clip_position == (1.0, 2.0, 3.0)
    assert placement.clip_orientation == (0.1, 0.2, 0.3)
    assert placement.clip_size == (10.0, 20.0)

def test_placement_parameters_validation():
    """Test that PlacementParameters validates correctly."""
    with pytest.raises(ValueError):
        # Invalid position
        PlacementParameters(
            clip_position=(1.0, 2.0),  # Only 2 coordinates
            clip_orientation=(0.1, 0.2, 0.3),
            clip_size=(10.0, 20.0),
            uncertainties={}
        )

def test_phenotype_classifier_initialization():
    """Test that PhenotypeClassifier initializes correctly."""
    classifier = PhenotypeClassifier()
    assert len(classifier.phenotypes) > 0

def test_classify_from_geometry():
    """Test the classification functionality."""
    classifier = PhenotypeClassifier()
    
    # Test with some sample geometric features
    features = {
        'annulus_narrowing_ratio': 0.7,
        'leaflet_thickness': 2.5,
        'valve_area_reduction': 0.6
    }
    
    results = classifier.classify_from_geometry(features)
    
    assert len(results) > 0
    assert isinstance(results[0], MorphologicalPhenotype)
    assert results[0].confidence >= 0.0
    assert results[0].confidence <= 1.0

def test_uncertainty_propagation():
    """Test uncertainty propagation in classification."""
    classifier = PhenotypeClassifier()
    
    features = {
        'annulus_narrowing_ratio': 0.7,
        'leaflet_thickness': 2.5,
        'valve_area_reduction': 0.6
    }
    
    uncertainties = {
        'annulus_narrowing_ratio': 0.05,
        'leaflet_thickness': 0.1,
        'valve_area_reduction': 0.05
    }
    
    results = classifier.classify_from_geometry(features, uncertainties)
    
    assert len(results) > 0
    # Check that uncertainties are properly propagated
    assert hasattr(results[0], 'uncertainties')

if __name__ == "__main__":
    pytest.main([__file__, "-v"])