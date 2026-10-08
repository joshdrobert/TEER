"""
Tests for ROM validation module
"""

import numpy as np
import pytest
from src.teer_cdss.rom_validator import ROMValidator, create_validation_report


class TestROMValidator:
    
    def test_conservation_validation(self):
        """Test conservation validation functionality."""
        validator = ROMValidator()
        
        # Create simple test data
        model_output = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        reference_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        time_points = np.array([0.0, 1.0])
        
        # Validate conservation
        metrics = validator.validate_conservation(model_output, reference_data, time_points)
        
        assert isinstance(metrics, dict)
        assert 'divergence_norm' in metrics
        
    def test_convergence_validation(self):
        """Test convergence validation functionality."""
        validator = ROMValidator()
        
        # Create test data at different refinement levels
        model_outputs = [
            np.array([[1.0, 2.0], [3.0, 4.0]]),
            np.array([[1.1, 2.1], [3.1, 4.1]])
        ]
        reference_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        refinement_levels = [10, 20]
        
        # Validate convergence
        metrics = validator.validate_convergence(model_outputs, reference_data, refinement_levels)
        
        assert isinstance(metrics, dict)
        assert 'convergence_rate' in metrics
        assert 'mean_error' in metrics
        
    def test_reference_validation(self):
        """Test validation against reference data."""
        validator = ROMValidator()
        
        # Create test data
        model_output = np.array([1.0, 2.0, 3.0])
        reference_data = np.array([1.05, 2.05, 3.05])
        validation_config = {
            'acceptable_range': (0.0, 5.0)
        }
        
        # Validate against reference
        metrics = validator.validate_against_reference(
            model_output, reference_data, validation_config
        )
        
        assert isinstance(metrics, dict)
        assert 'relative_error' in metrics
        assert 'max_absolute_error' in metrics
        assert 'mean_absolute_error' in metrics
        
    def test_model_validation_comprehensive(self):
        """Test comprehensive model validation."""
        validator = ROMValidator()
        
        # Create test data
        model_output = np.random.rand(50, 3)
        reference_data = np.random.rand(50, 3)
        
        # Validate complete model
        results = validator.validate_model(
            model_output=model_output,
            reference_data=reference_data,
            validation_config={'acceptable_range': (-1.0, 1.0)}
        )
        
        assert isinstance(results, dict)
        assert 'conservation' in results
        assert 'reference_comparison' in results
        
    def test_validation_report_creation(self):
        """Test creation of validation report."""
        validator = ROMValidator()
        
        # Create some test results
        model_output = np.array([1.0, 2.0, 3.0])
        reference_data = np.array([1.05, 2.05, 3.05])
        validation_config = {'acceptable_range': (0.0, 5.0)}
        
        results = validator.validate_against_reference(
            model_output, reference_data, validation_config
        )
        
        # Create report
        report = create_validation_report(results)
        
        assert isinstance(report, str)
        assert "Reduced-Order Model Validation Report" in report
        
    def test_load_validation_metrics(self):
        """Test loading of validation metrics from file."""
        validator = ROMValidator()
        
        # Load test metrics file
        metrics = validator.load_validation_metrics('research/physics_validation_metrics.json')
        
        assert isinstance(metrics, dict)
        assert 'conservation' in metrics
        assert 'convergence' in metrics
        assert 'reference_data' in metrics


def test_example_validation():
    """Test the example validation function."""
    # This should not raise any exceptions
    from src.teer_cdss.rom_validator import example_validation
    
    results = example_validation()
    
    assert isinstance(results, dict)
    assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])