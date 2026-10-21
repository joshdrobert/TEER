"""
Reduced-Order Model Validation Module

This module implements validation checks for reduced-order models including:
- Conservation checks
- Convergence comparisons against reference data
- Physics-based validation metrics
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import json


class ROMValidator:
    """Validates reduced-order models against conservation and convergence criteria."""
    
    def __init__(self):
        self.validation_metrics = {}
        
    def validate_conservation(self, model_output: np.ndarray, 
                            reference_data: np.ndarray,
                            time_points: np.ndarray) -> Dict[str, float]:
        """
        Validate conservation properties of the reduced-order model.
        
        Args:
            model_output: Model simulation results
            reference_data: Reference measurements or independent data
            time_points: Time points for comparison
            
        Returns:
            Dictionary of conservation metrics
        """
        conservation_metrics = {}
        
        # Check mass conservation (for fluid dynamics applications)
        if len(model_output.shape) > 1:
            # For vector fields, check divergence
            divergence = self._compute_divergence(model_output)
            conservation_metrics['divergence_norm'] = np.linalg.norm(divergence)
            
        # Check energy conservation (if applicable)
        energy_conservation = self._check_energy_conservation(
            model_output, reference_data, time_points
        )
        conservation_metrics.update(energy_conservation)
        
        return conservation_metrics
    
    def _compute_divergence(self, vector_field: np.ndarray) -> np.ndarray:
        """Compute divergence of a vector field."""
        # Simple finite difference divergence calculation
        if len(vector_field.shape) == 2 and vector_field.shape[1] > 1:
            # Assuming spatial dimensions are columns and time is rows
            div = np.zeros(vector_field.shape[0])
            for i in range(vector_field.shape[0]):
                grad = np.gradient(vector_field[i, :])
                div[i] = np.sum(grad)
            return div
        return np.zeros(vector_field.shape[0])
    
    def _check_energy_conservation(self, model_output: np.ndarray,
                                 reference_data: np.ndarray,
                                 time_points: np.ndarray) -> Dict[str, float]:
        """Check energy conservation properties."""
        metrics = {}
        
        # Compute kinetic energy (simplified)
        if len(model_output.shape) > 1:
            kinetic_energy = 0.5 * np.sum(model_output**2, axis=1)
            
            # Compare with reference
            if len(reference_data.shape) > 1:
                reference_kinetic_energy = 0.5 * np.sum(reference_data**2, axis=1)
                
                # Energy conservation error
                energy_error = np.abs(kinetic_energy - reference_kinetic_energy)
                metrics['energy_conservation_error'] = np.mean(energy_error)
                metrics['max_energy_error'] = np.max(energy_error)
        
        return metrics
    
    def validate_convergence(self, model_outputs: List[np.ndarray],
                           reference_data: np.ndarray,
                           refinement_levels: List[int]) -> Dict[str, float]:
        """
        Validate convergence of the reduced-order model.
        
        Args:
            model_outputs: List of model outputs at different refinement levels
            reference_data: High-fidelity reference data
            refinement_levels: List of refinement levels
            
        Returns:
            Dictionary of convergence metrics
        """
        convergence_metrics = {}
        
        # Compute convergence rate
        if len(model_outputs) >= 2:
            convergence_rate = self._compute_convergence_rate(
                model_outputs, reference_data, refinement_levels
            )
            convergence_metrics['convergence_rate'] = convergence_rate
            
        # Compute error norms
        errors = []
        for i, output in enumerate(model_outputs):
            error = np.linalg.norm(output - reference_data)
            errors.append(error)
            
        if len(errors) > 0:
            convergence_metrics['mean_error'] = np.mean(errors)
            convergence_metrics['max_error'] = np.max(errors)
            
        return convergence_metrics
    
    def _compute_convergence_rate(self, model_outputs: List[np.ndarray],
                                reference_data: np.ndarray,
                                refinement_levels: List[int]) -> float:
        """Compute the order of convergence."""
        if len(model_outputs) < 2 or len(refinement_levels) < 2:
            return 0.0
            
        # Compute error at each level
        errors = []
        for output in model_outputs:
            error = np.linalg.norm(output - reference_data)
            errors.append(error)
            
        # Simple convergence rate calculation using last two points
        if len(errors) >= 2:
            rate = np.log(errors[-2] / errors[-1]) / np.log(refinement_levels[-1] / refinement_levels[-2])
            return rate
            
        return 0.0
    
    def validate_against_reference(self, model_output: np.ndarray,
                                 reference_data: np.ndarray,
                                 validation_config: Dict[str, Any]) -> Dict[str, float]:
        """
        Validate model output against independent reference data.
        
        Args:
            model_output: Model simulation results
            reference_data: Independent reference measurements
            validation_config: Configuration for validation
            
        Returns:
            Dictionary of validation metrics
        """
        validation_metrics = {}
        
        # Compute relative error
        if np.linalg.norm(reference_data) > 0:
            relative_error = np.linalg.norm(model_output - reference_data) / np.linalg.norm(reference_data)
            validation_metrics['relative_error'] = relative_error
            
        # Compute maximum absolute error
        abs_error = np.abs(model_output - reference_data)
        validation_metrics['max_absolute_error'] = np.max(abs_error)
        
        # Compute mean absolute error
        validation_metrics['mean_absolute_error'] = np.mean(abs_error)
        
        # Check if results are within acceptable bounds
        if 'acceptable_range' in validation_config:
            range_min, range_max = validation_config['acceptable_range']
            within_bounds = np.all((reference_data >= range_min) & (reference_data <= range_max))
            validation_metrics['within_acceptable_range'] = within_bounds
            
        return validation_metrics
    
    def load_validation_metrics(self, metrics_file: str) -> Dict[str, Any]:
        """
        Load physics-based validation metrics from file.
        
        Args:
            metrics_file: Path to JSON file with validation metrics
            
        Returns:
            Dictionary of loaded metrics
        """
        try:
            with open(metrics_file, 'r') as f:
                self.validation_metrics = json.load(f)
            return self.validation_metrics
        except FileNotFoundError:
            print(f"Warning: Validation metrics file {metrics_file} not found")
            return {}
        except json.JSONDecodeError:
            print(f"Warning: Invalid JSON in {metrics_file}")
            return {}
    
    def validate_model(self, model_output: np.ndarray,
                     reference_data: np.ndarray,
                     time_points: Optional[np.ndarray] = None,
                     refinement_levels: Optional[List[int]] = None,
                     validation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive model validation.
        
        Args:
            model_output: Model simulation results
            reference_data: Reference measurements or independent data
            time_points: Time points for temporal validation (optional)
            refinement_levels: Refinement levels for convergence analysis (optional)
            validation_config: Configuration for validation
            
        Returns:
            Dictionary of all validation metrics
        """
        if validation_config is None:
            validation_config = {}
            
        results = {}
        
        # Conservation validation
        conservation_metrics = self.validate_conservation(
            model_output, reference_data, time_points or np.array([0])
        )
        results['conservation'] = conservation_metrics
        
        # Convergence validation (if refinement levels provided)
        if refinement_levels is not None:
            convergence_metrics = self.validate_convergence(
                [model_output], reference_data, refinement_levels
            )
            results['convergence'] = convergence_metrics
            
        # Reference validation
        reference_metrics = self.validate_against_reference(
            model_output, reference_data, validation_config
        )
        results['reference_comparison'] = reference_metrics
        
        return results


def create_validation_report(validation_results: Dict[str, Any]) -> str:
    """
    Create a human-readable validation report.
    
    Args:
        validation_results: Results from validate_model method
        
    Returns:
        Formatted validation report string
    """
    report = "Reduced-Order Model Validation Report\n"
    report += "=" * 40 + "\n\n"
    
    # Conservation metrics
    if 'conservation' in validation_results:
        report += "Conservation Metrics:\n"
        for metric, value in validation_results['conservation'].items():
            report += f"  {metric}: {value:.6f}\n"
        report += "\n"
    
    # Convergence metrics
    if 'convergence' in validation_results:
        report += "Convergence Metrics:\n"
        for metric, value in validation_results['convergence'].items():
            report += f"  {metric}: {value:.6f}\n"
        report += "\n"
    
    # Reference comparison metrics
    if 'reference_comparison' in validation_results:
        report += "Reference Comparison Metrics:\n"
        for metric, value in validation_results['reference_comparison'].items():
            report += f"  {metric}: {value:.6f}\n"
        report += "\n"
    
    return report


# Example usage function
def example_validation():
    """Example of how to use the ROMValidator."""
    validator = ROMValidator()
    
    # Create some sample data
    model_output = np.random.rand(100, 5)  # 100 time steps, 5 variables
    reference_data = np.random.rand(100, 5)  # Same structure as model output
    
    # Validate the model
    results = validator.validate_model(
        model_output=model_output,
        reference_data=reference_data,
        validation_config={
            'acceptable_range': (-1.0, 1.0)
        }
    )
    
    # Print report
    print(create_validation_report(results))
    
    return results


if __name__ == "__main__":
    example_validation()