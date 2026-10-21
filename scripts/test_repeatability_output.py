import json
import pytest
from pathlib import Path

def test_geometry_repeatability_output():
    """Test that geometry repeatability output contains expected structure and metrics."""
    # Read the generated file
    output_file = Path("research/geometry_repeatability.json")
    assert output_file.exists(), "geometry_repeatability.json should be created"

    with open(output_file, 'r') as f:
        data = json.load(f)

    # Check basic structure
    assert "version" in data
    assert "assessment_date" in data
    assert "meshes_assessed" in data
    assert "metrics" in data

    # Check that we have multiple meshes assessed
    assert len(data["meshes_assessed"]) == 5

    # Check each mesh has expected fields
    for mesh in data["meshes_assessed"]:
        assert "name" in mesh
        assert "metrics" in mesh
        assert "timestamp" in mesh

        metrics = mesh["metrics"]
        assert "alignment_invariance" in metrics
        assert "feature_invariance" in metrics
        assert "mesh_volume_raw" in metrics
        assert "mesh_surface_area_raw" in metrics
        assert "error" in metrics

    # Check that metrics section has expected structure
    assert "alignment_invariance" in data["metrics"]
    assert "feature_invariance" in data["metrics"]

    # Check alignment invariance metrics
    alignment_metrics = data["metrics"]["alignment_invariance"]
    assert "mean" in alignment_metrics
    assert "std" in alignment_metrics
    assert "min" in alignment_metrics
    assert "max" in alignment_metrics
    assert "outlier_flags" in alignment_metrics

    # Check feature invariance metrics
    feature_metrics = data["metrics"]["feature_invariance"]
    assert "mean" in feature_metrics
    assert "std" in feature_metrics
    assert "min" in feature_metrics
    assert "max" in feature_metrics
    assert "outlier_flags" in feature_metrics

    # Verify that values are reasonable (not null)
    assert alignment_metrics["mean"] is not None
    assert feature_metrics["mean"] is not None
    assert alignment_metrics["std"] is not None
    assert feature_metrics["std"] is not None

def test_geometry_repeatability_area_volume_invariance():
    """Test that area/volume invariant results are properly computed."""
    output_file = Path("research/geometry_repeatability.json")

    with open(output_file, 'r') as f:
        data = json.load(f)

    # Check that all meshes have valid volume and surface area values
    for mesh in data["meshes_assessed"]:
        metrics = mesh["metrics"]
        assert metrics["mesh_volume_raw"] > 0
        assert metrics["mesh_surface_area_raw"] > 0

    # Check that the computed metrics are reasonable
    alignment_metrics = data["metrics"]["alignment_invariance"]
    feature_metrics = data["metrics"]["feature_invariance"]

    # These should be small values (indicating good repeatability)
    assert alignment_metrics["mean"] < 0.01
    assert feature_metrics["mean"] < 0.01

def test_geometry_repeatability_outlier_warnings():
    """Test that outlier warnings are preserved in the output."""
    output_file = Path("research/geometry_repeatability.json")

    with open(output_file, 'r') as f:
        data = json.load(f)

    # Check that outlier_flags exists and is properly structured
    alignment_metrics = data["metrics"]["alignment_invariance"]
    feature_metrics = data["metrics"]["feature_invariance"]

    # Outlier flags should be present (could be empty list if no outliers)
    assert isinstance(alignment_metrics["outlier_flags"], list)
    assert isinstance(feature_metrics["outlier_flags"], list)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])