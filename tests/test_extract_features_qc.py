"""Tests for QC pipeline in extract_features.py."""

import tempfile
import csv
from pathlib import Path

import pytest
import trimesh
import numpy as np

from scripts.extract_features import extract_features, run


def test_qc_validity_checks():
    """Test that validity checks are computed correctly."""
    # Create a simple mesh for testing
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_path = Path(tmpdir) / "test.stl"
        mesh.export(mesh_path)
        
        # Test that features are extracted
        result = extract_features(mesh_path)
        
        # Check that QC fields are present
        assert "qc_valid" in result
        assert "qc_outlier_flags" in result
        assert "qc_zscore" in result


def test_qc_outlier_detection():
    """Test that outlier detection works correctly."""
    # Create a mesh with known properties
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_path = Path(tmpdir) / "test.stl"
        mesh.export(mesh_path)
        
        result = extract_features(mesh_path)
        
        # Should have some QC fields
        assert isinstance(result["qc_valid"], bool)
        assert isinstance(result["qc_outlier_flags"], list)
        assert isinstance(result["qc_zscore"], dict)


def test_qc_report_generation():
    """Test that QC report CSV is generated correctly."""
    # Create a simple mesh for testing
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])
    faces = np.array([
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3]
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        mesh_path = Path(tmpdir) / "test.stl"
        mesh.export(mesh_path)
        
        # Test run function
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir()
        
        run([mesh_path], output_dir)
        
        # Check that report was created
        report_file = output_dir / "qc_report.csv"
        assert report_file.exists()
        
        # Read the CSV to verify content
        with open(report_file, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])