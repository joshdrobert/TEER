"""Test cases for geometric feature repeatability assessment."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh

from extract_features import extract_features, _compute_qc_metrics


class TestFeatureRepeatability(unittest.TestCase):
    """Test feature repeatability functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.TemporaryDirectory()
        
    def tearDown(self):
        """Clean up test fixtures."""
        self.temp_dir.cleanup()
    
    def test_repeatability_metrics_computation(self):
        """Test that repeatability metrics are computed correctly."""
        # Create a simple mesh for testing
        vertices = np.array([
            [0, 0, 0],
            [1, 0, 0], 
            [0, 1, 0],
            [0, 0, 1]
        ])
        faces = np.array([[0, 1, 2], [0, 1, 3], [1, 2, 3], [0, 2, 3]])
        
        # Create a simple mesh
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        # Save the mesh to a temporary file
        test_file = Path(self.temp_dir.name) / "test.stl"
        mesh.export(test_file)
        
        # Extract features from the mesh
        result = extract_features(test_file)
        
        # Check that we have all expected basic metrics
        self.assertIn("subject_id", result)
        self.assertIn("vertex_count", result)
        self.assertIn("face_count", result)
        self.assertIn("surface_area", result)
        self.assertIn("bbox_x", result)
        self.assertIn("bbox_y", result)
        self.assertIn("bbox_z", result)
        self.assertIn("centroid_x", result)
        self.assertIn("centroid_y", result)
        self.assertIn("centroid_z", result)
        self.assertIn("axis_1_len", result)
        self.assertIn("axis_2_len", result)
        self.assertIn("axis_3_len", result)
        self.assertIn("component_count", result)
        self.assertIn("is_watertight", result)
        self.assertIn("max_extent", result)
        self.assertIn("volume", result)
        self.assertIn("v_comm", result)
        self.assertIn("v_open", result)
        self.assertIn("center_x", result)
        self.assertIn("center_y", result)
        
        # Check QC metrics
        self.assertIn("qc_valid", result)
        self.assertIn("qc_outlier_flags", result)
        self.assertIn("qc_zscore", result)
        
    def test_repeatability_with_invalid_mesh(self):
        """Test handling of invalid meshes."""
        # Create a mesh with no faces
        vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mesh = trimesh.Trimesh(vertices=vertices)
        
        # Save the mesh to a temporary file
        test_file = Path(self.temp_dir.name) / "test.stl"
        mesh.export(test_file)
        
        # Extract features from the mesh (should raise an exception)
        with self.assertRaises(ValueError):
            extract_features(test_file)
            
    def test_qc_metrics_validation(self):
        """Test QC metrics computation."""
        # Create a valid row
        valid_row = {
            "subject_id": "train_001",
            "vertex_count": 1000,
            "face_count": 500,
            "surface_area": 10.0,
            "bbox_x": 2.0,
            "bbox_y": 2.0,
            "bbox_z": 2.0,
            "centroid_x": 1.0,
            "centroid_y": 1.0,
            "centroid_z": 1.0,
            "axis_1_len": 2.0,
            "axis_2_len": 1.5,
            "axis_3_len": 1.0,
            "component_count": 1,
            "is_watertight": True,
            "max_extent": 2.0,
            "volume": 1.0,
            "v_comm": 0.0,
            "v_open": 2.0,
            "center_x": 1.0,
            "center_y": 1.0,
        }
        
        # Compute QC metrics
        valid, flags, zscores = _compute_qc_metrics(valid_row)
        self.assertTrue(valid)
        self.assertEqual(len(flags), 0)
        
        # Test with invalid face count
        invalid_row = valid_row.copy()
        invalid_row["face_count"] = 5  # Too low
        
        valid, flags, zscores = _compute_qc_metrics(invalid_row)
        self.assertFalse(valid)
        self.assertIn("low_face_count", flags)
        
        # Test with invalid surface area
        invalid_row = valid_row.copy()
        invalid_row["surface_area"] = 0.00001  # Too small
        
        valid, flags, zscores = _compute_qc_metrics(invalid_row)
        self.assertFalse(valid)
        self.assertIn("very_small_surface_area", flags)


if __name__ == "__main__":
    unittest.main()