"""Test cases for anatomy contract landmark extraction in extract_features."""

import tempfile
from pathlib import Path

import numpy as np
import trimesh

from scripts.extract_features import extract_features


def test_anatomy_landmarks_extraction():
    """Test that v_comm, v_open, center_x, and center_y fields are extracted."""
    
    # Create a simple test mesh (a basic cylinder)
    mesh = trimesh.primitives.Cylinder(radius=1.0, height=2.0, sections=32)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        mesh.export(tmp_path)
        
        # Extract features
        result = extract_features(tmp_path)
        
        # Check that anatomy contract fields are present
        assert "v_comm" in result, "v_comm field should be present"
        assert "v_open" in result, "v_open field should be present"
        assert "center_x" in result, "center_x field should be present"
        assert "center_y" in result, "center_y field should be present"
        
        # Check that fields have reasonable values
        assert isinstance(result["v_comm"], (int, float)), "v_comm should be numeric"
        assert isinstance(result["v_open"], (int, float)), "v_open should be numeric"
        assert isinstance(result["center_x"], (int, float)), "center_x should be numeric"
        assert isinstance(result["center_y"], (int, float)), "center_y should be numeric"
        
        # Clean up
        tmp_path.unlink()


def test_anatomy_landmarks_values():
    """Test that anatomy contract landmark values are computed correctly."""
    
    # Create a simple mesh for testing
    vertices = np.array([
        [0, 0, 0],   # bottom center
        [1, 0, 0],   # right point
        [0, 1, 0],   # front point  
        [0, 0, 1],   # top point
        [-1, 0, 0],  # left point
        [0, -1, 0],  # back point
    ])
    
    faces = np.array([
        [0, 1, 3],
        [0, 2, 3], 
        [0, 4, 3],
        [0, 5, 3],
        [1, 2, 0],
        [2, 4, 0],
        [4, 5, 0],
        [5, 1, 0]
    ])
    
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        mesh.export(tmp_path)
        
        # Extract features
        result = extract_features(tmp_path)
        
        # Check that fields exist and have values
        assert "v_comm" in result
        assert "v_open" in result
        assert "center_x" in result
        assert "center_y" in result
        
        # Clean up
        tmp_path.unlink()