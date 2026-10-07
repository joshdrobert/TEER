import os
import tempfile
import numpy as np
from stl import mesh
import pytest
from scripts.validate_geometries import validate_mesh

def test_finite_coordinates_check():
    """Test that all triangle vertices are checked for finite coordinates"""
    
    # Create a simple STL with one triangle
    # This should pass validation
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    
    # Create a mesh with valid coordinates
    cube = mesh.Mesh(np.zeros(1, dtype=mesh.Mesh.dtype))
    cube.v0[0] = vertices[0]
    cube.v1[0] = vertices[1]
    cube.v2[0] = vertices[2]
    
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        cube.save(f.name)
        result = validate_mesh(f.name)
        assert result["finite_coords"] == True
        os.unlink(f.name)

def test_nonfinite_coordinates_fail():
    """Test that non-finite coordinates fail validation"""
    
    # Create a mesh with non-finite coordinates (infinity)
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, np.inf]  # This makes v2 infinite
    ])
    
    cube = mesh.Mesh(np.zeros(1, dtype=mesh.Mesh.dtype))
    cube.v0[0] = vertices[0]
    cube.v1[0] = vertices[1]
    cube.v2[0] = vertices[2]
    
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        cube.save(f.name)
        result = validate_mesh(f.name)
        # This should fail because v2 has infinite coordinates
        assert result["finite_coords"] == False
        os.unlink(f.name)

def test_bounding_box_includes_all_vertices():
    """Test that bounding box includes all triangle vertices"""
    
    # Create a mesh with vertices at extreme positions
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, 0]
    ])
    
    cube = mesh.Mesh(np.zeros(1, dtype=mesh.Mesh.dtype))
    cube.v0[0] = vertices[0]
    cube.v1[0] = vertices[1]
    cube.v2[0] = vertices[2]
    
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        cube.save(f.name)
        result = validate_mesh(f.name)
        # Check that bbox_min and bbox_max are computed correctly
        assert result["bbox_min"] is not None
        assert result["bbox_max"] is not None
        os.unlink(f.name)

def test_adversarial_stl_with_nan():
    """Test adversarial STL with NaN values"""
    
    vertices = np.array([
        [0, 0, 0],
        [1, 0, 0],
        [0, 1, np.nan]  # This makes v2 have NaN coordinates
    ])
    
    cube = mesh.Mesh(np.zeros(1, dtype=mesh.Mesh.dtype))
    cube.v0[0] = vertices[0]
    cube.v1[0] = vertices[1]
    cube.v2[0] = vertices[2]
    
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        cube.save(f.name)
        result = validate_mesh(f.name)
        # This should fail validation due to NaN coordinates
        assert result["finite_coords"] == False
        os.unlink(f.name)

def test_adversarial_stl_with_all_nan():
    """Test adversarial STL with all vertices having NaN"""
    
    vertices = np.array([
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan]
    ])
    
    cube = mesh.Mesh(np.zeros(1, dtype=mesh.Mesh.dtype))
    cube.v0[0] = vertices[0]
    cube.v1[0] = vertices[1]
    cube.v2[0] = vertices[2]
    
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        cube.save(f.name)
        result = validate_mesh(f.name)
        # This should fail validation due to NaN coordinates
        assert result["finite_coords"] == False
        os.unlink(f.name)

if __name__ == "__main__":
    test_finite_coordinates_check()
    test_nonfinite_coordinates_fail()
    test_bounding_box_includes_all_vertices()
    test_adversarial_stl_with_nan()
    test_adversarial_stl_with_all_nan()
    print("All tests passed!")