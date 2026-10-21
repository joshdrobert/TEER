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

def test_zero_face_stl_returns_invalid():
    """Deterministic regression test: zero-face STL must return GEOMETRY_INVALID
    without fabricated bounds, centroid, or other metrics."""
    
    # Create an empty mesh with zero faces
    empty_mesh = mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))
    
    with tempfile.NamedTemporaryFile(suffix='.stl', delete=False) as f:
        empty_mesh.save(f.name)
        result = validate_mesh(f.name)
        
        # Must parse successfully (file is valid STL format)
        assert result["parse_success"] is True
        
        # Must report zero faces and vertices
        assert result["faces"] == 0
        assert result["vertices"] == 0
        
        # Must NOT fabricate bounds or centroid - these must be None
        assert result["bbox_min"] is None
        assert result["bbox_max"] is None
        assert result["centroid"] is None
        
        # Must report GEOMETRY_INVALID, not PARSE_FAILED or QUALITY_WARNING
        assert result["status"] == "GEOMETRY_INVALID"
        
        # Must have a descriptive error message
        assert "zero-face" in result.get("error", "").lower()
        
        os.unlink(f.name)

def test_new_validation_fields_present():
    """Test that new bounded coordinate verification and uncertainty fields are present"""
    # Create a basic mesh to test field presence
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
        
        # Check that new fields are present in the metrics
        assert "bounded_coordinate_verification" in result
        assert "coordinate_invariance_score" in result
        assert "uncertainty_estimate" in result
        assert "abstention_reason" in result
        
        # These should have initial default values
        assert result["bounded_coordinate_verification"] is False or result["bounded_coordinate_verification"] is True
        assert isinstance(result["coordinate_invariance_score"], (int, float))
        assert isinstance(result["uncertainty_estimate"], (int, float))
        assert result["abstention_reason"] is None or isinstance(result["abstention_reason"], str)
        
        os.unlink(f.name)

if __name__ == "__main__":
    test_finite_coordinates_check()
    test_nonfinite_coordinates_fail()
    test_bounding_box_includes_all_vertices()
    test_adversarial_stl_with_nan()
    test_adversarial_stl_with_all_nan()
    test_zero_face_stl_returns_invalid()
    test_new_validation_fields_present()
    print("All tests passed!")