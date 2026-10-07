import os
import glob
import csv
import numpy as np
from stl import mesh
import trimesh

def validate_mesh(file_path):
    metrics = {
        "filename": os.path.basename(file_path),
        "parse_success": False,
        "file_size": os.path.getsize(file_path),
        "vertices": 0,
        "faces": 0,
        "finite_coords": False,
        "degenerate_faces": 0,
        "duplicate_vertices": 0,
        "connected_components": 0,
        "bbox_min": None,
        "bbox_max": None,
        "centroid": None,
        "coord_range": 0,
        "surface_area": 0,
        "is_watertight": False,
        "is_manifold": False,
        "status": "UNKNOWN"
    }

    try:
        m = mesh.Mesh.from_file(file_path)
        metrics["parse_success"] = True

        # Zero-face STL: return truthful invalid result without fabricating bounds
        if m.v0.size == 0:
            metrics["faces"] = 0
            metrics["vertices"] = 0
            metrics["status"] = "GEOMETRY_INVALID"
            metrics["error"] = "Zero-face STL: no geometry to validate"
            return metrics

        all_coords = np.concatenate([m.v0.flatten(), m.v1.flatten(), m.v2.flatten()])
        if np.all(np.isfinite(all_coords)):
            metrics["finite_coords"] = True
        min_coords = np.min(m.v0, axis=0)
        max_coords = np.max(m.v0, axis=0)
        metrics["bbox_min"] = min_coords.tolist()
        metrics["bbox_max"] = max_coords.tolist()
        metrics["coord_range"] = float(np.max(max_coords - min_coords))
        metrics["centroid"] = np.mean(m.v0, axis=0).tolist()
        metrics["surface_area"] = float(m.areas.sum())

        t_mesh = trimesh.load(file_path)
        metrics["vertices"] = len(t_mesh.vertices)
        metrics["faces"] = len(t_mesh.faces)
        metrics["duplicate_vertices"] = len(t_mesh.vertices) - len(np.unique(t_mesh.vertices, axis=0))
        metrics["connected_components"] = len(t_mesh.split())
        metrics["is_watertight"] = t_mesh.is_watertight
        metrics["is_manifold"] = t_mesh.is_winding_consistent
        degenerate = np.sum(t_mesh.area_faces == 0)
        metrics["degenerate_faces"] = int(degenerate)

        if not metrics["parse_success"]:
            metrics["status"] = "PARSE_FAILED"
        elif not metrics["finite_coords"]:
            metrics["status"] = "GEOMETRY_INVALID"
        elif metrics["degenerate_faces"] > 0:
            metrics["status"] = "QUALITY_WARNING"
        else:
            metrics["status"] = "PARSE_VALID"
            
    except Exception as e:
        metrics["status"] = "PARSE_FAILED"
        metrics["error"] = str(e)

    return metrics

def run_validation():
    abs_valve_dir = os.path.expanduser("~/AI-Server/repos/TEER/valves/")
    files = glob.glob(os.path.join(abs_valve_dir, "*.stl"))
    
    if not files:
        print(f"No STL files found in {abs_valve_dir}")
        return

    print(f"Validating {len(files)} valve surfaces...")
    all_results = []
    for f in files:
        res = validate_mesh(f)
        all_results.append(res)
    
    output_file = os.path.expanduser("~/AI-Server/worktrees/TEER/geometry-audit/artifacts/geometry_validation.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if all_results:
        keys = all_results[0].keys()
        with open(output_file, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(all_results)
    
    parsed = sum(1 for r in all_results if r["parse_success"])
    finite = sum(1 for r in all_results if r["finite_coords"])
    warnings = sum(1 for r in all_results if r["status"] == "QUALITY_WARNING")
    watertight = sum(1 for r in all_results if r["is_watertight"])
    
    print("\n=== GEOMETRY VALIDATION SUMMARY ===")
    print(f"Meshes Parsed: {parsed}/{len(files)}")
    print(f"Finite Coordinates: {finite}/{len(files)}")
    print(f"Quality Warnings: {warnings}/{len(files)}")
    print(f"Watertight: {watertight}/{len(files)}")
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    run_validation()
