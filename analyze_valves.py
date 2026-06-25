import os
import json
import numpy as np
import pyvista as pv
from pathlib import Path

def analyze_valve(path: Path):
    mesh = pv.read(str(path))
    if not isinstance(mesh, pv.PolyData):
        mesh = mesh.extract_surface()
        
    pts = mesh.points.copy()
    z_min, z_max = pts[:, 2].min(), pts[:, 2].max()
    z_len = z_max - z_min
    
    # Flip Z just like index.html and mitral_mock.py
    pts[:, 2] = z_max + z_min - pts[:, 2]
    
    # Filter tip points (Z < z_min + 0.35 * z_len)
    tip_pts = pts[pts[:, 2] < z_min + 0.35 * z_len]
    if len(tip_pts) < 10:
        # Fallback to all points if tip filter fails
        tip_pts = pts
        
    cx = float(tip_pts[:, 0].mean())
    cy = float(tip_pts[:, 1].mean())
    
    # Covariance and PCA in XY
    xy = tip_pts[:, :2]
    cov = np.cov(xy.T)
    evals, evecs = np.linalg.eigh(cov)
    
    # PC1 (larger eigenvalue) is the commissure line
    pc1 = evecs[:, -1]
    theta = float(np.arctan2(pc1[1], pc1[0]))
    
    # PC2 (smaller eigenvalue) is orthogonal opening axis
    pc2 = evecs[:, 0]
    # Ensure pc2 (opening direction) points towards positive Y (anterior side)
    if pc2[1] < 0:
        pc2 = -pc2
        
    return {
        "x_min": float(pts[:, 0].min()),
        "x_max": float(pts[:, 0].max()),
        "y_min": float(pts[:, 1].min()),
        "y_max": float(pts[:, 1].max()),
        "z_min": float(z_min),
        "z_max": float(z_max),
        "cx": cx,
        "cy": cy,
        "angle_rad": theta,
        "angle_deg": float(np.degrees(theta)),
        "v_comm": [float(pc1[0]), float(pc1[1])],
        "v_open": [float(pc2[0]), float(pc2[1])]
    }

def main():
    valves_dir = Path("valves")
    results = {}
    
    stl_files = sorted(list(valves_dir.glob("train_*-label_surface.stl")))
    print(f"Found {len(stl_files)} STL meshes in valves/ directory.")
    
    for f in stl_files:
        name = f.name.split("-")[0] # e.g. train_001
        print(f"Analyzing {name}...")
        try:
            res = analyze_valve(f)
            results[name] = res
            print(f"  Centroid: [{res['cx']:.2f}, {res['cy']:.2f}], Angle: {res['angle_deg']:.1f}°")
        except Exception as e:
            print(f"  Error analyzing {name}: {e}")
            
    with open("valves_metadata.json", "w") as out:
        json.dump(results, out, indent=2)
    print("Done! Metadata written to valves_metadata.json")

if __name__ == "__main__":
    main()
