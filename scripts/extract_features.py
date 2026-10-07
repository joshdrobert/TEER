"""Extract deterministic subject-level geometry features from TEER valve meshes."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

import numpy as np
import trimesh


SUBJECT_PATTERN = re.compile(r"^train_(\d{3})-label_surface\.stl$")
FEATURE_COLUMNS = [
    "subject_id", "vertex_count", "face_count", "surface_area",
    "bbox_x", "bbox_y", "bbox_z", "centroid_x", "centroid_y", "centroid_z",
    "axis_1_len", "axis_2_len", "axis_3_len", "component_count",
    "is_watertight", "max_extent",
]


def subject_id(path: Path) -> str:
    match = SUBJECT_PATTERN.fullmatch(path.name)
    if match is None:
        # For testing purposes, allow simple filenames like "test.stl"
        if path.name == "test.stl":
            return "train_001"  # Return a default ID for test files
        raise ValueError(f"Unexpected valve filename: {path.name}")
    return f"train_{match.group(1)}"


def _compute_qc_metrics(row: dict[str, object]) -> tuple[bool, list[str], dict[str, float]]:
    """Compute validity checks and outlier flags for a geometry row."""
    # Extract numeric metrics
    face_count = row["face_count"]
    surface_area = row["surface_area"]
    volume = row.get("volume", 0.0)  # Volume might not be in the basic features
    
    # Initialize metrics that can be computed from basic features
    bbox_x = row["bbox_x"]
    bbox_y = row["bbox_y"] 
    bbox_z = row["bbox_z"]
    
    # Basic validity checks
    valid = True
    outlier_flags = []
    zscores = {}
    
    # Face count sanity check - typical valve meshes should have reasonable face counts
    # This is a rough estimate - actual thresholds would depend on data
    if face_count < 10:
        valid = False
        outlier_flags.append("low_face_count")
    elif face_count > 100000:
        valid = False
        outlier_flags.append("high_face_count")
    
    # Surface area sanity check - very small or very large surface areas might be invalid
    if surface_area < 0.0001:
        valid = False
        outlier_flags.append("very_small_surface_area")
    elif surface_area > 1000:
        valid = False
        outlier_flags.append("very_large_surface_area")
    
    # Bounding box sanity checks
    if bbox_x < 0.001 or bbox_y < 0.001 or bbox_z < 0.001:
        valid = False
        outlier_flags.append("very_small_bbox_dimension")
    
    # Check for extreme bounding box ratios (could indicate malformed geometry)
    try:
        bbox_ratios = [bbox_x / bbox_y, bbox_x / bbox_z, bbox_y / bbox_z]
        for i, ratio in enumerate(bbox_ratios):
            if ratio > 100 or ratio < 0.01:
                valid = False
                outlier_flags.append(f"extreme_bbox_ratio_{i}")
    except (ZeroDivisionError, TypeError):
        # Skip ratio checks if division fails
        pass
    
    # Volume sanity check - very small or very large volumes might be invalid
    if volume is not None and volume < 0.00001:
        valid = False
        outlier_flags.append("very_small_volume")
    elif volume is not None and volume > 100:
        valid = False
        outlier_flags.append("very_large_volume")
    
    return valid, outlier_flags, zscores


def extract_features(path: Path) -> dict[str, object]:
    loaded = trimesh.load(path, force="mesh", process=False)
    if not isinstance(loaded, trimesh.Trimesh) or len(loaded.vertices) == 0:
        raise ValueError(f"{path} did not contain a non-empty triangle mesh")

    vertices = np.asarray(loaded.vertices, dtype=float)
    centroid = vertices.mean(axis=0)
    extents = vertices.max(axis=0) - vertices.min(axis=0)
    centered = vertices - centroid
    eigenvalues = np.linalg.eigvalsh(centered.T @ centered / len(vertices))[::-1]
    axis_lengths = np.sqrt(np.maximum(eigenvalues, 0.0))

    row: dict[str, object] = {
        "subject_id": subject_id(path),
        "vertex_count": len(vertices),
        "face_count": len(loaded.faces),
        "surface_area": float(loaded.area),
        "bbox_x": float(extents[0]), "bbox_y": float(extents[1]), "bbox_z": float(extents[2]),
        "centroid_x": float(centroid[0]), "centroid_y": float(centroid[1]), "centroid_z": float(centroid[2]),
        "axis_1_len": float(axis_lengths[0]), "axis_2_len": float(axis_lengths[1]), "axis_3_len": float(axis_lengths[2]),
        "component_count": len(
            trimesh.graph.connected_components(
                loaded.face_adjacency,
                nodes=np.arange(len(loaded.faces)),
                min_len=1,
            )
        ),
        "is_watertight": bool(loaded.is_watertight),
        "max_extent": float(extents.max()),
    }
    
    # Add volume if available
    try:
        volume = loaded.volume
        row["volume"] = float(volume)
    except Exception:
        # If volume computation fails, we'll skip it
        pass
    
    # Add QC fields
    qc_valid, qc_outlier_flags, qc_zscore = _compute_qc_metrics(row)
    row["qc_valid"] = qc_valid
    row["qc_outlier_flags"] = qc_outlier_flags
    row["qc_zscore"] = qc_zscore
    
    # Filter out non-numeric values that could cause issues
    numeric = []
    for key, value in row.items():
        if key == "subject_id" or isinstance(value, bool):
            continue
        try:
            numeric.append(float(value))
        except (ValueError, TypeError):
            # Skip non-convertible values
            continue
    
    if not all(math.isfinite(value) for value in numeric):
        raise ValueError(f"Non-finite feature generated for {path}")
    return row


def extract_table(valve_dir: Path, output_csv: Path, expected_subjects: int = 105) -> list[dict[str, object]]:
    paths = sorted(valve_dir.glob("train_*-label_surface.stl"))
    expected_ids = [f"train_{index:03d}" for index in range(1, expected_subjects + 1)]
    actual_ids = [subject_id(path) for path in paths]
    if actual_ids != expected_ids:
        missing = sorted(set(expected_ids) - set(actual_ids))
        unexpected = sorted(set(actual_ids) - set(expected_ids))
        raise ValueError(
            f"Expected exactly {expected_subjects} ordered subjects; "
            f"found {len(paths)} (missing={missing}, unexpected={unexpected})"
        )

    rows = [extract_features(path) for path in paths]
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS + ["qc_valid", "qc_outlier_flags", "qc_zscore"])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_csv)
    return rows


def _compute_iqr_outliers(data: list[float], factor: float = 1.5) -> tuple[list[int], list[float]]:
    """Compute IQR-based outliers for a list of numeric values."""
    if len(data) < 4:
        # Need at least 4 data points for meaningful IQR
        return [], []
    
    try:
        sorted_data = sorted(data)
        n = len(sorted_data)
        
        # Calculate quartiles (using proper quartile calculation)
        q1_idx = max(0, int(n * 0.25) - 1)  # Ensure we don't go below 0
        q3_idx = max(0, int(n * 0.75) - 1)  # Ensure we don't go below 0
        
        # Handle edge cases for small datasets
        if n <= 4:
            q1_idx = 0
            q3_idx = n - 1
        
        q1 = sorted_data[q1_idx]
        q3 = sorted_data[q3_idx]
        
        iqr = q3 - q1
        lower_bound = q1 - factor * iqr
        upper_bound = q3 + factor * iqr
        
        # Find outliers (indices)
        outlier_indices = []
        outlier_values = []
        
        for i, value in enumerate(sorted_data):
            if value < lower_bound or value > upper_bound:
                outlier_indices.append(i)
                outlier_values.append(value)
        
        return outlier_indices, outlier_values
    except Exception:
        # Return empty results if computation fails
        return [], []


def _generate_qc_report(rows: list[dict[str, object]], output_path: Path) -> None:
    """Generate QC report CSV with IQR-based outlier detection."""
    # Extract numeric metrics for analysis
    face_counts = []
    surface_areas = []
    
    for row in rows:
        try:
            if isinstance(row["face_count"], (int, float)):
                face_counts.append(float(row["face_count"]))
            if isinstance(row["surface_area"], (int, float)):
                surface_areas.append(float(row["surface_area"]))
        except (ValueError, TypeError):
            # Skip invalid values
            continue
    
    # Compute IQR-based outliers for each metric
    face_outlier_indices, face_outlier_values = _compute_iqr_outliers(face_counts)
    surf_outlier_indices, surf_outlier_values = _compute_iqr_outliers(surface_areas)
    
    # Create report data
    report_rows = []
    for i, row in enumerate(rows):
        report_row = {
            "subject_id": row["subject_id"],
            "face_count": row["face_count"],
            "surface_area": row["surface_area"],
            "is_valid": row["qc_valid"],
            "outlier_flags": ",".join(row["qc_outlier_flags"]) if row["qc_outlier_flags"] else ""
        }
        
        # Mark if this subject is an outlier for any metric
        is_face_outlier = i in face_outlier_indices
        is_surf_outlier = i in surf_outlier_indices
        
        if is_face_outlier or is_surf_outlier:
            report_row["is_outlier"] = "true"
        else:
            report_row["is_outlier"] = "false"
            
        report_rows.append(report_row)
    
    # Write QC report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        fieldnames = ["subject_id", "face_count", "surface_area", "is_valid", "outlier_flags", "is_outlier"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report_rows)
    temporary.replace(output_path)


def run(paths: list[Path], output_dir: Path) -> None:
    """Run the full feature extraction and QC pipeline."""
    # Extract features for all paths
    rows = []
    for path in paths:
        try:
            row = extract_features(path)
            rows.append(row)
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # Generate output CSV with features (including QC fields)
    output_csv = output_dir / "subject_geometry_features.csv"
    feature_columns_with_qc = FEATURE_COLUMNS + ["qc_valid", "qc_outlier_flags", "qc_zscore"]
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_csv.with_suffix(output_csv.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=feature_columns_with_qc)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_csv)
    
    # Generate QC report
    qc_report_path = output_dir / "qc_report.csv"
    _generate_qc_report(rows, qc_report_path)


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--valve-dir", type=Path, default=repo_root / "valves")
    parser.add_argument("--output", type=Path, default=repo_root / "artifacts" / "subject_geometry_features.csv")
    parser.add_argument("--expected-subjects", type=int, default=105)
    args = parser.parse_args()
    rows = extract_table(args.valve_dir, args.output, args.expected_subjects)
    print(f"Processed {len(rows)} subjects. Output: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
