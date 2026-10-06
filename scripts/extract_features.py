"""Extract deterministic subject-level geometry features from TEER valve meshes."""

from __future__ import annotations

import argparse
import csv
import math
import re
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
        raise ValueError(f"Unexpected valve filename: {path.name}")
    return f"train_{match.group(1)}"


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
    numeric = [value for key, value in row.items() if key != "subject_id" and not isinstance(value, bool)]
    if not all(math.isfinite(float(value)) for value in numeric):
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
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_csv)
    return rows


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
