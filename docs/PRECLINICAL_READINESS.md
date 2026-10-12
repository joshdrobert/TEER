# Preclinical Readiness Packet

## Geometry Source

All geometry sources must be patient-specific anatomical meshes conforming to the TEER anatomy contract. Meshes are validated using `validate_mesh` function which ensures:

- All triangle vertices are checked for finite coordinates (not only v0)
- Zero-face STLs return truthful invalid result without fabricating bounds
- Coordinate system type is "patient-specific anatomical"
- Coordinate system units are "millimeters (mm)"

## Observability

The following anatomical features are observable and must be present in validated meshes:

- Annulus
- Leaflets  
- Commissures
- Opening angle

Landmarks that must be detected:
- v_comm (commissure point)
- v_open (opening point)
- center_x (x-coordinate of center)
- center_y (y-coordinate of center)

## Units

All geometric measurements are in millimeters (mm) as specified in the anatomy contract.

## Coordinate Frames

Coordinate system type: patient-specific anatomical
Coordinate system units: millimeters (mm)
Origin is at patient-specific anatomical reference point.

## Validation Prerequisites

Before mesh validation, the following prerequisites must be met:

1. Mesh file exists and is readable
2. Mesh contains valid triangle data 
3. All vertices have finite coordinates
4. Mesh has non-zero face count
5. Observable anatomy features are present
6. Required landmarks are detected

## Validation Metrics

### Synthetic vs Validated Metrics

- **Synthetic metrics**: Generated for testing purposes, such as adversarial STLs
- **Validated metrics**: Actual measurements from verified patient anatomies

### Provenance

All validation results include:
- File size
- Vertex and face counts  
- Bounding box coordinates
- Surface area
- Coordinate range
- Mesh topology (watertight, manifold)
- Anatomy contract compliance status

## Acceptance Criteria

Meshes are considered valid if:
1. All coordinate checks pass
2. No degenerate faces exist
3. Anatomy contract is compliant
4. Observable features and landmarks are detected