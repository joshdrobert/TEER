"""Core schemas shared across the TEER pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


Vector3 = Tuple[float, float, float]


class SegmentationLabel(int, Enum):
    """Semantic labels for mitral valve anatomy."""

    BACKGROUND = 0
    ANNULUS = 1
    ANTERIOR_LEAFLET = 2
    POSTERIOR_LEAFLET = 3
    COAPTATION_ZONE = 4


class MVSeg2023Label(int, Enum):
    """Native labels from the MICCAI 2023 MVSeg dataset."""

    BACKGROUND = 0
    POSTERIOR_LEAFLET = 1
    ANTERIOR_LEAFLET = 2


@dataclass
class DatasetResource:
    """Description of a source dataset and local storage contract."""

    name: str
    uri: str
    local_root: Path
    image_suffix: str
    label_suffix: str
    archive_name: Optional[str] = None
    split_archives: Dict[str, Path] = field(default_factory=dict)
    label_names: Dict[int, str] = field(default_factory=dict)


@dataclass
class VolumeMetadata:
    """Spatial and temporal metadata for an image volume."""

    subject_id: str
    spacing_mm: Vector3
    origin_mm: Vector3
    direction: Tuple[float, ...]
    frame_times_ms: List[float] = field(default_factory=list)
    cardiac_phase: Optional[str] = None


@dataclass
class NiftiPair:
    """Matched ultrasound volume and label paths."""

    image_path: Path
    label_path: Path
    subject_id: str


@dataclass
class DICOMFrame:
    """A single DICOM frame with metadata needed for 4D echo assembly."""

    sop_instance_uid: str
    trigger_time_ms: float
    frame_index: int
    array_shape: Tuple[int, int, int]


@dataclass
class AnonymizationRecord:
    """Track each field scrubbed or retained during HIPAA anonymization."""

    subject_hash: str
    removed_fields: List[str]
    retained_fields: List[str]
    operator: str


@dataclass
class PreprocessingConfig:
    """Voxel and temporal normalization configuration."""

    target_spacing_mm: Vector3 = (0.5, 0.5, 0.5)
    normalize_percentiles: Tuple[float, float] = (1.0, 99.0)
    target_frame_count: int = 20
    intensity_clip_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class UltrasoundAugmentationConfig:
    """Data augmentation parameters tailored to ultrasound physics."""

    speckle_scale: float = 0.08
    dropout_probability: float = 0.1
    elastic_alpha: float = 2.5
    elastic_sigma: float = 0.75


@dataclass
class SegmentationConfig:
    """Configuration for the 3D mitral segmentation network."""

    in_channels: int = 1
    out_channels: int = 5
    base_channels: int = 16
    attention_gates: bool = True
    temporal_latent_dim: int = 32


@dataclass
class MeshConfig:
    """Surface extraction and tetrahedralization parameters."""

    marching_cubes_isovalue: float = 0.5
    surface_smoothing_iterations: int = 30
    surface_decimation: float = 0.15
    tetrahedral_edge_length_mm: float = 0.6


@dataclass
class MitraClipSpecification:
    """Mechanical geometry inputs for a parametric MitraClip surrogate."""

    model_name: str = "MitraClip_G4_NT"
    arm_width_mm: float = 5.0
    grasp_length_mm: float = 9.0
    arm_thickness_mm: float = 1.2
    opening_angle_deg: float = 45.0


@dataclass
class FluidProperties:
    """Blood material parameters for the FSI domain."""

    density_kg_per_m3: float = 1060.0
    dynamic_viscosity_pa_s: float = 0.0035
    model: str = "carreau-yasuda"


@dataclass
class TissueProperties:
    """Leaflet constitutive parameters for a hyperelastic material model."""

    model: str = "Holzapfel-Gasser-Ogden"
    density_kg_per_m3: float = 1120.0
    fiber_angle_deg: float = 35.0
    c10_kpa: float = 60.0
    k1_kpa: float = 180.0
    k2_unitless: float = 7.5


@dataclass
class ClipPlacement:
    """Position and orientation of a MitraClip on the coaptation line."""

    x_mm: float
    y_mm: float
    z_mm: float
    theta_deg: float


@dataclass
class SimulationRequest:
    """FSI-ready configuration for a specific candidate intervention."""

    subject_id: str
    placements: List[ClipPlacement]
    fluid: FluidProperties
    tissue: TissueProperties
    cycle_duration_ms: float
    output_dir: Path


@dataclass
class StressMapSummary:
    """Compact summary of clinically relevant stress outputs."""

    max_von_mises_kpa: float
    percentile_95_kpa: float
    hotspot_coordinates_mm: List[Vector3] = field(default_factory=list)


@dataclass
class SimulationResult:
    """Key hemodynamic and structural outputs from the FSI solver."""

    regurgitant_volume_ml: float
    stress_summary: StressMapSummary
    convergence_iterations: int
    output_artifacts: Dict[str, Path]


@dataclass
class OptimizationWeights:
    """Clinical weighting factors for the objective function."""

    alpha: float = 0.6
    beta: float = 0.3
    gamma: float = 0.1


@dataclass
class CandidateOutcome:
    """Objective evaluation for one candidate clip configuration."""

    clip_count: int
    placements: List[ClipPlacement]
    objective_value: float
    regurgitant_volume_ml: float
    max_leaflet_stress_kpa: float
    stress_map_path: Optional[Path] = None
    simulation_artifact_dir: Optional[Path] = None


@dataclass
class ExportOverlayPoint:
    """Point projected into fusion-imaging coordinate space."""

    label: str
    coordinates_mm: Vector3
    orientation_deg: float


@dataclass
class ExportPayload:
    """Serializable payload for physician review and fusion overlay systems."""

    subject_id: str
    rank: int
    clip_count: int
    objective_value: float
    regurgitant_volume_ml: float
    max_leaflet_stress_kpa: float
    overlay_points: List[ExportOverlayPoint]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert nested dataclasses into a JSON-serializable dictionary."""
        return asdict(self)


@dataclass
class PipelineRunSummary:
    """Top-level return object for the orchestrated pipeline."""

    subject_id: str
    candidate_rankings: List[CandidateOutcome]
    export_payloads: List[ExportPayload]
    artifacts: Dict[str, Path]


@dataclass
class TEERPipelineConfig:
    """Aggregate configuration for all pipeline stages."""

    datasets: List[DatasetResource]
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    augmentations: UltrasoundAugmentationConfig = field(default_factory=UltrasoundAugmentationConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    mesh: MeshConfig = field(default_factory=MeshConfig)
    clip_spec: MitraClipSpecification = field(default_factory=MitraClipSpecification)
    fluid: FluidProperties = field(default_factory=FluidProperties)
    tissue: TissueProperties = field(default_factory=TissueProperties)
    objective: OptimizationWeights = field(default_factory=OptimizationWeights)
    top_k_recommendations: int = 3

    @classmethod
    def default(cls, workspace: Path) -> "TEERPipelineConfig":
        """Build a default configuration rooted in the current workspace."""
        mvseg_archives = {
            "train": workspace / "data" / "train.zip",
            "val": workspace / "data" / "val.zip",
            "test": workspace / "data" / "test.zip",
        }
        return cls(
            datasets=[
                DatasetResource(
                    name="MVSeg2023",
                    uri="local://data",
                    local_root=workspace / "data" / "mvseg2023",
                    image_suffix="-US.nii.gz",
                    label_suffix="-label.nii.gz",
                    split_archives=mvseg_archives,
                    label_names={
                        0: "background",
                        1: "posterior_leaflet",
                        2: "anterior_leaflet",
                    },
                ),
                DatasetResource(
                    name="MVAA2026",
                    uri="challenge://MVAA2026",
                    local_root=workspace / "data" / "mvaa2026",
                    image_suffix="_image.nii.gz",
                    label_suffix="_label.nii.gz",
                ),
            ]
        )

    @classmethod
    def mvseg2023_training(cls, workspace: Path) -> "TEERPipelineConfig":
        """Build a training-oriented config matching the native MVSeg2023 labels."""
        config = cls.default(workspace)
        config.segmentation = SegmentationConfig(out_channels=len(MVSeg2023Label))
        return config


def serialize_paths(mapping: Mapping[str, Path]) -> Dict[str, str]:
    """Render paths as strings for external serialization."""
    return {key: str(value) for key, value in mapping.items()}


def flatten_overlay_points(points: Sequence[ExportOverlayPoint]) -> List[Dict[str, Any]]:
    """Convert overlay points into simple dictionaries."""
    return [asdict(point) for point in points]
