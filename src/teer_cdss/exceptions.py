"""Custom exception hierarchy for the TEER CDSS."""


class TEERPipelineError(Exception):
    """Base class for TEER decision-support failures."""


class DatasetAcquisitionError(TEERPipelineError):
    """Raised when a research dataset cannot be located, downloaded, or parsed."""


class CorruptedSpatialHeaderError(TEERPipelineError):
    """Raised when image orientation or voxel spacing metadata is unusable."""


class AnonymizationError(TEERPipelineError):
    """Raised when protected health information cannot be safely removed."""


class SegmentationInferenceError(TEERPipelineError):
    """Raised when segmentation inference or tensor preparation fails."""


class MeshGenerationError(TEERPipelineError):
    """Raised when a watertight surface or volumetric mesh cannot be created."""


class FSINonConvergenceError(TEERPipelineError):
    """Raised when the external FSI solver fails to converge."""


class ContactResolutionError(TEERPipelineError):
    """Raised when clip-leaflet contact conditions cannot be resolved."""


class OptimizationSearchError(TEERPipelineError):
    """Raised when the combinatorial search process cannot produce candidates."""


class ExportSerializationError(TEERPipelineError):
    """Raised when physician-facing payloads cannot be serialized."""
