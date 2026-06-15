"""Clinical ingestion, anonymization, and preprocessing pipeline."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pydicom
import SimpleITK as sitk

from .exceptions import AnonymizationError, CorruptedSpatialHeaderError
from .schemas import AnonymizationRecord, DICOMFrame, PreprocessingConfig, VolumeMetadata


@dataclass(slots=True)
class ClinicalVolume:
    """Container for a preprocessed 3D or 4D TEE volume."""

    voxel_data: np.ndarray
    metadata: VolumeMetadata
    anonymization: AnonymizationRecord


class DICOMIngestor:
    """Read multi-frame TEE studies and assemble volume metadata."""

    REQUIRED_TAGS = ("PixelSpacing", "ImagePositionPatient", "ImageOrientationPatient")

    def read_series(self, dicom_paths: Sequence[Path]) -> Tuple[List[pydicom.Dataset], List[DICOMFrame]]:
        """Load DICOM frames and validate required spatial metadata."""
        datasets: List[pydicom.Dataset] = []
        frames: List[DICOMFrame] = []
        for index, path in enumerate(sorted(dicom_paths)):
            ds = pydicom.dcmread(str(path))
            for tag in self.REQUIRED_TAGS:
                if not hasattr(ds, tag):
                    raise CorruptedSpatialHeaderError(f"Missing spatial tag {tag} in {path}")
            datasets.append(ds)
            pixel_array = ds.pixel_array
            frames.append(
                DICOMFrame(
                    sop_instance_uid=str(ds.SOPInstanceUID),
                    trigger_time_ms=float(getattr(ds, "TriggerTime", index)),
                    frame_index=index,
                    array_shape=tuple(int(dim) for dim in pixel_array.shape[-3:]),
                )
            )
        return datasets, frames


class HIPAAAnonymizer:
    """Remove direct identifiers while recording each transformed field."""

    DEFAULT_SCRUB_FIELDS = (
        "PatientName",
        "PatientID",
        "PatientBirthDate",
        "PatientSex",
        "AccessionNumber",
        "InstitutionName",
        "ReferringPhysicianName",
    )

    def anonymize(self, dataset: pydicom.Dataset, operator: str) -> AnonymizationRecord:
        """Scrub configured DICOM fields and return an audit record."""
        removed_fields: List[str] = []
        retained_fields: List[str] = []
        patient_id = str(getattr(dataset, "PatientID", "unknown"))
        for field_name in self.DEFAULT_SCRUB_FIELDS:
            if hasattr(dataset, field_name):
                setattr(dataset, field_name, "")
                removed_fields.append(field_name)
            else:
                retained_fields.append(field_name)
        if not removed_fields:
            raise AnonymizationError("No PHI fields were scrubbed; refusing to continue.")
        subject_hash = hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:16]
        return AnonymizationRecord(
            subject_hash=subject_hash,
            removed_fields=removed_fields,
            retained_fields=retained_fields,
            operator=operator,
        )


class ClinicalPreprocessor:
    """Resample, normalize, and temporally align clinical echo volumes."""

    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config

    def preprocess(self, datasets: Sequence[pydicom.Dataset], anonymization: AnonymizationRecord) -> ClinicalVolume:
        """Convert DICOM datasets into a normalized isotropic volume."""
        images = [sitk.GetImageFromArray(ds.pixel_array.astype(np.float32)) for ds in datasets]
        image = sitk.JoinSeries(images) if len(images) > 1 else images[0]
        image = self._set_spatial_metadata(image, datasets[0])
        resampled = self._resample(image)
        array = sitk.GetArrayFromImage(resampled).astype(np.float32)
        array = self._normalize(array)
        aligned = self._temporal_align(array)
        metadata = VolumeMetadata(
            subject_id=anonymization.subject_hash,
            spacing_mm=tuple(float(v) for v in resampled.GetSpacing()[:3]),
            origin_mm=tuple(float(v) for v in resampled.GetOrigin()[:3]),
            direction=tuple(float(v) for v in resampled.GetDirection()),
            frame_times_ms=[float(getattr(ds, "TriggerTime", idx)) for idx, ds in enumerate(datasets)],
            cardiac_phase="aligned_systole_diastole",
        )
        return ClinicalVolume(voxel_data=aligned, metadata=metadata, anonymization=anonymization)

    def _set_spatial_metadata(self, image: sitk.Image, dataset: pydicom.Dataset) -> sitk.Image:
        pixel_spacing = [float(v) for v in dataset.PixelSpacing]
        spacing = tuple(pixel_spacing + [1.0])
        image.SetSpacing(spacing)
        image.SetOrigin(tuple(float(v) for v in dataset.ImagePositionPatient))
        image.SetDirection(tuple(float(v) for v in dataset.ImageOrientationPatient) + (0.0, 0.0, 1.0))
        return image

    def _resample(self, image: sitk.Image) -> sitk.Image:
        target_spacing = self.config.target_spacing_mm
        original_spacing = image.GetSpacing()
        original_size = image.GetSize()
        out_size = [
            int(round(size * spacing / target))
            for size, spacing, target in zip(original_size[:3], original_spacing[:3], target_spacing)
        ]
        if image.GetDimension() > 3:
            out_size.append(image.GetSize()[-1])
        resampler = sitk.ResampleImageFilter()
        resampler.SetOutputSpacing(target_spacing)
        resampler.SetSize(out_size)
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        return resampler.Execute(image)

    def _normalize(self, volume: np.ndarray) -> np.ndarray:
        lo, hi = np.percentile(volume, self.config.normalize_percentiles)
        clipped = np.clip(volume, lo, hi)
        normalized = (clipped - lo) / max(hi - lo, 1e-6)
        return np.clip(normalized, *self.config.intensity_clip_range)

    def _temporal_align(self, volume: np.ndarray) -> np.ndarray:
        if volume.ndim < 4:
            return volume
        frame_axis = 0
        target_frames = self.config.target_frame_count
        source_frames = volume.shape[frame_axis]
        indices = np.linspace(0, source_frames - 1, num=target_frames).round().astype(int)
        return volume[indices]
