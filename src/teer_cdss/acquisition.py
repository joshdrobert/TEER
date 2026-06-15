"""Dataset acquisition and PyTorch data loading for mitral valve imaging."""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset

from .exceptions import DatasetAcquisitionError
from .schemas import DatasetResource, NiftiPair, UltrasoundAugmentationConfig


@dataclass(slots=True)
class AcquisitionReport:
    """Result of dataset synchronization."""

    dataset_name: str
    local_root: Path
    downloaded_files: List[Path]


class HuggingFaceDatasetFetcher:
    """Download and unpack research datasets distributed via Hugging Face."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def fetch(self, resource: DatasetResource) -> AcquisitionReport:
        """Download the configured archive or raise a structural error."""
        resource.local_root.mkdir(parents=True, exist_ok=True)
        downloaded: List[Path] = []

        if not resource.uri.startswith("hf://"):
            raise DatasetAcquisitionError(f"Unsupported dataset URI for automatic fetch: {resource.uri}")

        if resource.archive_name is None:
            raise DatasetAcquisitionError(f"Missing archive_name for {resource.name}")

        _, repo_id = resource.uri.split("hf://", maxsplit=1)
        archive_path = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=resource.archive_name,
                repo_type="dataset",
                local_dir=self.cache_dir,
            )
        )
        downloaded.append(archive_path)
        self._unzip(archive_path, resource.local_root)
        return AcquisitionReport(dataset_name=resource.name, local_root=resource.local_root, downloaded_files=downloaded)

    @staticmethod
    def _unzip(archive_path: Path, output_dir: Path) -> None:
        """Extract a zip archive into the destination directory."""
        try:
            with zipfile.ZipFile(archive_path, "r") as handle:
                handle.extractall(output_dir)
        except zipfile.BadZipFile as exc:
            raise DatasetAcquisitionError(f"Corrupted archive: {archive_path}") from exc


class MVSegPairFinder:
    """Pair image and label volumes using challenge naming conventions."""

    def discover_pairs(self, root: Path, image_suffix: str, label_suffix: str) -> List[NiftiPair]:
        """Match `.nii` or `.nii.gz` volumes to their corresponding labels."""
        images = sorted(root.rglob(f"*{image_suffix}"))
        pairs: List[NiftiPair] = []
        for image_path in images:
            subject_id = image_path.name.replace(image_suffix, "")
            label_path = image_path.with_name(f"{subject_id}{label_suffix}")
            if label_path.exists():
                pairs.append(NiftiPair(image_path=image_path, label_path=label_path, subject_id=subject_id))
        if not pairs:
            raise DatasetAcquisitionError(f"No paired NIfTI volumes found under {root}")
        return pairs


class UltrasoundAugmentor:
    """Physics-informed transforms for volumetric ultrasound."""

    def __init__(self, config: UltrasoundAugmentationConfig) -> None:
        self.config = config

    def __call__(self, volume: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply speckle, dropout, and elastic warp consistently to the sample."""
        volume = self._inject_speckle(volume)
        volume = self._simulate_dropout(volume)
        volume, label = self._elastic_deformation(volume, label)
        return volume, label

    def _inject_speckle(self, volume: np.ndarray) -> np.ndarray:
        noise = np.random.normal(loc=0.0, scale=self.config.speckle_scale, size=volume.shape)
        return np.clip(volume + volume * noise, 0.0, 1.0)

    def _simulate_dropout(self, volume: np.ndarray) -> np.ndarray:
        mask = np.random.binomial(1, 1.0 - self.config.dropout_probability, size=volume.shape)
        return volume * mask

    def _elastic_deformation(self, volume: np.ndarray, label: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        shape = volume.shape
        random_state = np.random.RandomState()
        displacements = [
            gaussian_filter(
                (random_state.rand(*shape) * 2.0 - 1.0),
                self.config.elastic_sigma,
                mode="reflect",
            )
            * self.config.elastic_alpha
            for _ in range(3)
        ]
        coordinates = np.meshgrid(
            np.arange(shape[0]),
            np.arange(shape[1]),
            np.arange(shape[2]),
            indexing="ij",
        )
        warped_coords = [coord + disp for coord, disp in zip(coordinates, displacements)]
        warped_volume = map_coordinates(volume, warped_coords, order=1, mode="reflect")
        warped_label = map_coordinates(label, warped_coords, order=0, mode="nearest")
        return warped_volume, warped_label


class MitralValveVolumeDataset(Dataset[Dict[str, torch.Tensor]]):
    """PyTorch dataset for matched 3D TEE volumes and segmentation labels."""

    def __init__(
        self,
        pairs: Sequence[NiftiPair],
        augmentor: Optional[Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]] = None,
    ) -> None:
        self.pairs = list(pairs)
        self.augmentor = augmentor

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        pair = self.pairs[index]
        image = self._load_nifti(pair.image_path)
        label = self._load_nifti(pair.label_path)
        if self.augmentor is not None:
            image, label = self.augmentor(image, label)
        return {
            "image": torch.from_numpy(image[None, ...].astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "subject_hash": torch.tensor(self._subject_hash(pair.subject_id), dtype=torch.int64),
        }

    @staticmethod
    def _load_nifti(path: Path) -> np.ndarray:
        array = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        if array.max() > array.min():
            array = (array - array.min()) / (array.max() - array.min())
        return array

    @staticmethod
    def _subject_hash(subject_id: str) -> int:
        digest = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
        return int(digest[:15], 16)
