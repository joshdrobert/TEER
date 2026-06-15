"""Dataset acquisition and PyTorch data loading for mitral valve imaging."""

from __future__ import annotations

import hashlib
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import gaussian_filter, map_coordinates
from torch.utils.data import Dataset

from .exceptions import DatasetAcquisitionError
from .schemas import DatasetResource, NiftiPair, UltrasoundAugmentationConfig


@dataclass
class AcquisitionReport:
    """Result of dataset synchronization."""

    dataset_name: str
    local_root: Path
    downloaded_files: List[Path]


@dataclass
class DatasetSplitSummary:
    """Counts and paths for one prepared dataset split."""

    name: str
    root: Path
    pair_count: int
    image_suffix: str
    label_suffix: str


class HuggingFaceDatasetFetcher:
    """Download and unpack research datasets distributed via Hugging Face."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = cache_dir

    def fetch(self, resource: DatasetResource) -> AcquisitionReport:
        """Download the configured archive or raise a structural error."""
        from huggingface_hub import hf_hub_download

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


class LocalArchiveDatasetFetcher:
    """Import manually downloaded archives, such as Zenodo dataset exports."""

    def fetch(self, resource: DatasetResource) -> AcquisitionReport:
        """Copy or extract a local archive into the configured dataset root."""
        source = self._resolve_source(resource.uri)
        resource.local_root.mkdir(parents=True, exist_ok=True)

        if source.is_dir():
            return AcquisitionReport(dataset_name=resource.name, local_root=source, downloaded_files=[])

        if not source.exists():
            raise DatasetAcquisitionError(f"Dataset archive does not exist: {source}")

        copied = resource.local_root / source.name
        if source.resolve() != copied.resolve():
            shutil.copy2(source, copied)

        if zipfile.is_zipfile(copied):
            self._unzip(copied, resource.local_root)
        else:
            raise DatasetAcquisitionError(f"Unsupported local dataset archive format: {copied}")

        return AcquisitionReport(dataset_name=resource.name, local_root=resource.local_root, downloaded_files=[copied])

    @staticmethod
    def _resolve_source(uri: str) -> Path:
        if uri.startswith("file://"):
            return Path(uri.removeprefix("file://")).expanduser()
        if uri.startswith("manual://"):
            raise DatasetAcquisitionError(
                "This dataset must be downloaded manually first; set DatasetResource.uri to a file:// archive path."
            )
        return Path(uri).expanduser()

    @staticmethod
    def _unzip(archive_path: Path, output_dir: Path) -> None:
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


class MVSeg2023DatasetPreparer:
    """Prepare the bundled MVSeg2023 train/validation/test archives."""

    EXPECTED_SPLITS = ("train", "val", "test")

    def __init__(self, resource: DatasetResource) -> None:
        self.resource = resource
        self.pair_finder = MVSegPairFinder()

    def prepare(self) -> List[DatasetSplitSummary]:
        """Extract split archives as needed and return discovered pair counts."""
        self.resource.local_root.mkdir(parents=True, exist_ok=True)
        for split_name in self.EXPECTED_SPLITS:
            self._extract_split(split_name)
        return self.summarize()

    def summarize(self) -> List[DatasetSplitSummary]:
        """Discover prepared split contents without modifying the filesystem."""
        summaries: List[DatasetSplitSummary] = []
        for split_name in self.EXPECTED_SPLITS:
            split_root = self.resource.local_root / split_name
            pairs = self.pair_finder.discover_pairs(
                split_root,
                image_suffix=self.resource.image_suffix,
                label_suffix=self.resource.label_suffix,
            )
            summaries.append(
                DatasetSplitSummary(
                    name=split_name,
                    root=split_root,
                    pair_count=len(pairs),
                    image_suffix=self.resource.image_suffix,
                    label_suffix=self.resource.label_suffix,
                )
            )
        return summaries

    def discover_pairs(self, split_name: str) -> List[NiftiPair]:
        """Return image/label pairs for one prepared split."""
        if split_name not in self.EXPECTED_SPLITS:
            raise DatasetAcquisitionError(f"Unknown MVSeg2023 split: {split_name}")
        return self.pair_finder.discover_pairs(
            self.resource.local_root / split_name,
            image_suffix=self.resource.image_suffix,
            label_suffix=self.resource.label_suffix,
        )

    def discover_all_pairs(self) -> Dict[str, List[NiftiPair]]:
        """Return image/label pairs grouped by split name."""
        return {split_name: self.discover_pairs(split_name) for split_name in self.EXPECTED_SPLITS}

    def _extract_split(self, split_name: str) -> None:
        split_root = self.resource.local_root / split_name
        if self._split_is_prepared(split_root):
            return

        archive_path = self.resource.split_archives.get(split_name)
        if archive_path is None:
            raise DatasetAcquisitionError(f"Missing archive path for MVSeg2023 split: {split_name}")
        if not archive_path.exists():
            raise DatasetAcquisitionError(f"Missing MVSeg2023 archive: {archive_path}")

        try:
            with zipfile.ZipFile(archive_path, "r") as handle:
                handle.extractall(self.resource.local_root)
        except zipfile.BadZipFile as exc:
            raise DatasetAcquisitionError(f"Corrupted MVSeg2023 archive: {archive_path}") from exc

        if not self._split_is_prepared(split_root):
            raise DatasetAcquisitionError(f"Archive did not produce expected split directory: {split_root}")

    def _split_is_prepared(self, split_root: Path) -> bool:
        if not split_root.exists():
            return False
        return any(split_root.rglob(f"*{self.resource.image_suffix}")) and any(
            split_root.rglob(f"*{self.resource.label_suffix}")
        )


class MitralTEEVolumePairFinder:
    """Discover 3D TEE mitral volume/mask pairs across common archive layouts."""

    IMAGE_TOKENS = ("image", "images", "img", "volume", "volumes", "us", "ultrasound", "tee")
    LABEL_TOKENS = ("label", "labels", "mask", "masks", "seg", "segs", "segmentation", "segmentations")
    NIFTI_SUFFIXES = (".nii", ".nii.gz")

    def discover_pairs(self, root: Path) -> List[NiftiPair]:
        """Pair NIfTI volumes and masks without assuming one challenge naming scheme."""
        nifti_paths = sorted(path for path in root.rglob("*") if path.is_file() and self._is_nifti(path))
        image_paths = [path for path in nifti_paths if not self._looks_like_label(path)]
        label_paths = [path for path in nifti_paths if self._looks_like_label(path)]

        labels_by_subject: Dict[str, Path] = {}
        for label_path in label_paths:
            labels_by_subject[self._subject_key(label_path)] = label_path

        pairs: List[NiftiPair] = []
        for image_path in image_paths:
            subject_id = self._subject_key(image_path)
            label_path = labels_by_subject.get(subject_id)
            if label_path is not None:
                pairs.append(NiftiPair(image_path=image_path, label_path=label_path, subject_id=subject_id))

        if not pairs:
            raise DatasetAcquisitionError(
                f"No paired 3D TEE NIfTI volumes found under {root}. "
                "Expected image and label files with matching subject identifiers."
            )
        return pairs

    @classmethod
    def _is_nifti(cls, path: Path) -> bool:
        return any(str(path).endswith(suffix) for suffix in cls.NIFTI_SUFFIXES)

    @classmethod
    def _looks_like_label(cls, path: Path) -> bool:
        tokens = cls._path_tokens(path)
        return any(token in tokens for token in cls.LABEL_TOKENS)

    @classmethod
    def _subject_key(cls, path: Path) -> str:
        stem = cls._nifti_stem(path)
        parts = [part for part in cls._split_tokens(stem) if part not in cls.IMAGE_TOKENS + cls.LABEL_TOKENS]
        return "_".join(parts) if parts else stem

    @classmethod
    def _path_tokens(cls, path: Path) -> set[str]:
        tokens: set[str] = set()
        for part in path.parts:
            tokens.update(cls._split_tokens(part))
        return tokens

    @staticmethod
    def _split_tokens(value: str) -> List[str]:
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        return [part for part in normalized.split("_") if part]

    @staticmethod
    def _nifti_stem(path: Path) -> str:
        name = path.name
        if name.endswith(".nii.gz"):
            return name[:-7]
        return path.stem


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
        image = self._load_image_nifti(pair.image_path)
        label = self._load_label_nifti(pair.label_path)
        if self.augmentor is not None:
            image, label = self.augmentor(image, label)
        return {
            "image": torch.from_numpy(image[None, ...].astype(np.float32)),
            "label": torch.from_numpy(label.astype(np.int64)),
            "subject_hash": torch.tensor(self._subject_hash(pair.subject_id), dtype=torch.int64),
        }

    @staticmethod
    def _load_image_nifti(path: Path) -> np.ndarray:
        array = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        if array.max() > array.min():
            array = (array - array.min()) / (array.max() - array.min())
        return array

    @staticmethod
    def _load_label_nifti(path: Path) -> np.ndarray:
        return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.int64)

    @staticmethod
    def _subject_hash(subject_id: str) -> int:
        digest = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
        return int(digest[:15], 16)
