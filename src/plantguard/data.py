"""Archive-backed PyTorch dataset and shared image preprocessing.

The dataset stays in its source ZIP so data preparation does not create a
second multi-gigabyte copy. Each DataLoader worker opens its own archive handle.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

try:
    from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    import torch
except ImportError:  # Keep inspection utilities importable without PyTorch installed.
    torch = None
    DataLoader = WeightedRandomSampler = None
    Dataset = object


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = (224, 224)


def make_transform(training: bool = False, custom_transform: Optional[Callable] = None):
    """Create the shared RGB transform; callers may inject a custom transform.

    Validation and test use resize/center-crop/normalization only. Training
    adds a mild scale crop, horizontal flip, and restrained color jitter.
    Leaf symptom orientation is not label-defining, so horizontal
    flips are acceptable; vertical flips are omitted.
    """
    if custom_transform is not None:
        return custom_transform
    try:
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Install torch and torchvision to create transforms.") from exc

    ops = []
    if training:
        ops.extend([
            transforms.RandomResizedCrop(INPUT_SIZE, scale=(0.95, 1.0), ratio=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.10, contrast=0.10, saturation=0.08, hue=0.02),
        ])
    else:
        ops.extend([transforms.Resize(256), transforms.CenterCrop(INPUT_SIZE)])
    ops.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return transforms.Compose(ops)


class ArchiveImageDataset(Dataset):
    """Read one member per row from a split CSV without extracting the ZIP."""

    def __init__(self, split_csv: str | Path, archive_path: str | Path,
                 class_to_idx: dict[str, int], transform: Optional[Callable] = None):
        if torch is None:
            raise RuntimeError("Install PyTorch to use ArchiveImageDataset.")
        import pandas as pd

        self.rows = pd.read_csv(split_csv).to_dict("records")
        self.archive_path = str(archive_path)
        self.class_to_idx = class_to_idx
        self.transform = transform or make_transform(training=False)
        self._archive = None

    def __len__(self) -> int:
        return len(self.rows)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_archive"] = None
        return state

    def _zip(self):
        if self._archive is None:
            self._archive = zipfile.ZipFile(self.archive_path, "r")
        return self._archive

    def __getitem__(self, index: int):
        row = self.rows[index]
        with self._zip().open(row["path"]) as source:
            image = Image.open(io.BytesIO(source.read())).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, self.class_to_idx[row["class_name"]]


def class_weights(labels: list[str], class_to_idx: dict[str, int]):
    """Return inverse-frequency weights normalized to a mean of one."""
    if torch is None:
        raise RuntimeError("Install PyTorch to compute tensor class weights.")
    counts = torch.zeros(len(class_to_idx), dtype=torch.float32)
    for label in labels:
        counts[class_to_idx[label]] += 1
    weights = len(labels) / (len(class_to_idx) * counts.clamp_min(1))
    return weights


def make_weighted_sampler(labels: list[str], class_to_idx: dict[str, int]):
    """Build a replacement sampler with equal expected class contribution."""
    if torch is None:
        raise RuntimeError("Install PyTorch to create a weighted sampler.")
    counts = torch.zeros(len(class_to_idx), dtype=torch.float64)
    encoded = [class_to_idx[name] for name in labels]
    for idx in encoded:
        counts[idx] += 1
    sample_weights = torch.tensor([1.0 / counts[idx] for idx in encoded], dtype=torch.double)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def make_loader(dataset, batch_size: int, training: bool = False, num_workers: int = 0,
                sampler=None):
    """Create a configurable loader; validation remains sequential by default."""
    if torch is None:
        raise RuntimeError("Install PyTorch to create a DataLoader.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=training and sampler is None,
                      sampler=sampler, num_workers=num_workers, pin_memory=torch.cuda.is_available())
