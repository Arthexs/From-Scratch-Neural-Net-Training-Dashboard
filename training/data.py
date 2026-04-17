"""
Dataset classes and data utilities.

Each dataset is registered in DATASETS and follows the same pattern as layers/losses:
    @DATASETS.register("name", config=NameConfig)
    class NameDataset:
        def __init__(self, **kwargs): ...
        def get_loaders(self, val_split, batch_size) -> tuple[DataLoader, DataLoader]: ...

Utility functions:
    one_hot(y, num_classes)  — from-scratch one-hot encoding
    normalize(x, mean, std)  — from-scratch per-channel normalisation
"""

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from model.configs import MNISTConfig
from model.registry import DATASETS


# ── Utilities ──────────────────────────────────────────────────────────────────

def one_hot(y: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Encode a 1-D integer label tensor as a float one-hot matrix.

    Args:
        y:           (N,) integer class indices
        num_classes: total number of classes

    Returns:
        (N, num_classes) float tensor
    """
    out = torch.zeros(y.shape[0], num_classes)
    out.scatter_(1, y.unsqueeze(1), 1.0)
    return out


def normalize(x: torch.Tensor, mean: float | torch.Tensor, std: float | torch.Tensor) -> torch.Tensor:
    """Subtract mean and divide by std, broadcast over any leading batch dimensions.

    Args:
        x:    input tensor of any shape
        mean: scalar or tensor broadcastable to x
        std:  scalar or tensor broadcastable to x

    Returns:
        normalised tensor, same shape as x
    """
    return (x - mean) / std


# ── Dataset registry ───────────────────────────────────────────────────────────

@DATASETS.register("mnist", config=MNISTConfig)
class MNISTDataset:
    """MNIST handwritten digits (60 000 train / 10 000 test, 1×28×28, 10 classes).

    Applies ToTensor() then normalises with the configured mean and std.
    Labels are one-hot encoded as float tensors.
    """

    num_classes: int = 10

    def __init__(self, **kwargs: object) -> None:
        self._cfg = MNISTConfig(**kwargs)

    def get_loaders(self, val_split: float, batch_size: int) -> tuple[DataLoader, DataLoader]:
        """Download (if needed), split, and wrap in DataLoaders.

        Returns:
            (train_loader, val_loader)
        """
        raw = datasets.MNIST(root="data", train=True, download=True, transform=transforms.ToTensor())

        n_val = int(len(raw) * val_split)
        n_train = len(raw) - n_val
        train_split, val_split_data = random_split(raw, [n_train, n_val])

        train_set = _NormOneHotDataset(train_split, self._cfg.mean, self._cfg.std, self.num_classes)
        val_set = _NormOneHotDataset(val_split_data, self._cfg.mean, self._cfg.std, self.num_classes)

        train_loader: DataLoader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader: DataLoader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

        return train_loader, val_loader


# ── Internal wrapper ───────────────────────────────────────────────────────────

class _NormOneHotDataset(Dataset):
    """Wraps a raw image dataset: normalises x and one-hot encodes y on the fly."""

    def __init__(self, base: Dataset, mean: float, std: float, num_classes: int) -> None:
        self._base = base
        self._mean = mean
        self._std = std
        self._num_classes = num_classes

    def __len__(self) -> int:
        return len(self._base)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self._base[idx]
        x = normalize(x, self._mean, self._std)
        y = one_hot(torch.tensor([y]), self._num_classes).squeeze(0)
        return x, y
