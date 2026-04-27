"""
Exploratory analysis utilities for datasets.

Use these once when setting up a new dataset to derive normalisation stats
and check class balance. Results are then hardcoded into the dataset's config.

Functions:
    compute_mean_std(dataset)    — channel-wise mean and std
    class_distribution(dataset)  — sample count per class label
"""

import torch
from torch.utils.data import Dataset


def compute_mean_std(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute channel-wise mean and std over an entire dataset.

    Iterates once over all samples without loading everything into memory at once.
    Assumes each sample x has shape (C, ...) — i.e. channels-first.

    Returns:
        mean: (C,) tensor
        std:  (C,) tensor
    """
    # Two-pass: first accumulate sum and sum-of-squares, then derive stats.
    n = 0
    channel_sum: torch.Tensor | None = None
    channel_sum_sq: torch.Tensor | None = None

    for i in range(len(dataset)):  # type: ignore[arg-type]
        x, _ = dataset[i]
        x = x.float()
        c = x.shape[0]

        if channel_sum is None:
            channel_sum = torch.zeros(c)
            channel_sum_sq = torch.zeros(c)

        pixels = x.view(c, -1)  # (C, H*W)
        channel_sum += pixels.sum(dim=1)
        channel_sum_sq += pixels.pow(2).sum(dim=1)
        n += pixels.shape[1]

    assert channel_sum is not None and channel_sum_sq is not None, "Dataset is empty."

    mean = channel_sum / n
    std = (channel_sum_sq / n - mean.pow(2)).sqrt()
    return mean, std


def class_distribution(dataset: Dataset) -> dict[int, int]:
    """Count samples per integer class label.

    Returns:
        dict mapping class index → sample count, sorted by class index.
    """
    counts: dict[int, int] = {}

    for i in range(len(dataset)):  # type: ignore[arg-type]
        _, y = dataset[i]
        label = int(y) if isinstance(y, (int, torch.Tensor)) else y
        counts[label] = counts.get(label, 0) + 1

    return dict(sorted(counts.items()))
