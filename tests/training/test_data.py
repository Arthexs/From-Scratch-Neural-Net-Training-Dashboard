from unittest.mock import patch

import pytest
import torch
from torch.utils.data import DataLoader

from training.data import BaseDataset, MNISTDataset, normalize, one_hot

# ── Synthetic stand-in for torchvision MNIST ───────────────────────────────────


class _FakeMNIST:
    def __init__(self, n: int = 100) -> None:
        self._x = torch.rand(n, 1, 28, 28)
        self._y = torch.randint(0, 10, (n,))

    def __len__(self) -> int:
        return len(self._x)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self._x[idx], int(self._y[idx].item())


@pytest.fixture
def fake_mnist():
    return _FakeMNIST(n=100)


# ── one_hot ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "labels,num_classes",
    [
        ([0], 10),
        ([0, 1, 5, 9], 10),
        ([0, 1, 2], 3),
    ],
)
def test_one_hot_shape(labels, num_classes):
    out = one_hot(torch.tensor(labels), num_classes)
    assert out.shape == (len(labels), num_classes)


@pytest.mark.parametrize(
    "labels,num_classes",
    [
        ([0, 3, 7], 10),
        ([0, 1, 2], 3),
    ],
)
def test_one_hot_values(labels, num_classes):
    out = one_hot(torch.tensor(labels), num_classes)
    for i, cls in enumerate(labels):
        assert out[i, cls].item() == 1.0
        assert out[i].sum().item() == 1.0


def test_one_hot_dtype():
    assert one_hot(torch.tensor([0, 1]), 3).dtype == torch.float32


# ── normalize ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (0.5, 0.5), (0.1307, 0.3081)])
def test_normalize_zero_at_mean(mean, std):
    assert normalize(torch.tensor([mean]), mean, std).item() == pytest.approx(0.0)


@pytest.mark.parametrize("mean,std", [(0.0, 1.0), (0.5, 0.25)])
def test_normalize_unit_at_mean_plus_std(mean, std):
    assert normalize(torch.tensor([mean + std]), mean, std).item() == pytest.approx(1.0)


def test_normalize_preserves_shape():
    x = torch.rand(3, 1, 28, 28)
    assert normalize(x, 0.5, 0.5).shape == x.shape


# ── BaseDataset ────────────────────────────────────────────────────────────────


def test_base_dataset_is_abstract():
    with pytest.raises(TypeError):
        BaseDataset()  # type: ignore[abstract]


def test_mnist_dataset_is_subclass():
    assert issubclass(MNISTDataset, BaseDataset)


# ── MNISTDataset.get_loaders ───────────────────────────────────────────────────


@pytest.mark.parametrize("val_split,expect_val", [(0.0, False), (0.2, True), (0.5, True)])
def test_get_loaders_val_presence(val_split, expect_val, fake_mnist):
    with patch("training.data.datasets.MNIST", return_value=fake_mnist):
        train_loader, val_loader = MNISTDataset(val_split=val_split).get_loaders(batch_size=16)
    assert isinstance(train_loader, DataLoader)
    assert (val_loader is not None) == expect_val


@pytest.mark.parametrize("val_split", [0.2, 0.4])
def test_get_loaders_sizes_sum_to_total(val_split, fake_mnist):
    n = len(fake_mnist)
    with patch("training.data.datasets.MNIST", return_value=fake_mnist):
        train_loader, val_loader = MNISTDataset(val_split=val_split).get_loaders(batch_size=n)
    x_train, _ = next(iter(train_loader))
    x_val, _ = next(iter(val_loader))
    assert x_train.shape[0] + x_val.shape[0] == n


@pytest.mark.parametrize("batch_size", [8, 16, 32])
def test_get_loaders_batch_size(batch_size, fake_mnist):
    with patch("training.data.datasets.MNIST", return_value=fake_mnist):
        train_loader, _ = MNISTDataset(val_split=0.2).get_loaders(batch_size=batch_size)
    x, _ = next(iter(train_loader))
    assert x.shape[0] == batch_size


def test_get_loaders_feature_shape(fake_mnist):
    with patch("training.data.datasets.MNIST", return_value=fake_mnist):
        train_loader, _ = MNISTDataset(val_split=0.2).get_loaders(batch_size=16)
    x, y = next(iter(train_loader))
    assert x.shape[1:] == (1, 28, 28)
    assert y.shape[1] == 10


def test_get_loaders_label_dtype(fake_mnist):
    with patch("training.data.datasets.MNIST", return_value=fake_mnist):
        train_loader, _ = MNISTDataset(val_split=0.2).get_loaders(batch_size=16)
    _, y = next(iter(train_loader))
    assert y.dtype == torch.float32


# ── MNISTDataset.get_test_loader ───────────────────────────────────────────────


def test_get_test_loader_returns_dataloader(fake_mnist):
    with patch("training.data.datasets.MNIST", return_value=fake_mnist):
        loader = MNISTDataset(val_split=0.2).get_test_loader(batch_size=16)
    assert isinstance(loader, DataLoader)


def test_get_test_loader_feature_shape(fake_mnist):
    with patch("training.data.datasets.MNIST", return_value=fake_mnist):
        loader = MNISTDataset(val_split=0.2).get_test_loader(batch_size=16)
    x, y = next(iter(loader))
    assert x.shape[1:] == (1, 28, 28)
    assert y.shape[1] == 10


@pytest.mark.parametrize("batch_size", [8, 16])
def test_get_test_loader_batch_size(batch_size, fake_mnist):
    with patch("training.data.datasets.MNIST", return_value=fake_mnist):
        loader = MNISTDataset(val_split=0.2).get_test_loader(batch_size=batch_size)
    x, _ = next(iter(loader))
    assert x.shape[0] == batch_size
