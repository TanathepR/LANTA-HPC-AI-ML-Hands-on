"""MNIST — 70,000 handwritten digits (60k train / 10k test), 28x28 grayscale.

The dataset downloads itself automatically on first use (~12 MB).
On clusters where compute nodes have no internet access, download it
once on a login node first:

    python datasets/download.py --dataset mnist --root ./data
"""

from typing import Any

from torch.utils.data import DataLoader
from torchvision import datasets

from datasets.transforms import build_mnist_transforms


def build_mnist_dataloaders(
    config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Create MNIST train and test DataLoaders.

    Args:
        config: Full experiment configuration (parsed YAML).

    Returns:
        A tuple ``(train_loader, test_loader, class_names)`` where
        ``class_names`` is ``["0", ..., "9"]``.
    """
    ds_cfg = config["dataset"]
    root = str(ds_cfg.get("root", "./data"))
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(ds_cfg.get("num_workers", 4))
    pin_memory = bool(ds_cfg.get("pin_memory", True))

    train_set = datasets.MNIST(
        root, train=True, download=True, transform=build_mnist_transforms(train=True)
    )
    test_set = datasets.MNIST(
        root, train=False, download=True, transform=build_mnist_transforms(train=False)
    )

    # shuffle=True only for training: each epoch sees batches in a new order,
    # which improves convergence. Evaluation order does not matter.
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_names = [str(digit) for digit in range(10)]
    return train_loader, test_loader, class_names
