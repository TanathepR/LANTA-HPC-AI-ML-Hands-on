"""Dataset package.

`build_dataloaders` is the single entry point used by all scripts:
it reads the ``dataset`` section of the config and returns ready-to-use
PyTorch DataLoaders plus the list of class names.
"""

from typing import Any

from torch.utils.data import DataLoader

from datasets.mnist import build_mnist_dataloaders
from datasets.thfood100 import build_thfood_dataloaders


def build_dataloaders(
    config: dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Build DataLoaders for the dataset named in the config.

    Args:
        config: Full experiment configuration (parsed YAML).
        rank: This process's rank in a multi-GPU/multi-node run (see
            ``trainer.utils.init_distributed``). Ignored when
            ``world_size`` is 1.
        world_size: Total number of training processes. Currently only
            ``thfood100`` shards its training set across processes; passing
            ``world_size > 1`` for ``mnist`` trains on the full dataset
            redundantly on every process (no crash, just no speedup — Lab 1
            is small enough that this is not worth the added complexity).

    Returns:
        A tuple ``(train_loader, val_loader, test_loader, class_names)``.

    Raises:
        ValueError: If ``config["dataset"]["name"]`` is unknown.
    """
    name = str(config["dataset"]["name"]).lower()

    if name == "mnist":
        train_loader, test_loader, class_names = build_mnist_dataloaders(config)
        # MNIST has no official validation split, so the 10k test set is
        # used both for validation during training and for final testing.
        return train_loader, test_loader, test_loader, class_names

    if name == "thfood100":
        return build_thfood_dataloaders(config, rank=rank, world_size=world_size)

    raise ValueError(f"Unknown dataset '{name}'. Available: mnist, thfood100")
