"""THFOOD-100 — Thai food image classification (100 classes).

The dataset is loaded with ``torchvision.datasets.ImageFolder``, which
expects one sub-directory per class. Two layouts are supported:

Pre-split (the full dataset, as handed out for Lab 2)::

    <root>/
        train/
            pad_thai/      img001.jpg ...
            tom_yum/       img001.jpg ...
            ...
        val/
            pad_thai/      ...
        test/              (optional — falls back to val if missing)
            ...

Flat / unsplit (e.g. the ``THFOOD-100.sample`` preview, which ships only a
handful of images per class with no train/val/test folders)::

    <root>/
        0/    sample_1.jpg ...
        1/    sample_1.jpg ...
        ...
        class_labels.csv   (optional: Directory_Name -> Dish_Name_English)

For the flat layout, each class's images are deterministically split into
train/val/test (stratified per class, not shuffled) since there is no
split information on disk. If a ``class_labels.csv`` sits next to the
class folders, its ``Dish_Name_English`` column is used for readable class
names instead of the raw directory names.

Either way, the label <-> index mapping is deterministic.
"""

import csv
from pathlib import Path
from typing import Any

from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder

from datasets.transforms import build_image_transforms

_LAYOUT_HELP = """
Expected THFOOD-100 layout (torchvision ImageFolder), either:

    {root}/
        train/<class_name>/*.jpg
        val/<class_name>/*.jpg
        test/<class_name>/*.jpg   (optional)

or a flat/unsplit layout (e.g. the THFOOD-100.sample preview):

    {root}/
        <class_name>/*.jpg
        ...

Ask your instructor for the dataset location, or check the shared
project directory on LANTA, then set `dataset.root` in your config.
"""


def _stratified_split(
    targets: list[int], val_ratio: float = 0.2, test_ratio: float = 0.25
) -> tuple[list[int], list[int], list[int]]:
    """Deterministically split each class's sample indices into train/val/test.

    Used for the flat (unsplit) layout, where a directory has one folder per
    class with no split information on disk (e.g. the tiny sample/preview,
    or a full download that was not pre-split) — the last images of each
    class (by the dataset's sorted file order) are reserved for val/test in
    roughly ``val_ratio``/``test_ratio`` proportions, mirroring the ~20/25%
    validation/test split THFOOD-100 itself was built with. A fixed count
    (e.g. "last 2 images") would badly under-use classes with hundreds of
    images, so the split scales with class size instead, with at least one
    image per split whenever the class is large enough to spare it.
    """
    by_class: dict[int, list[int]] = {}
    for idx, label in enumerate(targets):
        by_class.setdefault(label, []).append(idx)

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for indices in by_class.values():
        n = len(indices)
        if n >= 4:
            n_val = max(1, round(n * val_ratio))
            n_test = max(1, round(n * test_ratio))
            while n_val + n_test >= n:  # keep at least one training image
                if n_test >= n_val:
                    n_test -= 1
                else:
                    n_val -= 1
            n_train = n - n_val - n_test
            train_idx += indices[:n_train]
            val_idx += indices[n_train : n_train + n_val]
            test_idx += indices[n_train + n_val :]
        elif n == 3:
            train_idx.append(indices[0])
            val_idx.append(indices[1])
            test_idx.append(indices[2])
        elif n == 2:
            train_idx.append(indices[0])
            val_idx.append(indices[1])
            test_idx.append(indices[1])
        else:
            train_idx += indices
            val_idx += indices
            test_idx += indices
    return train_idx, val_idx, test_idx


def _readable_class_names(root: Path, classes: list[str]) -> list[str]:
    """Translate ImageFolder directory names via ``class_labels.csv``, if present."""
    csv_path = root / "class_labels.csv"
    if not csv_path.is_file():
        return classes
    mapping: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            mapping[row["Directory_Name"]] = row["Dish_Name_English"]
    return [mapping.get(c, c) for c in classes]


def build_thfood_dataloaders(
    config: dict[str, Any],
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Create THFOOD-100 train/val/test DataLoaders.

    Args:
        config: Full experiment configuration (parsed YAML).
        rank: This process's rank, for multi-GPU/multi-node training via
            ``torchrun`` (see ``trainer.utils.init_distributed``). Ignored
            when ``world_size`` is 1.
        world_size: Total number of training processes. When greater than
            1, the training set is sharded across processes with a
            ``DistributedSampler`` so each GPU trains on a disjoint slice
            of the data instead of redundantly repeating the whole set.
            Validation/test loaders are left un-sharded — only the main
            process evaluates on them (see ``Trainer.fit``).

    Returns:
        A tuple ``(train_loader, val_loader, test_loader, class_names)``.

    Raises:
        FileNotFoundError: If ``root`` has neither a pre-split (train/val)
            nor a flat (class-subfolder) layout.
    """
    ds_cfg = config["dataset"]
    root = Path(ds_cfg.get("root", "./data/thfood100"))
    img_size = int(ds_cfg.get("img_size", 224))
    augmentation = str(ds_cfg.get("augmentation", "basic"))
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(ds_cfg.get("num_workers", 4))
    pin_memory = bool(ds_cfg.get("pin_memory", True))

    # Augmentation is applied only to the training split; validation and
    # test use a deterministic pipeline so results are comparable.
    train_tf = build_image_transforms(img_size, augmentation, train=True)
    eval_tf = build_image_transforms(img_size, "none", train=False)

    train_set: Dataset
    val_set: Dataset
    test_set: Dataset

    if (root / "train").is_dir():
        if not (root / "val").is_dir():
            raise FileNotFoundError(
                f"THFOOD-100 split not found: {root / 'val'}\n"
                + _LAYOUT_HELP.format(root=root)
            )
        train_set = ImageFolder(root / "train", transform=train_tf)
        val_set = ImageFolder(root / "val", transform=eval_tf)

        if (root / "test").is_dir():
            test_set = ImageFolder(root / "test", transform=eval_tf)
        else:
            print(f"NOTE: {root / 'test'} not found — using the val split as test.")
            test_set = val_set

        class_names = train_set.classes
    elif root.is_dir() and any(d.is_dir() for d in root.iterdir()):
        # Flat layout: <root>/<class>/*.jpg with no train/val/test split
        # (e.g. the THFOOD-100.sample preview) — stratify a split per class.
        print(
            f"NOTE: {root} has no train/val split — treating it as a flat, "
            "unsplit layout and deriving a per-class train/val/test split."
        )
        base_train = ImageFolder(root, transform=train_tf)
        base_eval = ImageFolder(root, transform=eval_tf)
        train_idx, val_idx, test_idx = _stratified_split(base_train.targets)
        train_set = Subset(base_train, train_idx)
        val_set = Subset(base_eval, val_idx)
        test_set = Subset(base_eval, test_idx)

        class_names = _readable_class_names(root, base_train.classes)
    else:
        raise FileNotFoundError(
            f"THFOOD-100 data not found under {root}\n"
            + _LAYOUT_HELP.format(root=root)
        )

    train_sampler = None
    if world_size > 1:
        # drop_last keeps every process's shard the same size each epoch,
        # so all processes run the same number of batches — required for
        # DDP's per-batch gradient all-reduce, which every process must
        # enter together or the run deadlocks.
        train_sampler = DistributedSampler(
            train_set, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True
        )
    # Keeps worker processes alive across epochs instead of respawning them
    # every epoch (fork + reimport torchvision/PIL each time) — pure
    # overhead reduction, no effect on what gets loaded. No-op when
    # num_workers is 0 (there are no worker processes to keep alive).
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    return train_loader, val_loader, test_loader, class_names
