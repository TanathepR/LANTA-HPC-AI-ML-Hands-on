"""Image transform pipelines.

A *transform* converts a raw PIL image into a normalized ``torch.Tensor``.
Training pipelines additionally apply **data augmentation** — random
modifications of the image (crops, flips, color changes) that make the
model generalize better because it never sees the exact same image twice.
"""

from typing import Any

from torchvision import transforms

# Channel statistics used for normalization: (pixel - mean) / std.
# MNIST values are computed over the MNIST training set;
# ImageNet values are the standard ones expected by pretrained backbones.
MNIST_MEAN, MNIST_STD = (0.1307,), (0.3081,)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_mnist_transforms(train: bool) -> transforms.Compose:
    """Transforms for MNIST (28x28 grayscale digits).

    MNIST is easy enough that no augmentation is needed, so the train
    and evaluation pipelines are identical.

    Args:
        train: Whether the pipeline is for the training split (unused,
            kept for symmetry with :func:`build_image_transforms`).

    Returns:
        A composed transform producing normalized 1x28x28 tensors.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(MNIST_MEAN, MNIST_STD),
        ]
    )


def build_image_transforms(
    img_size: int, augmentation: str, train: bool
) -> transforms.Compose:
    """Transforms for natural-image datasets such as THFOOD-100.

    Args:
        img_size: Final square input resolution (224 for ImageNet models).
        augmentation: ``"none"``, ``"basic"`` or ``"strong"``.
        train: If ``False``, augmentation is ignored and a deterministic
            resize + center-crop pipeline is returned.

    Returns:
        A composed transform producing normalized 3 x img_size x img_size tensors.

    Raises:
        ValueError: If ``augmentation`` is not a known level.
    """
    normalize = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

    # Evaluation: deterministic — resize the short side, crop the center.
    if not train or augmentation == "none":
        return transforms.Compose(
            [
                transforms.Resize(int(img_size / 0.875)),  # e.g. 224 -> 256
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if augmentation == "basic":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        )

    if augmentation == "strong":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                normalize,
                transforms.RandomErasing(p=0.25),
            ]
        )

    raise ValueError(
        f"Unknown augmentation level '{augmentation}'. Available: none, basic, strong"
    )


def build_eval_transform(config: dict[str, Any]) -> transforms.Compose:
    """Build the deterministic evaluation transform for a config.

    Used by ``predict.py`` so a single image is preprocessed exactly like
    the validation images were during training.

    Args:
        config: Full experiment configuration (parsed YAML).

    Returns:
        The evaluation transform for the configured dataset.
    """
    ds_cfg = config["dataset"]
    if str(ds_cfg["name"]).lower() == "mnist":
        return build_mnist_transforms(train=False)
    return build_image_transforms(
        img_size=int(ds_cfg.get("img_size", 224)), augmentation="none", train=False
    )
