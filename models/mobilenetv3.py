"""MobileNetV3 with transfer learning support (Howard et al., 2019).

MobileNetV3 is designed for efficiency: it reaches good accuracy with far
fewer FLOPs than ResNet-18, which makes it interesting to compare in the
benchmark script (accuracy vs. speed trade-off).
"""

from torch import nn
from torchvision.models import (
    MobileNet_V3_Large_Weights,
    MobileNet_V3_Small_Weights,
    mobilenet_v3_large,
    mobilenet_v3_small,
)


def build_mobilenetv3(
    num_classes: int,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    variant: str = "large",
) -> nn.Module:
    """Construct a MobileNetV3 adapted to ``num_classes`` outputs.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet weights for the backbone.
        freeze_backbone: If ``True``, only the new classification head trains.
        variant: ``"large"`` (default, more accurate) or ``"small"`` (faster).

    Returns:
        The adapted MobileNetV3.

    Raises:
        ValueError: If ``variant`` is not ``"large"`` or ``"small"``.
    """
    if variant == "large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        model = mobilenet_v3_large(weights=weights)
    elif variant == "small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        model = mobilenet_v3_small(weights=weights)
    else:
        raise ValueError(f"Unknown MobileNetV3 variant '{variant}'. Use: large, small")

    # Freeze BEFORE replacing the head so the new layer stays trainable.
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # The classifier is a Sequential; its last element is the final Linear.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model
