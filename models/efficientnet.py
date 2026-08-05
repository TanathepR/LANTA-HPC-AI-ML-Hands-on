"""EfficientNet-B0 with transfer learning support (Tan & Le, 2019).

EfficientNet scales depth, width, and input resolution together. B0 is
the smallest member of the family and a strong baseline for fine-grained
image classification such as THFOOD-100.
"""

from torch import nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


def build_efficientnet_b0(
    num_classes: int, pretrained: bool = True, freeze_backbone: bool = False
) -> nn.Module:
    """Construct an EfficientNet-B0 adapted to ``num_classes`` outputs.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet weights for the backbone.
        freeze_backbone: If ``True``, only the new classification head trains.

    Returns:
        The adapted EfficientNet-B0.
    """
    weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = efficientnet_b0(weights=weights)

    # Freeze BEFORE replacing the head so the new layer stays trainable.
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # classifier = [Dropout, Linear] — replace the final Linear.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model
