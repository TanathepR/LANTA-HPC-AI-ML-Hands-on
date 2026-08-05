"""ResNet-18 with transfer learning support (He et al., 2015).

Transfer learning recipe (used in Lab 2):
    1. Load a backbone pretrained on ImageNet (1.28M images, 1000 classes).
    2. Replace the final fully connected layer with a new one sized for
       our dataset (100 Thai food classes).
    3. Fine-tune: either the whole network, or only the new head
       (``freeze_backbone: true``).
"""

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(
    num_classes: int, pretrained: bool = True, freeze_backbone: bool = False
) -> nn.Module:
    """Construct a ResNet-18 adapted to ``num_classes`` outputs.

    Args:
        num_classes: Number of output classes.
        pretrained: Load ImageNet weights for the backbone.
        freeze_backbone: If ``True``, backbone weights are frozen and only
            the newly created classification head is trained.

    Returns:
        The adapted ResNet-18.
    """
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)

    # Freeze BEFORE replacing the head: the new layer below is created
    # afterwards, so it keeps requires_grad=True and stays trainable.
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    # Replace the 1000-class ImageNet head with our own.
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
