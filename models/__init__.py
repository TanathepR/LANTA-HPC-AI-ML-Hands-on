"""Model package.

`build_model` is the single entry point used by all scripts: it reads the
``model`` section of the config and returns a ready-to-train ``nn.Module``.

Available model names:
    lenet            Small CNN trained from scratch (Lab 1, MNIST)
    resnet18         ImageNet-pretrained ResNet-18 (Lab 2)
    mobilenetv3      ImageNet-pretrained MobileNetV3 (Lab 2)
    efficientnet_b0  ImageNet-pretrained EfficientNet-B0 (Lab 2)
"""

from typing import Any

from torch import nn

from models.efficientnet import build_efficientnet_b0
from models.lenet import build_lenet
from models.mobilenetv3 import build_mobilenetv3
from models.resnet18 import build_resnet18

MODEL_NAMES = ["lenet", "resnet18", "mobilenetv3", "efficientnet_b0"]


def build_model(config: dict[str, Any], num_classes: int) -> nn.Module:
    """Build the model named in the config.

    Args:
        config: Full experiment configuration (parsed YAML).
        num_classes: Number of output classes. Taken from the dataset
            (``len(class_names)``) so config and data can never disagree.

    Returns:
        The constructed model (still on CPU — the Trainer moves it).

    Raises:
        ValueError: If ``config["model"]["name"]`` is unknown.
    """
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()
    pretrained = bool(model_cfg.get("pretrained", False))
    freeze_backbone = bool(model_cfg.get("freeze_backbone", False))

    if name == "lenet":
        return build_lenet(
            num_classes=num_classes,
            in_channels=int(model_cfg.get("in_channels", 1)),
        )
    if name == "resnet18":
        return build_resnet18(num_classes, pretrained, freeze_backbone)
    if name == "mobilenetv3":
        return build_mobilenetv3(
            num_classes,
            pretrained,
            freeze_backbone,
            variant=str(model_cfg.get("variant", "large")),
        )
    if name in ("efficientnet_b0", "efficientnet"):
        return build_efficientnet_b0(num_classes, pretrained, freeze_backbone)

    raise ValueError(f"Unknown model '{name}'. Available: {', '.join(MODEL_NAMES)}")
