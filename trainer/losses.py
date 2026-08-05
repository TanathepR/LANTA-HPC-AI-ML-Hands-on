"""Loss-function factory.

Only classification losses are needed for these labs, but the factory
pattern shows how a config key maps to a PyTorch object.
"""

from typing import Any

from torch import nn


def build_loss(config: dict[str, Any]) -> nn.Module:
    """Build the loss function named in the config.

    Args:
        config: Full experiment configuration (parsed YAML).

    Returns:
        The loss module (a callable: ``loss = criterion(logits, targets)``).

    Raises:
        ValueError: If ``config["loss"]["name"]`` is unknown.
    """
    loss_cfg = config.get("loss", {}) or {}
    name = str(loss_cfg.get("name", "cross_entropy")).lower()

    if name in ("cross_entropy", "crossentropy", "ce"):
        # label_smoothing > 0 softens the one-hot targets: the model is asked
        # for e.g. 90% confidence instead of 100%, which reduces overfitting.
        return nn.CrossEntropyLoss(
            label_smoothing=float(loss_cfg.get("label_smoothing", 0.0))
        )

    raise ValueError(f"Unknown loss '{name}'. Available: cross_entropy")
