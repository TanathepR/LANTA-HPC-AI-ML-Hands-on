"""Metric helpers used during training and evaluation."""

import torch


class AverageMeter:
    """Track the running average of a value (e.g. loss over an epoch).

    Values can come from batches of different sizes, so each update is
    weighted by the number of samples it represents.

    Example:
        >>> meter = AverageMeter()
        >>> meter.update(0.5, n=32)   # batch of 32 with mean loss 0.5
        >>> meter.update(0.3, n=16)   # batch of 16 with mean loss 0.3
        >>> round(meter.avg, 4)
        0.4333
    """

    def __init__(self) -> None:
        self.sum: float = 0.0
        self.count: int = 0

    def reset(self) -> None:
        """Forget everything seen so far."""
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        """Add a new (already averaged) value observed over ``n`` samples.

        Args:
            value: Mean value over the batch (e.g. batch loss).
            n: Number of samples the value was averaged over.
        """
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        """The weighted average of everything seen so far."""
        return self.sum / max(self.count, 1)


def accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy: fraction of samples whose highest logit is correct.

    Args:
        outputs: Model logits of shape (N, num_classes).
        targets: Ground-truth class indices of shape (N,).

    Returns:
        Accuracy in [0, 1].
    """
    predictions = outputs.argmax(dim=1)
    return (predictions == targets).float().mean().item()


def topk_accuracy(outputs: torch.Tensor, targets: torch.Tensor, k: int = 5) -> float:
    """Top-k accuracy: the true class is among the k highest logits.

    Args:
        outputs: Model logits of shape (N, num_classes).
        targets: Ground-truth class indices of shape (N,).
        k: How many top predictions to consider.

    Returns:
        Top-k accuracy in [0, 1].
    """
    k = min(k, outputs.size(1))
    topk_predictions = outputs.topk(k, dim=1).indices  # (N, k)
    correct = topk_predictions.eq(targets.unsqueeze(1))  # (N, k) bools
    return correct.any(dim=1).float().mean().item()
