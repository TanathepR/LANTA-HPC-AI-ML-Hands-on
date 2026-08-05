"""LeNet-5 — the classic small CNN (LeCun et al., 1998), modernized with ReLU.

Trained **from scratch** (no pretraining) on MNIST in Lab 1. With only
~60k parameters it is small enough to train quickly even on a CPU, which
makes the CPU-vs-GPU comparison practical.
"""

import torch
from torch import nn


class LeNet5(nn.Module):
    """LeNet-5 for 28x28 inputs.

    Architecture:
        conv(5x5) -> ReLU -> maxpool -> conv(5x5) -> ReLU -> maxpool
        -> flatten -> fc(120) -> fc(84) -> fc(num_classes)

    Attributes:
        features: The convolutional feature extractor.
        classifier: The fully connected classification head.
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        """Initialize the network.

        Args:
            num_classes: Number of output classes.
            in_channels: Input image channels (1 for grayscale MNIST).
        """
        super().__init__()
        self.features = nn.Sequential(
            # padding=2 keeps 28x28 spatial size after the 5x5 convolution
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 28x28 -> 14x14
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 10x10 -> 5x5
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute class logits for a batch of images.

        Args:
            x: Input batch of shape (N, in_channels, 28, 28).

        Returns:
            Logits of shape (N, num_classes).
        """
        return self.classifier(self.features(x))


def build_lenet(num_classes: int = 10, in_channels: int = 1) -> LeNet5:
    """Construct a LeNet-5 model.

    Args:
        num_classes: Number of output classes.
        in_channels: Input image channels.

    Returns:
        A freshly initialized LeNet-5.
    """
    return LeNet5(num_classes=num_classes, in_channels=in_channels)
