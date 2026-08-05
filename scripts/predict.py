#!/usr/bin/env python
"""Predict the class of a single image with a trained checkpoint.

Usage:
    python scripts/predict.py \\
        --model outputs/thfood_baseline/best.pt \\
        --image sample.jpg
"""

import argparse
import sys
from pathlib import Path

# Make the repository root importable no matter where the script is run from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from PIL import Image

from datasets.transforms import build_eval_transform
from models import build_model
from trainer.utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Classify a single image.")
    parser.add_argument("--model", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument("--image", required=True, help="Path to the image file.")
    parser.add_argument("--topk", type=int, default=5, help="How many guesses to show.")
    parser.add_argument("--device", default="cuda", help="cuda | cpu.")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    """Load the checkpoint, preprocess the image, and print top-k guesses."""
    args = parse_args()
    device = get_device(args.device)

    checkpoint = load_checkpoint(args.model, device)
    config = checkpoint["config"]
    class_names: list[str] = checkpoint["class_names"]

    model = build_model(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    # Preprocess exactly like the validation images during training.
    # MNIST models expect grayscale ("L"); everything else expects RGB.
    is_grayscale = str(config["dataset"]["name"]).lower() == "mnist"
    image = Image.open(args.image).convert("L" if is_grayscale else "RGB")
    tensor = build_eval_transform(config)(image).unsqueeze(0).to(device)

    # Forward pass; softmax turns logits into probabilities.
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    top = probabilities.topk(min(args.topk, len(class_names)))

    print(f"\nPredictions for {args.image}:\n")
    print(f"{'rank':<6}{'class':<30}{'probability':>12}")
    print("-" * 48)
    for rank, (prob, idx) in enumerate(zip(top.values, top.indices), start=1):
        print(f"{rank:<6}{class_names[idx]:<30}{100 * prob.item():>11.2f}%")


if __name__ == "__main__":
    main()
