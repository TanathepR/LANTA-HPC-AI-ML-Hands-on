#!/usr/bin/env python
"""Evaluate a trained checkpoint on the validation or test split.

Usage:
    python scripts/evaluate.py --checkpoint outputs/thfood_baseline/best.pt
    python scripts/evaluate.py --checkpoint outputs/mnist_gpu/best.pt --split val

The checkpoint stores its own config and class names, so nothing else is
needed. Results (accuracy, loss, per-class report) are printed and saved
as JSON next to the checkpoint.
"""

import argparse
import json
import sys
from pathlib import Path

# Make the repository root importable no matter where the script is run from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sklearn.metrics import classification_report

from datasets import build_dataloaders
from models import build_model
from trainer.engine import predict_all, validate
from trainer.losses import build_loss
from trainer.utils import get_device, load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument(
        "--split", default="test", choices=["val", "test"], help="Which split to evaluate."
    )
    parser.add_argument("--data-root", help="Override dataset.root from the checkpoint.")
    parser.add_argument("--batch-size", type=int, help="Override the evaluation batch size.")
    parser.add_argument("--device", default="cuda", help="cuda | cpu.")
    return parser.parse_args()


def main() -> None:
    """Rebuild the model from the checkpoint and evaluate it."""
    args = parse_args()
    device = get_device(args.device)

    # The checkpoint carries the exact config it was trained with.
    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    class_names: list[str] = checkpoint["class_names"]
    if args.data_root:
        config["dataset"]["root"] = args.data_root
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size

    print(f"Checkpoint: {args.checkpoint} (epoch {checkpoint['epoch']})")
    print(f"Model     : {config['model']['name']}, {len(class_names)} classes")

    # Rebuild the data pipeline and the model, then load the weights.
    _, val_loader, test_loader, _ = build_dataloaders(config)
    loader = val_loader if args.split == "val" else test_loader

    model = build_model(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)

    # Overall loss and accuracy.
    criterion = build_loss(config)
    stats = validate(model, loader, criterion, device, desc=args.split)
    print(f"\n{args.split} loss    : {stats['loss']:.4f}")
    print(f"{args.split} accuracy: {100 * stats['accuracy']:.2f}%\n")

    # Per-class precision / recall / F1 with scikit-learn.
    y_true, y_pred = predict_all(model, loader, device)
    print(
        classification_report(
            y_true, y_pred, target_names=class_names, digits=4, zero_division=0
        )
    )

    # Save results next to the checkpoint.
    results = {
        "checkpoint": str(args.checkpoint),
        "split": args.split,
        "loss": stats["loss"],
        "accuracy": stats["accuracy"],
        "per_class": classification_report(
            y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0
        ),
    }
    out_path = Path(args.checkpoint).parent / f"eval_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
