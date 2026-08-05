#!/usr/bin/env python
"""Export a trained checkpoint to a deployable format.

Usage:
    python scripts/export.py --checkpoint outputs/thfood_baseline/best.pt
    python scripts/export.py --checkpoint outputs/mnist_gpu/best.pt --format onnx

TorchScript (default) produces a self-contained ``.pt`` file that runs
without the Python model source. ONNX requires the ``onnx`` package
(``pip install onnx``).
"""

import argparse
import sys
from pathlib import Path

# Make the repository root importable no matter where the script is run from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from models import build_model
from trainer.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export a trained checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to best.pt or last.pt.")
    parser.add_argument(
        "--format", default="torchscript", choices=["torchscript", "onnx"],
        help="Export format.",
    )
    parser.add_argument("--output", help="Output path (default: next to the checkpoint).")
    return parser.parse_args()


def main() -> None:
    """Rebuild the model from the checkpoint and export it."""
    args = parse_args()
    device = torch.device("cpu")  # export on CPU: portable and always available

    checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    class_names: list[str] = checkpoint["class_names"]

    model = build_model(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Example input matching the training resolution (needed for tracing).
    ds_cfg = config["dataset"]
    in_channels = 1 if str(ds_cfg["name"]).lower() == "mnist" else 3
    img_size = int(ds_cfg.get("img_size", 224))
    example = torch.randn(1, in_channels, img_size, img_size)

    checkpoint_dir = Path(args.checkpoint).parent
    if args.format == "torchscript":
        out_path = Path(args.output or checkpoint_dir / "model_torchscript.pt")
        traced = torch.jit.trace(model, example)
        traced.save(str(out_path))
    else:
        out_path = Path(args.output or checkpoint_dir / "model.onnx")
        try:
            torch.onnx.export(
                model,
                example,
                str(out_path),
                input_names=["image"],
                output_names=["logits"],
                dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            )
        except (ImportError, ModuleNotFoundError):
            sys.exit("ERROR: ONNX export requires the onnx package: pip install onnx")

    size_mb = out_path.stat().st_size / (1024**2)
    print(f"Exported {config['model']['name']} ({args.format}) -> {out_path} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
