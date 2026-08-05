#!/usr/bin/env python
"""Train a model from a YAML configuration.

Usage:
    python scripts/train.py --config configs/mnist_cpu.yaml

Common hyperparameters can be overridden from the command line without
editing the YAML — handy for quick experiments and Slurm job arrays:

    python scripts/train.py --config configs/thfood_baseline.yaml \\
        --name thfood_lr001 --lr 0.001 --epochs 10
"""

import argparse
import sys
from pathlib import Path
from typing import Any

# Make the repository root importable no matter where the script is run from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch

from datasets import build_dataloaders
from models import build_model
from trainer import Trainer
from trainer.losses import build_loss
from trainer.utils import (
    build_optimizer,
    build_scheduler,
    cleanup_distributed,
    count_parameters,
    init_distributed,
    load_config,
    set_seed,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a model from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to the YAML config file.")
    parser.add_argument("--name", help="Override experiment.name (results directory).")
    parser.add_argument("--device", help="Override device (cuda | cpu).")
    parser.add_argument("--epochs", type=int, help="Override training.epochs.")
    parser.add_argument("--batch-size", type=int, help="Override training.batch_size.")
    parser.add_argument("--lr", type=float, help="Override training.lr.")
    return parser.parse_args()


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> None:
    """Apply command-line overrides onto the loaded config (in place).

    Args:
        config: The configuration to modify.
        args: Parsed command-line arguments.
    """
    if args.name is not None:
        config["experiment"]["name"] = args.name
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["lr"] = args.lr


def main() -> None:
    """Load config, build every component, and run training."""
    args = parse_args()
    config = load_config(args.config)
    apply_overrides(config, args)

    # Reproducibility and device selection.
    exp_cfg = config.get("experiment", {}) or {}
    set_seed(int(exp_cfg.get("seed", 42)), bool(exp_cfg.get("deterministic", False)))

    # A plain `python scripts/train.py` run gets a single-process context
    # (same as before). Launching under `torchrun` — one node with
    # --nproc_per_node > 1, or several nodes — sets RANK/WORLD_SIZE/
    # LOCAL_RANK in the environment, and this picks them up automatically;
    # no config change is needed to go from 1 GPU to N. See jobs/
    # train_thfood_multigpu.sh and jobs/train_thfood_multinode.sh.
    dist_ctx = init_distributed(str(config.get("device", "cuda")))
    device = dist_ctx.device

    if dist_ctx.is_main_process:
        print("=" * 70)
        print(f"Config : {args.config}")
        if dist_ctx.enabled:
            print(
                f"Distributed: {dist_ctx.world_size} processes, "
                f"backend={dist_ctx.backend}"
            )
        if device.type == "cuda":
            print(f"Device : {device} ({torch.cuda.get_device_name(device)})")
        else:
            print(f"Device : {device}")

    # 1. Data ----------------------------------------------------------
    train_loader, val_loader, _, class_names = build_dataloaders(
        config, rank=dist_ctx.rank, world_size=dist_ctx.world_size
    )
    if dist_ctx.is_main_process:
        print(
            f"Dataset: {config['dataset']['name']} — "
            f"{len(train_loader.dataset):,} train / {len(val_loader.dataset):,} val images, "
            f"{len(class_names)} classes"
        )

    # 2. Model ---------------------------------------------------------
    model = build_model(config, num_classes=len(class_names))
    if dist_ctx.is_main_process:
        total = count_parameters(model)
        trainable = count_parameters(model, trainable_only=True)
        print(f"Model  : {config['model']['name']} — {total:,} parameters ({trainable:,} trainable)")
        print("=" * 70)

    # 3. Loss, optimizer, scheduler -----------------------------------
    criterion = build_loss(config)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    # 4. Train ---------------------------------------------------------
    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config,
        scheduler=scheduler,
        class_names=class_names,
        dist_ctx=dist_ctx,
    )
    try:
        trainer.fit()
    finally:
        cleanup_distributed(dist_ctx)


if __name__ == "__main__":
    main()
