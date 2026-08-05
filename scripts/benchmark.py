#!/usr/bin/env python
"""Benchmark a model: parameters, size, latency, and throughput.

Usage:
    python scripts/benchmark.py --config configs/thfood_baseline.yaml
    python scripts/benchmark.py --model efficientnet_b0 --num-classes 100
    python scripts/benchmark.py --model lenet --img-size 28 --in-channels 1

Measurements:
    * parameter count and model size (MB)
    * forward latency at batch size 1 (ms, mean +/- std)
    * forward throughput at the given batch size (images/sec)
    * training-step throughput: forward + backward + update (images/sec)
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable

# Make the repository root importable no matter where the script is run from.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from models import build_model
from trainer.utils import count_parameters, get_device, load_config, model_size_mb


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark a model.")
    parser.add_argument("--config", help="YAML config describing model + input size.")
    parser.add_argument("--model", help="Model name (alternative to --config).")
    parser.add_argument("--num-classes", type=int, default=100, help="Output classes (with --model).")
    parser.add_argument("--img-size", type=int, default=224, help="Input resolution (with --model).")
    parser.add_argument("--in-channels", type=int, default=3, help="Input channels (with --model).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for throughput tests.")
    parser.add_argument("--device", default="cuda", help="cuda | cpu.")
    parser.add_argument("--warmup", type=int, default=10, help="Untimed warm-up iterations.")
    parser.add_argument("--runs", type=int, default=50, help="Timed iterations.")
    return parser.parse_args()


def timed_runs(
    step: Callable[[], None], warmup: int, runs: int, device: torch.device
) -> list[float]:
    """Time a function accurately, GPU-aware.

    Warm-up runs are excluded because the first iterations include one-off
    costs (memory allocation, cuDNN kernel selection). GPU work is
    asynchronous, so we synchronize before reading the clock.

    Args:
        step: The zero-argument function to time.
        warmup: Number of untimed warm-up calls.
        runs: Number of timed calls.
        device: Device the work runs on.

    Returns:
        A list of per-call durations in seconds.
    """
    for _ in range(warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    durations: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        step()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        durations.append(time.perf_counter() - start)
    return durations


def main() -> None:
    """Build the model and run all benchmark measurements."""
    args = parse_args()
    if not args.config and not args.model:
        sys.exit("ERROR: pass either --config or --model. See --help.")
    device = get_device(args.device)

    # Build a config dict either from YAML or from the CLI shortcut.
    if args.config:
        config = load_config(args.config)
        model_name = str(config["model"]["name"])
        img_size = int(config["dataset"].get("img_size", 224))
        in_channels = int(config["model"].get("in_channels", 3))
        num_classes = 10 if config["dataset"]["name"] == "mnist" else args.num_classes
    else:
        model_name = args.model
        img_size = args.img_size
        in_channels = args.in_channels
        num_classes = args.num_classes
        config = {"model": {"name": model_name, "in_channels": in_channels}}

    # Random weights are fine — speed does not depend on the weight values,
    # and skipping the pretrained download lets this run on offline nodes.
    config["model"]["pretrained"] = False
    model = build_model(config, num_classes=num_classes).to(device)

    single = torch.randn(1, in_channels, img_size, img_size, device=device)
    batch = torch.randn(args.batch_size, in_channels, img_size, img_size, device=device)

    # --- Forward latency (batch size 1) --------------------------------
    model.eval()
    with torch.no_grad():
        latencies = timed_runs(lambda: model(single), args.warmup, args.runs, device)
    latency_ms = 1000 * statistics.mean(latencies)
    latency_std_ms = 1000 * statistics.stdev(latencies)

    # --- Forward throughput (batch) ------------------------------------
    with torch.no_grad():
        forward_times = timed_runs(lambda: model(batch), args.warmup, args.runs, device)
    forward_ips = args.batch_size / statistics.mean(forward_times)

    # --- Training-step throughput (forward + backward + update) --------
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    targets = torch.randint(0, num_classes, (args.batch_size,), device=device)

    def train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(batch), targets)
        loss.backward()
        optimizer.step()

    train_times = timed_runs(train_step, args.warmup, args.runs, device)
    train_ips = args.batch_size / statistics.mean(train_times)

    # --- Report --------------------------------------------------------
    device_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    results = {
        "model": model_name,
        "device": device_name,
        "input": f"{in_channels}x{img_size}x{img_size}",
        "batch_size": args.batch_size,
        "parameters": count_parameters(model),
        "model_size_mb": round(model_size_mb(model), 2),
        "latency_ms_bs1": round(latency_ms, 3),
        "latency_std_ms": round(latency_std_ms, 3),
        "forward_images_per_sec": round(forward_ips, 1),
        "train_images_per_sec": round(train_ips, 1),
    }

    print(f"\nBenchmark: {model_name} on {device_name}")
    print("-" * 56)
    print(f"{'input size':<28}{results['input']:>28}")
    print(f"{'parameters':<28}{results['parameters']:>28,}")
    print(f"{'model size':<28}{results['model_size_mb']:>25.2f} MB")
    print(f"{'latency (batch=1)':<28}{latency_ms:>17.3f} +/- {latency_std_ms:.3f} ms")
    print(f"{'forward throughput':<28}{forward_ips:>22.1f} img/s")
    print(f"{'training throughput':<28}{train_ips:>22.1f} img/s")
    print("-" * 56)

    out_path = Path("outputs") / f"benchmark_{model_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
