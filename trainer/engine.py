"""Core training and evaluation loops.

These are plain functions (not classes) so the entire forward/backward
pass can be read top-to-bottom in one place. The Trainer class calls
them once per epoch.
"""

import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainer.metrics import AverageMeter, accuracy


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int = 1,
    total_epochs: int = 1,
    disable_progress: bool = False,
) -> dict[str, float]:
    """Run one full pass over the training set.

    Each batch goes through the classic five steps:
        1. move data to the device
        2. forward pass  -> predictions and loss
        3. backward pass -> gradients
        4. optimizer step -> update weights
        5. record metrics

    Args:
        model: The model to train (moved to ``device`` already).
        loader: Training DataLoader.
        criterion: Loss function.
        optimizer: Optimizer updating the model weights.
        device: Device to run on.
        scaler: GradScaler for mixed precision, or ``None`` for full FP32.
        epoch: Current epoch number (for the progress bar only).
        total_epochs: Total epochs (for the progress bar only).
        disable_progress: Hide the tqdm bar — used in multi-GPU runs so only
            the main process prints one, instead of one per process.

    Returns:
        Dict with ``loss``, ``accuracy``, ``epoch_time`` (seconds) and
        ``images_per_sec`` (training throughput).
    """
    model.train()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    images_seen = 0

    start = time.perf_counter()
    progress = tqdm(
        loader,
        desc=f"Epoch {epoch}/{total_epochs} [train]",
        leave=False,
        dynamic_ncols=True,
        disable=disable_progress,
    )
    for images, targets in progress:
        # 1. Move the batch to the training device (CPU or GPU).
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # Clear gradients from the previous step.
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            # Mixed precision: the forward pass runs in float16 where safe,
            # and the scaler rescales the loss so small gradients do not
            # underflow. On A100 GPUs this is typically 1.5-3x faster.
            with torch.autocast(device_type=device.type):
                outputs = model(images)          # 2. forward
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()        # 3. backward
            scaler.step(optimizer)               # 4. update
            scaler.update()
        else:
            outputs = model(images)              # 2. forward
            loss = criterion(outputs, targets)
            loss.backward()                      # 3. backward
            optimizer.step()                     # 4. update

        # 5. Record metrics for this batch.
        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(outputs.detach(), targets), batch_size)
        images_seen += batch_size
        progress.set_postfix(
            loss=f"{loss_meter.avg:.4f}", acc=f"{100 * acc_meter.avg:.2f}%"
        )

    if device.type == "cuda":
        # GPU work is asynchronous — wait for it before stopping the clock,
        # otherwise the timing would be wrong.
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    return {
        "loss": loss_meter.avg,
        "accuracy": acc_meter.avg,
        "epoch_time": elapsed,
        "images_per_sec": images_seen / elapsed,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    desc: str = "val",
) -> dict[str, float]:
    """Evaluate the model on a validation or test set.

    Same as training minus the backward pass: ``@torch.no_grad()`` disables
    gradient tracking (faster, less memory) and ``model.eval()`` switches
    layers like BatchNorm and Dropout to inference behavior.

    Args:
        model: The model to evaluate.
        loader: Validation/test DataLoader.
        criterion: Loss function.
        device: Device to run on.
        desc: Progress-bar label.

    Returns:
        Dict with ``loss`` and ``accuracy``.
    """
    model.eval()
    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    for images, targets in tqdm(loader, desc=f"[{desc}]", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        batch_size = targets.size(0)
        loss_meter.update(loss.item(), batch_size)
        acc_meter.update(accuracy(outputs, targets), batch_size)

    return {"loss": loss_meter.avg, "accuracy": acc_meter.avg}


@torch.no_grad()
def predict_all(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    """Collect true and predicted labels for a whole DataLoader.

    Used by ``evaluate.py`` to build per-class reports with scikit-learn.

    Args:
        model: The model to run (set to eval mode here).
        loader: DataLoader to iterate (must not shuffle).
        device: Device to run on.

    Returns:
        A tuple ``(y_true, y_pred)`` of 1-D integer arrays.
    """
    model.eval()
    all_targets: list[np.ndarray] = []
    all_predictions: list[np.ndarray] = []

    for images, targets in tqdm(loader, desc="[predict]", leave=False, dynamic_ncols=True):
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        all_predictions.append(outputs.argmax(dim=1).cpu().numpy())
        all_targets.append(targets.numpy())

    return np.concatenate(all_targets), np.concatenate(all_predictions)
