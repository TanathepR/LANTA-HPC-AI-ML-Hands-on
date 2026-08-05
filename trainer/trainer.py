"""The Trainer: ties model, data, optimizer, logging and checkpoints together.

For every experiment the Trainer creates:

    outputs/<experiment_name>/
        config.yaml      exact copy of the config used (reproducibility)
        best.pt          checkpoint with the highest validation accuracy
        last.pt          checkpoint after the most recent epoch
        metrics.json     per-epoch metrics (loss, accuracy, time, throughput)

    logs/<experiment_name>/
        events.out.*     TensorBoard event files
"""

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from trainer.engine import train_one_epoch, validate
from trainer.utils import (
    DistributedContext,
    broadcast_object,
    reduce_max,
    reduce_mean,
    reduce_sum,
    save_config,
    unwrap_model,
)


class Trainer:
    """Supervised-classification training driver.

    Responsibilities:
        * the epoch loop (training + validation)
        * learning-rate scheduling
        * TensorBoard logging
        * checkpoint saving (``best.pt`` / ``last.pt``)
        * metrics logging (``metrics.json``)
        * optional early stopping

    Attributes:
        history: One dict of metrics per completed epoch.
        best_val_acc: Highest validation accuracy seen so far.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict[str, Any],
        scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
        class_names: list[str] | None = None,
        dist_ctx: DistributedContext | None = None,
    ) -> None:
        """Set up the experiment directories, TensorBoard, and AMP.

        Args:
            model: The model to train.
            criterion: Loss function.
            optimizer: Optimizer.
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            device: Device to train on.
            config: Full experiment configuration (saved for reproducibility).
            scheduler: Optional learning-rate scheduler (stepped per epoch).
            class_names: Class names, stored inside checkpoints so that
                evaluation and prediction scripts can label outputs.
            dist_ctx: Multi-GPU/multi-node process info from
                ``trainer.utils.init_distributed``. ``None`` (the default)
                behaves exactly like a single-process run.
        """
        self.dist_ctx = dist_ctx or DistributedContext(
            enabled=False, rank=0, world_size=1, local_rank=0, device=device, backend="none"
        )
        self.is_main_process = self.dist_ctx.is_main_process

        self.model = model.to(device)
        if self.dist_ctx.enabled:
            # Wraps the model so every backward() call all-reduces (averages)
            # gradients across processes before the optimizer step — the
            # actual mechanism behind data-parallel multi-GPU training.
            ddp_kwargs = {"device_ids": [device.index]} if device.type == "cuda" else {}
            self.model = DistributedDataParallel(self.model, **ddp_kwargs)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        self.class_names = class_names or []

        exp_cfg = config.get("experiment", {}) or {}
        self.experiment_name = str(exp_cfg.get("name", "experiment"))
        self.epochs = int(config["training"]["epochs"])

        # Create outputs/<name>/ and logs/<name>/, save the config copy.
        # Only the main process touches disk — every rank would otherwise
        # race to create the same directories and write the same files.
        self.output_dir = Path(exp_cfg.get("output_dir", "outputs")) / self.experiment_name
        self.log_dir = Path(exp_cfg.get("log_dir", "logs")) / self.experiment_name
        if self.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.log_dir.mkdir(parents=True, exist_ok=True)
            save_config(config, self.output_dir / "config.yaml")
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
        else:
            self.writer = None

        # Mixed precision only makes sense on GPU.
        use_amp = bool(config["training"].get("amp", False)) and device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device.type) if use_amp else None

        # Early stopping (optional).
        es_cfg = config["training"].get("early_stopping", {}) or {}
        self.early_stopping = bool(es_cfg.get("enabled", False))
        self.patience = int(es_cfg.get("patience", 5))
        self.min_delta = float(es_cfg.get("min_delta", 0.0))

        self.best_val_acc: float = 0.0
        self.history: list[dict[str, float]] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def fit(self) -> list[dict[str, float]]:
        """Train for the configured number of epochs.

        Returns:
            The per-epoch metrics history.
        """
        amp_note = "on" if self.scaler is not None else "off"
        if self.is_main_process:
            print(
                f"\nTraining '{self.experiment_name}' for {self.epochs} epochs "
                f"on {self.device} (AMP {amp_note})"
            )
            if self.dist_ctx.enabled:
                print(
                    f"  distributed -> {self.dist_ctx.world_size} processes "
                    f"({self.dist_ctx.backend}), batch size below is PER PROCESS"
                )
            print(f"  outputs -> {self.output_dir}")
            print(f"  logs    -> {self.log_dir}\n")

        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            # Reshuffles differently each epoch and reshards across ranks;
            # without this every rank would see the exact same order/subset
            # of data every epoch.
            if isinstance(self.train_loader.sampler, DistributedSampler):
                self.train_loader.sampler.set_epoch(epoch)

            train_stats = train_one_epoch(
                self.model,
                self.train_loader,
                self.criterion,
                self.optimizer,
                self.device,
                scaler=self.scaler,
                epoch=epoch,
                total_epochs=self.epochs,
                disable_progress=not self.is_main_process,
            )
            train_stats = self._reduce_train_stats(train_stats)

            # Validation runs once, on the main process only, over the full
            # (un-sharded) val set — then broadcasts the result to every
            # other rank. This keeps scheduler stepping and the
            # early-stopping decision below identical on every process;
            # letting them diverge would deadlock the next DDP backward()
            # call, since some ranks would stop while others kept training.
            if self.is_main_process:
                val_stats = validate(self.model, self.val_loader, self.criterion, self.device)
            else:
                val_stats = {"loss": 0.0, "accuracy": 0.0}
            val_stats = broadcast_object(val_stats, src=0)

            self._step_scheduler(val_stats["loss"])
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Record metrics: history list, metrics.json, TensorBoard.
            record = {
                "epoch": epoch,
                "train_loss": train_stats["loss"],
                "train_acc": train_stats["accuracy"],
                "val_loss": val_stats["loss"],
                "val_acc": val_stats["accuracy"],
                "epoch_time_sec": train_stats["epoch_time"],
                "images_per_sec": train_stats["images_per_sec"],
                "lr": current_lr,
            }
            self.history.append(record)
            if self.is_main_process:
                self._log_epoch(record)

            # Early-stopping bookkeeping (checked against the OLD best).
            if val_stats["accuracy"] > self.best_val_acc + self.min_delta:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Checkpoints: best.pt on improvement, last.pt every epoch.
            if val_stats["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_stats["accuracy"]
                if self.is_main_process:
                    self._save_checkpoint("best.pt", epoch)
            if self.is_main_process:
                self._save_checkpoint("last.pt", epoch)

            if self.is_main_process:
                print(
                    f"Epoch {epoch:>3}/{self.epochs} | "
                    f"train loss {train_stats['loss']:.4f} acc {100 * train_stats['accuracy']:5.2f}% | "
                    f"val loss {val_stats['loss']:.4f} acc {100 * val_stats['accuracy']:5.2f}% | "
                    f"{train_stats['epoch_time']:6.1f}s  {train_stats['images_per_sec']:7.0f} img/s | "
                    f"lr {current_lr:.2e}"
                )

            if self.early_stopping and epochs_without_improvement >= self.patience:
                if self.is_main_process:
                    print(
                        f"Early stopping: no val-accuracy improvement "
                        f"for {self.patience} epochs."
                    )
                break

        if self.is_main_process:
            self.writer.close()
            print(f"\nDone. Best validation accuracy: {100 * self.best_val_acc:.2f}%")
            print(f"Checkpoints and metrics: {self.output_dir}")
            print(f"TensorBoard logs:        {self.log_dir}")
        return self.history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _reduce_train_stats(self, train_stats: dict[str, float]) -> dict[str, float]:
        """Combine per-process training stats into cluster-wide numbers.

        Each process only trains on its own shard of the data, so loss and
        accuracy are averaged across processes, while throughput is summed
        (total images/sec across every GPU is the actual scaling number
        this feature exists to demonstrate). A no-op on a single process.
        """
        if not self.dist_ctx.enabled:
            return train_stats
        images_seen = train_stats["images_per_sec"] * train_stats["epoch_time"]
        images_seen_total = reduce_sum(images_seen, self.device)
        epoch_time = reduce_max(train_stats["epoch_time"], self.device)
        return {
            "loss": reduce_mean(train_stats["loss"], self.device),
            "accuracy": reduce_mean(train_stats["accuracy"], self.device),
            "epoch_time": epoch_time,
            "images_per_sec": images_seen_total / epoch_time,
        }

    def _step_scheduler(self, val_loss: float) -> None:
        """Advance the LR scheduler by one epoch (if one is configured)."""
        if self.scheduler is None:
            return
        if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            # This scheduler reacts to a metric instead of the epoch count.
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

    def _log_epoch(self, record: dict[str, float]) -> None:
        """Write one epoch of metrics to TensorBoard and metrics.json."""
        epoch = int(record["epoch"])
        self.writer.add_scalar("Loss/train", record["train_loss"], epoch)
        self.writer.add_scalar("Loss/val", record["val_loss"], epoch)
        self.writer.add_scalar("Accuracy/train", record["train_acc"], epoch)
        self.writer.add_scalar("Accuracy/val", record["val_acc"], epoch)
        self.writer.add_scalar("Time/epoch_seconds", record["epoch_time_sec"], epoch)
        self.writer.add_scalar("Throughput/images_per_sec", record["images_per_sec"], epoch)
        self.writer.add_scalar("LR", record["lr"], epoch)

        # Rewriting the full file each epoch keeps it valid even if the
        # job is killed mid-training (e.g. by the Slurm time limit).
        metrics = {
            "experiment": self.experiment_name,
            "best_val_acc": self.best_val_acc,
            "epochs": self.history,
        }
        with open(self.output_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    def _save_checkpoint(self, filename: str, epoch: int) -> None:
        """Save model + training state so a run can be resumed or evaluated."""
        checkpoint = {
            "epoch": epoch,
            "model_state": unwrap_model(self.model).state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler else None,
            "best_val_acc": self.best_val_acc,
            "config": self.config,
            "class_names": self.class_names,
        }
        torch.save(checkpoint, self.output_dir / filename)
