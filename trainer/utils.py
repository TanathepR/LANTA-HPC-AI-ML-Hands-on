"""Shared helpers: config I/O, reproducibility, device selection, and
factories that turn config sections into PyTorch objects."""

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import yaml
from torch import nn
from torch.nn.parallel import DistributedDataParallel


# ---------------------------------------------------------------------------
# Config I/O
# ---------------------------------------------------------------------------
def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        The configuration as a nested dictionary.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_config(config: dict[str, Any], path: str | Path) -> None:
    """Save a configuration to YAML (a copy is stored with every experiment
    so results are always reproducible).

    Args:
        config: Configuration dictionary.
        path: Destination file path (parent dirs are created).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


# ---------------------------------------------------------------------------
# Reproducibility & device
# ---------------------------------------------------------------------------
def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every random number generator used during training.

    Args:
        seed: The seed value.
        deterministic: If ``True``, force cuDNN into deterministic mode.
            Runs become bit-exact reproducible but slower, because cuDNN
            can no longer auto-select the fastest convolution kernels.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Let cuDNN benchmark kernels once and reuse the fastest.
        torch.backends.cudnn.benchmark = True


def get_device(name: str) -> torch.device:
    """Resolve the device requested in the config, falling back gracefully.

    Args:
        name: ``"cuda"``, ``"cuda:0"``, or ``"cpu"``.

    Returns:
        A usable ``torch.device`` (CPU if CUDA was requested but is absent).
    """
    if name.startswith("cuda") and not torch.cuda.is_available():
        print("WARNING: config requests CUDA but no GPU is available — using CPU.")
        return torch.device("cpu")
    return torch.device(name)


# ---------------------------------------------------------------------------
# Multi-GPU / multi-node (DistributedDataParallel via torchrun)
# ---------------------------------------------------------------------------
@dataclass
class DistributedContext:
    """Process-group info for one training process.

    A plain ``python scripts/train.py ...`` invocation gets a context with
    ``enabled=False`` (rank 0 of 1), so the rest of the code — Trainer,
    dataset builders — can be written once and behave correctly whether or
    not the run is distributed, instead of branching everywhere.
    """

    enabled: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str

    @property
    def is_main_process(self) -> bool:
        """Whether this process should do the one-off work (logging,
        checkpointing, directory creation) that must not be duplicated."""
        return self.rank == 0


def init_distributed(requested_device: str) -> DistributedContext:
    """Initialize ``torch.distributed`` if launched under ``torchrun``.

    ``torchrun`` (single- or multi-node) sets ``RANK``, ``WORLD_SIZE``, and
    ``LOCAL_RANK`` in the environment before the script starts. A plain
    ``python scripts/train.py ...`` invocation has none of these, so
    ``WORLD_SIZE`` defaults to 1 and this is a no-op that returns a
    single-process context — same training path, no source change needed
    to go from 1 GPU to N.

    Args:
        requested_device: The config's ``device`` value (``"cuda"`` or
            ``"cpu"``).

    Returns:
        A ``DistributedContext`` describing this process's role.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return DistributedContext(
            enabled=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=get_device(requested_device),
            backend="none",
        )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    use_cuda = requested_device.startswith("cuda") and torch.cuda.is_available()
    # NCCL is the fast GPU-to-GPU backend; gloo is the CPU fallback so a
    # multi-process CPU run (e.g. for debugging) still works under torchrun.
    backend = "nccl" if use_cuda else "gloo"

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    return DistributedContext(
        enabled=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
        backend=backend,
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    """Tear down the process group opened by ``init_distributed`` (if any)."""
    if ctx.enabled and dist.is_initialized():
        dist.destroy_process_group()


def unwrap_model(model: nn.Module) -> nn.Module:
    """Strip the ``DistributedDataParallel`` wrapper, if present.

    Checkpoints must always store the plain model's ``state_dict``: a DDP
    wrapper prefixes every key with ``module.``, which would silently break
    ``evaluate.py`` / ``predict.py`` / ``export.py`` — they load checkpoints
    into a plain, non-distributed model.
    """
    return model.module if isinstance(model, DistributedDataParallel) else model


def reduce_mean(value: float, device: torch.device) -> float:
    """Average a scalar across all processes (no-op outside a distributed run)."""
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return (tensor / dist.get_world_size()).item()


def reduce_sum(value: float, device: torch.device) -> float:
    """Sum a scalar across all processes (no-op outside a distributed run)."""
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.item()


def reduce_max(value: float, device: torch.device) -> float:
    """Take the max of a scalar across all processes (no-op outside a distributed run)."""
    if not dist.is_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def broadcast_scalars(
    values: dict[str, float], device: torch.device, src: int = 0
) -> dict[str, float]:
    """Broadcast a dict of float scalars from rank ``src`` to every process.

    Used to keep validation metrics identical on every rank (validation runs
    only on the main process — see ``Trainer.fit``) so scheduler stepping and
    early-stopping decisions never diverge between ranks, which would
    deadlock the next collective call inside DDP's backward pass.

    Deliberately a plain tensor broadcast rather than
    ``dist.broadcast_object_list`` (which pickles arbitrary Python objects):
    that path hits a PyTorch/NCCL bug on some builds
    (``RuntimeError: SymIntArrayRef expected to contain only concrete
    integers`` inside ``torch.distributed.distributed_c10d``). Since the
    values here are always plain floats, a tensor broadcast sidesteps the
    bug entirely and is the more idiomatic tool for the job anyway.
    """
    if not dist.is_initialized():
        return values
    keys = sorted(values.keys())
    tensor = torch.tensor([values[k] for k in keys], device=device, dtype=torch.float64)
    dist.broadcast(tensor, src=src)
    return dict(zip(keys, tensor.tolist()))


# ---------------------------------------------------------------------------
# Model inspection
# ---------------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count model parameters.

    Args:
        model: The model to inspect.
        trainable_only: Count only parameters with ``requires_grad=True``
            (useful to see the effect of ``freeze_backbone``).

    Returns:
        The number of parameters.
    """
    params = (
        p for p in model.parameters() if p.requires_grad or not trainable_only
    )
    return sum(p.numel() for p in params)


def model_size_mb(model: nn.Module) -> float:
    """Size of all parameters and buffers in megabytes.

    Args:
        model: The model to inspect.

    Returns:
        Model size in MB (1 MB = 1024 * 1024 bytes).
    """
    n_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    n_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return n_bytes / (1024**2)


# ---------------------------------------------------------------------------
# Optimizer / scheduler factories
# ---------------------------------------------------------------------------
def build_optimizer(model: nn.Module, config: dict[str, Any]) -> torch.optim.Optimizer:
    """Build the optimizer named in the config.

    Only parameters with ``requires_grad=True`` are passed to the
    optimizer, so frozen backbone weights are skipped automatically.

    Args:
        model: The model whose parameters will be optimized.
        config: Full experiment configuration (parsed YAML).

    Returns:
        The configured optimizer.

    Raises:
        ValueError: If ``config["optimizer"]["name"]`` is unknown.
    """
    opt_cfg = config.get("optimizer", {}) or {}
    name = str(opt_cfg.get("name", "AdamW")).lower()
    lr = float(config["training"]["lr"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    params = [p for p in model.parameters() if p.requires_grad]

    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=float(opt_cfg.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=bool(opt_cfg.get("nesterov", False)),
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    raise ValueError(f"Unknown optimizer '{name}'. Available: SGD, Adam, AdamW")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict[str, Any]
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the learning-rate scheduler named in the config.

    All schedulers here are stepped **once per epoch** by the Trainer.

    Args:
        optimizer: The optimizer whose learning rate will be scheduled.
        config: Full experiment configuration (parsed YAML).

    Returns:
        The configured scheduler, or ``None`` for ``name: none``.

    Raises:
        ValueError: If ``config["scheduler"]["name"]`` is unknown.
    """
    sch_cfg = config.get("scheduler", {}) or {}
    name = str(sch_cfg.get("name", "none")).lower()
    epochs = int(config["training"]["epochs"])

    if name in ("none", "null", ""):
        return None
    if name == "steplr":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg.get("step_size", 10)),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )
    if name == "multisteplr":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(sch_cfg.get("milestones", [10, 15])),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )
    if name == "cosineannealinglr":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=epochs,
            eta_min=float(sch_cfg.get("min_lr", 0.0)),
        )
    if name == "reducelronplateau":
        # Steps on the validation loss (handled specially by the Trainer).
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(sch_cfg.get("gamma", 0.1)),
            patience=int(sch_cfg.get("patience", 3)),
        )

    raise ValueError(
        f"Unknown scheduler '{name}'. Available: none, StepLR, MultiStepLR, "
        "CosineAnnealingLR, ReduceLROnPlateau"
    )


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint saved by the Trainer.

    Args:
        path: Path to ``best.pt`` or ``last.pt``.
        device: Device to map the tensors onto.

    Returns:
        The checkpoint dictionary (model_state, config, class_names, ...).
    """
    # weights_only=False because our checkpoints also store the config dict
    # and class names. Only load checkpoint files you trust (i.e. your own).
    return torch.load(path, map_location=device, weights_only=False)
