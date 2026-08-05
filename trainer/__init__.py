"""Training package.

Modules:
    trainer   The Trainer class (epoch loop, checkpoints, TensorBoard).
    engine    train_one_epoch / validate — the raw forward/backward loops.
    losses    Loss-function factory.
    metrics   AverageMeter and accuracy helpers.
    utils     Config I/O, seeding, device selection, optimizer/scheduler factories.
"""

from trainer.trainer import Trainer

__all__ = ["Trainer"]
