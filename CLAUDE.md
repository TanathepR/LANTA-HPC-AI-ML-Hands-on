# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

A teaching codebase for an HPC AI/ML workshop (training on the LANTA supercomputer). It is **config-driven by design**: students edit YAML files, never Python, to run experiments. Two labs live here — MNIST (CPU vs. GPU comparison) and THFOOD-100 Thai food classification (transfer learning + hyperparameter tuning). When making changes, preserve the property that both labs run through the *exact same* training code with only the config differing — that's the pedagogical point.

There is no test suite and no lint/format tooling configured in this repo (no pytest, ruff, flake8, or pyproject.toml). Validate changes by actually running training scripts (see below).

## Commands

Environment setup:
```bash
conda env create -f environment.yml && conda activate hpc-ai
# or: pip install -r requirements.txt   (Python 3.11)
```

Pre-download data (do this before training; compute nodes on LANTA have no internet):
```bash
python datasets/download.py --dataset mnist --root ./data
python datasets/download.py --dataset thfood100 --root ./data/thfood100   # verifies ImageFolder layout only
```

Train (the primary "does it work" check for any change to `datasets/`, `models/`, or `trainer/`):
```bash
python scripts/train.py --config configs/mnist_cpu.yaml
python scripts/train.py --config configs/thfood_baseline.yaml
# CLI overrides (for sweeps, forwarded by jobs/train_thfood.sh):
python scripts/train.py --config configs/thfood_baseline.yaml --name exp2 --lr 0.001 --batch-size 128 --epochs 10 --device cpu
```

Other scripts, all operating on a checkpoint (`outputs/<name>/best.pt`):
```bash
python scripts/evaluate.py --checkpoint outputs/<name>/best.pt --split val|test
python scripts/predict.py  --model outputs/<name>/best.pt --image sample.jpg
python scripts/benchmark.py --config configs/thfood_baseline.yaml   # or --model <name>, uses random weights, offline-safe
python scripts/export.py   --checkpoint outputs/<name>/best.pt --format torchscript|onnx
```

TensorBoard (`--logdir logs`, not a single run subdir, overlays all experiments):
```bash
tensorboard --logdir logs
```

Slurm (LANTA), in `jobs/`: `sbatch jobs/train_cpu.sh`, `train_gpu.sh`, `train_thfood.sh` (forwards extra args to `train.py`), `benchmark.sh`. Edit the `#SBATCH --account=` line before submitting.

For a class sharing one LANTA project quota (home dirs are small, `/project` is huge): instructor runs `setup_project.sh` once on `/project` (builds the conda env at `./envs/hpc-ai` inside the checkout, never under `$HOME`); each student runs `setup_user.sh` once to get a personal `~/hpc-ai-workshop/` with their own `configs/`/`jobs/` plus empty `checkpoints/`/`outputs/`/`logs/`, and a `project.env` pointing `jobs/*.sh` at the shared code+env. Students submit from `~/hpc-ai-workshop/`, not the `/project` checkout.

## Architecture

Full technical detail lives in [ARCHITECTURE.md](ARCHITECTURE.md) — read it before non-trivial changes. Summary:

**Data flow:** `scripts/train.py` loads a YAML config → `build_dataloaders` (datasets/) + `build_model` (models/) + `build_loss`/`build_optimizer`/`build_scheduler` (trainer/) → all handed to `Trainer.fit()` → writes `outputs/<name>/` (config.yaml, best.pt, last.pt, metrics.json) and `logs/<name>/` (TensorBoard). Downstream scripts (`evaluate.py`, `predict.py`, `export.py`) consume only the checkpoint file — nothing else.

**Factory pattern:** each package (`datasets/`, `models/`, `trainer/utils.py`) exposes one `build_*` entry point that dispatches on a config field to construct the right PyTorch object. `Trainer` is the only class with real state; the hot loops (`train_one_epoch`, `validate`) in `trainer/engine.py` are plain functions — this split is intentional, so students read `engine.py` for the training loop itself and treat `Trainer` as bookkeeping infrastructure.

**Dependency direction is strict:** `trainer/` never imports `datasets` or `models` — it only knows `nn.Module` and `DataLoader`. This is what lets Lab 1 and Lab 2 share identical training code. Don't break this by importing dataset/model specifics into `trainer/`.

**Checkpoints are self-contained:** a `.pt` file bundles `model_state`, `optimizer_state`, `scheduler_state`, `best_val_acc`, the full `config` dict, and `class_names`. This is why `evaluate.py`/`predict.py`/`export.py` need no other input. Preserve this contract when touching checkpoint save/load code (`trainer/trainer.py`, `trainer/utils.py:load_checkpoint`).

**Configs are flat and self-contained** (no inheritance/merging). Code reads them defensively via `.get(key, default)` so missing optional keys don't crash — keep this pattern when adding new config fields. Schema sections: `experiment`, `dataset`, `model`, `training`, `loss`, `optimizer`, `scheduler`, `device`. `configs/default.yaml` is the fully-commented reference.

**Transfer-learning contract** for pretrained models (`models/resnet18.py`, `mobilenetv3.py`, `efficientnet.py`): load ImageNet weights → optionally freeze backbone → *then* replace the final layer, so the new head stays trainable. Follow this order when adding a model.

**Extension points** (each is a single new file + one dispatch branch, immediately YAML-selectable, no script changes needed):
- Dataset: `datasets/<name>.py` + branch in `datasets/__init__.py`; return loaders + `class_names`.
- Model: `models/<name>.py` + branch in `models/__init__.py`; builder signature `f(num_classes, pretrained, freeze_backbone) -> nn.Module`.
- Optimizer/scheduler: add a branch in `trainer/utils.py` (`build_optimizer`/`build_scheduler`).
- Loss: `trainer/losses.py`.
- Metric: `trainer/metrics.py`, then log it in `Trainer._log_epoch`.

**HPC-specific concerns to respect:** GPU timing must use `torch.cuda.synchronize()` before measuring (see `engine.py`, `benchmark.py`) or throughput numbers will be wrong; `num_workers`/`pin_memory` are config-exposed, not hardcoded; scripts must stay offline-safe after initial download (no surprise network calls on compute nodes).
