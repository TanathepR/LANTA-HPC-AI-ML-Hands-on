# Project Architecture — Technical Overview

This document explains **how the repository is structured internally**: what
each module does, how a training run flows through the code, and where to
extend it. For lab instructions and usage, see [README.md](README.md).

---

## 1. Design Principles

1. **Config-driven** — one YAML file fully describes an experiment. Students
   change YAML, never Python. Every knob (dataset, model, optimizer,
   scheduler, batch size, LR, epochs, device) lives in the config.
2. **Same code, different config** — Lab 1 (CPU vs. GPU) works because the
   code path is identical; only `device:` differs between configs.
3. **Factory pattern, minimal OOP** — each package exposes one `build_*`
   entry point that maps a config section to a PyTorch object. The only
   class with real state is `Trainer`; the hot loops are plain functions.
4. **Self-contained artifacts** — a checkpoint carries its own config and
   class names, so `evaluate.py` / `predict.py` / `export.py` need nothing
   but the `.pt` file. Every experiment directory is fully reproducible.a
5. **HPC-aware** — GPU-correct timing (`torch.cuda.synchronize`), AMP
   support, `num_workers`/`pin_memory` exposed in config, Slurm scripts
   that log GPU utilization, offline-friendly data/weight handling, and a
   quota-aware split between a shared `/project` checkout and each
   student's small `$HOME` workspace (see `jobs/` below).
6. **Scales to multi-GPU / multi-node without a config or code change**
   (Lab 2 / THFOOD-100) — launch the exact same config with `torchrun`
   instead of `python` and `Trainer` trains with `DistributedDataParallel`
   across every GPU it was given. See §9.

---

## 2. High-Level Data Flow

```
configs/*.yaml
     │  load_config()                        scripts/train.py
     ▼
┌─────────────────────────────────────────────────────────────┐
│  build_dataloaders(config)   → train/val/test loaders,      │   datasets/
│                                class_names                  │
│  build_model(config, n_cls)  → nn.Module                    │   models/
│  build_loss(config)          → criterion                    │   trainer/losses.py
│  build_optimizer(model, cfg) → optimizer                    │   trainer/utils.py
│  build_scheduler(opt, cfg)   → scheduler | None             │   trainer/utils.py
└─────────────────────────────────────────────────────────────┘
     │  all handed to
     ▼
Trainer(model, criterion, optimizer, loaders, device, config, ...)
     │  .fit()  — per epoch:
     │     train_one_epoch()  ─┐
     │     validate()          │  trainer/engine.py (plain functions)
     │     scheduler step      │
     │     TensorBoard + metrics.json + best.pt/last.pt
     ▼
outputs/<experiment_name>/          logs/<experiment_name>/
  config.yaml  best.pt  last.pt       events.out.* (TensorBoard)
  metrics.json
```

Downstream scripts consume only the checkpoint:

```
best.pt ──► evaluate.py   (per-class report, eval_<split>.json)
        ──► predict.py    (top-k for one image)
        ──► export.py     (TorchScript / ONNX)
```

---

## 3. Module Responsibilities

### `configs/` — experiment definitions

| File | Purpose |
|------|---------|
| `default.yaml` | Fully commented reference: documents every key and its allowed values |
| `mnist_cpu.yaml` / `mnist_gpu.yaml` | Lab 1 pair — identical except `device:` and `experiment.name` |
| `thfood_baseline.yaml` | Lab 2 baseline: pretrained ResNet-18, AdamW, cosine schedule, AMP |
| `thfood_competition.yaml` | Lab 2 tuning template with `# <-- tune me` markers |
| `thfood_sample.yaml` | Smoke test on the bundled `THFOOD-100.sample` preview (flat layout, ~5 images/class) — confirms the pipeline runs, not for real accuracy |

Configs are **self-contained** (no inheritance/merging). Code reads them
defensively with `.get(key, default)`, so a missing optional key never
crashes a run.

Top-level schema: `experiment`, `dataset`, `model`, `training`, `loss`,
`optimizer`, `scheduler`, `device`.

### `datasets/` — data pipeline

| Module | Key symbol | Role |
|--------|-----------|------|
| `__init__.py` | `build_dataloaders(config)` | Single entry point; dispatches on `dataset.name`; returns `(train_loader, val_loader, test_loader, class_names)` |
| `mnist.py` | `build_mnist_dataloaders` | Auto-downloading MNIST; the official 10k test set doubles as the val split |
| `thfood100.py` | `build_thfood_dataloaders` | `ImageFolder` over `root/{train,val,test}`; `test` falls back to `val`. If `root` has no `train/` folder but has class sub-directories directly (the flat `THFOOD-100.sample` preview layout), derives a stratified per-class train/val/test split instead, and reads `class_labels.csv` (if present) for readable class names. Raises a layout-help error if neither layout is found |
| `transforms.py` | `build_mnist_transforms`, `build_image_transforms`, `build_eval_transform` | Normalization constants + augmentation levels `none / basic / strong`; eval pipeline is always deterministic (resize → center-crop) |
| `download.py` | CLI | Login-node tool: downloads MNIST, verifies THFOOD-100 layout with per-split counts |

`class_names` is derived from the data (ImageFolder directory names, sorted),
and `num_classes` is always taken as `len(class_names)` — config and data can
never disagree.

### `models/` — model zoo

| Module | Builder | Notes |
|--------|---------|-------|
| `lenet.py` | `build_lenet` → `LeNet5(nn.Module)` | ~60k params, trained from scratch; the only hand-written architecture (Lab 1) |
| `resnet18.py` | `build_resnet18` | torchvision backbone; replaces `model.fc` |
| `mobilenetv3.py` | `build_mobilenetv3` | `large`/`small` variants; replaces `classifier[3]` |
| `efficientnet.py` | `build_efficientnet_b0` | Replaces `classifier[1]` |
| `__init__.py` | `build_model(config, num_classes)` | Name → builder dispatch |

Transfer-learning contract (all pretrained models): load ImageNet weights →
optionally freeze the backbone (**before** swapping the head, so the new
`nn.Linear` stays trainable) → replace the final layer with a
`num_classes`-sized one.

### `trainer/` — training core

| Module | Contents |
|--------|----------|
| `trainer.py` | `Trainer` class — the only stateful component. Owns: epoch loop, scheduler stepping (incl. the `ReduceLROnPlateau` metric-based special case), TensorBoard writer, `best.pt`/`last.pt` checkpointing, `metrics.json`, optional early stopping (patience on val accuracy). Creates `outputs/<name>/` + `logs/<name>/` and snapshots the config at construction time. |
| `engine.py` | The hot loops as **plain functions**: `train_one_epoch` (the 5-step batch cycle: move → forward → backward → update → record; AMP branch via `GradScaler`; GPU-synchronized timing), `validate` (`@torch.no_grad` + `model.eval()`), `predict_all` (label collection for sklearn reports) |
| `losses.py` | `build_loss` — CrossEntropyLoss with configurable label smoothing |
| `metrics.py` | `AverageMeter` (sample-weighted running mean), `accuracy` (top-1), `topk_accuracy` |
| `utils.py` | Config I/O (`load_config`/`save_config`), `set_seed` (+ cuDNN determinism toggle), `get_device` (CUDA→CPU fallback with warning), `count_parameters`, `model_size_mb`, `build_optimizer` (SGD/Adam/AdamW; skips frozen params), `build_scheduler` (StepLR/MultiStepLR/CosineAnnealingLR/ReduceLROnPlateau/none), `load_checkpoint`, and the DDP helpers described in §9 (`init_distributed`, `cleanup_distributed`, `unwrap_model`, `reduce_mean`/`reduce_sum`/`reduce_max`, `broadcast_scalars`) |

Class/function split rationale: the *loop bodies* (what students must read)
are flat functions in `engine.py`; the *bookkeeping* (what students can treat
as infrastructure) is encapsulated in `Trainer`.

### `scripts/` — CLI entry points

All scripts insert the repo root into `sys.path`, so they run from any
working directory. All accept `--help`.

| Script | Input | Output |
|--------|-------|--------|
| `train.py` | `--config` + optional overrides (`--name --device --epochs --batch-size --lr`) | Trained experiment in `outputs/<name>/` + `logs/<name>/` |
| `evaluate.py` | `--checkpoint` (+ `--split val\|test`) | Console loss/accuracy + sklearn per-class report; `eval_<split>.json` next to the checkpoint |
| `predict.py` | `--model` + `--image` | Top-k classes with probabilities (grayscale handling for MNIST models) |
| `benchmark.py` | `--config` or `--model` | Params, size (MB), batch-1 latency (mean±std), forward and train-step throughput; `outputs/benchmark_<model>.json`. Uses random weights (offline-safe) and warmup + `cuda.synchronize` for honest numbers |
| `export.py` | `--checkpoint` (+ `--format torchscript\|onnx`) | Deployable model file traced on CPU |

The CLI overrides in `train.py` exist for Slurm-based hyperparameter sweeps:
`jobs/train_thfood.sh` forwards `"$@"`, so
`sbatch jobs/train_thfood.sh --name exp2 --lr 0.001` needs no file edits.

### `jobs/` — Slurm integration (LANTA)

Shared pattern in all four scripts: `#SBATCH` resources → `cd
$SLURM_SUBMIT_DIR` → `source ./project.env` → `module load Mamba` +
`conda activate hpc-ai` → `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` → run
`"$HPCAI_PROJECT_DIR/scripts/<script>.py"`. `conda activate hpc-ai` resolves
by *name* — even though the environment physically lives on `/project` — 
because `setup_user.sh`/`setup_project.sh` register `/project/.../envs` in
`~/.condarc` via `conda config --append envs_dirs`; only `HPCAI_PROJECT_DIR`
(for locating the code) still needs the full path from `project.env`.

| Script | Partition | Extras |
|--------|-----------|--------|
| `train_cpu.sh` | `compute` | Lab 1 CPU half |
| `train_gpu.sh` | `gpu` (1× A100) | Background `nvidia-smi -l 5` sampler → `logs/gpu-usage-<jobid>.csv` |
| `train_thfood.sh` | `gpu` | Forwards extra args to `train.py` for sweeps |
| `train_thfood_multigpu.sh` | `gpu` | Same as `train_thfood.sh`, but launches via `torchrun --standalone --nproc_per_node=<N>` for `N` GPUs on one node (DDP, see §9) |
| `train_thfood_multinode.sh` | `gpu` | `srun torchrun` with a `c10d` rendezvous across multiple nodes (DDP, see §9) |
| `benchmark.sh` | `gpu` | Benchmarks all four models in one job |

#### Quota split: shared `/project` vs. personal `$HOME`

These scripts are meant to be submitted from a student's **personal
workspace** (`~/hpc-ai-workshop/`, created once by `setup_user.sh`), not
from the shared repo checkout on `/project`. This matters because LANTA
home directories carry a small quota (e.g. 100 GB / 600k inodes) while
`/project` is much larger (e.g. 30 TB / 300M inodes):

- **On `/project`, set up once** via `setup_project.sh`: the repo itself
  (`scripts/`, `datasets/`, `models/`, `trainer/`), the conda environment
  (built with `conda env create -p ./envs/hpc-ai`, i.e. a path-prefixed env
  *inside* the checkout, never under `$HOME/.conda`), and the datasets.
- **In `$HOME`, per student** via `setup_user.sh`: only the small, personal
  parts — copies of `configs/` and `jobs/` (meant to be edited) and empty
  `checkpoints/`, `outputs/`, `logs/` directories (where a student's own
  runs land), plus a `scripts/` symlink back to the shared `/project`
  checkout (so `python scripts/train.py ...` works with a relative path
  from the student's workspace, without duplicating the code). A generated
  `project.env` records `HPCAI_PROJECT_DIR` so the
  copied `jobs/*.sh` scripts know where the shared code lives, while still
  running relative to the student's own workspace — so `configs/`,
  `outputs/`, `logs/`, `checkpoints/` all resolve to `$HOME`, never to
  `/project`. `setup_user.sh` also runs
  `conda config --append envs_dirs "$PROJECT_DIR/envs"` once, which edits
  only the student's own `~/.condarc` (a few bytes) so `conda activate
  hpc-ai` finds the shared, path-prefixed environment by name — no path to
  remember or type. Because jobs run with the student's workspace (not
  `/project`) as the working directory, `setup_user.sh` also `sed`-rewrites
  each freshly-copied config's `dataset.root` from the default relative
  `./data/...` to an absolute path under the shared project's `data/` —
  otherwise it would silently resolve to a nonexistent folder under
  `$HOME` and fail with `FileNotFoundError`. This only happens on the
  initial copy, never on a config a student has already edited.

Similarly, `project.env` exports `TORCH_HOME="$PROJECT_DIR/cache/torch"`.
PyTorch's default pretrained-weight cache (`$HOME/.cache/torch`) is
per-user, so without this override, every pretrained model (`resnet18`,
`mobilenetv3`, `efficientnet_b0`) would try to re-download on each
student's first run — and fail outright, since compute nodes have no
internet. `setup_project.sh` prints the matching `export TORCH_HOME=...`
for the instructor's one-time login-node download, so the cache the
instructor populates is the same one every student's job reads from.

This split is optional: `setup.sh` (creating a plain named `hpc-ai` env) still
works for a single person with their own quota who doesn't need it.

---

## 4. Artifact Formats

### Checkpoint (`best.pt` / `last.pt`)

Saved with `torch.save`; loaded with `weights_only=False` (it contains
plain-Python metadata, so only load trusted files):

```python
{
    "epoch":           int,          # epoch this checkpoint was written at
    "model_state":     state_dict,
    "optimizer_state": state_dict,
    "scheduler_state": state_dict | None,
    "best_val_acc":    float,        # fraction, 0..1
    "config":          dict,         # the full YAML config
    "class_names":     list[str],    # index -> label mapping
}
```

`config` + `class_names` are what make downstream scripts checkpoint-only.

### `metrics.json`

Rewritten in full after **every** epoch (stays valid if Slurm kills the job):

```json
{
  "experiment": "mnist_gpu",
  "best_val_acc": 0.9912,
  "epochs": [
    {"epoch": 1, "train_loss": 0.24, "train_acc": 0.925,
     "val_loss": 0.07, "val_acc": 0.977,
     "epoch_time_sec": 11.2, "images_per_sec": 5357.0, "lr": 0.001}
  ]
}
```

### TensorBoard scalars (per epoch)

`Loss/train`, `Loss/val`, `Accuracy/train`, `Accuracy/val`,
`Time/epoch_seconds`, `Throughput/images_per_sec`, `LR` — the shared parent
`logs/` directory means `tensorboard --logdir logs` overlays all experiments.

---

## 5. Reproducibility Model

- `experiment.seed` seeds Python `random`, NumPy, and PyTorch (CPU + all GPUs).
- `experiment.deterministic: true` additionally forces deterministic cuDNN
  kernels (bit-exact reruns, slower); default `false` enables
  `cudnn.benchmark` for speed.
- The exact config is snapshotted to `outputs/<name>/config.yaml` at Trainer
  construction **after** CLI overrides are applied — what ran is what's saved.
- Residual nondeterminism (educational caveat): DataLoader worker scheduling
  and some CUDA atomics can still vary unless deterministic mode is on.

---

## 6. Extension Points

| To add… | Touch | Pattern |
|---------|-------|---------|
| A dataset | `datasets/<name>.py` + one branch in `datasets/__init__.py` | Return loaders + `class_names`; reuse `transforms.py` |
| A model | `models/<name>.py` + one branch in `models/__init__.py` | Builder `f(num_classes, pretrained, freeze_backbone) -> nn.Module` |
| An optimizer / scheduler | `trainer/utils.py` (`build_optimizer` / `build_scheduler`) | One `if name == ...` branch |
| A loss | `trainer/losses.py` | Same |
| A metric | `trainer/metrics.py`, log it in `Trainer._log_epoch` | Keep `engine.py` returns as plain dicts |

Everything new becomes YAML-selectable immediately — no script changes.

---

## 7. Dependency Map (imports between packages)

```
scripts/*  ──►  datasets, models, trainer          (top-level glue)
trainer/trainer.py ──► trainer/engine.py ──► trainer/metrics.py
trainer/*  ──►  (never imports datasets or models)
datasets/* ──►  torchvision only
models/*   ──►  torchvision only
```

`trainer/` is dataset- and model-agnostic: it only sees `nn.Module` and
`DataLoader`. That separation is what makes Lab 1 and Lab 2 run through the
exact same training code. `trainer/` importing `torch.distributed` (for DDP,
§9) doesn't change this — it's still only reasoning about generic PyTorch
objects. `datasets/thfood100.py` takes `rank`/`world_size` as plain `int`
parameters rather than importing `trainer.utils.DistributedContext`, keeping
`datasets/* ──► torchvision only` intact.

---

## 8. Stack

Python 3.11 · PyTorch ≥ 2.3 · torchvision ≥ 0.18 · PyYAML · TensorBoard ·
tqdm · scikit-learn (evaluation reports) · NumPy — intentionally **no**
Lightning or other training frameworks: the loop itself is the curriculum.
`torch.distributed` (DDP) ships with PyTorch itself — no extra dependency.

---

## 9. Multi-GPU / Multi-Node Training (DistributedDataParallel)

Lab 2 / THFOOD-100 scales to multiple GPUs and multiple nodes with **no
config or source changes** — the same `configs/thfood_baseline.yaml` that
runs on one GPU also runs under `torchrun`. Lab 1 / MNIST does not get this
(see below) — its models are tiny enough that multi-GPU would add
complexity without a real speedup to demonstrate.

**How it's wired in:**

- `scripts/train.py` calls `trainer.utils.init_distributed(device)` right
  after seeding. It reads `RANK` / `WORLD_SIZE` / `LOCAL_RANK` from the
  environment — set by `torchrun`, absent for a plain `python
  scripts/train.py` run — and returns a `DistributedContext` (rank,
  world_size, local_rank, resolved device, `is_main_process`). Everything
  downstream branches on this one object instead of re-deriving distributed
  state.
- `datasets/thfood100.py::build_thfood_dataloaders` takes `rank`/
  `world_size` and, when `world_size > 1`, shards the **training** set with
  a `DistributedSampler(shuffle=True, drop_last=True)`. `drop_last=True`
  keeps every process's shard the same size each epoch — required so every
  process runs the same number of batches and enters DDP's per-batch
  gradient all-reduce together (a mismatch would deadlock). Validation/test
  loaders are left un-sharded.
- `Trainer.__init__` wraps the model in `nn.parallel.DistributedDataParallel`
  when `dist_ctx.enabled`. DDP is what actually implements data parallelism:
  every `.backward()` call all-reduces (averages) gradients across processes
  before the optimizer step, so training math is identical to single-GPU,
  just faster.
- **Only the main process (`rank == 0`) touches shared state**: creates
  `outputs/<name>/` and `logs/<name>/`, writes TensorBoard/`metrics.json`,
  prints progress, and saves checkpoints. Every other process trains
  silently.
- **Validation runs once**, on the main process, over the full (un-sharded)
  val set, then broadcasts the result (`trainer.utils.broadcast_scalars`) to
  every other process as a plain tensor — not `dist.broadcast_object_list`,
  which pickles arbitrary Python objects and hit a PyTorch/NCCL bug
  (`SymIntArrayRef expected to contain only concrete integers`) on some
  builds; since the broadcast values here are always plain floats, a tensor
  broadcast avoids that bug and is the more idiomatic tool anyway. This
  keeps LR-scheduler stepping and the early-stopping decision **identical
  on every process** — if they diverged, some process could `break` out of
  the epoch loop while others kept training, and the next DDP
  `.backward()` call would deadlock waiting for a gradient all-reduce that
  never comes from the process that already
  exited.
- **Per-epoch training stats are reduced across processes**
  (`Trainer._reduce_train_stats`) before logging: loss/accuracy are
  averaged, and `images_per_sec` is summed (this is the actual scaling
  number multi-GPU training exists to demonstrate — compare it against a
  single-GPU run's throughput).
- **Checkpoints stay self-contained and DDP-agnostic**: `_save_checkpoint`
  calls `trainer.utils.unwrap_model` before `.state_dict()`. A DDP-wrapped
  model prefixes every state-dict key with `module.` — saving it directly
  would silently break `evaluate.py` / `predict.py` / `export.py`, which
  load checkpoints into a plain, non-distributed model. This preserves the
  checkpoint contract described in §4 regardless of how a checkpoint was
  trained.
- `scripts/train.py` calls `trainer.utils.cleanup_distributed(dist_ctx)` in
  a `finally` block after `trainer.fit()`, tearing down the process group
  it opened.

**`training.batch_size` is per-process (per-GPU), not global** — the
effective batch size is `batch_size × world_size`. This is the standard DDP
convention and matches how `dataset.num_workers` already scales per-process
(each GPU process gets its own DataLoader workers).

**Launching it**, from `jobs/train_thfood_multigpu.sh` (one node) and
`jobs/train_thfood_multinode.sh` (multiple nodes, via `srun torchrun` with a
`c10d` rendezvous) — see §3's `jobs/` table and the README's "Multi-GPU /
multi-node training" section for the commands.
