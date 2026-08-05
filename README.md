# HPC AI/ML Training Workshop

Hands-on labs for **AI training on the LANTA supercomputer**: GPU acceleration,
Slurm job submission, reproducible experiments, and hyperparameter tuning —
all with plain PyTorch.

| Lab | Task | What you learn |
|-----|------|----------------|
| **Lab 1** | MNIST digit classification | CPU vs. GPU: training time, throughput, GPU utilization — same code, only the config changes |
| **Lab 2** | THFOOD-100 Thai food classification | Transfer learning + hyperparameter tuning — no source code edits, only YAML changes |

Everything is driven by **YAML configuration files**. Students never need to
edit Python source code to complete the labs.

---

## Folder Structure

```
training_ai_ml/
├── README.md
├── requirements.txt         # pip dependencies
├── environment.yml          # conda environment (Python 3.11)
├── setup.sh                 # one-time environment setup (solo use)
├── setup_project.sh         # one-time SHARED setup for a class (see below)
├── setup_user.sh            # one-time personal workspace setup for each student
│
├── configs/                 # one YAML file = one experiment
│   ├── default.yaml         #   fully documented reference config
│   ├── mnist_cpu.yaml       #   Lab 1: CPU
│   ├── mnist_gpu.yaml       #   Lab 1: GPU (identical except `device`)
│   ├── thfood_baseline.yaml #   Lab 2: ResNet-18 baseline
│   ├── thfood_sample.yaml   #   Lab 2: smoke test on the bundled sample data
│   └── thfood_competition.yaml  # Lab 2: your tuning playground
│
├── datasets/                # data loading (MNIST auto-download, THFOOD ImageFolder)
├── models/                  # LeNet-5, ResNet-18, MobileNetV3, EfficientNet-B0
├── trainer/                 # Trainer class, train/val loops, losses, metrics, utils
├── scripts/                 # train / evaluate / predict / benchmark / export
├── jobs/                    # Slurm submission scripts for LANTA
│
├── checkpoints/             # (optional) long-term storage for good checkpoints
├── logs/                    # TensorBoard events + Slurm output + GPU-usage CSVs
└── outputs/                 # per-experiment results (checkpoints, metrics, config copy)
```

---

## Installation

### On LANTA, solo (one person, one project quota)

```bash
module load Mamba/23.11.0-0     # LANTA's conda distribution
bash setup.sh                   # creates the 'hpc-ai' conda env (Python 3.11)
conda activate hpc-ai
```

### On LANTA, for a class (shared project quota — recommended)

Home directories on LANTA have a small quota (e.g. **100 GB / 600k inodes**),
while `/project` typically has a much larger one (e.g. **30 TB / 300M
inodes**). For a workshop with many students sharing one project account,
put the heavy, shared parts — the code, the conda environment, and the
datasets — on `/project` **once**, and give each student only a small
personal workspace in their own `$HOME` for the things they actually edit
and produce (configs, Slurm scripts, checkpoints, run outputs, logs).

**Instructor / project owner, once:**

```bash
git clone <repo-url> /project/tn999996-north/hpc-ai-workshop
cd /project/tn999996-north/hpc-ai-workshop
module load Mamba/23.11.0-0
bash setup_project.sh           # builds the conda env inside ./envs/hpc-ai
                                 # (never touches anyone's home quota)
```

`setup_project.sh` prints the next steps: pre-downloading MNIST and the
ImageNet weights on a login node, and where to place the full THFOOD-100
dataset (`data/thfood100/`) — all under the shared project directory.
PyTorch's pretrained-weight cache (`TORCH_HOME`) is per-user by default, so
this download must be pointed at a shared path under `/project` (the
printed instructions show the exact `export TORCH_HOME=...` command) —
otherwise every student's job would try and fail to re-download the same
weights from a no-internet compute node. `setup_user.sh` sets this up
automatically for each student via `project.env`.

**Each student, once:**

```bash
bash /project/tn999996-north/hpc-ai-workshop/setup_user.sh
```

This creates `~/hpc-ai-workshop/` containing your own copies of `configs/`
and `jobs/` (edit these freely) plus empty `checkpoints/`, `outputs/`, and
`logs/` directories for your runs. It also registers the project's `envs/`
directory in your own `~/.condarc`, so **`conda activate hpc-ai` works by
name from anywhere** — the environment itself still lives entirely on
`/project`, never in your home quota. A `project.env` file records where
the shared code lives, so the job scripts in `jobs/` know where to find it.
Jobs run with this workspace (not `/project`) as the working directory, so
`setup_user.sh` also rewrites the copied configs' `dataset.root` from the
default relative `./data/...` to an absolute path under the shared
project — otherwise it would resolve to a nonexistent folder under
`$HOME`. From then on, work entirely from your personal workspace:

```bash
cd ~/hpc-ai-workshop
# edit jobs/*.sh: set #SBATCH --account=ltXXXXXX to your LANTA account
sbatch jobs/train_cpu.sh
```

### Anywhere else

```bash
conda env create -f environment.yml && conda activate hpc-ai
# or, with plain pip in a Python 3.11 environment:
pip install -r requirements.txt
```

### Pre-download data and weights (important on clusters!)

LANTA **compute nodes have no internet access**, so download everything once
on a **login node**. In the shared class setup above, the instructor does
this once under the shared project directory (`setup_project.sh` prints
these same commands) — students don't need to repeat it.

```bash
# MNIST (~12 MB)
python datasets/download.py --dataset mnist --root ./data

# ImageNet weights for Lab 2 (cached in ~/.cache/torch)
python -c "import torchvision.models as m; \
    m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1); \
    m.mobilenet_v3_large(weights=m.MobileNet_V3_Large_Weights.IMAGENET1K_V2); \
    m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.IMAGENET1K_V1)"
```

---

## Lab 1 — CPU vs. GPU (MNIST)

The two configs [mnist_cpu.yaml](configs/mnist_cpu.yaml) and
[mnist_gpu.yaml](configs/mnist_gpu.yaml) are **identical except for
`device`**. Train with both and compare.

### Running locally / interactively

```bash
python scripts/train.py --config configs/mnist_cpu.yaml
python scripts/train.py --config configs/mnist_gpu.yaml
```

### Running on LANTA via Slurm

Edit the `#SBATCH --account=ltXXXXXX` line in the job scripts first, then:

```bash
sbatch jobs/train_cpu.sh     # queued on the 'compute' (CPU) partition
sbatch jobs/train_gpu.sh     # queued on the 'gpu' partition (1x A100)
squeue --me                  # watch your jobs
```

### What to observe

Each epoch prints one summary line:

```
Epoch   1/5 | train loss 0.2431 acc 92.51% | val loss 0.0705 acc 97.72% |   11.2s    5357 img/s | lr 1.00e-03
```

Fill in a comparison table from your two runs:

| Metric | Where to find it | CPU | GPU |
|--------|------------------|-----|-----|
| Time per epoch (s) | epoch summary line / `metrics.json` | | |
| Throughput (img/s) | epoch summary line / `metrics.json` | | |
| Final val accuracy | end-of-training summary | | |
| GPU utilization (%) | `logs/gpu-usage-<jobid>.csv` (GPU job only) | — | |

Questions to discuss:
1. Accuracy is (almost) identical — why?
2. The speedup is large but the GPU utilization is low — what is the bottleneck for such a tiny model?
3. What happens to throughput if you double `training.batch_size`? If you set `training.amp: true`?

---

## Lab 2 — Thai Food Classification (THFOOD-100)

### 1. Prepare the dataset

THFOOD-100 is **not** downloaded automatically. Get it from your instructor
(or the shared project directory on LANTA) and arrange it as
`torchvision.datasets.ImageFolder` splits:

```
data/thfood100/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg      # optional — falls back to val
```

Verify the layout:

```bash
python datasets/download.py --dataset thfood100 --root ./data/thfood100
```

Don't have the full dataset yet? This repo bundles a tiny preview at
`data/THFOOD-100.sample/` (flat layout, no train/val/test folders, only a
handful of images per class) — enough to smoke-test the pipeline with
`configs/thfood_sample.yaml`, though not enough for meaningful accuracy.
See [ARCHITECTURE.md](ARCHITECTURE.md) for how the flat layout is handled.

### 2. Train the baseline

```bash
python scripts/train.py --config configs/thfood_baseline.yaml
# or on LANTA:
sbatch jobs/train_thfood.sh
```

The baseline fine-tunes an **ImageNet-pretrained ResNet-18** — the backbone
already knows edges, textures, and shapes, so only a few epochs are needed
to adapt it to 100 Thai dishes (this is *transfer learning*).

### 3. Tune hyperparameters — YAML only!

Copy [thfood_competition.yaml](configs/thfood_competition.yaml), rename the
experiment for every attempt, and tune **only** these knobs:

| Knob | Config key | Things to try |
|------|------------|---------------|
| Model | `model.name` | `resnet18`, `mobilenetv3`, `efficientnet_b0` |
| Batch size | `training.batch_size` | 32, 64, 128, 256 |
| Learning rate | `training.lr` | 0.0001 … 0.01 |
| Epochs | `training.epochs` | 5 … 30 (early stopping saves wasted time) |
| Optimizer | `optimizer.name` | `SGD`, `Adam`, `AdamW` |
| Scheduler | `scheduler.name` | `none`, `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau` |

Quick experiments without editing any file (extra args are forwarded to
`train.py`):

```bash
sbatch jobs/train_thfood.sh --name thfood_lr001  --lr 0.001
sbatch jobs/train_thfood.sh --name thfood_bs128  --batch-size 128 --lr 0.0006
```

Every run gets its own `outputs/<name>/` and `logs/<name>/` — compare them
all at once in TensorBoard.

### 4. Evaluate and predict

```bash
python scripts/evaluate.py --checkpoint outputs/thfood_baseline/best.pt          # test split + per-class report
python scripts/predict.py  --model outputs/thfood_baseline/best.pt --image sample.jpg
python scripts/benchmark.py --config configs/thfood_baseline.yaml                # speed & size
python scripts/export.py   --checkpoint outputs/thfood_baseline/best.pt          # TorchScript
```

### 5. Multi-GPU / multi-node training

The same [thfood_baseline.yaml](configs/thfood_baseline.yaml) trains on
several GPUs — one node or several — with **no config or code changes**,
using PyTorch's `DistributedDataParallel` (DDP): each GPU trains on its own
shard of the data, and gradients are averaged across GPUs every step, so the
math is identical to a single-GPU run, just faster.

```bash
# Locally / interactively, e.g. 2 GPUs on one machine:
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/thfood_baseline.yaml

# On LANTA via Slurm:
sbatch jobs/train_thfood_multigpu.sh     # 1 node, several GPUs (edit --gpus-per-node)
sbatch jobs/train_thfood_multinode.sh    # several nodes (edit --nodes / --gpus-per-node)
```

`training.batch_size` in the config is the **per-GPU** batch size — the
effective (global) batch size is `batch_size × total GPUs`. When comparing
against a single-GPU baseline, scale the learning rate up accordingly (a
common starting point is linear scaling with the global batch size).

`images/sec` in the epoch summary line and TensorBoard is the **aggregate**
throughput across every GPU — that's the number to compare against the
single-GPU run from step 2 to see how well training scales.

---

## TensorBoard

Every run logs loss, accuracy, learning rate, epoch time, and throughput to
`logs/<experiment_name>/`.

**Locally:**

```bash
tensorboard --logdir logs
# open http://localhost:6006
```

**On LANTA** (TensorBoard runs on the login node, viewed through an SSH tunnel):

```bash
# terminal 1 — on LANTA:
conda activate hpc-ai
tensorboard --logdir logs --port 6006 --bind_all

# terminal 2 — on your laptop:
ssh -L 6006:localhost:6006 <username>@lanta.nstda.or.th
# then open http://localhost:6006 in your browser
```

Pointing `--logdir` at `logs/` (not a single run) overlays **all**
experiments in one dashboard — ideal for comparing tuning attempts.

---

## Checkpoints & Outputs

Each experiment writes a self-contained results directory:

```
outputs/<experiment_name>/
├── config.yaml      # exact config used  -> full reproducibility
├── best.pt          # weights with the highest validation accuracy
├── last.pt          # weights after the most recent epoch
├── metrics.json     # per-epoch: loss, accuracy, epoch time, images/sec, lr
└── eval_test.json   # written by evaluate.py
```

Checkpoints store the model weights **plus** the optimizer/scheduler state,
the config, and the class names — so `evaluate.py`, `predict.py`, and
`export.py` need nothing but the `.pt` file. Copy checkpoints worth keeping
into `checkpoints/` (files in `outputs/` may be overwritten by re-runs).

---

## Expected Outputs

Numbers vary with hardware and node load — these are ballpark figures to
sanity-check your runs.

**Lab 1 — MNIST, LeNet-5, 5 epochs, batch 128:**

| | CPU (16 cores) | 1x A100 GPU |
|--|--|--|
| Time per epoch | ~30–60 s | ~5–10 s |
| Throughput | ~1,000–2,000 img/s | ~6,000–12,000 img/s |
| Final val accuracy | ~99% | ~99% (same math, same result) |

**Lab 2 — THFOOD-100 baseline (ResNet-18, 5 epochs, 1x A100):**

| | Value |
|--|--|
| Time per epoch | a few minutes (depends on dataset size and I/O) |
| Val accuracy after 5 epochs | roughly 70–85% |
| Well-tuned (competition) | higher — that's your job! |

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Job hangs / download error on a compute node | Download MNIST and ImageNet weights on a **login node** first (see Installation) |
| `URLError: Network is unreachable` downloading ResNet/MobileNet/EfficientNet weights | In the shared class setup, `TORCH_HOME` must point at the shared cache under `/project` (see Installation) — re-run `setup_user.sh` to regenerate `project.env` if it's missing `TORCH_HOME` |
| `config requests CUDA but no GPU is available` | You are on a CPU node — the run continues on CPU; use the `gpu` partition for GPU runs |
| `THFOOD-100 split not found` | Check the ImageFolder layout with `python datasets/download.py --dataset thfood100` |
| Out-of-memory on GPU | Reduce `training.batch_size` (halve it until it fits) |
| DataLoader is the bottleneck (low GPU utilization) | Raise `dataset.num_workers` to match `--cpus-per-task` |
| Two runs overwrite each other | Give every run a unique `experiment.name` (or `--name`) |
