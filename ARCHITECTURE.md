# Project Architecture — ภาพรวมทางเทคนิค (Technical Overview)

เอกสารนี้อธิบาย **โครงสร้างภายในของ repository** ว่าแต่ละโมดูลทำหน้าที่อะไร
ข้อมูลไหลผ่านโค้ดอย่างไรในระหว่างการฝึกโมเดลหนึ่งครั้ง (training run) และจะขยาย
(extend) ระบบได้จากจุดไหนบ้าง สำหรับคำแนะนำการใช้งานแลป ดูที่ [README.md](README.md)

---

## 1. หลักการออกแบบ (Design Principles)

1. **Config-driven** — ไฟล์ YAML หนึ่งไฟล์อธิบายการทดลองหนึ่งครั้งอย่างครบถ้วน
   นักเรียนแก้ไขแค่ YAML ไม่แตะ Python เลย ทุกพารามิเตอร์ที่ปรับได้ (dataset,
   model, optimizer, scheduler, batch size, LR, epochs, device) อยู่ใน config
   ทั้งหมด
2. **Same code, different config** — Lab 1 (เปรียบเทียบ CPU กับ GPU) ทำงานได้
   เพราะ code path เหมือนกันทุกประการ ต่างกันแค่ `device:` ระหว่าง config
   เท่านั้น
3. **Factory pattern, minimal OOP** — แต่ละ package มี entry point เดียวคือ
   `build_*` ที่แปลงส่วนหนึ่งของ config ให้กลายเป็น PyTorch object คลาสเดียวที่
   มี state จริงคือ `Trainer` ส่วนลูปการทำงานหลัก (hot loops) เป็นแค่ฟังก์ชัน
   ธรรมดา
4. **Self-contained artifacts** — checkpoint หนึ่งไฟล์พก config และชื่อคลาส
   (class names) ของตัวเองไปด้วย ทำให้ `evaluate.py` / `predict.py` /
   `export.py` ต้องการแค่ไฟล์ `.pt` เท่านั้น โฟลเดอร์ของแต่ละการทดลองทำซ้ำ
   (reproducible) ได้อย่างสมบูรณ์
5. **HPC-aware** — จับเวลาบน GPU อย่างถูกต้อง (`torch.cuda.synchronize`),
   รองรับ AMP, เปิดให้ปรับ `num_workers`/`pin_memory` ผ่าน config, สคริปต์
   Slurm ที่บันทึกการใช้งาน GPU, การจัดการข้อมูล/weights แบบ offline-friendly
   และการแบ่ง workspace ระหว่าง shared `/project` checkout กับ `$HOME`
   ส่วนตัวของนักเรียนที่คำนึงถึง quota (ดู `jobs/` ด้านล่าง)
6. **Scale ไปสู่ multi-GPU / multi-node ได้โดยไม่ต้องแก้ config หรือโค้ด**
   (Lab 2 / THFOOD-100) — รัน config ไฟล์เดิมด้วย `torchrun` แทน `python`
   แล้ว `Trainer` จะฝึกโมเดลด้วย `DistributedDataParallel` ข้ามทุก GPU ที่
   ได้รับมา ดู §9

---

## 2. Data Flow ระดับสูง (High-Level Data Flow)

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

สคริปต์ปลายทาง (downstream scripts) ใช้แค่ checkpoint เป็น input:

```
best.pt ──► evaluate.py   (per-class report, eval_<split>.json)
        ──► predict.py    (top-k for one image)
        ──► export.py     (TorchScript / ONNX)
```

---

## 3. หน้าที่ของแต่ละโมดูล (Module Responsibilities)

### `configs/` — นิยามการทดลอง (experiment definitions)

| ไฟล์ | หน้าที่ |
|------|---------|
| `default.yaml` | config อ้างอิงที่มีคำอธิบายครบทุกคีย์และค่าที่เป็นไปได้ |
| `mnist_cpu.yaml` / `mnist_gpu.yaml` | คู่ config ของ Lab 1 — เหมือนกันทุกอย่าง ต่างแค่ `device:` และ `experiment.name` |
| `thfood_baseline.yaml` | baseline ของ Lab 2: ResNet-18 แบบ pretrained, AdamW, cosine schedule, AMP |
| `thfood_competition.yaml` | template สำหรับปรับแต่ง (tuning) ของ Lab 2 มีเครื่องหมาย `# <-- tune me` กำกับ |
| `thfood_sample.yaml` | ทดสอบระบบเบื้องต้น (smoke test) บนข้อมูลตัวอย่าง `THFOOD-100.sample` ที่แนบมาให้ (flat layout, ~5 รูป/คลาส) — ยืนยันว่า pipeline รันได้ ไม่ได้ใช้วัดความแม่นยำจริง |

config แต่ละไฟล์ **self-contained** (ไม่มีการสืบทอด/inheritance หรือ merge กัน)
โค้ดอ่านค่าจาก config แบบป้องกันความผิดพลาด (defensive) ด้วย `.get(key,
default)` ดังนั้นคีย์ทางเลือก (optional) ที่ขาดหายไปจะไม่ทำให้การรัน crash

โครงสร้างระดับบนสุด (top-level schema): `experiment`, `dataset`, `model`,
`training`, `loss`, `optimizer`, `scheduler`, `device`

### `datasets/` — data pipeline

| โมดูล | สัญลักษณ์หลัก | หน้าที่ |
|--------|-----------|------|
| `__init__.py` | `build_dataloaders(config)` | entry point เดียว; เลือกเส้นทางตาม `dataset.name`; คืนค่า `(train_loader, val_loader, test_loader, class_names)` |
| `mnist.py` | `build_mnist_dataloaders` | MNIST ที่ดาวน์โหลดอัตโนมัติ; test set ทางการขนาด 10k ใช้เป็น val split ไปด้วย |
| `thfood100.py` | `build_thfood_dataloaders` | `ImageFolder` บน `root/{train,val,test}`; `test` จะ fallback ไปใช้ `val` ถ้าไม่มี ถ้า `root` ไม่มีโฟลเดอร์ `train/` แต่มี sub-directory ของแต่ละคลาสอยู่ตรงๆ (โครงสร้างแบบ flat ของ `THFOOD-100.sample`) จะคำนวณ stratified train/val/test split ต่อคลาสให้แทน และอ่าน `class_labels.csv` (ถ้ามี) เพื่อใช้ชื่อคลาสที่อ่านเข้าใจง่าย จะแจ้ง error พร้อมคำแนะนำโครงสร้างไฟล์ ถ้าไม่พบทั้งสอง layout |
| `transforms.py` | `build_mnist_transforms`, `build_image_transforms`, `build_eval_transform` | ค่าคงที่สำหรับ normalization + ระดับการทำ augmentation `none / basic / strong`; pipeline สำหรับ eval เป็นแบบ deterministic เสมอ (resize → center-crop) |
| `download.py` | CLI | เครื่องมือสำหรับ login node: ดาวน์โหลด MNIST, ตรวจสอบโครงสร้าง THFOOD-100 พร้อมนับจำนวนรูปแต่ละ split |

`class_names` ได้มาจากข้อมูลจริง (ชื่อโฟลเดอร์ของ ImageFolder เรียงตามลำดับ)
และ `num_classes` ใช้ค่าจาก `len(class_names)` เสมอ — config กับข้อมูลจริง
จึงไม่มีทางขัดแย้งกันได้

### `models/` — model zoo

| โมดูล | Builder | หมายเหตุ |
|--------|---------|-------|
| `lenet.py` | `build_lenet` → `LeNet5(nn.Module)` | พารามิเตอร์ ~60,000 ตัว ฝึกจากศูนย์ (from scratch); เป็นสถาปัตยกรรมเดียวที่เขียนเองในโปรเจกต์ (Lab 1) |
| `resnet18.py` | `build_resnet18` | backbone จาก torchvision; แทนที่ `model.fc` |
| `mobilenetv3.py` | `build_mobilenetv3` | มีรุ่น `large`/`small`; แทนที่ `classifier[3]` |
| `efficientnet.py` | `build_efficientnet_b0` | แทนที่ `classifier[1]` |
| `__init__.py` | `build_model(config, num_classes)` | เลือก builder ตามชื่อโมเดล |

ข้อตกลงสำหรับ transfer learning (โมเดล pretrained ทุกตัว): โหลด weights ของ
ImageNet ก่อน → เลือกได้ว่าจะ freeze backbone หรือไม่ (**ก่อน** เปลี่ยน head
ใหม่ เพื่อให้ `nn.Linear` ตัวใหม่ยังเทรนได้) → แทนที่ layer สุดท้ายด้วย layer
ขนาดเท่ากับ `num_classes`

### `trainer/` — แกนหลักของการฝึกโมเดล (training core)

| โมดูล | เนื้อหา |
|--------|----------|
| `trainer.py` | คลาส `Trainer` — ส่วนประกอบเดียวที่มี state จริง มีหน้าที่: ควบคุมลูปแต่ละ epoch, สั่ง scheduler ทำงาน (รวมกรณีพิเศษของ `ReduceLROnPlateau` ที่อิงตาม metric), TensorBoard writer, บันทึก checkpoint `best.pt`/`last.pt`, `metrics.json`, early stopping แบบเลือกเปิดได้ (patience บน val accuracy) สร้าง `outputs/<name>/` + `logs/<name>/` และบันทึกสำเนา config ไว้ตอนสร้างออบเจกต์ |
| `engine.py` | ลูปการทำงานหลักในรูปแบบ **ฟังก์ชันธรรมดา**: `train_one_epoch` (วงจร 5 ขั้นตอนต่อ batch: ย้ายข้อมูล → forward → backward → update → บันทึกผล; มี branch สำหรับ AMP ผ่าน `GradScaler`; จับเวลาแบบ sync กับ GPU), `validate` (`@torch.no_grad` + `model.eval()`), `predict_all` (รวบรวม label สำหรับรายงานของ sklearn) |
| `losses.py` | `build_loss` — CrossEntropyLoss ที่ปรับ label smoothing ได้ผ่าน config |
| `metrics.py` | `AverageMeter` (ค่าเฉลี่ยแบบถ่วงน้ำหนักตามจำนวนตัวอย่าง), `accuracy` (top-1), `topk_accuracy` |
| `utils.py` | Config I/O (`load_config`/`save_config`), `set_seed` (พร้อมสวิตช์ cuDNN determinism), `get_device` (fallback จาก CUDA ไป CPU พร้อม warning), `count_parameters`, `model_size_mb`, `build_optimizer` (SGD/Adam/AdamW; ข้าม parameter ที่ freeze ไว้), `build_scheduler` (StepLR/MultiStepLR/CosineAnnealingLR/ReduceLROnPlateau/none), `load_checkpoint` และฟังก์ชันช่วยสำหรับ DDP ตามที่อธิบายใน §9 (`init_distributed`, `cleanup_distributed`, `unwrap_model`, `reduce_mean`/`reduce_sum`/`reduce_max`, `broadcast_scalars`) |

เหตุผลที่แยกเป็นคลาส/ฟังก์ชัน: *ตัวลูปการทำงาน* (ส่วนที่นักเรียนต้องอ่าน) เป็น
ฟังก์ชัน flat ใน `engine.py` ส่วน *งานบัญชี/บันทึกผล* (bookkeeping — สิ่งที่
นักเรียนมองเป็น infrastructure ได้เลย) ถูกห่อหุ้มไว้ใน `Trainer`

### `scripts/` — จุดเข้าใช้งานผ่าน CLI (CLI entry points)

ทุกสคริปต์แทรก repo root เข้าไปใน `sys.path` จึงรันได้จาก working directory
ใดก็ได้ ทุกสคริปต์รองรับ `--help`

| สคริปต์ | Input | Output |
|--------|-------|--------|
| `train.py` | `--config` + override ทางเลือก (`--name --device --epochs --batch-size --lr`) | ผลการทดลองที่ฝึกแล้วใน `outputs/<name>/` + `logs/<name>/` |
| `evaluate.py` | `--checkpoint` (+ `--split val\|test`) | loss/accuracy ทาง console + รายงานรายคลาสจาก sklearn; `eval_<split>.json` ไว้ข้างๆ checkpoint |
| `predict.py` | `--model` + `--image` | คลาส top-k พร้อมความน่าจะเป็น (มีการจัดการภาพขาวดำสำหรับโมเดล MNIST) |
| `benchmark.py` | `--config` หรือ `--model` | จำนวนพารามิเตอร์, ขนาด (MB), latency ของ batch-1 (mean±std), throughput ของ forward และ train-step; `outputs/benchmark_<model>.json` ใช้ weights แบบสุ่ม (offline-safe) พร้อม warmup + `cuda.synchronize` เพื่อให้ตัวเลขน่าเชื่อถือ |
| `export.py` | `--checkpoint` (+ `--format torchscript\|onnx`) | ไฟล์โมเดลพร้อม deploy ที่ trace บน CPU |

override ผ่าน CLI ใน `train.py` มีไว้สำหรับการทำ hyperparameter sweep บน
Slurm: `jobs/train_thfood.sh` ส่งต่อ `"$@"` ให้ ดังนั้น
`sbatch jobs/train_thfood.sh --name exp2 --lr 0.001` ไม่ต้องแก้ไฟล์ใดๆ เลย

### `jobs/` — การผสานกับ Slurm (LANTA)

รูปแบบที่ใช้ร่วมกันในสคริปต์ทั้งสี่ตัว: ขอทรัพยากร `#SBATCH` → `cd
$SLURM_SUBMIT_DIR` → `source ./project.env` → `module load Mamba` +
`conda activate hpc-ai` → `OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK` → รัน
`"$HPCAI_PROJECT_DIR/scripts/<script>.py"` คำสั่ง `conda activate hpc-ai`
ทำงานผ่าน *ชื่อ* — แม้ว่า environment จะอยู่บน `/project` จริงๆ — เพราะ
`setup_user.sh`/`setup_project.sh` ลงทะเบียน `/project/.../envs` ไว้ใน
`~/.condarc` ผ่าน `conda config --append envs_dirs`; มีแค่
`HPCAI_PROJECT_DIR` (สำหรับหาตำแหน่งโค้ด) เท่านั้นที่ยังต้องใช้ path เต็มจาก
`project.env`

| สคริปต์ | Partition | ส่วนเสริม |
|--------|-----------|--------|
| `train_cpu.sh` | `compute` | ครึ่ง CPU ของ Lab 1 |
| `train_gpu.sh` | `gpu` (1× A100) | ตัวเก็บข้อมูล `nvidia-smi -l 5` ที่รันเบื้องหลัง → `logs/gpu-usage-<jobid>.csv` |
| `train_thfood.sh` | `gpu` | ส่งต่อ argument เพิ่มเติมให้ `train.py` สำหรับ sweep |
| `train_thfood_multigpu.sh` | `gpu` | เหมือน `train_thfood.sh` แต่รันผ่าน `torchrun --standalone --nproc_per_node=<N>` สำหรับ `N` GPU บน node เดียว (DDP, ดู §9) |
| `train_thfood_multinode.sh` | `gpu` | `srun python` — หนึ่ง process ต่อหนึ่ง GPU โดยตรง ไม่ผ่าน `torchrun` (DDP, ดู §9) |
| `benchmark.sh` | `gpu` | Benchmark โมเดลทั้งสี่ตัวในหนึ่ง job |

#### การแบ่ง quota: shared `/project` กับ personal `$HOME`

สคริปต์เหล่านี้ถูกออกแบบให้ submit จาก **workspace ส่วนตัว** ของนักเรียน
(`~/hpc-ai-workshop/` ที่สร้างครั้งเดียวโดย `setup_user.sh`) ไม่ใช่จาก shared
repo checkout บน `/project` เหตุผลคือ home directory บน LANTA มี quota จำกัด
(เช่น 100 GB / 600k inodes) ในขณะที่ `/project` ใหญ่กว่ามาก (เช่น 30 TB /
300M inodes):

- **บน `/project` ตั้งค่าครั้งเดียว** ผ่าน `setup_project.sh`: ตัว repo เอง
  (`scripts/`, `datasets/`, `models/`, `trainer/`), conda environment
  (สร้างด้วย `conda env create -p ./envs/hpc-ai` คือ environment แบบ
  path-prefixed *ภายใน* checkout ไม่ใช่ใต้ `$HOME/.conda`) และชุดข้อมูล
- **ใน `$HOME` ต่อนักเรียนหนึ่งคน** ผ่าน `setup_user.sh`: มีแค่ส่วนเล็กๆ ที่
  เป็นของส่วนตัว — สำเนาของ `configs/` และ `jobs/` (สำหรับแก้ไข) และโฟลเดอร์
  `checkpoints/`, `outputs/`, `logs/` ที่ว่างเปล่า (ที่ผลการรันของนักเรียนจะ
  ไปอยู่) พร้อม symlink `scripts/` ย้อนกลับไปยัง shared `/project` checkout
  (เพื่อให้ `python scripts/train.py ...` ใช้ path สัมพัทธ์จาก workspace
  ของนักเรียนได้เลย โดยไม่ต้องคัดลอกโค้ดซ้ำ) ไฟล์ `project.env` ที่สร้างขึ้น
  จะบันทึก `HPCAI_PROJECT_DIR` ไว้ เพื่อให้สคริปต์ `jobs/*.sh` ที่คัดลอกมา
  รู้ว่าโค้ดที่ใช้ร่วมกันอยู่ที่ไหน ในขณะที่ยังรันโดยอ้างอิงกับ workspace
  ของนักเรียนเอง — ดังนั้น `configs/`, `outputs/`, `logs/`, `checkpoints/`
  จึง resolve ไปที่ `$HOME` เสมอ ไม่ใช่ `/project` `setup_user.sh` ยังรัน
  `conda config --append envs_dirs "$PROJECT_DIR/envs"` ครั้งเดียว ซึ่งแก้
  แค่ `~/.condarc` ของนักเรียนเอง (ไม่กี่ byte) เพื่อให้ `conda activate
  hpc-ai` หา environment แบบ path-prefixed ที่ใช้ร่วมกันเจอด้วยชื่อ — ไม่
  ต้องจำหรือพิมพ์ path เอง เนื่องจาก job รันโดยใช้ workspace ของนักเรียน
  (ไม่ใช่ `/project`) เป็น working directory `setup_user.sh` จึงต้อง
  `sed`-rewrite `dataset.root` ของแต่ละ config ที่เพิ่งคัดลอกมา จาก path
  สัมพัทธ์เดิม `./data/...` ให้เป็น absolute path ภายใต้ `data/` ของ shared
  project แทน — มิเช่นนั้นจะ resolve ไปที่โฟลเดอร์ที่ไม่มีอยู่จริงใน `$HOME`
  อย่างเงียบๆ แล้วล้มเหลวด้วย `FileNotFoundError` การ rewrite นี้เกิดขึ้น
  แค่ตอนคัดลอกครั้งแรกเท่านั้น ไม่เกิดขึ้นซ้ำกับ config ที่นักเรียนแก้ไขไป
  แล้ว

ในทำนองเดียวกัน `project.env` export ค่า `TORCH_HOME="$PROJECT_DIR/cache/torch"`
ไว้ด้วย เพราะ cache ของ pretrained weights ใน PyTorch
(`$HOME/.cache/torch`) เป็นแบบต่อ user โดย default ถ้าไม่ override ค่านี้
โมเดล pretrained ทุกตัว (`resnet18`, `mobilenetv3`, `efficientnet_b0`) จะ
พยายามดาวน์โหลดใหม่ในการรันครั้งแรกของนักเรียนแต่ละคน — และล้มเหลวทันที
เพราะ compute node ไม่มีอินเทอร์เน็ต `setup_project.sh` จะแสดงคำสั่ง
`export TORCH_HOME=...` ที่ตรงกันสำหรับให้ผู้สอนดาวน์โหลดครั้งเดียวบน login
node เพื่อให้ cache ที่ผู้สอนสร้างไว้เป็น cache เดียวกับที่ job ของนักเรียน
ทุกคนอ่านได้

การแบ่งแบบนี้เป็นทางเลือก: `setup.sh` (สร้าง environment ชื่อ `hpc-ai`
ธรรมดา) ยังใช้งานได้สำหรับคนที่ใช้งานคนเดียวและมี quota เป็นของตัวเอง โดยไม่
จำเป็นต้องแบ่งแบบนี้

---

## 4. รูปแบบของ Artifact (Artifact Formats)

### Checkpoint (`best.pt` / `last.pt`)

บันทึกด้วย `torch.save`; โหลดด้วย `weights_only=False` (เพราะมี metadata
แบบ plain-Python อยู่ด้วย ดังนั้นให้โหลดเฉพาะไฟล์ที่เชื่อถือได้เท่านั้น):

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

`config` + `class_names` คือสิ่งที่ทำให้สคริปต์ปลายทางใช้แค่ checkpoint ไฟล์
เดียวได้

### `metrics.json`

เขียนทับทั้งไฟล์ใหม่หลัง **ทุก** epoch (ยังใช้งานได้แม้ Slurm จะ kill job
กลางคัน):

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

### TensorBoard scalars (ต่อ epoch)

`Loss/train`, `Loss/val`, `Accuracy/train`, `Accuracy/val`,
`Time/epoch_seconds`, `Throughput/images_per_sec`, `LR` — การใช้โฟลเดอร์แม่
`logs/` ร่วมกัน ทำให้ `tensorboard --logdir logs` แสดงผลทุกการทดลองซ้อนกันได้

---

## 5. โมเดลของ Reproducibility (Reproducibility Model)

- `experiment.seed` ใช้ seed ค่าเดียวกันกับ `random` ของ Python, NumPy, และ
  PyTorch (ทั้ง CPU และทุก GPU)
- `experiment.deterministic: true` จะบังคับให้ cuDNN kernel ทำงานแบบ
  deterministic เพิ่มเติม (รันซ้ำได้ผลลัพธ์เป๊ะ (bit-exact) แต่ช้าลง);
  ค่า default `false` จะเปิด `cudnn.benchmark` เพื่อความเร็ว
- config ที่ใช้จริงจะถูกบันทึกสำเนาไปที่ `outputs/<name>/config.yaml` ตอน
  สร้าง Trainer **หลังจาก** ใช้ CLI override แล้ว — สิ่งที่รันจริงคือสิ่งที่
  ถูกบันทึกไว้
- ความไม่แน่นอนที่ยังหลงเหลืออยู่ (ข้อควรระวังเชิงการศึกษา): ลำดับการทำงาน
  ของ DataLoader worker และ CUDA atomic บางตัวยังแปรผันได้ ถ้าไม่เปิดโหมด
  deterministic

---

## 6. จุดขยายระบบ (Extension Points)

| ต้องการเพิ่ม… | แก้ไขที่ | รูปแบบ |
|---------|-------|---------|
| Dataset | `datasets/<name>.py` + เพิ่ม branch หนึ่งใน `datasets/__init__.py` | คืนค่า loaders + `class_names`; ใช้ `transforms.py` ซ้ำได้ |
| Model | `models/<name>.py` + เพิ่ม branch หนึ่งใน `models/__init__.py` | Builder รูปแบบ `f(num_classes, pretrained, freeze_backbone) -> nn.Module` |
| Optimizer / Scheduler | `trainer/utils.py` (`build_optimizer` / `build_scheduler`) | เพิ่ม branch `if name == ...` หนึ่งอัน |
| Loss | `trainer/losses.py` | เหมือนกัน |
| Metric | `trainer/metrics.py` แล้ว log ใน `Trainer._log_epoch` | ให้ `engine.py` คืนค่าเป็น dict ธรรมดาเหมือนเดิม |

ทุกอย่างที่เพิ่มใหม่จะเลือกใช้ผ่าน YAML ได้ทันที — ไม่ต้องแก้สคริปต์เลย

---

## 7. แผนผัง Dependency (imports ระหว่าง package)

```
scripts/*  ──►  datasets, models, trainer          (top-level glue)
trainer/trainer.py ──► trainer/engine.py ──► trainer/metrics.py
trainer/*  ──►  (never imports datasets or models)
datasets/* ──►  torchvision only
models/*   ──►  torchvision only
```

`trainer/` ไม่รู้จักทั้ง dataset และ model ที่ใช้จริง (dataset- and
model-agnostic) — มองเห็นแค่ `nn.Module` และ `DataLoader` เท่านั้น การแยก
แบบนี้เองที่ทำให้ Lab 1 และ Lab 2 รันผ่านโค้ดฝึกโมเดลชุดเดียวกันได้ การที่
`trainer/` import `torch.distributed` (สำหรับ DDP, §9) ไม่ได้ทำให้หลักการนี้
เปลี่ยนไป — เพราะยังคงจัดการกับ PyTorch object ทั่วไปเหมือนเดิม
`datasets/thfood100.py` รับ `rank`/`world_size` เป็นพารามิเตอร์ `int` ธรรมดา
แทนที่จะ import `trainer.utils.DistributedContext` เพื่อรักษากฎ
`datasets/* ──► torchvision only` ไว้

---

## 8. Stack

Python 3.11 · PyTorch ≥ 2.3 · torchvision ≥ 0.18 · PyYAML · TensorBoard ·
tqdm · scikit-learn (สำหรับรายงานผลการประเมิน) · NumPy — ตั้งใจ **ไม่ใช้**
Lightning หรือ training framework อื่นๆ เพราะตัว training loop เองคือ
เนื้อหาหลักของหลักสูตร `torch.distributed` (DDP) มากับ PyTorch อยู่แล้ว
ไม่ต้องเพิ่ม dependency ใดๆ

---

## 9. การฝึกแบบ Multi-GPU / Multi-Node (DistributedDataParallel)

Lab 2 / THFOOD-100 สามารถ scale ไปสู่หลาย GPU และหลาย node ได้โดย **ไม่ต้อง
แก้ config หรือ source code เลย** — `configs/thfood_baseline.yaml` ไฟล์เดิม
ที่รันบน GPU เดียวได้ ก็รันผ่าน `torchrun` ได้เช่นกัน ส่วน Lab 1 / MNIST ไม่มี
ความสามารถนี้ (ดูเหตุผลด้านล่าง) — เพราะโมเดลมีขนาดเล็กเกินกว่าที่ multi-GPU
จะเพิ่มความซับซ้อนแล้วคุ้มค่ากับ speedup ที่จะแสดงให้เห็น

**สอง launch style ที่รองรับ**, ทั้งคู่เข้า `trainer.utils.init_distributed`
จุดเดียวกัน:

- **`torchrun`** (`jobs/train_thfood_multigpu.sh`, node เดียว) — ตั้งค่า
  `RANK` / `WORLD_SIZE` / `LOCAL_RANK` ให้เองผ่าน rendezvous ภายในของมันเอง
- **`srun python` โดยตรง** (`jobs/train_thfood_multinode.sh`, ข้ามหลาย
  node) — Slurm สั่ง process หนึ่งตัวต่อหนึ่ง GPU โดยตรง แล้ว rank มาจาก
  `SLURM_PROCID`/`SLURM_LOCALID`/`SLURM_NTASKS` แทน; job script เป็นคน
  export `MASTER_ADDR`/`MASTER_PORT` เอง วิธีนี้ตรงกับ pattern ที่ ThaiSC/
  LANTA ใช้เป็นทางการสำหรับ PyTorch multi-GPU/multi-node — เลี่ยงปัญหาที่
  rendezvous ของ `torchrun` เอง (เป็น TCP connection คนละอันกับ process
  group จริง) connect ข้าม node บนเครือข่ายของ LANTA ไม่ได้ ทั้งที่ตัว
  process group จริง (ผ่าน `env://` init เหมือนกัน) ใช้งานได้ปกติ

**วิธีที่ต่อเชื่อมเข้าไปในระบบ:**

- `scripts/train.py` เรียก `trainer.utils.init_distributed(device)` ทันที
  หลังจาก seed ค่าต่างๆ แล้ว ฟังก์ชันนี้ตรวจว่ามี `RANK` (จาก `torchrun`)
  อยู่ใน environment variable หรือไม่ ถ้าไม่มีจะ fallback ไปอ่าน
  `SLURM_PROCID`/`SLURM_LOCALID`/`SLURM_NTASKS` แทน (กรณี `srun python`
  โดยตรง) รันแบบ `python scripts/train.py` ธรรมดาจะไม่มีตัวแปรเหล่านี้เลย
  ฟังก์ชันจึงคืนค่าเป็น context แบบ single-process แทน แล้วคืนค่าเป็น
  `DistributedContext` (rank, world_size, local_rank, device ที่ resolve
  แล้ว, `is_main_process`) โค้ดส่วนถัดไปทั้งหมดจะแยกเงื่อนไขตามออบเจกต์นี้
  อันเดียว แทนที่จะคำนวณสถานะ distributed ซ้ำเองในแต่ละจุด สำหรับการเลือก
  GPU: ถ้า `torch.cuda.device_count()` เห็นแค่ 1 ตัว (Slurm จำกัด
  `CUDA_VISIBLE_DEVICES` ต่อ task ไว้แล้ว) จะใช้ index 0 เสมอ ไม่งั้นจะใช้
  `local_rank` เป็น index — รองรับได้ทั้งสองรูปแบบการตั้งค่า GPU binding ของ
  Slurm โดยไม่ต้องรู้ล่วงหน้าว่าคลัสเตอร์ตั้งค่าแบบไหน
- `datasets/thfood100.py::build_thfood_dataloaders` รับ `rank`/`world_size`
  และเมื่อ `world_size > 1` จะแบ่ง (shard) ชุดข้อมูล **train** ด้วย
  `DistributedSampler(shuffle=True, drop_last=True)` การใช้ `drop_last=True`
  ทำให้ shard ของทุก process มีขนาดเท่ากันทุก epoch — จำเป็นเพื่อให้ทุก
  process รันจำนวน batch เท่ากันและเข้าสู่ gradient all-reduce ของ DDP
  พร้อมกันในแต่ละ batch (ถ้าจำนวนไม่ตรงกันจะเกิด deadlock) ส่วน loader ของ
  validation/test จะไม่ถูกแบ่ง (un-sharded)
- `Trainer.__init__` จะห่อโมเดลด้วย `nn.parallel.DistributedDataParallel`
  เมื่อ `dist_ctx.enabled` เป็นจริง DDP คือสิ่งที่ทำหน้าที่ data parallelism
  จริงๆ: ทุกครั้งที่เรียก `.backward()` จะ all-reduce (เฉลี่ย) gradient ข้าม
  ทุก process ก่อนที่ optimizer จะ update ค่า ทำให้หลักการทางคณิตศาสตร์
  เหมือนกับการรันบน GPU เดียวทุกประการ เพียงแต่เร็วกว่า
- **มีแค่ main process (`rank == 0`) เท่านั้นที่แตะ shared state**: สร้าง
  `outputs/<name>/` และ `logs/<name>/`, เขียน TensorBoard/`metrics.json`,
  print progress, และบันทึก checkpoint ส่วน process อื่นทั้งหมดฝึกโมเดลไป
  เงียบๆ โดยไม่ทำสิ่งเหล่านี้ซ้ำ
- **Validation รันแค่ครั้งเดียว** บน main process เท่านั้น บนชุด val แบบเต็ม
  (ไม่ถูกแบ่ง) แล้ว broadcast ผลลัพธ์ (`trainer.utils.broadcast_scalars`)
  ไปยังทุก process ที่เหลือในรูปแบบ tensor ธรรมดา — ไม่ใช่
  `dist.broadcast_object_list` ซึ่ง pickle Python object ทั่วไปและเคยชนกับ
  บั๊กของ PyTorch/NCCL (`SymIntArrayRef expected to contain only concrete
  integers`) บางเวอร์ชัน เนื่องจากค่าที่ broadcast ในที่นี้เป็น float ธรรมดา
  เสมอ การ broadcast แบบ tensor จึงเลี่ยงบั๊กนั้นได้และเหมาะสมกว่าอยู่แล้ว
  วิธีนี้ทำให้การ step ของ LR scheduler และการตัดสินใจ early-stopping
  **เหมือนกันทุก process** — ถ้าค่าต่างกัน บาง process อาจ `break` ออกจาก
  epoch loop ในขณะที่ process อื่นยังฝึกต่อ แล้วการเรียก `.backward()` ของ
  DDP ครั้งถัดไปจะ deadlock เพราะรอ gradient all-reduce จาก process ที่
  ออกจากลูปไปแล้ว
- **สถิติของแต่ละ epoch ระหว่างการฝึกถูกรวมข้าม process**
  (`Trainer._reduce_train_stats`) ก่อนจะ log: loss/accuracy ใช้ค่าเฉลี่ย
  ส่วน `images_per_sec` ใช้ค่าผลรวม (นี่คือตัวเลข scaling จริงที่ multi-GPU
  training มีไว้เพื่อแสดงให้เห็น — เอาไปเทียบกับ throughput ของการรันบน
  GPU เดียวได้เลย)
- **Checkpoint ยังคง self-contained และไม่ยึดติดกับ DDP**: `_save_checkpoint`
  เรียก `trainer.utils.unwrap_model` ก่อน `.state_dict()` เพราะโมเดลที่ถูก
  ห่อด้วย DDP จะมี prefix `module.` นำหน้าทุกคีย์ใน state dict — ถ้าบันทึก
  ตรงๆ จะทำให้ `evaluate.py` / `predict.py` / `export.py` พังแบบเงียบๆ
  เพราะสคริปต์เหล่านี้โหลด checkpoint เข้าโมเดลแบบธรรมดาที่ไม่ใช่
  distributed วิธีนี้จึงรักษาข้อตกลงเรื่อง checkpoint ตามที่อธิบายไว้ใน §4
  ไว้ได้ ไม่ว่า checkpoint จะถูกฝึกมาแบบไหนก็ตาม
- `scripts/train.py` เรียก `trainer.utils.cleanup_distributed(dist_ctx)`
  ใน block `finally` หลังจาก `trainer.fit()` เพื่อปิด process group ที่เปิด
  ไว้

**`training.batch_size` เป็นค่าต่อ process (ต่อ GPU หนึ่งตัว) ไม่ใช่ค่ารวม
ทั้งหมด (global)** — batch size ที่ใช้จริงคือ `batch_size × world_size`
นี่คือธรรมเนียมมาตรฐานของ DDP และสอดคล้องกับที่ `dataset.num_workers` scale
ตาม process อยู่แล้ว (แต่ละ GPU process มี DataLoader worker เป็นของตัวเอง)

**วิธีรัน**: ผ่าน `jobs/train_thfood_multigpu.sh` (node เดียว, ผ่าน
`torchrun --standalone`) และ `jobs/train_thfood_multinode.sh` (หลาย node,
ผ่าน `srun python` โดยตรงหนึ่ง process ต่อหนึ่ง GPU — ดูเหตุผลด้านบน) — ดู
ตาราง `jobs/` ใน §3 และหัวข้อ "การฝึกแบบ Multi-GPU / Multi-node" ใน README
สำหรับคำสั่งที่ใช้
