# LANTA-HPC-AI-ML-Hands-on

แล็บฝึกปฏิบัติสำหรับ **AI/ML Training บน LANTA Supercomputer**: การเร่งความเร็วด้วย GPU,
การส่งงานผ่าน Slurm, การทดลองที่ทำซ้ำได้ (reproducible), และการปรับจูนไฮเปอร์พารามิเตอร์ —
ทั้งหมดใช้ PyTorch แบบพื้นฐาน

| แล็บ | งาน | สิ่งที่จะได้เรียนรู้ |
|-----|------|----------------|
| **Demo** | จำแนกตัวเลขลายมือ MNIST | เปรียบเทียบ CPU กับ GPU: เวลาที่ใช้เทรน, throughput, การใช้งาน GPU — โค้ดเดิมทุกอย่าง เปลี่ยนแค่ config |
| **Hands-on** | จำแนกอาหารไทย THFOOD-100 | Transfer learning + การปรับจูนไฮเปอร์พารามิเตอร์ — ไม่ต้องแก้ source code แก้แค่ YAML |

ทุกอย่างถูกควบคุมด้วย **ไฟล์คอนฟิก YAML** ผู้เรียนไม่จำเป็นต้องแก้ไข
โค้ด Python เพื่อทำแล็บให้เสร็จ

---

## โครงสร้างโฟลเดอร์

```
training_ai_ml/
├── README.md
├── requirements.txt         # pip dependencies
├── environment.yml          # conda environment (Python 3.11)
├── setup.sh                 # ติดตั้งสภาพแวดล้อมครั้งเดียว (ใช้คนเดียว)
├── setup_project.sh         # ติดตั้งส่วนกลาง (SHARED) สำหรับทั้งคลาส ครั้งเดียว (ดูรายละเอียดด้านล่าง)
├── setup_user.sh            # ติดตั้ง workspace ส่วนตัวครั้งเดียวสำหรับผู้เรียนแต่ละคน
├── cache                    # เก็บโมเดล checkpoint ResNet-18, MobileNetV3, EfficientNet-B0
│
├── configs/                 # ไฟล์ YAML 1 ไฟล์ = 1 การทดลอง
│   ├── default.yaml         #   คอนฟิกอ้างอิงที่มีคำอธิบายครบถ้วน
│   ├── mnist_cpu.yaml       #   Demo: CPU
│   ├── mnist_gpu.yaml       #   Demo: GPU (เหมือนกันทุกอย่าง ต่างแค่ `device`)
│   ├── thfood_baseline.yaml #   Hands-on: ResNet-18 baseline
│   ├── thfood_sample.yaml   #   Hands-on: ทดสอบระบบด้วยข้อมูลตัวอย่างที่แนบมาให้
│   └── thfood_competition.yaml  # Lab 2: พื้นที่ปรับจูนของคุณ
│
├── datasets/                # การโหลดข้อมูล (MNIST ดาวน์โหลดอัตโนมัติ, THFOOD แบบ ImageFolder)
├── data/                    # ข้อมูลดิบ MNIST และ THFOOD
├── models/                  # LeNet-5, ResNet-18, MobileNetV3, EfficientNet-B0
├── trainer/                 # คลาส Trainer, ลูปเทรน/วาลิเดต, loss, metrics, utils
├── scripts/                 # train / evaluate / predict / benchmark / export
├── jobs/                    # สคริปต์ส่งงาน Slurm สำหรับ LANTA
├── 
├── checkpoints/             # (ไม่บังคับ) พื้นที่เก็บ checkpoint ที่ดีไว้ระยะยาว
├── logs/                    # TensorBoard events + Slurm output + CSV การใช้งาน GPU
└── outputs/                 # ผลลัพธ์ของแต่ละการทดลอง (checkpoint, metrics, สำเนา config)
```

---

## การติดตั้ง Environmet

### บน LANTA แบบคนเดียว (บัญชีโปรเจกต์เดียว คนเดียวใช้)

```bash
module load Mamba/23.11.0-0     # ระบบ conda ของ LANTA
bash setup.sh                   # สร้าง conda env ชื่อ 'hpc-ai' (Python 3.11)
conda activate hpc-ai
```

### บน LANTA สำหรับทั้งคลาส (ใช้โควตาโปรเจกต์ร่วมกัน — แนะนำวิธีนี้)

โฮมไดเรกทอรีบน LANTA มีโควตาค่อนข้างน้อย (เช่น **100 GB / 600,000 inode**)
ในขณะที่ `/project` มักมีโควตาที่มากกว่ามาก (เช่น **30 TB / 300 ล้าน
inode**) สำหรับเวิร์กช็อปที่มีผู้เรียนหลายคนใช้บัญชีโปรเจกต์ร่วมกัน
ให้นำส่วนที่หนักและใช้ร่วมกัน — โค้ด, conda environment, และชุดข้อมูล —
ไปไว้บน `/project` **เพียงครั้งเดียว** แล้วให้ผู้เรียนแต่ละคนมี workspace
ส่วนตัวขนาดเล็กใน `$HOME` ของตัวเอง สำหรับสิ่งที่พวกเขาต้องแก้ไขและ
สร้างผลลัพธ์เอง (config, สคริปต์ Slurm, checkpoint, ผลลัพธ์การรัน, log)

**ผู้สอน / เจ้าของโปรเจกต์ ทำครั้งเดียว:**

```bash
git clone <repo-url> /project/tn999996-north/hpc-ai-workshop
cd /project/tn999996-north/hpc-ai-workshop
module load Mamba/23.11.0-0
bash setup_project.sh           # สร้าง conda env ไว้ใน ./envs/hpc-ai
                                 # (ไม่แตะโควตาโฮมของใครเลย)
```

`setup_project.sh` จะแสดงขั้นตอนถัดไป: การดาวน์โหลด MNIST และ
น้ำหนักโมเดล ImageNet ล่วงหน้าบน login node และตำแหน่งที่ควรวาง
ชุดข้อมูล THFOOD-100 แบบเต็ม (`data/thfood100/`) — ทั้งหมดอยู่ภายใต้
ไดเรกทอรีโปรเจกต์ที่ใช้ร่วมกัน โดยค่าเริ่มต้นแคชน้ำหนักโมเดลที่เทรนไว้แล้ว
ของ PyTorch (`TORCH_HOME`) จะแยกตามผู้ใช้แต่ละคน ดังนั้นการดาวน์โหลดนี้
ต้องชี้ไปยังพาธที่ใช้ร่วมกันภายใต้ `/project` (คำสั่ง `export TORCH_HOME=...`
ที่ถูกต้องจะแสดงในขั้นตอนที่พิมพ์ออกมา) — มิฉะนั้นงานของผู้เรียนแต่ละคน
จะพยายามดาวน์โหลดน้ำหนักโมเดลชุดเดียวกันซ้ำ ๆ และล้มเหลว เพราะ
compute node ไม่มีอินเทอร์เน็ต `setup_user.sh` จะตั้งค่านี้ให้อัตโนมัติ
สำหรับผู้เรียนแต่ละคนผ่านไฟล์ `project.env`

**ผู้เรียนแต่ละคน ทำครั้งเดียว:**

```bash
bash /project/tn999996-north/hpc-ai-workshop/setup_user.sh
```

คำสั่งนี้จะสร้าง `~/hpc-ai-workshop/` ซึ่งมีสำเนา `configs/` และ `jobs/`
ของคุณเอง (แก้ไขได้อย่างอิสระ) พร้อมทั้งไดเรกทอรีว่าง `checkpoints/`,
`outputs/`, และ `logs/` สำหรับการรันของคุณ นอกจากนี้ยังลงทะเบียน
ไดเรกทอรี `envs/` ของโปรเจกต์ไว้ใน `~/.condarc` ของคุณเอง ทำให้
**`conda activate hpc-ai` ใช้งานได้จากชื่อ env จากที่ไหนก็ได้** — ตัว
environment เองยังคงอยู่บน `/project` ทั้งหมด ไม่เคยอยู่ในโควตาโฮมของคุณ
ไฟล์ `project.env` จะบันทึกตำแหน่งของโค้ดที่ใช้ร่วมกัน เพื่อให้สคริปต์งาน
ใน `jobs/` รู้ว่าจะไปหาโค้ดได้ที่ไหน งานจะรันโดยใช้ workspace นี้ (ไม่ใช่
`/project`) เป็น working directory ดังนั้น `setup_user.sh` จึงเขียนทับ
ค่า `dataset.root` ในสำเนา config ให้เปลี่ยนจากพาธสัมพัทธ์เริ่มต้น
`./data/...` เป็นพาธสัมบูรณ์ภายใต้โปรเจกต์ที่ใช้ร่วมกัน — มิฉะนั้นค่านี้จะ
ชี้ไปยังโฟลเดอร์ที่ไม่มีอยู่จริงใต้ `$HOME` จากนั้นให้ทำงานทั้งหมดจาก
workspace ส่วนตัวของคุณ:

```bash
cd ~/hpc-ai-workshop
# แก้ไข jobs/*.sh: ตั้งค่า #SBATCH --account=ltXXXXXX ให้เป็นบัญชี LANTA ของคุณ
sbatch jobs/train_cpu.sh
```

### ที่อื่น ๆ

```bash
conda env create -f environment.yml && conda activate hpc-ai
# หรือใช้ pip แบบธรรมดาใน Python 3.11 environment:
pip install -r requirements.txt
```

### ดาวน์โหลดข้อมูลและน้ำหนักโมเดลล่วงหน้า (สำคัญมากบนคลัสเตอร์!)

**compute node ของ LANTA ไม่มีอินเทอร์เน็ต** ดังนั้นให้ดาวน์โหลดทุกอย่าง
ครั้งเดียวบน **login node** ในการติดตั้งแบบคลาสข้างต้น ผู้สอนจะทำ
ขั้นตอนนี้ครั้งเดียวภายใต้ไดเรกทอรีโปรเจกต์ที่ใช้ร่วมกัน (`setup_project.sh`
จะแสดงคำสั่งชุดเดียวกันนี้) — ผู้เรียนไม่จำเป็นต้องทำซ้ำ

```bash
# MNIST (~12 MB)
python datasets/download.py --dataset mnist --root ./data

# น้ำหนักโมเดล ImageNet สำหรับ Lab 2 (แคชไว้ที่ ~/.cache/torch)
python -c "import torchvision.models as m; \
    m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1); \
    m.mobilenet_v3_large(weights=m.MobileNet_V3_Large_Weights.IMAGENET1K_V2); \
    m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.IMAGENET1K_V1)"
```

---

## Lab 1 — CPU กับ GPU (MNIST)

คอนฟิกทั้งสองไฟล์ [mnist_cpu.yaml](configs/mnist_cpu.yaml) และ
[mnist_gpu.yaml](configs/mnist_gpu.yaml) **เหมือนกันทุกอย่าง ต่างกันแค่
`device`** ให้เทรนทั้งสองแบบแล้วเปรียบเทียบกัน

### รันแบบโลคอล / อินเทอร์แอกทีฟ

```bash
python scripts/train.py --config configs/mnist_cpu.yaml
python scripts/train.py --config configs/mnist_gpu.yaml
```

### รันบน LANTA ผ่าน Slurm

แก้ไขบรรทัด `#SBATCH --account=ltXXXXXX` ในสคริปต์งานก่อน จากนั้น:

```bash
sbatch jobs/train_cpu.sh     # เข้าคิวใน partition 'compute' (CPU)
sbatch jobs/train_gpu.sh     # เข้าคิวใน partition 'gpu' (1x A100)
squeue --me                  # ดูสถานะงานของคุณ
```

### สิ่งที่ต้องสังเกต

แต่ละ epoch จะพิมพ์สรุปหนึ่งบรรทัด:

```
Epoch   1/5 | train loss 0.2431 acc 92.51% | val loss 0.0705 acc 97.72% |   11.2s    5357 img/s | lr 1.00e-03
```

กรอกตารางเปรียบเทียบจากการรันทั้งสองแบบ:

| ตัวชี้วัด | หาได้จากไหน | CPU | GPU |
|--------|------------------|-----|-----|
| เวลาต่อ epoch (วินาที) | บรรทัดสรุป epoch / `metrics.json` | | |
| Throughput (img/s) | บรรทัดสรุป epoch / `metrics.json` | | |
| ความแม่นยำ (accuracy) สุดท้ายของ val | สรุปท้ายการเทรน | | |
| การใช้งาน GPU (%) | `logs/gpu-usage-<jobid>.csv` (เฉพาะงาน GPU) | — | |

คำถามสำหรับพูดคุย:
1. ความแม่นยำ (เกือบ) เท่ากัน — เพราะอะไร?
2. ความเร็วเพิ่มขึ้นมาก แต่การใช้งาน GPU กลับต่ำ — อะไรคือคอขวดของโมเดลขนาดเล็กเช่นนี้?
3. หากเพิ่ม `training.batch_size` เป็นสองเท่า throughput จะเป็นอย่างไร? ถ้าตั้ง `training.amp: true` ล่ะ?

---

## Lab 2 — การจำแนกอาหารไทย (THFOOD-100)

### 1. เตรียมชุดข้อมูล

THFOOD-100 **ไม่ได้** ดาวน์โหลดให้อัตโนมัติ ให้ขอจากผู้สอน
(หรือจากไดเรกทอรีโปรเจกต์ที่ใช้ร่วมกันบน LANTA) แล้วจัดเรียงตามรูปแบบ
`torchvision.datasets.ImageFolder`:

```
data/thfood100/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg      # ไม่บังคับ — ถ้าไม่มีจะใช้ val แทน
```

ตรวจสอบโครงสร้าง:

```bash
python datasets/download.py --dataset thfood100 --root ./data/thfood100
```

ยังไม่มีชุดข้อมูลเต็ม? รีโพนี้แนบตัวอย่างขนาดเล็กไว้ที่
`data/THFOOD-100.sample/` (โครงสร้างแบบแบน ไม่มีโฟลเดอร์ train/val/test
มีรูปเพียงไม่กี่รูปต่อคลาส) — เพียงพอสำหรับทดสอบระบบเบื้องต้นด้วย
`configs/thfood_sample.yaml` แต่ยังไม่พอสำหรับความแม่นยำที่มีความหมาย
ดู [ARCHITECTURE.md](ARCHITECTURE.md) สำหรับวิธีจัดการโครงสร้างแบบแบนนี้

### 2. เทรน baseline

```bash
python scripts/train.py --config configs/thfood_baseline.yaml
# หรือบน LANTA:
sbatch jobs/train_thfood.sh
```

baseline นี้ปรับจูน (fine-tune) โมเดล **ResNet-18 ที่เทรนไว้แล้วบน
ImageNet** — เนื่องจาก backbone รู้จักขอบ ลวดลาย และรูปทรงต่าง ๆ อยู่แล้ว
จึงต้องการเพียงไม่กี่ epoch ในการปรับให้เข้ากับอาหารไทย 100 ชนิด
(นี่คือหลักการของ *transfer learning*)

### 3. ปรับจูนไฮเปอร์พารามิเตอร์ — แก้แค่ YAML!

คัดลอก [thfood_competition.yaml](configs/thfood_competition.yaml)
ตั้งชื่อการทดลองใหม่ทุกครั้งที่ลอง แล้วปรับจูน **เฉพาะ** ค่าต่อไปนี้:

| ตัวปรับ | คีย์ใน config | ค่าที่ลองได้ |
|------|------------|---------------|
| โมเดล | `model.name` | `resnet18`, `mobilenetv3`, `efficientnet_b0` |
| Batch size | `training.batch_size` | 32, 64, 128, 256 |
| Learning rate | `training.lr` | 0.0001 … 0.01 |
| Epochs | `training.epochs` | 5 … 30 (early stopping ช่วยประหยัดเวลา) |
| Optimizer | `optimizer.name` | `SGD`, `Adam`, `AdamW` |
| Scheduler | `scheduler.name` | `none`, `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau` |

ทดลองแบบรวดเร็วโดยไม่ต้องแก้ไฟล์ (อาร์กิวเมนต์เพิ่มเติมจะถูกส่งต่อไปยัง
`train.py`):

```bash
sbatch jobs/train_thfood.sh --name thfood_lr001  --lr 0.001
sbatch jobs/train_thfood.sh --name thfood_bs128  --batch-size 128 --lr 0.0006
```

แต่ละการรันจะมี `outputs/<name>/` และ `logs/<name>/` เป็นของตัวเอง —
เปรียบเทียบทั้งหมดพร้อมกันได้ใน TensorBoard

### 4. ประเมินผลและทำนาย

```bash
python scripts/evaluate.py --checkpoint outputs/thfood_baseline/best.pt          # test split + รายงานแยกตามคลาส
python scripts/predict.py  --model outputs/thfood_baseline/best.pt --image sample.jpg
python scripts/benchmark.py --config configs/thfood_baseline.yaml                # ความเร็วและขนาดโมเดล
python scripts/export.py   --checkpoint outputs/thfood_baseline/best.pt          # TorchScript
```

---

## TensorBoard

ทุกการรันจะบันทึก loss, accuracy, learning rate, เวลาต่อ epoch,
และ throughput ไว้ที่ `logs/<experiment_name>/`

**แบบโลคอล:**

```bash
tensorboard --logdir logs
# เปิด http://localhost:6006
```

**บน LANTA** (TensorBoard รันบน login node ดูผ่าน SSH tunnel):

```bash
# เทอร์มินัลที่ 1 — บน LANTA:
conda activate hpc-ai
tensorboard --logdir logs --port 6006 --bind_all

# เทอร์มินัลที่ 2 — บนเครื่องของคุณ:
ssh -L 6006:localhost:6006 <username>@lanta.nstda.or.th
# จากนั้นเปิด http://localhost:6006 ในเบราว์เซอร์
```

การชี้ `--logdir` ไปที่ `logs/` (ไม่ใช่การรันเดี่ยว) จะซ้อนทับ
การทดลอง **ทั้งหมด** ไว้ใน dashboard เดียว — เหมาะสำหรับเปรียบเทียบ
ความพยายามในการปรับจูนต่าง ๆ

---

## Checkpoint และ Outputs

แต่ละการทดลองจะเขียนไดเรกทอรีผลลัพธ์ที่สมบูรณ์ในตัวเอง:

```
outputs/<experiment_name>/
├── config.yaml      # config ที่ใช้จริง -> ทำซ้ำผลลัพธ์ได้เต็มรูปแบบ
├── best.pt          # น้ำหนักโมเดลที่ให้ความแม่นยำ validation สูงสุด
├── last.pt           # น้ำหนักโมเดลหลัง epoch ล่าสุด
├── metrics.json      # ต่อ epoch: loss, accuracy, เวลาต่อ epoch, images/sec, lr
└── eval_test.json    # เขียนโดย evaluate.py
```

Checkpoint เก็บทั้งน้ำหนักโมเดล **และ** สถานะของ optimizer/scheduler,
config, และชื่อคลาส — ดังนั้น `evaluate.py`, `predict.py`, และ
`export.py` ต้องการเพียงไฟล์ `.pt` เท่านั้น คัดลอก checkpoint ที่ควรเก็บไว้
ไปยัง `checkpoints/` (ไฟล์ใน `outputs/` อาจถูกเขียนทับเมื่อรันซ้ำ)

---

## ผลลัพธ์ที่คาดหวัง

ตัวเลขจะแตกต่างกันไปตามฮาร์ดแวร์และภาระงานของโหนด — นี่เป็นตัวเลข
คร่าว ๆ สำหรับตรวจสอบความสมเหตุสมผลของผลการรันของคุณ

**Lab 1 — MNIST, LeNet-5, 5 epoch, batch 128:**

| | CPU (16 คอร์) | GPU 1x A100 |
|--|--|--|
| เวลาต่อ epoch | ~30–60 วินาที | ~5–10 วินาที |
| Throughput | ~1,000–2,000 img/s | ~6,000–12,000 img/s |
| ความแม่นยำ (accuracy) สุดท้ายของ val | ~99% | ~99% (คำนวณแบบเดียวกัน ผลลัพธ์เดียวกัน) |

**Lab 2 — THFOOD-100 baseline (ResNet-18, 5 epoch, GPU 1x A100):**

| | ค่า |
|--|--|
| เวลาต่อ epoch | ไม่กี่นาที (ขึ้นอยู่กับขนาดชุดข้อมูลและ I/O) |
| ความแม่นยำ val หลังจาก 5 epoch | ประมาณ 70–85% |
| หากปรับจูนดี (competition) | สูงกว่านี้ — นั่นคืองานของคุณ! |

---

## การแก้ไขปัญหา

| อาการ | วิธีแก้ |
|---------|-----|
| งานค้าง / ดาวน์โหลดผิดพลาดบน compute node | ดาวน์โหลด MNIST และน้ำหนักโมเดล ImageNet บน **login node** ก่อน (ดูหัวข้อการติดตั้ง) |
| `URLError: Network is unreachable` เมื่อดาวน์โหลดน้ำหนักโมเดล ResNet/MobileNet/EfficientNet | ในการติดตั้งแบบคลาสร่วมกัน `TORCH_HOME` ต้องชี้ไปยังแคชที่ใช้ร่วมกันภายใต้ `/project` (ดูหัวข้อการติดตั้ง) — รัน `setup_user.sh` ใหม่เพื่อสร้าง `project.env` ใหม่หากไม่มี `TORCH_HOME` |
| `config requests CUDA but no GPU is available` | คุณกำลังอยู่บน CPU node — งานจะรันต่อบน CPU; ใช้ partition `gpu` สำหรับการรันแบบ GPU |
| `THFOOD-100 split not found` | ตรวจสอบโครงสร้าง ImageFolder ด้วย `python datasets/download.py --dataset thfood100` |
| หน่วยความจำ GPU ไม่พอ (Out-of-memory) | ลด `training.batch_size` (ลดครึ่งหนึ่งจนกว่าจะพอดี) |
| DataLoader เป็นคอขวด (การใช้งาน GPU ต่ำ) | เพิ่ม `dataset.num_workers` ให้สอดคล้องกับ `--cpus-per-task` |
| การรันสองครั้งเขียนทับกัน | ตั้ง `experiment.name` (หรือ `--name`) ให้ไม่ซ้ำกันในแต่ละการรัน |
