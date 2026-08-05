# HPC AI/ML Training Workshop

เอกสารประกอบการฝึกปฏิบัติ (hands-on labs) สำหรับการฝึกโมเดล AI บนซูเปอร์คอมพิวเตอร์
**LANTA** ครอบคลุมหัวข้อการเร่งความเร็วด้วย GPU (GPU acceleration) การส่งงานผ่าน
Slurm (Slurm job submission) การทดลองที่ทำซ้ำได้ (reproducible experiments) และ
การปรับแต่งไฮเปอร์พารามิเตอร์ (hyperparameter tuning) โดยใช้ PyTorch แบบพื้นฐานทั้งหมด

| Lab | หัวข้อ | สิ่งที่จะได้เรียนรู้ |
|-----|------|----------------|
| **Demo** | การจำแนกตัวเลขลายมือ MNIST (MNIST digit classification) | เปรียบเทียบ CPU กับ GPU: เวลาในการฝึกโมเดล (training time), throughput, การใช้งาน GPU (GPU utilization) — ใช้โค้ดชุดเดียวกัน เปลี่ยนแค่ config |
| **Hands-on** | การจำแนกอาหารไทย THFOOD-100 (THFOOD-100 Thai food classification) | Transfer learning ร่วมกับการปรับแต่งไฮเปอร์พารามิเตอร์ — ไม่ต้องแก้ source code แก้เพียงไฟล์ YAML |

ทุกการทดลองถูกควบคุมด้วย **ไฟล์การตั้งค่า YAML (YAML configuration files)** ทั้งหมด
นักเรียนไม่จำเป็นต้องแก้ไข source code ภาษา Python เพื่อทำแลปให้สำเร็จ

---

## โครงสร้างโฟลเดอร์

```
training_ai_ml/
├── README.md
├── requirements.txt         # รายการ dependency สำหรับ pip
├── environment.yml          # environment ของ conda (Python 3.11)
├── setup.sh                 # ตั้งค่า environment ครั้งเดียว (สำหรับใช้งานคนเดียว)
├── setup_project.sh         # ตั้งค่าแบบ SHARED สำหรับทั้งคลาสเรียน ครั้งเดียว (ดูรายละเอียดด้านล่าง)
├── setup_user.sh            # ตั้งค่า workspace ส่วนตัวให้นักเรียนแต่ละคน ครั้งเดียว
│
├── configs/                 # ไฟล์ YAML หนึ่งไฟล์ = การทดลองหนึ่งครั้ง
│   ├── default.yaml         #   config อ้างอิงที่มีคำอธิบายครบทุกฟิลด์
│   ├── mnist_cpu.yaml       #   Demo: CPU
│   ├── mnist_gpu.yaml       #   Demo: GPU (เหมือนกันทุกอย่าง ต่างแค่ `device`)
│   ├── thfood_baseline.yaml #   Hands-on: baseline ด้วย ResNet-18
│   ├── thfood_sample.yaml   #   Hands-on: ทดสอบระบบเบื้องต้น (smoke test) ด้วยข้อมูลตัวอย่างที่แนบมาให้
│   └── thfood_competition.yaml  # Hands-on: พื้นที่สำหรับทดลองปรับแต่ง (tuning) ของคุณเอง
│
├── datasets/                # การโหลดข้อมูล (MNIST ดาวน์โหลดอัตโนมัติ, THFOOD แบบ ImageFolder)
├── models/                  # LeNet-5, ResNet-18, MobileNetV3, EfficientNet-B0
├── trainer/                 # คลาส Trainer, ลูปการฝึก/ตรวจสอบ (train/val loops), losses, metrics, utils
├── scripts/                 # train / evaluate / predict / benchmark / export
├── jobs/                    # สคริปต์สำหรับส่งงานผ่าน Slurm บน LANTA
│
├── checkpoints/             # (ทางเลือก) สำหรับเก็บ checkpoint ที่ดีไว้ใช้ในระยะยาว
├── logs/                    # ข้อมูล TensorBoard + ผลลัพธ์ของ Slurm + ไฟล์ CSV การใช้งาน GPU
└── outputs/                 # ผลลัพธ์ของแต่ละการทดลอง (checkpoint, metrics, สำเนา config)
```

---

## การติดตั้ง (Installation)

### บน LANTA แบบใช้งานคนเดียว (คนเดียว, หนึ่ง project quota)

ทำตามขั้นตอนต่อไปนี้ตามลำดับ:

1. โหลดโมดูล conda distribution ของ LANTA:
   ```bash
   module load Mamba/23.11.0-0
   ```
2. รันสคริปต์ติดตั้งเพื่อสร้าง conda environment ชื่อ `hpc-ai` (Python 3.11):
   ```bash
   bash setup.sh
   ```
3. เปิดใช้งาน environment:
   ```bash
   conda activate hpc-ai
   ```

### บน LANTA สำหรับทั้งคลาสเรียน (ใช้ project quota ร่วมกัน — แนะนำ)

Home directory บน LANTA มี quota ค่อนข้างจำกัด (เช่น **100 GB / 600,000 inodes**)
ในขณะที่ `/project` มักมี quota ที่ใหญ่กว่ามาก (เช่น **30 TB / 300 ล้าน inodes**)
สำหรับเวิร์กช็อปที่มีนักเรียนหลายคนใช้ project account ร่วมกัน ให้นำส่วนที่หนักและ
ใช้ร่วมกัน — โค้ด, conda environment, และชุดข้อมูล (datasets) — ไปไว้ที่ `/project`
**เพียงครั้งเดียว** แล้วให้นักเรียนแต่ละคนมี workspace ส่วนตัวขนาดเล็กใน `$HOME`
ของตนเอง สำหรับสิ่งที่ต้องแก้ไขและสร้างผลลัพธ์จริง (configs, สคริปต์ Slurm,
checkpoints, ผลลัพธ์การรัน, logs)

**ขั้นตอนสำหรับผู้สอน / เจ้าของ project (ทำครั้งเดียว):**

1. Clone repository ไปยัง `/project`:
   ```bash
   git clone <repo-url> /project/tn999996-north/training_ai_ml
   cd /project/tn999996-north/training_ai_ml
   ```
2. โหลดโมดูล conda distribution ของ LANTA:
   ```bash
   module load Mamba/23.11.0-0
   ```
3. รันสคริปต์ตั้งค่า project เพื่อสร้าง conda environment ไว้ใน `./envs/hpc-ai`
   (ไม่แตะ home quota ของใครเลย):
   ```bash
   bash setup_project.sh
   ```
4. ทำตามขั้นตอนถัดไปที่ `setup_project.sh` แสดงไว้: ดาวน์โหลด MNIST และ
   ImageNet weights ล่วงหน้าบน login node และวางชุดข้อมูล THFOOD-100 แบบเต็ม
   ไว้ที่ `data/thfood100/` — ทั้งหมดนี้อยู่ภายใต้ shared project directory
   โดยปกติ cache ของ pretrained weights ใน PyTorch (`TORCH_HOME`) จะแยกตาม
   user แต่ละคน ดังนั้นการดาวน์โหลดนี้ต้องชี้ไปยัง path ที่ใช้ร่วมกันภายใต้
   `/project` (คำสั่ง `export TORCH_HOME=...` ที่ถูกต้องจะถูกแสดงในขั้นตอนนั้น)
   — มิเช่นนั้นงานของนักเรียนแต่ละคนจะพยายามดาวน์โหลด weights ชุดเดิมซ้ำและ
   ล้มเหลว เนื่องจาก compute node ไม่มีอินเทอร์เน็ต `setup_user.sh` จะตั้งค่า
   นี้ให้อัตโนมัติสำหรับนักเรียนแต่ละคนผ่านไฟล์ `project.env`

**ขั้นตอนสำหรับนักเรียนแต่ละคน (ทำครั้งเดียว):**

1. รันสคริปต์ตั้งค่า workspace ส่วนตัว:
   ```bash
   bash /project/tn999996-north/training_ai_ml/setup_user.sh
   ```
   คำสั่งนี้จะสร้าง `~/hpc-ai-workshop/` ซึ่งมีสำเนา `configs/` และ `jobs/`
   ของคุณเอง (แก้ไขได้อย่างอิสระ) พร้อมโฟลเดอร์ `checkpoints/`, `outputs/`,
   และ `logs/` ที่ยังว่างเปล่าไว้สำหรับผลการรันของคุณ นอกจากนี้ยังลงทะเบียน
   โฟลเดอร์ `envs/` ของ project ไว้ใน `~/.condarc` ของคุณเอง ทำให้
   **`conda activate hpc-ai` ใช้งานได้จากทุกที่ด้วยชื่อ environment โดยตรง**
   — ตัว environment เองยังคงอยู่บน `/project` ทั้งหมด ไม่แตะ home quota ของ
   คุณเลย ไฟล์ `project.env` จะบันทึกตำแหน่งของโค้ดที่ใช้ร่วมกันไว้ เพื่อให้
   สคริปต์ใน `jobs/` รู้ว่าจะไปหาโค้ดได้จากที่ใด เนื่องจาก job จะรันโดยใช้
   workspace นี้ (ไม่ใช่ `/project`) เป็น working directory ดังนั้น
   `setup_user.sh` จะแก้ไข `dataset.root` ในสำเนา config ให้เปลี่ยนจาก path
   สัมพัทธ์ (relative path) เดิม `./data/...` เป็น absolute path ภายใต้
   shared project แทน — มิเช่นนั้นจะไป resolve เป็นโฟลเดอร์ที่ไม่มีอยู่จริงใน
   `$HOME`
2. ย้ายไปทำงานที่ workspace ส่วนตัว แก้ไข account แล้วส่งงานทดสอบ:
   ```bash
   cd ~/hpc-ai-workshop
   # แก้ไข jobs/*.sh: ตั้งค่า #SBATCH --account=ltXXXXXX ให้เป็น account ของคุณบน LANTA
   sbatch jobs/train_cpu.sh
   ```
   จากนั้นให้ทำงานทั้งหมดจาก workspace ส่วนตัวนี้ (`~/hpc-ai-workshop/`) ต่อไป

### ในเครื่องอื่นๆ (Anywhere else)

เลือกทำวิธีใดวิธีหนึ่งต่อไปนี้:

1. สร้างและเปิดใช้งาน environment ด้วย conda:
   ```bash
   conda env create -f environment.yml && conda activate hpc-ai
   ```
   หรือ
2. ใช้ pip ธรรมดาใน environment ของ Python 3.11 ที่มีอยู่แล้ว:
   ```bash
   pip install -r requirements.txt
   ```

### ดาวน์โหลดข้อมูลและ weights ล่วงหน้า (สำคัญมากบนคลัสเตอร์!)

**compute node ของ LANTA ไม่มีการเชื่อมต่ออินเทอร์เน็ต** ดังนั้นให้ดาวน์โหลดทุก
อย่างล่วงหน้าเพียงครั้งเดียวบน **login node** ก่อนเริ่มฝึกโมเดล ในกรณีตั้งค่าแบบ
คลาสเรียนร่วมกันด้านบน ผู้สอนจะทำขั้นตอนนี้เพียงครั้งเดียวภายใต้ shared project
directory (`setup_project.sh` จะแสดงคำสั่งชุดเดียวกันนี้) — นักเรียนไม่จำเป็นต้อง
ทำซ้ำ

1. ดาวน์โหลด MNIST (~12 MB):
   ```bash
   python datasets/download.py --dataset mnist --root ./data
   ```
2. ดาวน์โหลด ImageNet weights สำหรับ Hands-on (cache ไว้ที่ `~/.cache/torch`):
   ```bash
   python -c "import torchvision.models as m; \
       m.resnet18(weights=m.ResNet18_Weights.IMAGENET1K_V1); \
       m.mobilenet_v3_large(weights=m.MobileNet_V3_Large_Weights.IMAGENET1K_V2); \
       m.efficientnet_b0(weights=m.EfficientNet_B0_Weights.IMAGENET1K_V1)"
   ```

---

## Demo — CPU vs. GPU (MNIST)

config ทั้งสองไฟล์ [mnist_cpu.yaml](configs/mnist_cpu.yaml) และ
[mnist_gpu.yaml](configs/mnist_gpu.yaml) **เหมือนกันทุกอย่าง ต่างกันเพียง
`device`** ให้ฝึกโมเดลด้วยทั้งสองไฟล์แล้วนำผลมาเปรียบเทียบกัน

### 1. ฝึกโมเดลด้วยทั้งสอง config (CPU และ GPU)

เลือกวิธีใดวิธีหนึ่งต่อไปนี้ตามสภาพแวดล้อมที่ใช้งาน:

**แบบ local / interactive:**

```bash
python scripts/train.py --config configs/mnist_cpu.yaml
python scripts/train.py --config configs/mnist_gpu.yaml
```

**แบบบน LANTA ผ่าน Slurm:**

แก้ไขบรรทัด `#SBATCH --account=ltXXXXXX` ในสคริปต์ก่อน จากนั้น:

```bash
sbatch jobs/train_cpu.sh     # เข้าคิวใน partition 'compute' (CPU)
sbatch jobs/train_gpu.sh     # เข้าคิวใน partition 'gpu' (1x A100)
squeue --me                  # ดูสถานะงานของคุณ
```

### 2. สังเกตผลลัพธ์

แต่ละ epoch จะพิมพ์สรุปผลออกมาหนึ่งบรรทัด:

```
Epoch   1/5 | train loss 0.2431 acc 92.51% | val loss 0.0705 acc 97.72% |   11.2s    5357 img/s | lr 1.00e-03
```

### 3. กรอกตารางเปรียบเทียบ

กรอกตารางเปรียบเทียบจากผลการรันทั้งสองครั้งของคุณ:

| ตัวชี้วัด (Metric) | ดูได้จากที่ไหน | CPU | GPU |
|--------|------------------|-----|-----|
| เวลาต่อ epoch (วินาที) | บรรทัดสรุปผลของแต่ละ epoch / `metrics.json` | | |
| Throughput (img/s) | บรรทัดสรุปผลของแต่ละ epoch / `metrics.json` | | |
| Val accuracy สุดท้าย | บทสรุปท้ายการฝึก | | |
| การใช้งาน GPU (%) | `logs/gpu-usage-<jobid>.csv` (เฉพาะงาน GPU) | — | |

### 4. ตอบคำถามอภิปราย

1. ความแม่นยำ (accuracy) ใกล้เคียงกัน (แทบจะเท่ากัน) — เพราะเหตุใด?
2. อัตราเร่ง (speedup) สูงมาก แต่การใช้งาน GPU (GPU utilization) กลับต่ำ — คอขวด (bottleneck) ของโมเดลขนาดเล็กเช่นนี้คืออะไร?
3. ถ้าเพิ่ม `training.batch_size` เป็นสองเท่า throughput จะเปลี่ยนไปอย่างไร? แล้วถ้าตั้ง `training.amp: true` ล่ะ?

---

## Hands-on — การจำแนกอาหารไทย (THFOOD-100)

### 1. เตรียมชุดข้อมูล

THFOOD-100 **ไม่ได้** ถูกดาวน์โหลดให้อัตโนมัติ ให้ขอชุดข้อมูลจากผู้สอน (หรือจาก
shared project directory บน LANTA) แล้วจัดเรียงในรูปแบบ
`torchvision.datasets.ImageFolder`:

```
data/thfood100/
├── train/<class_name>/*.jpg
├── val/<class_name>/*.jpg
└── test/<class_name>/*.jpg      # ทางเลือก — ถ้าไม่มีจะใช้ val แทน
```

ตรวจสอบโครงสร้างไฟล์:

```bash
python datasets/download.py --dataset thfood100 --root ./data/thfood100
```

ยังไม่มีชุดข้อมูลแบบเต็มใช่ไหม? repo นี้มีตัวอย่างข้อมูลขนาดเล็กแนบมาให้ที่
`data/THFOOD-100.sample/` (โครงสร้างแบบแบน (flat layout) ไม่มีโฟลเดอร์
train/val/test แยก มีเพียงไม่กี่รูปต่อคลาส) — เพียงพอสำหรับทดสอบระบบเบื้องต้น
(smoke test) ด้วย `configs/thfood_sample.yaml` แต่ยังไม่เพียงพอสำหรับวัดความแม่นยำ
ที่มีความหมาย ดู [ARCHITECTURE.md](ARCHITECTURE.md) สำหรับรายละเอียดวิธีจัดการ
โครงสร้างแบบแบนนี้

### 2. ฝึกโมเดล baseline

```bash
python scripts/train.py --config configs/thfood_baseline.yaml
# หรือบน LANTA:
sbatch jobs/train_thfood.sh
```

baseline นี้ทำ fine-tuning บน **ResNet-18 ที่ผ่านการฝึกด้วย ImageNet มาแล้ว
(pretrained)** — backbone รู้จักขอบ พื้นผิว และรูปทรงพื้นฐานอยู่แล้ว จึงต้องการ
เพียงไม่กี่ epoch ในการปรับให้เข้ากับอาหารไทย 100 ชนิด (นี่คือหลักการของ
*transfer learning*)

### 3. ปรับแต่งไฮเปอร์พารามิเตอร์ — ใช้ YAML เท่านั้น!

คัดลอกไฟล์ [thfood_competition.yaml](configs/thfood_competition.yaml)
เปลี่ยนชื่อการทดลอง (experiment name) ทุกครั้งที่ลอง แล้วปรับแต่ง **เฉพาะ** ค่า
เหล่านี้:

| พารามิเตอร์ | คีย์ใน config | ค่าที่ลองปรับได้ |
|------|------------|---------------|
| โมเดล | `model.name` | `resnet18`, `mobilenetv3`, `efficientnet_b0` |
| Batch size | `training.batch_size` | 32, 64, 128, 256 |
| Learning rate | `training.lr` | 0.0001 … 0.01 |
| จำนวน epoch | `training.epochs` | 5 … 30 (early stopping ช่วยลด epoch ที่เสียเปล่า) |
| Optimizer | `optimizer.name` | `SGD`, `Adam`, `AdamW` |
| Scheduler | `scheduler.name` | `none`, `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau` |

ทดลองอย่างรวดเร็วโดยไม่ต้องแก้ไฟล์ใดๆ (argument เพิ่มเติมจะถูกส่งต่อไปยัง
`train.py`):

```bash
sbatch jobs/train_thfood.sh --name thfood_lr001  --lr 0.001
sbatch jobs/train_thfood.sh --name thfood_bs128  --batch-size 128 --lr 0.0006
```

การรันแต่ละครั้งจะได้ `outputs/<name>/` และ `logs/<name>/` ของตัวเอง — สามารถ
เปรียบเทียบทุกการทดลองพร้อมกันได้ใน TensorBoard

### 4. ประเมินผลและทำนาย (Evaluate and predict)

```bash
python scripts/evaluate.py --checkpoint outputs/thfood_baseline/best.pt          # ประเมินบน test split พร้อมรายงานผลรายคลาส
python scripts/predict.py  --model outputs/thfood_baseline/best.pt --image sample.jpg
python scripts/benchmark.py --config configs/thfood_baseline.yaml                # วัดความเร็วและขนาดโมเดล
python scripts/export.py   --checkpoint outputs/thfood_baseline/best.pt          # ส่งออกเป็น TorchScript
```

### 5. การฝึกแบบ Multi-GPU / Multi-node

config เดิม [thfood_baseline.yaml](configs/thfood_baseline.yaml) สามารถฝึกบน
GPU หลายตัว — ไม่ว่าจะบน node เดียวหรือหลาย node — ได้โดย **ไม่ต้องแก้ config
หรือแก้โค้ดใดๆ เลย** โดยใช้ `DistributedDataParallel` (DDP) ของ PyTorch: GPU
แต่ละตัวจะฝึกบนข้อมูลส่วนของตัวเอง (shard) แล้วค่า gradient จะถูกเฉลี่ย
(average) ข้าม GPU ทุก step ทำให้หลักการทางคณิตศาสตร์เหมือนกับการรันบน GPU
เดียวทุกประการ เพียงแต่เร็วกว่า

```bash
# รันแบบ local / interactive เช่น 2 GPU บนเครื่องเดียว:
torchrun --standalone --nproc_per_node=2 scripts/train.py --config configs/thfood_baseline.yaml

# บน LANTA ผ่าน Slurm:
sbatch jobs/train_thfood_multigpu.sh     # 1 node หลาย GPU (แก้ไข --gpus-per-node)
sbatch jobs/train_thfood_multinode.sh    # หลาย node (แก้ไข --nodes / --gpus-per-node / --ntasks-per-node ให้ตรงกัน)
```

ทั้ง `jobs/train_thfood_multigpu.sh` (node เดียว) และ
`jobs/train_thfood_multinode.sh` (ข้ามหลาย node) รันผ่าน `srun python`
โดยตรง หนึ่ง process ต่อหนึ่ง GPU — ตาม pattern ที่ ThaiSC/LANTA แนะนำสำหรับ
PyTorch แบบ multi-GPU/multi-node แทนที่จะใช้ `torchrun` (rendezvous ของ
`torchrun` เองไม่สามารถ connect ข้าม node บนเครือข่ายของ LANTA ได้ — ดู
[ARCHITECTURE.md](ARCHITECTURE.md) §9 สำหรับรายละเอียด) ส่วนคำสั่ง
`torchrun` ด้านบนยังใช้ได้ปกติสำหรับรันแบบ local/interactive ที่ไม่ผ่าน
Slurm เท่านั้น

`training.batch_size` ใน config คือ batch size **ต่อ GPU หนึ่งตัว (per-GPU)**
— batch size ที่ใช้จริง (effective/global) คือ `batch_size × จำนวน GPU
ทั้งหมด` เมื่อเปรียบเทียบกับผลจาก single-GPU baseline ควรปรับ learning rate
ให้สูงขึ้นตามสัดส่วน (จุดเริ่มต้นที่นิยมใช้คือ linear scaling ตาม global batch
size)

`images/sec` ในบรรทัดสรุปผลของแต่ละ epoch และใน TensorBoard คือ throughput
**รวม (aggregate)** ของทุก GPU — ตัวเลขนี้แหละที่ควรนำไปเปรียบเทียบกับผลจาก
single-GPU run ในขั้นตอนที่ 2 เพื่อดูว่าการฝึกแบบขนาน scale ได้ดีเพียงใด

---

## TensorBoard

ทุกการรันจะบันทึก loss, accuracy, learning rate, เวลาต่อ epoch, และ throughput
ไว้ที่ `logs/<experiment_name>/`

**แบบ local:**

```bash
tensorboard --logdir logs
# เปิด http://localhost:6006
```

**บน LANTA** (TensorBoard รันบน login node และดูผลผ่าน SSH tunnel):

```bash
# terminal ที่ 1 — บน LANTA:
conda activate hpc-ai
tensorboard --logdir logs --port 6006 --bind_all

# terminal ที่ 2 — บนเครื่องของคุณ:
ssh -L 6006:localhost:6006 <username>@lanta.nstda.or.th
# จากนั้นเปิด http://localhost:6006 ในเบราว์เซอร์
```

การชี้ `--logdir` ไปที่ `logs/` (ไม่ใช่โฟลเดอร์ของ run เดียว) จะรวมทุกการทดลอง
ไว้ใน dashboard เดียวกัน — เหมาะสำหรับเปรียบเทียบผลการปรับแต่งไฮเปอร์พารามิเตอร์
หลายๆ ครั้ง

---

## Checkpoints และ Outputs

การทดลองแต่ละครั้งจะสร้างโฟลเดอร์ผลลัพธ์ที่สมบูรณ์ในตัวเอง (self-contained):

```
outputs/<experiment_name>/
├── config.yaml      # สำเนา config ที่ใช้จริง -> ทำให้ผลลัพธ์ทำซ้ำได้อย่างสมบูรณ์
├── best.pt          # weights ที่ให้ validation accuracy สูงสุด
├── last.pt          # weights หลัง epoch ล่าสุด
├── metrics.json     # ข้อมูลรายเอพอค: loss, accuracy, เวลาต่อ epoch, images/sec, lr
└── eval_test.json   # เขียนโดย evaluate.py
```

checkpoint จะเก็บทั้ง model weights **และ** สถานะของ optimizer/scheduler,
config, และชื่อคลาส (class names) ไว้ด้วย — ดังนั้น `evaluate.py`,
`predict.py`, และ `export.py` จึงต้องการเพียงไฟล์ `.pt` เท่านั้น ให้คัดลอก
checkpoint ที่ต้องการเก็บไว้ไปยัง `checkpoints/` (ไฟล์ใน `outputs/` อาจถูก
เขียนทับได้เมื่อรันซ้ำ)

---

## ผลลัพธ์ที่คาดหวัง (Expected Outputs)

ตัวเลขจะแตกต่างกันไปตามฮาร์ดแวร์และภาระงานของ node — ตัวเลขเหล่านี้เป็นเพียง
ค่าประมาณสำหรับตรวจสอบว่าผลการรันของคุณอยู่ในเกณฑ์ที่สมเหตุสมผล

**Demo — MNIST, LeNet-5, 5 epochs, batch 128:**

| | CPU (16 cores) | 1x A100 GPU |
|--|--|--|
| เวลาต่อ epoch | ~30–60 วินาที | ~5–10 วินาที |
| Throughput | ~1,000–2,000 img/s | ~6,000–12,000 img/s |
| Val accuracy สุดท้าย | ~99% | ~99% (หลักการคำนวณเดียวกัน ผลลัพธ์เดียวกัน) |

**Hands-on — THFOOD-100 baseline (ResNet-18, 5 epochs, 1x A100):**

| | ค่า |
|--|--|
| เวลาต่อ epoch | ไม่กี่นาที (ขึ้นกับขนาดชุดข้อมูลและความเร็ว I/O) |
| Val accuracy หลัง 5 epoch | ประมาณ 70–85% |
| ปรับแต่งอย่างดี (competition) | สูงกว่านี้ — นั่นคือภารกิจของคุณ! |

---

## การแก้ปัญหา (Troubleshooting)

| อาการ | วิธีแก้ |
|---------|-----|
| Job ค้าง / เกิด error ตอนดาวน์โหลดบน compute node | ให้ดาวน์โหลด MNIST และ ImageNet weights บน **login node** ก่อน (ดูหัวข้อการติดตั้ง) |
| `URLError: Network is unreachable` ตอนดาวน์โหลด weights ของ ResNet/MobileNet/EfficientNet | ในการตั้งค่าแบบคลาสเรียนร่วมกัน `TORCH_HOME` ต้องชี้ไปยัง cache ที่ใช้ร่วมกันภายใต้ `/project` (ดูหัวข้อการติดตั้ง) — ลองรัน `setup_user.sh` ใหม่เพื่อสร้าง `project.env` ใหม่ หากไฟล์นั้นขาด `TORCH_HOME` ไป |
| `config requests CUDA but no GPU is available` | คุณกำลังอยู่บน CPU node — การรันจะดำเนินต่อไปบน CPU แทน ให้ใช้ partition `gpu` สำหรับการรันแบบ GPU |
| `THFOOD-100 split not found` | ตรวจสอบโครงสร้างแบบ ImageFolder ด้วยคำสั่ง `python datasets/download.py --dataset thfood100` |
| GPU หน่วยความจำไม่พอ (Out-of-memory) | ลด `training.batch_size` ลง (ลดครึ่งหนึ่งไปเรื่อยๆ จนกว่าจะพอดี) |
| DataLoader เป็นคอขวด (การใช้งาน GPU ต่ำ) | เพิ่ม `dataset.num_workers` ให้สอดคล้องกับ `--cpus-per-task` |
| การรันสองครั้งเขียนทับกัน | ตั้ง `experiment.name` (หรือ `--name`) ให้ไม่ซ้ำกันในแต่ละการรัน |
