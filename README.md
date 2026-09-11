# AMOT (YOLO Integration Fork)

> Official code for multi-object tracker AMOT, AAAI, 2026

![](readme/MOT.png)

> [**Tracking the Unstable: Appearance-Guided Motion Modeling for Robust Multi-Object Tracking in UAV-Captured Videos**](https://arxiv.org/abs/2508.01730),
> Jianbo Ma, Hui Luo, Qi Chen, Yuankai Qi, Yumei Sun, Amin Beheshti, Jianlin Zhang, Ming-Hsuan Yang

Repo นี้เป็น fork ของ AMOT ต้นฉบับ (borrow มาจาก [FairMOT](https://github.com/ifzhang/FairMOT) และ [STCMOT](https://github.com/ydhcg-BoBo/STCMOT)) ที่เพิ่มเติม:

- **YOLO integration** — สลับ detector จาก DLA heatmap (CenterNet-style) เดิมของ AMOT มาใช้ YOLO ได้ โดยยังใช้ ReID feature/motion model เดิมของ AMOT ทั้งหมด
- **สคริปต์รันบนเครื่อง local** (`run_tracking.py`) — รันกับวิดีโอของตัวเองได้ ไม่ต้องพึ่ง Colab/Kaggle และไม่ต้องแก้ hardcode path ในไฟล์
- **เครื่องมือวิเคราะห์ผล** — เทียบ baseline กับ YOLO, ทำ confidence-threshold ablation, วินิจฉัย track fragmentation โดยไม่ต้องมี Ground Truth

สารบัญ:
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)
- [การติดตั้ง](#การติดตั้ง)
- [เตรียมข้อมูล](#เตรียมข้อมูล)
- [โมเดลที่เทรนไว้แล้ว](#โมเดลที่เทรนไว้แล้ว)
- [วิธีรัน Tracking](#วิธีรัน-tracking)
- [YOLO Integration](#yolo-integration)
- [Confidence Threshold Ablation Study](#confidence-threshold-ablation-study)
- [เครื่องมือวิเคราะห์ผล](#เครื่องมือวิเคราะห์ผล)
- [การเทรนโมเดล](#การเทรนโมเดล)
- [ปัญหาที่พบบ่อย](#ปัญหาที่พบบ่อย)
- [Citation](#citation)

---

## โครงสร้างโปรเจกต์

```
AMOT/
├── DCNv2/                       # custom CUDA op (Deformable Convolution v2) ที่ backbone DLA ต้องใช้
├── src/
│   ├── _init_paths.py           # ตั้งค่า sys.path ให้ import lib/ ได้
│   ├── train.py                 # เทรนโมเดล (DLA + ReID heads)
│   ├── track_AMOT.py            # entry point เดิมของ AMOT — รัน benchmark บน VisDrone test set
│   ├── run_tracking.py          # entry point ใหม่ — รันกับวิดีโอของตัวเองบนเครื่อง local (baseline หรือ YOLO ก็ได้)
│   ├── run_conf_ablation.sh     # ไล่ค่า --yolo-conf หลายค่าบนคลิปเดียวกัน
│   ├── analyze_conf_ablation.py # รวมผลจาก ablation มาเทียบในตารางเดียว
│   ├── compare_tracking_results.py  # เทียบผล baseline vs YOLO (ไม่ต้องมี GT)
│   ├── diagnose_fragmentation.py    # วิเคราะห์ว่า track ID ใหม่คือวัตถุใหม่จริง หรือ fragment ของ track เดิม
│   ├── gen_dataset_visdrone.py  # แปลง VisDrone annotation → label สำหรับเทรน (มี cls2id/id2cls ที่ใช้ทั้งระบบ)
│   ├── test_import.py           # เทสต์เร็วๆ ว่า import lib ได้ครบ (ไว้ debug ปัญหา DCNv2/_ext)
│   └── lib/
│       ├── opts.py              # config กลางของทั้งโปรเจกต์ (argparse รวมทุก option)
│       ├── cfg/                 # config path ของแต่ละ dataset (visdrone.json, UAVDT.json, VTMOT.json)
│       ├── models/              # backbone (pose_dla_dcn.py), loss, decode
│       ├── tracker/
│       │   ├── multitracker.py  # หัวใจของ tracker: MCJDETracker, MCTrack, detector switch (DLA/YOLO)
│       │   ├── matching.py      # cost matrix: IoU, embedding distance, ReID-motion fusion
│       │   └── basetrack.py
│       ├── tracking_utils/      # evaluation, visualization, kalman filter, io
│       ├── datasets/            # dataloader (jde.py)
│       └── trains/              # training loop
├── requirements.txt
└── README.md
```

---

## การติดตั้ง

### 1. Python environment

```bash
git clone https://github.com/Suraphus/YOLO-integration-AMOT.git
cd YOLO-integration-AMOT
pip install -r requirements.txt
pip install torch torchvision   # เลือกเวอร์ชันให้ตรงกับ CUDA ของเครื่อง
```

ถ้าต้องการใช้ YOLO เป็น detector ให้ติดตั้งเพิ่ม:

```bash
pip install ultralytics
```

(ถ้าไม่ติดตั้ง `ultralytics` ก็ยังรันโหมด baseline เดิมของ AMOT ได้ปกติ — โค้ดเช็ค `import` แบบ optional ไว้ให้แล้ว)

### 2. Build DCNv2

DLA backbone ของ AMOT ใช้ Deformable Convolution v2 ซึ่งต้อง compile เป็น custom CUDA extension ก่อน:

```bash
cd DCNv2
python3 setup.py build develop
cd ..
```

> **หมายเหตุเรื่อง compatibility:** โค้ด DCNv2 ใน repo นี้ถูกแก้ให้ใช้ได้กับ PyTorch/GCC เวอร์ชันใหม่แล้ว (`.data<T>()` → `.data_ptr<T>()`, `.type().is_cuda()` → `.is_cuda()`, เพิ่ม `-Xcompiler -fpermissive` ใน `setup.py`) ถ้ายัง build ไม่ผ่านบนเครื่องคุณ ให้เช็คเวอร์ชัน CUDA toolkit ตรงกับ PyTorch ที่ติดตั้งไว้ก่อน

ถ้าไม่มี GPU/CUDA เลย โค้ดจะ fallback ไปใช้ CPU implementation ของ DCNv2 อัตโนมัติ (ช้ากว่ามาก แต่ยังรันได้)

---

## เตรียมข้อมูล

รองรับ 3 dataset: **VisDrone**, **UAVDT**, **VT-MOT-UAV** (subset ของ [VTMOT](https://github.com/wqw123wqw/PFTrack))

1. ดาวน์โหลด dataset และแตกไฟล์ไว้ในโฟลเดอร์ที่ต้องการ
2. แก้ path ให้ตรงกับเครื่องตัวเองในไฟล์ config ที่ [`src/lib/cfg/`](src/lib/cfg/) (`visdrone.json`, `UAVDT.json`, `VTMOT.json`) — ค่า `root`/path ใน config เหล่านี้เป็นของเครื่องผู้เขียนต้นฉบับ (`jianbo`) **ต้องแก้เป็น path จริงของเครื่องคุณก่อนใช้งาน**
3. ถ้าต้องแปลง VisDrone annotation เป็น label สำหรับเทรน ใช้ [`src/gen_dataset_visdrone.py`](src/gen_dataset_visdrone.py) — ไฟล์นี้มี `cls2id`/`id2cls` ที่เป็น class mapping มาตรฐานของทั้งระบบ (10 คลาส: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor) **ทุกส่วนของโค้ด (รวมถึง YOLO integration) อ้างอิง mapping นี้เป็นหลัก**

รายละเอียดเพิ่มเติมเรื่อง environment/data ดูได้จาก [STCMOT](https://github.com/ydhcg-BoBo/STCMOT) ซึ่ง AMOT สืบทอด pipeline มา

---

## โมเดลที่เทรนไว้แล้ว

| Dataset    | Link                                                                                     |
|------------|-------------------------------------------------------------------------------------------|
| Visdrone   | [BaiduDrive](https://pan.baidu.com/s/1BjGru5Xoi8tuejdG8VsnMw?pwd=2026) (password: 2026)   |
| UAVDT      | [BaiduDrive](https://pan.baidu.com/s/1WX2zLrNSTRlekhWUhcJLQw?pwd=2026) (password: 2026)   |
| VT-MOT-UAV | [BaiduDrive](https://pan.baidu.com/s/1FcPbXRnFAiNAc-oY2m1a3g?pwd=2026) (password: 2026)   |

---

## วิธีรัน Tracking

มี 2 entry point ใช้งานคนละสถานการณ์กัน:

### แบบที่ 1: `track_AMOT.py` (benchmark เดิมของ AMOT)

ใช้สำหรับรัน evaluation บน VisDrone test set (เทียบ MOTA/IDF1 กับ Ground Truth) ตามที่ paper รายงานผล

```bash
cd src
python track_AMOT.py --load_model /path/to/visdrone.pth --data_dir /path/to/UAVdata
```

- `--load_model` และ `--data_dir` มี default เป็น path ส่วนตัวของผู้เขียนต้นฉบับ (`opts.py`) **ต้องระบุเองทุกครั้งด้วย flag ข้างต้น** ไม่งั้นจะหา path ไม่เจอ
- สคริปต์นี้ hardcode รายชื่อ sequence ของ VisDrone test_dev ไว้ในตัวอยู่แล้ว ไม่ต้องแก้อะไรเพิ่ม
- ผลลัพธ์ (ภาพ/วิดีโอ/ไฟล์ track) จะถูกเซฟไว้ตาม `--save_dir_result`

### แบบที่ 2: `run_tracking.py` (รันกับวิดีโอของตัวเองบนเครื่อง local)

เหมาะกับการทดลองเร็วๆ กับคลิปของตัวเอง ไม่ต้องมี Ground Truth และไม่ต้องแก้ hardcode ใดๆ ในไฟล์ — ทุก path ส่งผ่าน argument หมด

```bash
cd src
python run_tracking.py \
    --video /path/to/video.mp4 \
    --model /path/to/visdrone.pth \
    --output output.mp4
```

Argument ที่มี:

| Flag | Default | ความหมาย |
|---|---|---|
| `--video` | (required) | path วิดีโอที่จะ track |
| `--model` | (required) | path checkpoint ของ AMOT (`.pth`) |
| `--output` | `output.mp4` | path วิดีโอผลลัพธ์ |
| `--frames-dir` | `./demo_output_frames` | โฟลเดอร์เก็บเฟรมรายภาพ (จะถูกลบของเก่าทิ้งทุกครั้งที่รันใหม่) |
| `--conf-thres` | `0.4` | เกณฑ์ confidence สำหรับ tracking/เปิด track ใหม่ — ลดถ้าเจอ missing detection |
| `--track-buffer` | `30` | เพิ่มถ้าเจอ ID switch บ่อยตอนวัตถุถูกบัง |
| `--use_yolo` | `False` | เปิดใช้ YOLO เป็น detector แทน DLA heatmap เดิม (ดูหัวข้อ [YOLO Integration](#yolo-integration)) |
| `--yolo-model` | `''` | path weight ของ YOLO (`.pt`) — **บังคับต้องระบุถ้าตั้ง `--use_yolo`** |
| `--yolo-conf` | `0.1` | confidence threshold ของตัว YOLO เอง |

ผลลัพธ์ที่ได้:
- วิดีโอที่วาด box/track id ทับแล้ว (`--output`, codec mp4v) และเวอร์ชัน H.264 ที่เล่นได้ทุกที่ (`*_web.mp4`, ต้องมี `ffmpeg` ในเครื่อง)
- เฟรมรายภาพใน `--frames-dir`
- ไฟล์ track ผลลัพธ์รูปแบบ MOT (`*_tracks.txt`) — เอาไปใช้กับสคริปต์วิเคราะห์ผลต่อได้เลย

ถ้าไม่มี GPU สคริปต์จะแจ้งเตือนและรันบน CPU อัตโนมัติ (ช้ามากและผลอาจต่างจากที่ทดสอบบน GPU)

---

## YOLO Integration

ค่า default ของ AMOT ใช้ heatmap จาก DLA backbone (CenterNet-style) เป็น detector ในตัว ฟีเจอร์นี้เพิ่มทางเลือกให้สลับไปใช้ **YOLO** เป็น detector แทนได้ โดย **ReID feature และ motion model (Kalman filter, GMC, ReID-motion fusion) ยังเป็นของ AMOT เดิมทั้งหมด** — สิ่งที่เปลี่ยนมีแค่แหล่งที่มาของกล่อง detection

วิธีเปิดใช้งาน:

```bash
python run_tracking.py \
    --video clip.mp4 --model visdrone.pth \
    --use_yolo --yolo-model /path/to/yolo_best.pt --yolo-conf 0.1
```

กลไกภายใน (สรุปจาก [`multitracker.py`](src/lib/tracker/multitracker.py)):
1. รัน YOLO detect บนภาพต้นฉบับ (`imgsz=960`) ได้กล่อง + confidence + class
2. แปลงพิกัดกล่องจากภาพต้นฉบับไปเป็นพิกัดบน feature grid ของ AMOT (`orig2map`) เพื่อ sample ReID feature ตรงตำแหน่งที่ YOLO เจอ
3. ส่งต่อเข้า pipeline matching/tracking เดิมของ AMOT ทุกขั้นตอน (association, re-id, Kalman) เหมือนโหมด baseline ทุกประการ

### ⚠️ ข้อควรระวังสำคัญ: Class-index mapping

YOLO ที่เทรนมาเองอาจมีลำดับ class index **ไม่ตรงกับ `cls2id`/`id2cls` ของ AMOT** (กำหนดไว้ใน [`gen_dataset_visdrone.py`](src/gen_dataset_visdrone.py)) ถ้าลำดับไม่ตรงกัน detection จะถูกจัดเข้าคลาสผิดแบบเงียบๆ (ไม่มี error) ทำให้ผลลัพธ์การ track ผิดเพี้ยนทั้งหมดโดยไม่รู้ตัว

ระบบมีการเช็คอัตโนมัติให้แล้ว — ทุกครั้งที่โหลด YOLO detector จะ print ตารางเทียบ class name ระหว่าง YOLO กับ AMOT ออกมาให้ดู เช่น:

```
ตรวจสอบ class mapping ระหว่าง YOLO กับ AMOT (id2cls):
  id 0: yolo="pedestrian"  amot="pedestrian"
  id 4: yolo="truck"       amot="van"        <-- ไม่ตรงกัน!
  ...
```

- ถ้าจำนวนคลาสไม่เท่ากัน → โปรแกรมจะหยุดทันที (`ValueError`)
- ถ้าจำนวนเท่ากันแต่ชื่อ/ลำดับไม่ตรงกัน → โปรแกรม**เตือนแต่ยังรันต่อ** — **ต้องตรวจตารางนี้ด้วยตาเองทุกครั้ง** ก่อนเชื่อผลลัพธ์
- ถ้าเจอ mismatch: **ไม่จำเป็นต้องเทรน YOLO ใหม่** ปัญหานี้แก้ที่โค้ดได้ (map index ตามชื่อคลาส) เพราะ mapping ที่แท้จริงถูกฝังอยู่ใน `.pt` weight อยู่แล้ว (`model.names`)

---

## Confidence Threshold Ablation Study

ใช้หา conf threshold ที่สมดุลระหว่าง recall กับ fragmentation (track ขาดแล้วถูกนับเป็น ID ใหม่)

### ขั้นตอน

1. แก้ path วิดีโอ/โมเดล/tag ใน [`run_conf_ablation.sh`](src/run_conf_ablation.sh) ให้ตรงกับเครื่องตัวเอง แล้วรัน:

```bash
bash run_conf_ablation.sh
```

สคริปต์จะไล่รัน `run_tracking.py --use_yolo` ด้วยค่า `--yolo-conf` หลายค่า (`0.1, 0.15, 0.2, 0.25, 0.3, 0.35` โดย default) บนคลิปเดียวกัน แล้วเซฟไฟล์ `*_tracks.txt` ของแต่ละค่าไว้

2. รวมผลมาเทียบในตารางเดียว:

```bash
python analyze_conf_ablation.py --dir /path/to/conf_ablation --tag clip2
```

จะได้ตารางเทียบ detection count เฉลี่ย/เฟรม, จำนวน track, % track ที่น่าจะเป็น fragment ของ track เดิม (ไม่ใช่วัตถุใหม่จริง) ในแต่ละค่า conf — พร้อมคำแนะนำวิธีอ่านผลท้ายตาราง

---

## เครื่องมือวิเคราะห์ผล

ทั้งสองสคริปต์นี้วิเคราะห์จาก**ไฟล์ track output อย่างเดียว ไม่ต้องมี Ground Truth** (ใช้ proxy metric จากความต่อเนื่องของ track ไม่ใช่ MOTA/IDF1 จริง — ถ้าต้องการตัวเลขทางการสำหรับรายงาน ต้องมี GT annotation แล้วใช้ `track_AMOT.py` ประเมินแทน)

### `compare_tracking_results.py` — เทียบ baseline vs YOLO

```bash
python compare_tracking_results.py \
    --baseline baseline_tracks.txt \
    --yolo yolo_tracks.txt \
    --short-track-thresh 10
```

รายงาน: จำนวน detection เฉลี่ย/เฟรม, ความยาว track เฉลี่ย/สูงสุด, % track สั้น (สัญญาณ fragmentation), จำนวนช่วงที่ track ขาด — พร้อมสรุป % เปลี่ยนแปลงระหว่างสองไฟล์

### `diagnose_fragmentation.py` — วัตถุใหม่จริง vs track ที่ขาดแล้วนับ ID ใหม่

```bash
python diagnose_fragmentation.py --input yolo_tracks.txt --gap-window 30 --dist-thresh 100
```

สำหรับ track ที่ "เกิดกลางคลิป" (ไม่ได้เริ่มที่เฟรมแรกสุด) จะเช็คว่ามี track อื่นที่จบไปไม่นานก่อนหน้าและอยู่ตำแหน่งใกล้กันไหม — ถ้ามีแปลว่าน่าจะเป็น fragment ของ object เดิม ถ้าไม่มีแปลว่าน่าจะเป็นวัตถุใหม่จริง

---

## การเทรนโมเดล

```bash
cd src
python train.py --data_cfg lib/cfg/visdrone.json --arch dla_34
```

- `--data_cfg` เลือก config dataset (`visdrone.json` / `UAVDT.json` / `VTMOT.json` ใน `src/lib/cfg/`) — ต้องแก้ path ข้างในไฟล์ config ให้ตรงกับเครื่องตัวเองก่อน (ดู [เตรียมข้อมูล](#เตรียมข้อมูล))
- default architecture คือ `dla_34` (รองรับ `resdcn_18/34/50`, `resfpndcn_34`, `hrnet_32/18`, `cspdarknet_53` ด้วย แต่ paper ใช้ `dla_34`)
- option อื่นๆ ทั้งหมดดูได้จาก [`src/lib/opts.py`](src/lib/opts.py) (ไฟล์ config กลางที่ทุก entry point ใช้ร่วมกัน)

---

## ปัญหาที่พบบ่อย

**`ModuleNotFoundError: No module named 'dcn_v2'` หรือ `'_ext'`**
มักเจอบน Colab/Kaggle เพราะ `setup.py build develop` บางเวอร์ชันไม่ลงทะเบียน path ให้ครบ แก้โดย build DCNv2 ใหม่ในเครื่อง/สภาพแวดล้อมนั้นโดยตรง หรือรัน [`test_import.py`](src/test_import.py) เพื่อ debug ว่า import ตรงไหนพัง

**DCNv2 build ไม่ผ่านบน PyTorch/GCC เวอร์ชันใหม่**
repo นี้แก้ compatibility ไว้แล้วในระดับหนึ่ง (ดูหัวข้อ [การติดตั้ง](#การติดตั้ง)) ถ้ายังพัง ให้เช็คว่า CUDA toolkit ที่ใช้ compile ตรงกับเวอร์ชันที่ PyTorch build มาด้วย

**`--use_yolo` แล้ว error ว่าไม่มี `--yolo_model`**
ต้องระบุ `--yolo-model /path/to/yolo_best.pt` เสมอเมื่อเปิด `--use_yolo` ไม่มี default path ให้ (เดิมเคย fallback ไป path ส่วนตัวของผู้เขียน ตอนนี้เอาออกแล้วเพื่อไม่ให้พังแบบงงๆ บนเครื่องอื่น)

**ผล tracking ดูผิดปกติตอนใช้ YOLO** (เช่นจับผิดประเภทวัตถุ)
เช็คตาราง "class mapping" ที่ print ออกมาตอนเริ่มโปรแกรมก่อน (ดูหัวข้อ [YOLO Integration](#yolo-integration)) — สาเหตุที่พบบ่อยที่สุดคือลำดับคลาสของ YOLO ไม่ตรงกับ `cls2id` ของ AMOT

---

## Citation

ถ้าเอาโค้ดนี้ไปใช้ กรุณา star โปรเจกต์และพิจารณา cite:

```bibtex
@article{ma2025tracking,
  title={Tracking the Unstable: Appearance-Guided Motion Modeling for Robust Multi-Object Tracking in UAV-Captured Videos},
  author={Ma, Jianbo and Luo, Hui and Chen, Qi and Qi, Yuankai and Sun, Yumei and Beheshti, Amin and Zhang, Jianlin and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2508.01730},
  year={2025}
}
```

## Acknowledgement

โค้ดส่วนใหญ่ borrow มาจาก [FairMOT](https://github.com/ifzhang/FairMOT) และ [STCMOT](https://github.com/ydhcg-BoBo/STCMOT) ขอบคุณสำหรับงานที่ยอดเยี่ยมครับ

หากมีคำถามเกี่ยวกับ paper หรือโค้ด ติดต่อผู้เขียนต้นฉบับได้ตามที่ระบุใน [arXiv paper](https://arxiv.org/abs/2508.01730)
