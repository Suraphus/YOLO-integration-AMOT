# AMOT (YOLO Integration Fork)

> Official code for multi-object tracker AMOT, AAAI, 2026

![](readme/MOT.png)

> [**Tracking the Unstable: Appearance-Guided Motion Modeling for Robust Multi-Object Tracking in UAV-Captured Videos**](https://arxiv.org/abs/2508.01730),
> Jianbo Ma, Hui Luo, Qi Chen, Yuankai Qi, Yumei Sun, Amin Beheshti, Jianlin Zhang, Ming-Hsuan Yang

Fork ของ AMOT ต้นฉบับ (borrow จาก [FairMOT](https://github.com/ifzhang/FairMOT) และ [STCMOT](https://github.com/ydhcg-BoBo/STCMOT)) ที่เพิ่ม:
- สลับ detector จาก DLA heatmap เดิมไปใช้ **YOLO** ได้ (ยังใช้ ReID/motion model เดิมของ AMOT ทั้งหมด)
- `run_tracking.py` — รันวิดีโอของตัวเองบนเครื่อง local ได้โดยไม่ต้องแก้ hardcode ในไฟล์
- เครื่องมือ ablation/วิเคราะห์ track ที่ไม่ต้องมี Ground Truth

## ติดตั้ง

ต้องใช้ **Python 3.10–3.12** (PyTorch ยังไม่มี wheel รองรับ Python เวอร์ชันใหม่กว่านี้) แนะนำสร้าง conda env แยกต่างหาก:

```bash
conda create -n amot python=3.11 -y
conda activate amot

git clone https://github.com/Suraphus/YOLO-integration-AMOT.git
cd YOLO-integration-AMOT
pip install -r requirements.txt
pip install ultralytics             # จำเป็นเฉพาะถ้าจะใช้ --use_yolo

# เช็ค CUDA toolkit ของเครื่องก่อน (nvcc --version) แล้วเลือก index ให้ตรงกัน เช่น cu124 ถ้า nvcc เป็น 12.4
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

cd DCNv2 && pip install -e . --no-build-isolation && cd ..   # compile custom DCN op
```

ไม่มี GPU ก็รันได้ (fallback เป็น CPU implementation ของ DCNv2 อัตโนมัติ แต่ช้ากว่ามาก)

> หมายเหตุ: ใช้ `pip install -e . --no-build-isolation` แทน `python setup.py build develop` (คำสั่งเดิม deprecated แล้วในเวอร์ชัน setuptools ใหม่ๆ และ default ของ pip จะสร้าง isolated build env ที่ไม่มี torch ติดตั้งอยู่ ทำให้ build fail)

เตรียมข้อมูล (VisDrone / UAVDT / VT-MOT-UAV) และ environment เพิ่มเติม ดูตาม [STCMOT](https://github.com/ydhcg-BoBo/STCMOT) ที่ AMOT สืบทอด pipeline มา — แก้ path ให้ตรงเครื่องตัวเองใน `src/lib/cfg/*.json` ก่อนใช้งาน

### Trained Models
| Dataset    | Link                                                                                     |
|------------|-------------------------------------------------------------------------------------------|
| Visdrone   | [BaiduDrive](https://pan.baidu.com/s/1BjGru5Xoi8tuejdG8VsnMw?pwd=2026) (password: 2026)   |
| UAVDT      | [BaiduDrive](https://pan.baidu.com/s/1WX2zLrNSTRlekhWUhcJLQw?pwd=2026) (password: 2026)   |
| VT-MOT-UAV | [BaiduDrive](https://pan.baidu.com/s/1FcPbXRnFAiNAc-oY2m1a3g?pwd=2026) (password: 2026)   |
| YOLO detector (`yolo_best.pt`) + `visdrone.pth` ที่ใช้ทดสอบใน fork นี้ | [Google Drive](https://drive.google.com/drive/folders/1PP_hzTe_zri7i2YmqNKfhtMgavjXLxiG?usp=sharing) |

วาง `.pth`/`.pt` ที่ดาวน์โหลดมาไว้ในโฟลเดอร์ `models/` ที่ root ของ repo (สร้างเองถ้ายังไม่มี — `.gitignore` กัน `models/*.pth` และ `models/*.pt` ไว้ให้แล้ว จะไม่หลุดเข้า git):

```bash
mkdir -p models
mv ~/Downloads/visdrone.pth ~/Downloads/yolo_best.pt models/
```

ตำแหน่งไม่ได้ล็อกตายตัว — ทุกคำสั่งด้านล่างรับ path ผ่าน `--load_model`/`--model`/`--yolo-model` โดยตรง ใช้ `models/` แค่เป็น convention กลางให้ทั้งทีม path ตรงกัน

## รัน

**Benchmark เดิม (VisDrone test set, เทียบ GT):**
```bash
cd src
python track_AMOT.py --load_model ../models/visdrone.pth --data_dir /path/to/UAVdata
```

**รันวิดีโอของตัวเอง (local, ไม่ต้องมี GT):**
```bash
cd src
python run_tracking.py --video clip.mp4 --model ../models/visdrone.pth --output out.mp4
```
ดู `python run_tracking.py --help` สำหรับ argument ทั้งหมด (conf-thres, track-buffer, ฯลฯ)

## YOLO Integration

เพิ่ม `--use_yolo --yolo-model ../models/yolo_best.pt` เข้าไปที่คำสั่ง `run_tracking.py` ด้านบน (`--yolo-conf` ปรับ threshold ของ YOLO เอง default 0.1)

> ⚠️ **ทุกครั้งที่โหลด YOLO detector โปรแกรมจะ print ตารางเทียบ class mapping ระหว่าง YOLO กับ `id2cls` ของ AMOT (`gen_dataset_visdrone.py`) ให้ตรวจดูก่อนเสมอ** — ถ้าลำดับคลาสไม่ตรงกัน detection จะถูกจัดเข้าคลาสผิดแบบเงียบๆ โดยไม่มี error ถ้าเจอ mismatch ไม่ต้องเทรน YOLO ใหม่ แก้ที่โค้ด (remap ตามชื่อคลาส) ได้

## เครื่องมือเสริม

รายละเอียดวิธีใช้ดูได้จาก docstring ในแต่ละไฟล์ (`--help` ใช้ได้ทุกตัว):
- `run_conf_ablation.sh` + `analyze_conf_ablation.py` — ไล่หาค่า `--yolo-conf` ที่สมดุลระหว่าง recall กับ fragmentation
- `compare_tracking_results.py` — เทียบผล baseline (DLA) กับ YOLO จากไฟล์ track output อย่างเดียว
- `diagnose_fragmentation.py` — วิเคราะห์ว่า track ID ใหม่คือวัตถุใหม่จริง หรือ track เดิมที่ขาดแล้วถูกนับ ID ใหม่

## Citation

```bibtex
@article{ma2025tracking,
  title={Tracking the Unstable: Appearance-Guided Motion Modeling for Robust Multi-Object Tracking in UAV-Captured Videos},
  author={Ma, Jianbo and Luo, Hui and Chen, Qi and Qi, Yuankai and Sun, Yumei and Beheshti, Amin and Zhang, Jianlin and Yang, Ming-Hsuan},
  journal={arXiv preprint arXiv:2508.01730},
  year={2025}
}
```

## Acknowledgement

โค้ดส่วนใหญ่ borrow มาจาก [Fairmot](https://github.com/ifzhang/FairMOT) และ [STCMOT](https://github.com/ydhcg-BoBo/STCMOT) ขอบคุณสำหรับงานที่ยอดเยี่ยม

หากมีคำถามเกี่ยวกับ paper และโค้ด ติดต่อผู้เขียนต้นฉบับได้ตามที่ระบุใน [arXiv paper](https://arxiv.org/abs/2508.01730)
