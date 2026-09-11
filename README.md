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

```bash
git clone https://github.com/Suraphus/YOLO-integration-AMOT.git
cd YOLO-integration-AMOT
pip install -r requirements.txt
pip install torch torchvision       # เลือกเวอร์ชันให้ตรงกับ CUDA ของเครื่อง
pip install ultralytics             # จำเป็นเฉพาะถ้าจะใช้ --use_yolo

cd DCNv2 && python3 setup.py build develop && cd ..   # compile custom DCN op
```

ไม่มี GPU ก็รันได้ (fallback เป็น CPU implementation ของ DCNv2 อัตโนมัติ แต่ช้ากว่ามาก)

เตรียมข้อมูล (VisDrone / UAVDT / VT-MOT-UAV) และ environment เพิ่มเติม ดูตาม [STCMOT](https://github.com/ydhcg-BoBo/STCMOT) ที่ AMOT สืบทอด pipeline มา — แก้ path ให้ตรงเครื่องตัวเองใน `src/lib/cfg/*.json` ก่อนใช้งาน

### Trained Models
| Dataset    | Link                                                                                     |
|------------|-------------------------------------------------------------------------------------------|
| Visdrone   | [BaiduDrive](https://pan.baidu.com/s/1BjGru5Xoi8tuejdG8VsnMw?pwd=2026) (password: 2026)   |
| UAVDT      | [BaiduDrive](https://pan.baidu.com/s/1WX2zLrNSTRlekhWUhcJLQw?pwd=2026) (password: 2026)   |
| VT-MOT-UAV | [BaiduDrive](https://pan.baidu.com/s/1FcPbXRnFAiNAc-oY2m1a3g?pwd=2026) (password: 2026)   |

## รัน

**Benchmark เดิม (VisDrone test set, เทียบ GT):**
```bash
cd src
python track_AMOT.py --load_model /path/to/visdrone.pth --data_dir /path/to/UAVdata
```

**รันวิดีโอของตัวเอง (local, ไม่ต้องมี GT):**
```bash
cd src
python run_tracking.py --video clip.mp4 --model visdrone.pth --output out.mp4
```
ดู `python run_tracking.py --help` สำหรับ argument ทั้งหมด (conf-thres, track-buffer, ฯลฯ)

## YOLO Integration

เพิ่ม `--use_yolo --yolo-model /path/to/yolo_best.pt` เข้าไปที่คำสั่ง `run_tracking.py` ด้านบน (`--yolo-conf` ปรับ threshold ของ YOLO เอง default 0.1)

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
