#!/bin/bash
# run_conf_ablation.sh
# รัน run_tracking.py ไล่หลายค่า --yolo-conf บนคลิปเดียวกัน
# เพื่อดูว่า conf threshold กระทบ fragmentation / detection count ยังไง

set -e  # หยุดทันทีถ้าเจอ error ระหว่างทาง (กันรันต่อด้วยผลลัพธ์ที่ไม่สมบูรณ์)

VIDEO=~/amot_workspace/AMOT/videos/8349314-uhd_3840_2160_25fps.mp4          # แก้เป็นคลิปที่ต้องการ (แนะนำคลิป 3 ก่อน)
MODEL=~/amot_workspace/AMOT/models/visdrone.pth
YOLO_MODEL=~/amot_workspace/AMOT/models/yolo_best.pt
OUTDIR=~/amot_workspace/AMOT/output_videos/conf_ablation
CLIP_TAG="clip2"   # เปลี่ยนชื่อ tag ตามคลิปที่ใช้จริง กันไฟล์ทับกันตอนรันหลายคลิป

mkdir -p "$OUTDIR"
cd ~/amot_workspace/AMOT/src

CONF_VALUES=(0.1 0.15 0.2 0.25 0.3 0.35)

for c in "${CONF_VALUES[@]}"; do
    echo ""
    echo "========================================"
    echo "  รันที่ conf = $c"
    echo "========================================"

    python run_tracking.py \
        --video "$VIDEO" \
        --model "$MODEL" \
        --output "$OUTDIR/${CLIP_TAG}_conf${c}.mp4" \
        --frames-dir "$OUTDIR/${CLIP_TAG}_conf${c}_frames" \
        --use_yolo --yolo-model "$YOLO_MODEL" --yolo-conf "$c"
done

echo ""
echo "รันครบทุกค่า conf แล้ว ผลลัพธ์อยู่ที่: $OUTDIR"
echo "ไฟล์ .txt ที่จะเอาไปวิเคราะห์:"
ls "$OUTDIR"/*_tracks.txt
