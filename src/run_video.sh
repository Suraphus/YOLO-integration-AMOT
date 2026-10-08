#!/bin/bash
# run_video.sh
# รัน AMOT + YOLO กับวิดีโอไฟล์เดียว ด้วยค่าเดียวกับที่ใช้วัดผลบน VisDrone
# (track_AMOT.py --use_yolo --conf_thres <conf>, yolo-conf 0.1, imgsz 960)
#
# วิธีใช้:
#   bash run_video.sh <video> [conf] [yolo_weight]
#
# ตัวอย่าง:
#   bash run_video.sh videos/clip.mp4                        # yolo26m, conf 0.6
#   bash run_video.sh videos/clip.mp4 0.7                    # MOTA สูงสุด / false positive น้อย
#   bash run_video.sh videos/clip.mp4 0.6 models/yolo26s.pt  # เทียบกับ 26s

set -e

if [ "$#" -lt 1 ]; then
    echo "วิธีใช้: bash run_video.sh <video> [conf=0.6] [yolo_weight=models/yolo26m.pt]"
    exit 1
fi

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VIDEO="$(realpath "$1")"
CONF="${2:-0.6}"
YOLO="${3:-models/yolo26m.pt}"

cd "$SRC_DIR"
for f in "$VIDEO" "$YOLO" models/visdrone.pth; do
    if [ ! -f "$f" ]; then
        echo "ไม่พบไฟล์: $f"
        exit 1
    fi
done

CLIP="$(basename "${VIDEO%.*}")"
TAG="$(basename "${YOLO%.*}")_c${CONF}"
OUT_DIR="$SRC_DIR/../output_videos"
OUT="$OUT_DIR/${CLIP}_${TAG}.mp4"
mkdir -p "$OUT_DIR"

python run_tracking.py \
    --video "$VIDEO" \
    --model models/visdrone.pth \
    --output "$OUT" \
    --conf-thres "$CONF" \
    --use_yolo --yolo-model "$YOLO" --yolo-imgsz 960

echo ""
echo "เสร็จแล้ว:"
ls -lh "${OUT%.mp4}"*
