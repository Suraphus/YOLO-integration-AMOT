#!/bin/bash
# compare_videos.sh
# รวม 2 วิดีโอมาวางซ้าย-ขวา พร้อม label กำกับ สำหรับเทียบ Baseline vs YOLO
#
# วิธีใช้:
#   bash compare_videos.sh <baseline.mp4> <yolo.mp4> <output.mp4>
#
# ตัวอย่าง:
#   bash compare_videos.sh clip2_baseline.mp4 clip2_yolo.mp4 clip2_compare.mp4

set -e

if [ "$#" -lt 3 ]; then
    echo "วิธีใช้: bash compare_videos.sh <baseline.mp4> <yolo.mp4> <output.mp4>"
    echo ""
    echo "ตัวอย่าง:"
    echo "  bash compare_videos.sh \\"
    echo "      ~/test/YOLO-integration-AMOT/output_videos/clip2_baseline.mp4 \\"
    echo "      ~/test/YOLO-integration-AMOT/output_videos/clip2_yolo.mp4 \\"
    echo "      ~/test/YOLO-integration-AMOT/output_videos/clip2_compare.mp4"
    exit 1
fi

LEFT_VIDEO="$1"
RIGHT_VIDEO="$2"
OUTPUT="$3"
LEFT_LABEL="${4:-YOLO Integration}"
RIGHT_LABEL="${5:-Baseline (DLA-only)}"

# เช็คว่าไฟล์มีอยู่จริงก่อนเริ่ม
for f in "$LEFT_VIDEO" "$RIGHT_VIDEO"; do
    if [ ! -f "$f" ]; then
        echo "ไม่พบไฟล์: $f"
        exit 1
    fi
done

# เช็คความยาวของทั้ง 2 คลิป (เตือนถ้าต่างกันมาก)
DUR_L=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$LEFT_VIDEO")
DUR_R=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$RIGHT_VIDEO")
echo "ความยาว: ซ้าย ${DUR_L}s / ขวา ${DUR_R}s"
echo ""

echo "กำลังรวมวิดีโอ..."
echo "  ซ้าย  : $LEFT_VIDEO  [$LEFT_LABEL]"
echo "  ขวา   : $RIGHT_VIDEO  [$RIGHT_LABEL]"
echo "  output: $OUTPUT"
echo ""

ffmpeg -i "$LEFT_VIDEO" -i "$RIGHT_VIDEO" -filter_complex \
"[0:v]scale=960:540,format=yuv420p,drawtext=text='${LEFT_LABEL}':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[left]; \
[1:v]scale=960:540,format=yuv420p,drawtext=text='${RIGHT_LABEL}':x=20:y=20:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=8[right]; \
[left][right]hstack=inputs=2[v]" \
-map "[v]" -c:v libx264 -crf 20 -preset medium \
-colorspace bt709 -color_primaries bt709 -color_trc bt709 \
-pix_fmt yuv420p -shortest "$OUTPUT" -y

echo ""
echo "เสร็จแล้ว: $OUTPUT"
ls -lh "$OUTPUT"
