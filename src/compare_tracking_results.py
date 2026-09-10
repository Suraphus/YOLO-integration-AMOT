"""
compare_tracking_results.py

เปรียบเทียบผลลัพธ์ tracking ระหว่าง baseline (DLA-only) กับ YOLO integration
โดยไม่ต้องมี Ground Truth — วัดจาก "ความต่อเนื่องของ track" เอง

Metric ที่คำนวณ (ทั้งหมดคำนวณได้จากไฟล์ track output อย่างเดียว ไม่ต้องมี GT):
  - จำนวน track ID ทั้งหมดที่เกิดขึ้น (ยิ่งเยอะเกินจริง = อาจมี fragmentation เยอะ)
  - ความยาวเฉลี่ยของแต่ละ track (เฟรม) — ยิ่งยาวยิ่งดี (แปลว่า track ต่อเนื่อง ไม่ขาด)
  - จำนวน "short-lived track" (track ที่อยู่ไม่กี่เฟรมแล้วหายไป) — สัญญาณของการ fragment
  - จำนวนเฟรมที่มี detection รวม (proxy สำหรับ recall)

วิธีใช้:
  python compare_tracking_results.py \
      --baseline /path/baseline_tracks.txt \
      --yolo /path/yolo_tracks.txt \
      --short-track-thresh 10
"""

import argparse
from collections import defaultdict


def load_mot_txt(path):
    """
    อ่านไฟล์ MOT format: frame,id,x1,y1,w,h,1,-1,-1,-1
    คืนค่า dict: {track_id: [frame_id, frame_id, ...]}  (เรียงลำดับ)
    """
    track_frames = defaultdict(list)
    total_dets = 0
    frames_seen = set()

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 2:
                continue
            frame_id = int(parts[0])
            track_id = int(parts[1])
            track_frames[track_id].append(frame_id)
            total_dets += 1
            frames_seen.add(frame_id)

    return track_frames, total_dets, len(frames_seen)


def analyze(track_frames):
    """
    คำนวณสถิติความต่อเนื่องของแต่ละ track
    """
    track_lengths = {}
    fragment_counts = {}  # จำนวนช่วงที่ไม่ต่อเนื่อง (gap) ในแต่ละ track

    for track_id, frames in track_frames.items():
        frames_sorted = sorted(frames)
        track_lengths[track_id] = len(frames_sorted)

        # นับ gap: ถ้า frame ถัดไปไม่ใช่ frame+1 แปลว่าเกิดการขาดช่วง
        gaps = 0
        for i in range(1, len(frames_sorted)):
            if frames_sorted[i] - frames_sorted[i - 1] > 1:
                gaps += 1
        fragment_counts[track_id] = gaps

    return track_lengths, fragment_counts


def print_report(name, track_frames, total_dets, num_frames, short_thresh):
    track_lengths, fragment_counts = analyze(track_frames)

    num_tracks = len(track_lengths)
    avg_length = sum(track_lengths.values()) / num_tracks if num_tracks > 0 else 0
    max_length = max(track_lengths.values()) if track_lengths else 0
    short_tracks = sum(1 for l in track_lengths.values() if l < short_thresh)
    total_gaps = sum(fragment_counts.values())
    avg_det_per_frame = total_dets / num_frames if num_frames > 0 else 0

    print(f"\n{'=' * 50}")
    print(f"  {name}")
    print(f"{'=' * 50}")
    print(f"  จำนวนเฟรมทั้งหมด          : {num_frames}")
    print(f"  จำนวน detection รวม        : {total_dets}")
    print(f"  detection เฉลี่ย/เฟรม       : {avg_det_per_frame:.2f}")
    print(f"  จำนวน track ID ทั้งหมด      : {num_tracks}")
    print(f"  ความยาว track เฉลี่ย (เฟรม)  : {avg_length:.2f}")
    print(f"  ความยาว track ที่ยาวสุด      : {max_length}")
    print(f"  track สั้น (< {short_thresh} เฟรม)      : {short_tracks} ({100*short_tracks/num_tracks:.1f}% ของทั้งหมด)" if num_tracks else "  track สั้น: N/A")
    print(f"  จำนวนช่วงที่ track ขาด (gap) : {total_gaps}")

    return {
        'num_tracks': num_tracks,
        'avg_length': avg_length,
        'max_length': max_length,
        'short_tracks': short_tracks,
        'total_gaps': total_gaps,
        'avg_det_per_frame': avg_det_per_frame,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', required=True, help='path ของ baseline_tracks.txt')
    parser.add_argument('--yolo', required=True, help='path ของ yolo_tracks.txt')
    parser.add_argument('--short-track-thresh', type=int, default=10,
                         help='จำนวนเฟรมขั้นต่ำที่ถือว่า track ไม่ใช่ short-lived (default: 10)')
    args = parser.parse_args()

    base_frames, base_dets, base_nframes = load_mot_txt(args.baseline)
    yolo_frames, yolo_dets, yolo_nframes = load_mot_txt(args.yolo)

    base_stats = print_report("BASELINE (DLA-only)", base_frames, base_dets, base_nframes,
                               args.short_track_thresh)
    yolo_stats = print_report("YOLO INTEGRATION", yolo_frames, yolo_dets, yolo_nframes,
                               args.short_track_thresh)

    print(f"\n{'=' * 50}")
    print("  สรุปเปรียบเทียบ")
    print(f"{'=' * 50}")
    print(f"{'Metric':<30}{'Baseline':>12}{'YOLO':>12}{'เปลี่ยนแปลง':>15}")
    print("-" * 69)

    def diff_pct(base, new):
        if base == 0:
            return "N/A"
        pct = (new - base) / base * 100
        sign = "+" if pct >= 0 else ""
        return f"{sign}{pct:.1f}%"

    print(f"{'จำนวน detection เฉลี่ย/เฟรม':<30}{base_stats['avg_det_per_frame']:>12.2f}"
          f"{yolo_stats['avg_det_per_frame']:>12.2f}"
          f"{diff_pct(base_stats['avg_det_per_frame'], yolo_stats['avg_det_per_frame']):>15}")
    print(f"{'ความยาว track เฉลี่ย (เฟรม)':<30}{base_stats['avg_length']:>12.2f}"
          f"{yolo_stats['avg_length']:>12.2f}"
          f"{diff_pct(base_stats['avg_length'], yolo_stats['avg_length']):>15}")
    print(f"{'ความยาว track ยาวสุด':<30}{base_stats['max_length']:>12}"
          f"{yolo_stats['max_length']:>12}"
          f"{diff_pct(base_stats['max_length'], yolo_stats['max_length']):>15}")
    print(f"{'% track สั้น (fragment)':<30}"
          f"{100*base_stats['short_tracks']/max(base_stats['num_tracks'],1):>11.1f}%"
          f"{100*yolo_stats['short_tracks']/max(yolo_stats['num_tracks'],1):>11.1f}%"
          f"{'':>15}")
    print(f"{'จำนวนช่วง track ขาด (gap)':<30}{base_stats['total_gaps']:>12}"
          f"{yolo_stats['total_gaps']:>12}"
          f"{diff_pct(base_stats['total_gaps'], yolo_stats['total_gaps']):>15}")

    print(f"\nสรุปการอ่านผล:")
    print(f"  - 'ความยาว track เฉลี่ย' สูงขึ้น = tracking ต่อเนื่องดีขึ้น (ไม่ขาดบ่อย)")
    print(f"  - '% track สั้น' ลดลง = fragmentation น้อยลง (มี track จิ๋วๆ ที่เกิดจาก detect ไม่เจอ+เจอใหม่กลายเป็นคนละ ID น้อยลง)")
    print(f"  - 'จำนวนช่วง track ขาด' ลดลง = ตรงกับปัญหาหลักที่แก้ (detect ไม่เจอตั้งแต่แรก = track ขาด)")
    print(f"  - หมายเหตุ: นี่คือ proxy metric จากพฤติกรรม track เอง ไม่ใช่ MOTA/IDF1 จริง")
    print(f"    เพราะไม่มี Ground Truth ถ้าต้องการตัวเลขทางการสำหรับรายงาน ต้องมี GT annotation")


if __name__ == '__main__':
    main()
