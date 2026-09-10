"""
analyze_conf_ablation.py

รวมผลจากทุกค่า conf ที่ทดสอบ (จาก run_conf_ablation.sh) มาเทียบในตารางเดียว
ใช้ logic เดียวกับ diagnose_fragmentation.py + compare_tracking_results.py
แต่รวมทุก conf level ไว้ในผลลัพธ์เดียว เพื่อหาจุดสมดุลระหว่าง recall กับ fragmentation

วิธีใช้:
  python analyze_conf_ablation.py --dir ~/amot_workspace/AMOT/output_videos/conf_ablation --tag clip1
"""

import argparse
import math
from collections import defaultdict
import glob
import os
import re


def load_mot_txt(path):
    track_points = defaultdict(list)
    total_dets = 0
    frames_seen = set()

    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            frame_id = int(float(parts[0]))
            uid = int(float(parts[1]))
            x, y, w, h = map(float, parts[2:6])
            cx, cy = x + w / 2.0, y + h / 2.0
            track_points[uid].append((frame_id, cx, cy))
            total_dets += 1
            frames_seen.add(frame_id)

    for uid in track_points:
        track_points[uid].sort(key=lambda p: p[0])

    return track_points, total_dets, len(frames_seen)


def analyze_fragmentation(track_points, gap_window=30, dist_thresh=100):
    all_first_frames = [pts[0][0] for pts in track_points.values()]
    global_start = min(all_first_frames) if all_first_frames else 1

    track_info = []
    for uid, pts in track_points.items():
        first_frame, fx, fy = pts[0]
        last_frame, lx, ly = pts[-1]
        track_info.append({
            'uid': uid, 'first_frame': first_frame, 'first_pos': (fx, fy),
            'last_frame': last_frame, 'last_pos': (lx, ly), 'length': len(pts)
        })

    mid_video_births = [t for t in track_info if t['first_frame'] > global_start]
    fragments = []

    for t in mid_video_births:
        for other in track_info:
            if other['uid'] == t['uid']:
                continue
            frame_gap = t['first_frame'] - other['last_frame']
            if 0 < frame_gap <= gap_window:
                dist = math.sqrt(
                    (other['last_pos'][0] - t['first_pos'][0]) ** 2 +
                    (other['last_pos'][1] - t['first_pos'][1]) ** 2
                )
                if dist <= dist_thresh:
                    fragments.append(t)
                    break

    return track_info, mid_video_births, fragments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', required=True, help='โฟลเดอร์ที่เก็บผลจาก run_conf_ablation.sh')
    parser.add_argument('--tag', required=True, help='CLIP_TAG ที่ตั้งไว้ตอนรัน (เช่น clip3)')
    parser.add_argument('--param', default='conf', choices=['conf', 'amotconf'],
                         help="'conf' = ไล่ค่า --yolo-conf, 'amotconf' = ไล่ค่า --conf-thres ฝั่ง AMOT")
    args = parser.parse_args()

    pattern = os.path.join(args.dir, f"{args.tag}_{args.param}*_tracks.txt")
    files = sorted(glob.glob(pattern), key=lambda p: float(re.search(rf'{args.param}([\d.]+)_tracks', p).group(1)))

    if not files:
        print(f"ไม่พบไฟล์ที่ตรงกับ pattern: {pattern}")
        print("เช็คว่ารัน run_conf_ablation.sh เสร็จแล้ว และ --tag ตรงกับที่ตั้งไว้ในสคริปต์นั้น")
        return

    print(f"\n{'value':>6} {'detections':>11} {'avg/frame':>10} {'tracks':>7} "
          f"{'mid-births':>11} {'%fragment':>10} {'%genuine':>9}")
    print("-" * 72)

    rows = []
    for f in files:
        val = re.search(rf'{args.param}([\d.]+)_tracks', f).group(1)
        track_points, total_dets, n_frames = load_mot_txt(f)
        track_info, mid_births, fragments = analyze_fragmentation(track_points)

        avg_det = total_dets / n_frames if n_frames else 0
        pct_frag = 100 * len(fragments) / len(mid_births) if mid_births else 0
        pct_genuine = 100 - pct_frag if mid_births else 0

        rows.append({
            'val': val, 'total_dets': total_dets, 'avg_det': avg_det,
            'n_tracks': len(track_info), 'mid_births': len(mid_births),
            'pct_frag': pct_frag, 'pct_genuine': pct_genuine,
        })

        print(f"{val:>6} {total_dets:>11} {avg_det:>10.2f} {len(track_info):>7} "
              f"{len(mid_births):>11} {pct_frag:>9.1f}% {pct_genuine:>8.1f}%")

    print("\nวิธีอ่านผล:")
    print("  - 'avg/frame' ลดลงเมื่อ conf สูงขึ้น = คาดหวังได้ (กรองเข้มขึ้น เห็นน้อยลง)")
    print("  - '%fragment' ควรลดลงเมื่อ conf สูงขึ้น (ถ้า noise คือสาเหตุจริง)")
    print("  - จุดที่ดี = conf ที่ %fragment เริ่มลดลงชัดเจน แต่ avg/frame ยังไม่ตกลงมาก")
    print("    (ถ้า avg/frame ตกลงมาใกล้ baseline เดิม แปลว่าเสีย recall ที่เป็นจุดประสงค์หลักไปแล้ว)")


if __name__ == '__main__':
    main()
