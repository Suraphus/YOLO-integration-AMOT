"""
diagnose_fragmentation.py

วิเคราะห์ลึกกว่าเดิม: track ID ที่เพิ่มขึ้นในเวอร์ชัน YOLO เป็น
  (A) วัตถุใหม่จริงที่ DLA เดิมไม่เคยเห็นเลย  หรือ
  (B) object เดิมที่ track ขาดแล้วถูกนับเป็น ID ใหม่ (fragmentation)

วิธีคิด: สำหรับ track ที่ "เกิดกลางคลิป" (ไม่ได้เริ่มที่เฟรม 1)
  ดูว่ามี track อื่นที่ "จบไปไม่นานก่อนหน้า" และ "อยู่ตำแหน่งใกล้ๆ กัน" ไหม
  ถ้ามี = น่าจะเป็น fragment ของ object เดิม (กรณี B)
  ถ้าไม่มี = น่าจะเป็นวัตถุใหม่จริงที่โผล่เข้าเฟรม (กรณี A)

วิธีใช้:
  python diagnose_fragmentation.py --input yolo_tracks.txt --gap-window 30 --dist-thresh 100
  python diagnose_fragmentation.py --input baseline_tracks.txt --gap-window 30 --dist-thresh 100
"""

import argparse
import math
from collections import defaultdict


def load_mot_txt(path):
    """
    คืนค่า dict: {unique_id: [(frame, cx, cy), ...]} เรียงตาม frame
    unique_id = cls_id*10000 + track_id (ตามที่แก้ไว้ใน run_tracking.py)
    """
    track_points = defaultdict(list)

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

    for uid in track_points:
        track_points[uid].sort(key=lambda p: p[0])

    return track_points


def analyze_fragmentation(track_points, gap_window, dist_thresh):
    """
    สำหรับ track ที่เริ่มไม่ใช่เฟรมแรกสุดของคลิป
    เช็คว่ามี track อื่นที่จบใกล้ๆ (เวลา+ตำแหน่ง) ก่อนหน้าไหม
    """
    # หาเฟรมแรกสุดในไฟล์ทั้งหมด (ใช้เป็น reference ว่า "เริ่มตั้งแต่ต้นคลิป" คืออะไร)
    all_first_frames = [pts[0][0] for pts in track_points.values()]
    global_start = min(all_first_frames) if all_first_frames else 1

    # เตรียมข้อมูล track แต่ละตัว: (uid, first_frame, first_pos, last_frame, last_pos)
    track_info = []
    for uid, pts in track_points.items():
        first_frame, fx, fy = pts[0]
        last_frame, lx, ly = pts[-1]
        track_info.append({
            'uid': uid, 'first_frame': first_frame, 'first_pos': (fx, fy),
            'last_frame': last_frame, 'last_pos': (lx, ly), 'length': len(pts)
        })

    # track ที่ "เกิดกลางคลิป" (ไม่ได้เริ่มที่เฟรมแรกสุด)
    mid_video_births = [t for t in track_info if t['first_frame'] > global_start]

    likely_fragment = []
    likely_genuine_new = []

    for t in mid_video_births:
        found_predecessor = False
        for other in track_info:
            if other['uid'] == t['uid']:
                continue
            # predecessor ต้องจบ "ก่อนหน้า" ภายใน gap_window เฟรม
            frame_gap = t['first_frame'] - other['last_frame']
            if 0 < frame_gap <= gap_window:
                dist = math.sqrt(
                    (other['last_pos'][0] - t['first_pos'][0]) ** 2 +
                    (other['last_pos'][1] - t['first_pos'][1]) ** 2
                )
                if dist <= dist_thresh:
                    found_predecessor = True
                    break

        if found_predecessor:
            likely_fragment.append(t)
        else:
            likely_genuine_new.append(t)

    return mid_video_births, likely_fragment, likely_genuine_new, track_info


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='path ของไฟล์ tracks.txt (baseline หรือ yolo)')
    parser.add_argument('--gap-window', type=int, default=30,
                         help='จำนวนเฟรมสูงสุดที่ถือว่า "ใกล้เวลากัน" (default: 30 เท่ากับ track_buffer)')
    parser.add_argument('--dist-thresh', type=float, default=100,
                         help='ระยะพิกเซลสูงสุดที่ถือว่า "ตำแหน่งใกล้กัน" (default: 100px)')
    args = parser.parse_args()

    track_points = load_mot_txt(args.input)
    mid_births, fragments, genuine_new, all_tracks = analyze_fragmentation(
        track_points, args.gap_window, args.dist_thresh
    )

    print(f"\n{'=' * 55}")
    print(f"  วิเคราะห์: {args.input}")
    print(f"{'=' * 55}")
    print(f"  จำนวน track ทั้งหมด                : {len(all_tracks)}")
    print(f"  track ที่เกิด 'กลางคลิป' (ไม่ใช่เฟรมแรก): {len(mid_births)}")
    print(f"    - น่าจะเป็น FRAGMENT ของ object เดิม : {len(fragments)}"
          f" ({100*len(fragments)/max(len(mid_births),1):.1f}% ของ mid-births)")
    print(f"    - น่าจะเป็นวัตถุใหม่จริง (genuine)     : {len(genuine_new)}"
          f" ({100*len(genuine_new)/max(len(mid_births),1):.1f}% ของ mid-births)")

    if fragments:
        print(f"\n  ตัวอย่าง track ที่คาดว่าเป็น fragment (5 อันดับแรก):")
        for t in sorted(fragments, key=lambda x: x['first_frame'])[:5]:
            print(f"    uid={t['uid']}, เกิดที่เฟรม {t['first_frame']}, "
                  f"ตำแหน่งเริ่ม ({t['first_pos'][0]:.0f}, {t['first_pos'][1]:.0f}), "
                  f"ความยาว {t['length']} เฟรม")

    print(f"\n  หมายเหตุ: นี่คือ heuristic (ระยะเวลา+ตำแหน่งใกล้กัน) ไม่ใช่การพิสูจน์ 100%")
    print(f"  ใช้เป็นตัวช่วยตัดสินใจว่าควรปรับ threshold หรือปรับ embedding matching ต่อไหม")


if __name__ == '__main__':
    main()
