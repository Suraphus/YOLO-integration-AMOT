"""
viz_raw_dets.py

ตอบคำถามบนวิดีโอที่ไม่มี GT: วัตถุที่ AMOT เดิมไม่ตีกรอบ แต่ AMOT + YOLO ตีกรอบ
เป็นเพราะ DLA "มองไม่เห็น" หรือ "เห็นแต่ให้คะแนนต่ำจนไม่ผ่าน threshold"

รัน tracker ทั้งสองแบบ (DLA เดิม และ YOLO) บนวิดีโอเดียวกันทีละเฟรม แล้วอ่านกล่องดิบก่อนเข้าขั้นจับคู่
(tracker.frame_dets ตัวเดียวกับที่ track_AMOT.py --dump_dets ใช้)
  1. ใช้กล่อง YOLO ที่คะแนน >= --ref-thres เป็น "วัตถุที่ YOLO ตีกรอบ" แล้วดูว่า DLA มีกล่องทับตรงนั้นไหม
     (IoU >= --iou ไม่สนคลาส) แบ่งเป็น 3 กลุ่ม:
       pass = DLA ก็มีกล่องคะแนน >= --conf-thres (ผ่าน threshold ของ tracker เหมือนกัน)
       low  = DLA มีกล่อง แต่คะแนนอยู่ระหว่าง --min-score ถึง --conf-thres (เห็นแต่ไม่มั่นใจ)
       none = DLA ไม่มีกล่องคะแนน >= --min-score เลย (มองไม่เห็น)
  2. บันทึกภาพเทียบซ้าย-ขวา (DLA | YOLO) ทุก --every เฟรม
     สีเขียว = คะแนน >= --conf-thres, สีส้ม = --min-score ถึง --conf-thres

ข้อควรระวัง: กล่อง YOLO บางส่วนเป็นกล่องขยะ (บน VisDrone ที่คะแนน 0.4 ราว 3 ใน 10 กล่องผิด)
กลุ่ม none จึงมีทั้ง "DLA พลาด" และ "YOLO ตีกรอบของที่ไม่มีอยู่จริง" ต้องดูภาพประกอบ
หรือใช้ --ref-thres 0.6 ให้กล่องอ้างอิงสะอาดขึ้น

วิธีใช้ (ใน src/):
  python viz_raw_dets.py --video videos/Input2.mp4 --out-dir ~/raw_dets
  python viz_raw_dets.py --video videos/*.mp4 --out-dir ~/raw_dets --max-frames 300 --every 60

ผลลัพธ์:
  <out-dir>/<ชื่อวิดีโอ>/frame_XXXXX.jpg   ภาพเทียบ (ครั้งละ 2560x720 ราว 0.5 MB)
  <out-dir>/summary.csv                     จำนวนวัตถุต่อ วิดีโอ / กลุ่ม / คลาส / ขนาด
"""

import argparse
import os
import os.path as osp
import sys
from collections import Counter

import cv2
import numpy as np
import torch

# แก้ปัญหา ModuleNotFoundError: No module named 'dcn_v2' (เหมือน run_tracking.py)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_here, '..', 'DCNv2')))

import _init_paths
from lib.tracker.multitracker import MCJDETracker, id2cls
import lib.datasets.dataset.jde as datasets
from lib.opts import opts
from reeval_visdrone import iou_matrix

GROUPS = ['pass', 'low', 'none']
SIZE_BINS = [(0, 32, '<32px'), (32, 64, '32-64px'), (64, float('inf'), '>=64px')]
GREEN, ORANGE = (0, 200, 0), (0, 140, 255)  # BGR
PANEL_W, PANEL_H = 1280, 720


def parse_args():
    parser = argparse.ArgumentParser(description='DLA มองไม่เห็น หรือเห็นแต่คะแนนต่ำ? (วิดีโอที่ไม่มี GT)')
    parser.add_argument('--video', nargs='+', required=True, help='วิดีโอ 1 ไฟล์หรือมากกว่า')
    parser.add_argument('--model', default='models/visdrone.pth', help='DLA-34 weight')
    parser.add_argument('--yolo-model', default='models/best.pt', help='YOLO weight')
    parser.add_argument('--out-dir', required=True, help='โฟลเดอร์เก็บภาพและ summary.csv')
    parser.add_argument('--conf-thres', type=float, default=0.4,
                        help='threshold ของ tracker: กล่องต้องได้คะแนนเท่านี้ถึงเริ่ม track ใหม่ได้')
    parser.add_argument('--ref-thres', type=float, default=0.4,
                        help='กล่อง YOLO คะแนนเท่านี้ขึ้นไปนับเป็น "วัตถุที่ YOLO ตีกรอบ"')
    parser.add_argument('--min-score', type=float, default=0.1,
                        help='กล่องคะแนนต่ำกว่านี้ถือว่า "ไม่มีกล่อง" (ใช้เป็น yolo_conf ด้วย)')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU ขั้นต่ำที่นับว่ากล่อง DLA ทับวัตถุเดียวกัน')
    parser.add_argument('--max-frames', type=int, default=300, help='ใช้แค่ N เฟรมแรกของแต่ละวิดีโอ (0 = ทั้งคลิป)')
    parser.add_argument('--every', type=int, default=60, help='บันทึกภาพทุก N เฟรม')
    return parser.parse_args()


def build_opt(args, use_yolo):
    """สร้าง opt ของ AMOT แบบเดียวกับ run_tracking.py"""
    sys.argv = [sys.argv[0]]  # ป้องกัน opts().init() ไป parse args ของเราเอง
    opt = opts().init()
    opt.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    opt.load_model = args.model
    opt.conf_thres = args.conf_thres
    opt.use_yolo = use_yolo
    opt.yolo_model = args.yolo_model
    opt.yolo_conf = args.min_score
    return opt


def xyxy2tlwh(b):
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], axis=1)


def size_bin(w, h):
    s = np.sqrt(max(w * h, 0))
    return next(name for lo, hi, name in SIZE_BINS if lo <= s < hi)


def classify(dla, yolo, args, counts, low_scores):
    """นับว่าวัตถุที่ YOLO ตีกรอบแต่ละตัว DLA ทำอะไรกับมัน (pass / low / none)"""
    ref = yolo[yolo[:, 4] >= args.ref_thres]
    if len(ref) == 0:
        return
    if len(dla):
        iou = iou_matrix(xyxy2tlwh(ref[:, :4]), xyxy2tlwh(dla[:, :4]))
        # คะแนนสูงสุดของกล่อง DLA ที่ทับวัตถุนี้ (-1 = ไม่มีกล่องไหนทับถึง --iou)
        best = np.where(iou >= args.iou, dla[None, :, 4], -1.0).max(axis=1)
    else:
        best = np.full(len(ref), -1.0)
    for box, s in zip(ref, best):
        group = 'pass' if s >= args.conf_thres else ('low' if s >= 0 else 'none')
        counts[(group, id2cls[int(box[5])], size_bin(box[2] - box[0], box[3] - box[1]))] += 1
        if group == 'low':
            low_scores.append(s)


def draw_panel(img0, dets, title, args):
    sx, sy = PANEL_W / img0.shape[1], PANEL_H / img0.shape[0]
    im = cv2.resize(img0, (PANEL_W, PANEL_H))
    high = dets[dets[:, 4] >= args.conf_thres]
    low = dets[dets[:, 4] < args.conf_thres]
    # วาดกล่องคะแนนต่ำก่อน กล่องคะแนนสูงจะได้ทับอยู่ด้านบน
    for boxes, color, thick in ((low, ORANGE, 1), (high, GREEN, 2)):
        for x1, y1, x2, y2, s, _ in boxes:
            p1, p2 = (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy))
            cv2.rectangle(im, p1, p2, color, thick)
            cv2.putText(im, '{:.2f}'.format(s), (p1[0], max(p1[1] - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    cv2.rectangle(im, (0, 0), (PANEL_W, 34), (0, 0, 0), -1)
    cv2.putText(im, '{}   green: score >= {:.2f} ({})   orange: {:.2f}-{:.2f} ({})'.format(
        title, args.conf_thres, len(high), args.min_score, args.conf_thres, len(low)),
        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return im


def run_video(path, args, opt_dla, opt_yolo):
    name = osp.splitext(osp.basename(path))[0]
    out_dir = osp.join(args.out_dir, name)
    os.makedirs(out_dir, exist_ok=True)

    loader = datasets.LoadVideo(path, opt_dla.img_size)
    frame_rate = loader.frame_rate if loader.frame_rate > 0 else 30
    trackers = {'DLA-34 (AMOT)': MCJDETracker(opt_dla, frame_rate),
                'YOLO11s': MCJDETracker(opt_yolo, frame_rate)}

    counts, low_scores, n_frames = Counter(), [], 0
    for i, img, img0 in loader:
        if args.max_frames and i >= args.max_frames:
            break
        blob = torch.from_numpy(img).unsqueeze(0).to(opt_dla.device)
        dets = []
        for tracker in trackers.values():
            tracker.update_tracking(blob, img0)  # รันเต็ม flow ให้เหมือนตอนทำวิดีโอ แต่ใช้แค่กล่องดิบ
            d = tracker.frame_dets
            dets.append(d[d[:, 4] >= args.min_score])
        classify(dets[0], dets[1], args, counts, low_scores)
        n_frames += 1

        if i % args.every == 0:
            panels = [draw_panel(img0, d, title, args) for d, title in zip(dets, trackers)]
            cv2.imwrite(osp.join(out_dir, 'frame_{:05d}.jpg'.format(i + 1)), np.hstack(panels),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
    return name, n_frames, counts, low_scores


def report(title, counts, low_scores, args):
    total = sum(counts.values())
    print('\n=== {} | วัตถุที่ YOLO ตีกรอบ (คะแนน >= {:.2f}): {:,}'.format(title, args.ref_thres, total))
    if total == 0:
        return
    labels = {'pass': 'pass  DLA ก็ผ่าน threshold {:.2f}'.format(args.conf_thres),
              'low': 'low   DLA เห็น แต่คะแนน {:.2f}-{:.2f}'.format(args.min_score, args.conf_thres),
              'none': 'none  DLA ไม่มีกล่อง (คะแนน < {:.2f})'.format(args.min_score)}
    for g in GROUPS:
        n = sum(v for k, v in counts.items() if k[0] == g)
        print('  {}: {:,} ({:.1f}%)'.format(labels[g], n, 100.0 * n / total))
    if low_scores:
        print('  คะแนนกลาง (median) ที่ DLA ให้วัตถุกลุ่ม low: {:.2f}'.format(np.median(low_scores)))

    classes = sorted({k[1] for k in counts}, key=lambda c: -sum(v for k, v in counts.items() if k[1] == c))
    for idx, head, rows in ((1, 'class', classes), (2, 'size', [b[2] for b in SIZE_BINS])):
        print('  {:<16s}{:>9s}{:>8s}{:>8s}{:>8s}'.format(head, 'objects', 'pass', 'low', 'none'))
        for r in rows:
            n = {g: sum(v for k, v in counts.items() if k[0] == g and k[idx] == r) for g in GROUPS}
            tot = sum(n.values())
            if tot:
                print('  {:<16s}{:>9,}{:>7.1f}%{:>7.1f}%{:>7.1f}%'.format(
                    r, tot, *(100.0 * n[g] / tot for g in GROUPS)))


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    opt_dla, opt_yolo = build_opt(args, False), build_opt(args, True)

    all_counts, all_low, rows = Counter(), [], []
    for path in args.video:
        name, n_frames, counts, low_scores = run_video(path, args, opt_dla, opt_yolo)
        report('{} ({} เฟรม)'.format(name, n_frames), counts, low_scores, args)
        all_counts.update(counts)
        all_low += low_scores
        rows += [(name,) + k + (v,) for k, v in sorted(counts.items())]
    if len(args.video) > 1:
        report('รวมทุกวิดีโอ', all_counts, all_low, args)

    with open(osp.join(args.out_dir, 'summary.csv'), 'w') as f:
        f.write('video,group,class,size,count\n')
        f.writelines('{},{},{},{},{}\n'.format(*r) for r in rows)
    print('\nภาพและ summary.csv อยู่ที่ {}'.format(args.out_dir))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
