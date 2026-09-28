"""
eval_detection.py

วัด "ตา" อย่างเดียว ไม่ผ่าน tracker — เอากล่องดิบที่ track_AMOT.py --dump_dets บันทึกไว้
(results/<exp_name>/dets/<seq>.txt) มาเทียบกับ GT ทีละเฟรม เพื่อตอบว่า
  - ที่ recall เท่ากัน detector ไหนให้ precision สูงกว่า (= แยกของจริงออกจากกล่องขยะได้ดีกว่า)
  - score เท่ากันของ DLA กับ YOLO มีความหมายต่างกันแค่ไหน

กติกา (ตรงกับ reeval_visdrone.py โหมด +ignore):
  - วัดเฉพาะ 5 คลาส (pedestrian, car, van, truck, bus) และ GT ที่ flag = 1
  - ตัดกล่องทั้ง detection และ GT ที่พื้นที่ทับ ignored region (category 0 / 11) เกิน --ioa
  - detection คู่กับ GT ได้เมื่อคลาสตรงกันและ IoU >= --iou ในเฟรมเดียวกัน แต่ละ GT คู่ได้ครั้งเดียว
    จับคู่จาก score สูงไปต่ำ (คู่กับ GT ที่ยังว่างและ IoU สูงสุด) กล่องที่ไม่ได้คู่ = FP

ผลที่แสดง:
  1. AP@IoU รายคลาส + ค่าเฉลี่ย 5 คลาส
  2. precision ที่ recall เท่ากัน (รวม 5 คลาส) พร้อม score ที่ทำให้ได้ recall นั้น — ตัวตอบสมมติฐานโดยตรง
  3. recall / precision ที่ score threshold ต่างๆ (ค่า default ตรงกับ threshold ที่ tracker ใช้)
  4. recall แยกตามขนาดวัตถุในภาพจริง (ด้านเฉลี่ย √(w×h) เป็น pixel)

หมายเหตุ: กล่องดิบยังไม่ผ่าน min_box_area ของ track_AMOT.py และตัดกล่องที่ score ต่ำกว่า --min_score (0.1)
ทุก run เท่ากัน เพราะ YOLO ส่งออกมาต่ำสุดแค่ yolo_conf = 0.1 ส่วน DLA ส่งต่ำถึง ~0.01 — PR curve จึงหยุดที่
recall ที่ score 0.1 และ AP นับส่วนที่ไปไม่ถึงเป็น 0

วิธีใช้ (ใน env amot):
  python eval_detection.py --gt_dir ~/datasets/VisDrone2019-MOT-test-dev/annotations \
      --run DLA=~/UAVdata/VisDrone2019/test_dev/results/dets_dla/dets \
      --run YOLO=~/UAVdata/VisDrone2019/test_dev/results/dets_yolo/dets
"""

import argparse
import glob
import os
import os.path as osp

import numpy as np

from reeval_visdrone import (read_rows, split_by_frame, ignore_keep_masks, print_table, iou_matrix,
                             EVAL_CATEGORIES, IGNORE_CATEGORIES, FRAME, X, W, H, SCORE, CLS)

CLASS_NAMES = {1: 'pedestrian', 4: 'car', 5: 'van', 6: 'truck', 9: 'bus'}
# ขนาดวัตถุ = √(w×h) ในภาพจริง (pixel)
SIZE_BUCKETS = [('<16px', 0, 16), ('16-32px', 16, 32), ('32-64px', 32, 64), ('>64px', 64, np.inf)]
EMPTY = np.zeros((0, 8))


def match_frame(dets, gts, iou_thr):
    """
    จับคู่กล่องในเฟรมเดียว คลาสเดียว จาก score สูงไปต่ำ
    คืน tp ของแต่ละ det และ score ของ det ที่มาคู่กับแต่ละ GT (0 = ไม่มีใครคู่)
    ลำดับจากสูงไปต่ำทำให้ "GT ที่มี det score >= T มาคู่" เท่ากับผลของการจับคู่เฉพาะกล่อง score >= T
    """
    tp = np.zeros(len(dets), dtype=bool)
    gt_score = np.zeros(len(gts))
    if len(dets) == 0 or len(gts) == 0:
        return tp, gt_score
    order = np.argsort(-dets[:, SCORE], kind='stable')
    iou = iou_matrix(dets[order, X:H + 1], gts[:, X:H + 1])
    matched = np.zeros(len(gts), dtype=bool)
    for k, i in enumerate(order):
        row = np.where(matched, -1.0, iou[k])
        j = int(np.argmax(row))
        if row[j] >= iou_thr:
            matched[j] = True
            tp[i] = True
            gt_score[j] = dets[i, SCORE]
    return tp, gt_score


def evaluate_run(gt_dir, det_dir, seqs, iou_thr, ioa, min_score):
    det_score, det_tp, det_cls = [], [], []    # ต่อกล่อง detection
    gt_cls, gt_size, gt_best = [], [], []      # ต่อกล่อง GT
    for seq in seqs:
        gt_all = read_rows(osp.join(gt_dir, seq + '.txt'))
        gt = gt_all[np.isin(gt_all[:, CLS], EVAL_CATEGORIES) & (gt_all[:, SCORE] == 1)]
        ign = gt_all[np.isin(gt_all[:, CLS], IGNORE_CATEGORIES)]
        det = read_rows(osp.join(det_dir, seq + '.txt'))
        det = det[np.isin(det[:, CLS], EVAL_CATEGORIES) & (det[:, SCORE] >= min_score)]
        keep_gt, keep_det = ignore_keep_masks([gt, det], ign, ioa)
        gt, det = gt[keep_gt], det[keep_det]

        gt_by_f, det_by_f = split_by_frame(gt), split_by_frame(det)
        for f in sorted(set(gt_by_f) | set(det_by_f)):
            g_f, d_f = gt_by_f.get(f, EMPTY), det_by_f.get(f, EMPTY)
            for c in EVAL_CATEGORIES:
                g, d = g_f[g_f[:, CLS] == c], d_f[d_f[:, CLS] == c]
                tp, g_score = match_frame(d, g, iou_thr)
                det_score.append(d[:, SCORE])
                det_tp.append(tp)
                det_cls.append(d[:, CLS])
                gt_cls.append(g[:, CLS])
                gt_size.append(np.sqrt(g[:, W] * g[:, H]))
                gt_best.append(g_score)
    cat = np.concatenate
    return {'det_score': cat(det_score), 'det_tp': cat(det_tp), 'det_cls': cat(det_cls),
            'gt_cls': cat(gt_cls), 'gt_size': cat(gt_size), 'gt_best': cat(gt_best)}


def pr_curve(score, tp, n_gt):
    order = np.argsort(-score, kind='stable')
    tp_cum = np.cumsum(tp[order])
    fp_cum = np.cumsum(~tp[order])
    recall = tp_cum / max(n_gt, 1)
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1)
    return recall, precision, score[order]


def average_precision(recall, precision):
    """AP แบบ all-point interpolation (PASCAL VOC 2010+) — ส่วนที่ recall ไปไม่ถึงนับเป็น 0"""
    r = np.concatenate([[0.0], recall, [1.0]])
    p = np.concatenate([[0.0], precision, [0.0]])
    p = np.maximum.accumulate(p[::-1])[::-1]
    idx = np.nonzero(r[1:] != r[:-1])[0]
    return float(np.sum((r[idx + 1] - r[idx]) * p[idx + 1]))


def at_recall(recall, precision, scores, r):
    """precision (แบบ interpolated) ที่ recall >= r และ score ของกล่องที่ทำให้ recall ถึง r"""
    idx = np.nonzero(recall >= r)[0]
    if len(idx) == 0:
        return None, None
    return float(precision[idx[0]:].max()), float(scores[idx[0]])


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gt_dir', required=True,
                        help='โฟลเดอร์ annotations ดิบของ VisDrone2019-MOT (ไม่ใช่ annotations_eval)')
    parser.add_argument('--run', action='append', required=True, metavar='NAME=DIR',
                        help='ชื่อ=โฟลเดอร์ dets/ ที่ได้จาก track_AMOT.py --dump_dets ใส่ได้หลายครั้ง')
    parser.add_argument('--iou', type=float, default=0.5, help='IoU ขั้นต่ำที่นับว่าเจอ (default 0.5)')
    parser.add_argument('--ioa', type=float, default=0.5,
                        help='ตัดกล่องที่สัดส่วนพื้นที่ใน ignored region เกินค่านี้ (default 0.5)')
    parser.add_argument('--min_score', type=float, default=0.1,
                        help='ตัดกล่องที่ score ต่ำกว่านี้ทุก run ให้เทียบกันแฟร์ (default 0.1 = yolo_conf '
                             'และต่ำกว่าที่ tracker ใช้อยู่แล้ว; DLA ส่งออกมาต่ำถึง ~0.01)')
    parser.add_argument('--recalls', type=float, nargs='+', default=[0.4, 0.5, 0.55, 0.6, 0.65, 0.7],
                        help='ระดับ recall ที่จะเทียบ precision')
    parser.add_argument('--thresholds', type=float, nargs='+', default=[0.2, 0.4, 0.6],
                        help='score threshold ที่จะดู recall / precision (0.2 = พื้นกล่องต่ำ, 0.4/0.6 = conf_thres)')
    parser.add_argument('--out_dir', default='', help='ถ้าระบุ จะบันทึกตารางและ PR curve เป็น CSV')
    args = parser.parse_args()

    gt_dir = osp.abspath(osp.expanduser(args.gt_dir))
    runs = []
    for spec in args.run:
        name, sep, path = spec.partition('=')
        if not sep or not name or not path:
            parser.error('--run ต้องอยู่ในรูป NAME=DIR เช่น DLA=~/UAVdata/.../results/dets_dla/dets')
        runs.append((name, osp.abspath(osp.expanduser(path))))

    def seq_names(d):
        return {osp.splitext(osp.basename(p))[0] for p in glob.glob(osp.join(d, '*.txt'))}

    seqs = seq_names(gt_dir)
    for name, path in runs:
        have = seq_names(path)
        if not have:
            raise FileNotFoundError('ไม่พบไฟล์ใน {} ({}) — รัน track_AMOT.py --dump_dets ก่อน'.format(path, name))
        seqs &= have
    seqs = sorted(seqs)
    print('วัด detection {} sequence | runs: {} | IoU >= {} | score >= {}\n'.format(
        len(seqs), ', '.join(n for n, _ in runs), args.iou, args.min_score))

    results = {}
    for name, path in runs:
        print('กำลังจับคู่: {}'.format(name), flush=True)
        results[name] = evaluate_run(gt_dir, path, seqs, args.iou, args.ioa, args.min_score)

    # ----- 1. AP รายคลาส (คลาสที่ไม่มี GT เลยไม่นับ ไม่งั้นได้ AP 0 ดึงค่าเฉลี่ยลง) -----
    print('\n=== AP@{} รายคลาส ==='.format(args.iou))
    ap_rows = []
    for name, _ in runs:
        r = results[name]
        aps = []
        for c in EVAL_CATEGORIES:
            n_gt = int((r['gt_cls'] == c).sum())
            if n_gt == 0:
                aps.append(float('nan'))
                continue
            sel = r['det_cls'] == c
            rec, prec, _ = pr_curve(r['det_score'][sel], r['det_tp'][sel], n_gt)
            aps.append(100 * average_precision(rec, prec))
        ap_rows.append([name] + aps + [float(np.nanmean(aps))])
    print_table(['run'] + [CLASS_NAMES[c] for c in EVAL_CATEGORIES] + ['mean'],
                [[row[0]] + ['-' if np.isnan(v) else '{:.1f}'.format(v) for v in row[1:]] for row in ap_rows],
                [8, 10, 6, 6, 6, 6, 6])

    # ----- 2. precision ที่ recall เท่ากัน -----
    print('\n=== precision ที่ recall เท่ากัน (รวม 5 คลาส) — ในวงเล็บคือ score ที่ทำให้ได้ recall นั้น ===')
    curves, eq_rows = {}, []
    for name, _ in runs:
        r = results[name]
        rec, prec, sc = pr_curve(r['det_score'], r['det_tp'], len(r['gt_cls']))
        curves[name] = (rec, prec, sc)
        cells = []
        for level in args.recalls:
            p, s = at_recall(rec, prec, sc, level)
            cells.append('-' if p is None else '{:.1f} ({:.2f})'.format(100 * p, s))
        eq_rows.append([name] + cells + ['{:.1f}'.format(100 * rec[-1] if len(rec) else 0)])
    print_table(['run'] + ['R={:.2f}'.format(level) for level in args.recalls] + ['maxR'],
                eq_rows, [8] + [12] * len(args.recalls) + [5])

    # ----- 3. recall / precision ที่ score threshold -----
    print('\n=== recall / precision ที่ score threshold (รวม 5 คลาส) ===')
    thr_rows = []
    for name, _ in runs:
        r = results[name]
        n_gt = len(r['gt_cls'])
        for t in args.thresholds:
            sel = r['det_score'] >= t
            tp = int(r['det_tp'][sel].sum())
            thr_rows.append([name, t, 100.0 * (r['gt_best'] >= t).sum() / max(n_gt, 1),
                             100.0 * tp / max(int(sel.sum()), 1), int(sel.sum()), int(sel.sum()) - tp])
    print_table(['run', 'score>=', 'recall', 'precision', 'boxes', 'FP'],
                [[row[0], '{:.1f}'.format(row[1]), '{:.1f}'.format(row[2]), '{:.1f}'.format(row[3]),
                  '{:,}'.format(row[4]), '{:,}'.format(row[5])] for row in thr_rows],
                [8, 7, 7, 9, 10, 10])

    # ----- 4. recall แยกตามขนาด -----
    print('\n=== recall แยกตามขนาดวัตถุ (√(w×h) ในภาพจริง) ===')
    size_rows = []
    for name, _ in runs:
        r = results[name]
        for t in args.thresholds:
            cells = []
            for _, lo, hi in SIZE_BUCKETS:
                sel = (r['gt_size'] >= lo) & (r['gt_size'] < hi)
                cells.append(100.0 * (r['gt_best'][sel] >= t).sum() / max(int(sel.sum()), 1))
            size_rows.append([name, t] + cells)
    counts = [int(((results[runs[0][0]]['gt_size'] >= lo) & (results[runs[0][0]]['gt_size'] < hi)).sum())
              for _, lo, hi in SIZE_BUCKETS]
    print_table(['run', 'score>='] + ['{} (n={:,})'.format(b[0], n) for b, n in zip(SIZE_BUCKETS, counts)],
                [[row[0], '{:.1f}'.format(row[1])] + ['{:.1f}'.format(v) for v in row[2:]] for row in size_rows],
                [8, 7] + [17] * len(SIZE_BUCKETS))

    if args.out_dir:
        import pandas as pd
        out_dir = osp.abspath(osp.expanduser(args.out_dir))
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(ap_rows, columns=['run'] + [CLASS_NAMES[c] for c in EVAL_CATEGORIES] + ['mean']
                     ).to_csv(osp.join(out_dir, 'ap.csv'), index=False)
        pd.DataFrame(thr_rows, columns=['run', 'score_thr', 'recall', 'precision', 'boxes', 'FP']
                     ).to_csv(osp.join(out_dir, 'thresholds.csv'), index=False)
        pd.DataFrame(size_rows, columns=['run', 'score_thr'] + [b[0] for b in SIZE_BUCKETS]
                     ).to_csv(osp.join(out_dir, 'recall_by_size.csv'), index=False)
        for name, (rec, prec, sc) in curves.items():
            step = max(len(rec) // 1000, 1)    # ย่อ PR curve เหลือ ~1,000 จุดสำหรับวาดกราฟ
            pd.DataFrame({'recall': rec[::step], 'precision': prec[::step], 'score': sc[::step]}
                         ).to_csv(osp.join(out_dir, 'pr_{}.csv'.format(name)), index=False)
        print('\nบันทึก CSV ไว้ที่ {}'.format(out_dir))


if __name__ == '__main__':
    main()
