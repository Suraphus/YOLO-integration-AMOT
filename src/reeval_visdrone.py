"""
reeval_visdrone.py

วัดผล tracking ใหม่จากไฟล์ใน results/ ที่ track_AMOT.py เขียนไว้แล้ว (ไม่ต้องรัน tracking ใหม่)
แล้วแสดงทีละขั้นว่าการแก้กติกาการวัดผลแต่ละข้อเปลี่ยนตัวเลขไปเท่าไร

โหมด (สะสมทีละขั้น จากบนลงล่าง):
  legacy   เหมือน Evaluator เดิม (lib/tracking_utils/evaluation.py) — ไม่ดูคลาส, ใช้ track ID ดิบ,
           ไม่ตัด ignored region และนับเฉพาะเฟรมที่มีผล tracking
  +frames  นับเฟรมที่มี GT แต่ tracker ไม่ output อะไรเลยด้วย (GT ในเฟรมนั้นเป็น FN)
  +uid     ทำ track ID ไม่ให้ชนกันข้ามคลาส (cls*100000 + id) — ID ของ AMOT นับแยกคลาส
  +class   match แยกคลาส (ผลคลาส car match ได้กับ GT car เท่านั้น) แล้วรวม count ทุกคลาส
  +ignore  ตัดกล่องทั้งผลและ GT ที่พื้นที่ทับ ignored region (category 0 / 11) เกิน --ioa
           = โหมดที่แก้ครบ

ใช้ annotation ดิบของ VisDrone (ไม่ใช่ annotations_eval ที่ setup_visdrone_eval.py กรองไว้)
เพราะต้องใช้ ignored region และคลาสอื่นๆ ด้วย

--hota: คำนวณ HOTA / DetA / AssA ด้วย TrackEval (โค้ดทางการของ HOTA) โดยใช้กติกาเดียวกับ +ignore
  (ต้องติดตั้งก่อน: pip install git+https://github.com/JonathonLuiten/TrackEval.git)

นอกจากนี้จะนับจำนวน GT ต่อเฟรม (คลาส 1-10 ที่ DLA ทาย) เทียบกับเพดาน --K
เพราะ mot_decode เลือก top-K รวมทุกคลาสต่อเฟรม (default --K 200) ส่วน YOLO ตัดที่ max_det=300

หมายเหตุ: กติกา +class / +ignore เขียนให้ใกล้ toolkit ทางการของ VisDrone แต่ไม่ใช่ตัวเดียวกัน
ตัวเลขจึงอาจยังไม่ตรงกับ paper เป๊ะ

วิธีใช้ (ใน env amot):
  python reeval_visdrone.py \
      --gt_dir ~/datasets/VisDrone2019-MOT-test-dev/annotations \
      --run baseline=~/UAVdata/VisDrone2019/test_dev/results/baseline \
      --run yolo=~/UAVdata/VisDrone2019/test_dev/results/yolo
"""

import argparse
import glob
import os
import os.path as osp

import numpy as np
import motmetrics as mm

mm.lap.default_solver = 'lap'

# motmetrics 1.4.0 ยังเรียก np.asfarray ซึ่งถูกลบไปใน NumPy 2.0
if not hasattr(np, 'asfarray'):
    np.asfarray = lambda a, dtype=np.float64: np.asarray(a, dtype=dtype)

# category ของ VisDrone ที่ track_AMOT.py เขียนผลออกมา: pedestrian, car, van, truck, bus
EVAL_CATEGORIES = (1, 4, 5, 6, 9)
# ignored region กับ others (ใน annotation ของ test-dev สองคลาสนี้คือแถวที่ flag=0 ทั้งหมด)
IGNORE_CATEGORIES = (0, 11)
# 10 คลาสที่ DLA ของ AMOT ทาย ใช้นับความแน่นต่อเฟรมเทียบกับ --K
AMOT_CATEGORIES = tuple(range(1, 11))

MODES = [
    ('legacy',  dict(all_frames=False, unique_id=False, per_class=False, ignore=False)),
    ('+frames', dict(all_frames=True,  unique_id=False, per_class=False, ignore=False)),
    ('+uid',    dict(all_frames=True,  unique_id=True,  per_class=False, ignore=False)),
    ('+class',  dict(all_frames=True,  unique_id=True,  per_class=True,  ignore=False)),
    ('+ignore', dict(all_frames=True,  unique_id=True,  per_class=True,  ignore=True)),
]

# ค่าที่แสดงจาก HOTA (แต่ละตัวเฉลี่ยจาก IoU threshold 0.05–0.95 ตามนิยามของ HOTA)
HOTA_FIELDS = ['HOTA', 'DetA', 'AssA', 'DetRe', 'DetPr', 'AssRe', 'AssPr']

COUNT_METRICS = ['num_objects', 'num_predictions', 'num_detections', 'num_false_positives',
                 'num_misses', 'num_switches', 'idtp']

# คอลัมน์ของ array ที่อ่านมา (GT และผล tracking ใช้ 8 คอลัมน์แรกตรงกัน:
# คอลัมน์ 6 ของ GT คือ flag, ของผลคือ score / คอลัมน์ 7 คือ category ของ VisDrone ทั้งคู่)
FRAME, TID, X, Y, W, H, SCORE, CLS = range(8)
EMPTY = np.zeros((0, 8))


def read_rows(path):
    """อ่านไฟล์ MOT/VisDrone เป็น array (N, 8)"""
    rows = []
    with open(path, 'r') as f:
        for line in f:
            cols = line.strip().split(',')
            if len(cols) < 8:
                continue
            rows.append([float(c) for c in cols[:8]])
    return np.asarray(rows, dtype=np.float64).reshape(-1, 8)


def split_by_frame(rows):
    order = np.argsort(rows[:, FRAME], kind='stable')
    rows = rows[order]
    frames, starts = np.unique(rows[:, FRAME].astype(np.int64), return_index=True)
    ends = list(starts[1:]) + [len(rows)]
    return {int(f): rows[s:e] for f, s, e in zip(frames, starts, ends)}


def iou_matrix(a, b):
    """IoU ระหว่างกล่อง tlwh (N, 4) กับ (M, 4) — ใช้ทั้ง HOTA ในไฟล์นี้และ eval_detection.py"""
    ax1, ay1 = a[:, 0:1], a[:, 1:2]
    ax2, ay2 = ax1 + a[:, 2:3], ay1 + a[:, 3:4]
    bx1, by1 = b[:, 0], b[:, 1]
    bx2, by2 = bx1 + b[:, 2], by1 + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(ax1, bx1), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(ay1, by1), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def to_pixels(tlwhs):
    x1 = np.floor(tlwhs[:, 0]).astype(np.int64)
    y1 = np.floor(tlwhs[:, 1]).astype(np.int64)
    x2 = np.ceil(tlwhs[:, 0] + tlwhs[:, 2]).astype(np.int64)
    y2 = np.ceil(tlwhs[:, 1] + tlwhs[:, 3]).astype(np.int64)
    return x1, y1, x2, y2


class IgnoreRegion(object):
    """
    union ของ ignored region ในเฟรมเดียว เก็บเป็น integral image
    ใช้ union แทนการรวมทีละกล่อง เพราะ ignored region ซ้อนกันเองได้ (ไม่งั้นพื้นที่ทับจะถูกนับซ้ำ)
    """

    def __init__(self, tlwhs):
        x1, y1, x2, y2 = to_pixels(tlwhs)
        self.ox, self.oy = x1.min(), y1.min()
        w, h = x2.max() - self.ox, y2.max() - self.oy
        mask = np.zeros((h, w), dtype=bool)
        for a, b, c, d in zip(x1 - self.ox, y1 - self.oy, x2 - self.ox, y2 - self.oy):
            mask[b:d, a:c] = True
        self.integral = np.zeros((h + 1, w + 1), dtype=np.int64)
        self.integral[1:, 1:] = mask.cumsum(0).cumsum(1)

    def ioa(self, tlwhs):
        """สัดส่วนพื้นที่ของแต่ละกล่องที่อยู่ใน ignored region (intersection over box area)"""
        x1, y1, x2, y2 = to_pixels(tlwhs)
        area = np.maximum((x2 - x1) * (y2 - y1), 1)
        h, w = self.integral.shape[0] - 1, self.integral.shape[1] - 1
        cx1, cx2 = np.clip(x1 - self.ox, 0, w), np.clip(x2 - self.ox, 0, w)
        cy1, cy2 = np.clip(y1 - self.oy, 0, h), np.clip(y2 - self.oy, 0, h)
        ii = self.integral
        inter = ii[cy2, cx2] - ii[cy1, cx2] - ii[cy2, cx1] + ii[cy1, cx1]
        return inter / area


def ignore_keep_masks(row_sets, ign_rows, thr):
    """คืน keep mask ของแต่ละชุดกล่อง (False = ทับ ignored region เกิน thr) สร้าง region ครั้งเดียวต่อเฟรม"""
    keeps = [np.ones(len(rows), dtype=bool) for rows in row_sets]
    frame_ids = [rows[:, FRAME].astype(np.int64) for rows in row_sets]
    for f, ign in split_by_frame(ign_rows).items():
        region = None
        for keep, fids, rows in zip(keeps, frame_ids, row_sets):
            idx = np.nonzero(fids == f)[0]
            if len(idx) == 0:
                continue
            if region is None:
                region = IgnoreRegion(ign[:, X:H + 1])
            keep[idx] = region.ioa(rows[idx, X:H + 1]) <= thr
    return keeps


def eval_sequence(mh, gt, res, frames, unique_id, per_class):
    """รวม count ของทั้ง sequence (ถ้า per_class = แยก accumulator ต่อคลาสแล้วรวม count)"""
    gt_by_f, res_by_f = split_by_frame(gt), split_by_frame(res)
    total = dict.fromkeys(COUNT_METRICS, 0)
    for cls in (EVAL_CATEGORIES if per_class else (None,)):
        if cls is not None and not (np.any(gt[:, CLS] == cls) or np.any(res[:, CLS] == cls)):
            continue
        acc = mm.MOTAccumulator(auto_id=True)
        for f in frames:
            g = gt_by_f.get(f, EMPTY)
            r = res_by_f.get(f, EMPTY)
            if cls is not None:
                g = g[g[:, CLS] == cls]
                r = r[r[:, CLS] == cls]
            hyp_ids = r[:, TID] + (r[:, CLS] * 100000 if unique_id else 0)
            dist = mm.distances.iou_matrix(g[:, X:H + 1], r[:, X:H + 1], max_iou=0.5)
            acc.update(g[:, TID].astype(np.int64), hyp_ids.astype(np.int64), dist)
        counts = mh.compute(acc, metrics=COUNT_METRICS, name='acc')
        for k in COUNT_METRICS:
            total[k] += int(counts[k].iloc[0])
    return total


def hota_sequence(metric, gt, res, frames):
    """
    ส่ง 1 sequence × 1 คลาสให้ trackeval.metrics.HOTA (โค้ดทางการของ HOTA) คำนวณ
    TrackEval ใช้ ID เป็นเลขช่องของตาราง จึงต้องแปลง ID ดิบ (เช่น 7, 152, 3001) ให้เรียงเป็น 0, 1, 2, ...
    similarity ระหว่าง GT กับ track ในแต่ละเฟรม = IoU (เหมือนที่ TrackEval ใช้กับ MOTChallenge)
    """
    gt, res = gt.copy(), res.copy()
    gt[:, TID] = np.unique(gt[:, TID], return_inverse=True)[1]
    res[:, TID] = np.unique(res[:, TID], return_inverse=True)[1]
    gt_by_f, res_by_f = split_by_frame(gt), split_by_frame(res)
    data = {'num_gt_dets': len(gt), 'num_tracker_dets': len(res),
            'num_gt_ids': int(gt[:, TID].max()) + 1 if len(gt) else 0,
            'num_tracker_ids': int(res[:, TID].max()) + 1 if len(res) else 0,
            'gt_ids': [], 'tracker_ids': [], 'similarity_scores': []}
    for f in frames:
        g, r = gt_by_f.get(f, EMPTY), res_by_f.get(f, EMPTY)
        data['gt_ids'].append(g[:, TID].astype(int))
        data['tracker_ids'].append(r[:, TID].astype(int))
        data['similarity_scores'].append(iou_matrix(g[:, X:H + 1], r[:, X:H + 1]))
    return metric.eval_sequence(data)


def summarize(c):
    n_gt = max(c['num_objects'], 1)
    det = c['num_detections']
    # IDF1 ตามสูตรของ motmetrics (idf1_m) — ใช้จำนวนกล่องทั้งหมดเป็นตัวหาร
    # ไม่ใช้ idfp เพราะถ้า ID ซ้ำกันในเฟรมเดียว (โหมด legacy) idfp จะไม่เท่ากับ num_predictions - idtp
    return {
        'IDF1': 100.0 * 2 * c['idtp'] / max(c['num_objects'] + c['num_predictions'], 1),
        'MOTA': 100.0 * (1.0 - (c['num_misses'] + c['num_switches'] + c['num_false_positives']) / n_gt),
        'Rcll': 100.0 * det / n_gt,
        'Prcn': 100.0 * det / max(det + c['num_false_positives'], 1),
        'FP': c['num_false_positives'],
        'FN': c['num_misses'],
        'IDs': c['num_switches'],
        'GT': c['num_objects'],
    }


def add_counts(dicts):
    total = dict.fromkeys(COUNT_METRICS, 0)
    for d in dicts:
        for k in COUNT_METRICS:
            total[k] += d[k]
    return total


def gt_density(gt_all, eval_gt, ks):
    """จำนวนวัตถุ (คลาส 1-10) ต่อเฟรม และสัดส่วน GT ที่ใช้วัดผลซึ่งอยู่ในเฟรมที่แน่นเกินเพดาน K"""
    objs = gt_all[np.isin(gt_all[:, CLS], AMOT_CATEGORIES)]
    per_frame = np.bincount(objs[:, FRAME].astype(np.int64))
    out = {'frames': int(gt_all[:, FRAME].max()), 'max_per_frame': int(per_frame.max())}
    eval_frames = eval_gt[:, FRAME].astype(np.int64)
    for k in ks:
        crowded = np.nonzero(per_frame > k)[0]
        out['frames>{}'.format(k)] = len(crowded)
        out['evalGT_in>{}'.format(k)] = int(np.isin(eval_frames, crowded).sum())
    out['evalGT'] = len(eval_gt)
    return out


def print_table(header, rows, widths):
    fmt = '  '.join('{:>' + str(w) + '}' for w in widths)
    print(fmt.format(*header))
    for row in rows:
        print(fmt.format(*row))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--gt_dir', required=True,
                        help='โฟลเดอร์ annotations ดิบของ VisDrone2019-MOT (ไม่ใช่ annotations_eval)')
    parser.add_argument('--run', action='append', required=True, metavar='NAME=DIR',
                        help='ชื่อ=โฟลเดอร์ results/<exp_name> ใส่ได้หลายครั้ง')
    parser.add_argument('--modes', default=','.join(name for name, _ in MODES),
                        help='โหมดที่จะวัด คั่นด้วย , (default: ทุกโหมด)')
    parser.add_argument('--seqs', default='',
                        help='วัดเฉพาะ sequence เหล่านี้ คั่นด้วย , (default: ทุก sequence ที่มีครบ)')
    parser.add_argument('--ioa', type=float, default=0.5,
                        help='ตัดกล่องที่สัดส่วนพื้นที่ใน ignored region เกินค่านี้ (default 0.5)')
    parser.add_argument('--K', type=int, nargs='+', default=[200, 300],
                        help='เพดานจำนวน detection ต่อเฟรมที่จะเทียบ (DLA --K 200, YOLO max_det 300)')
    parser.add_argument('--out_dir', default='',
                        help='ถ้าระบุ จะบันทึกตารางเป็น CSV ไว้ที่นี่')
    parser.add_argument('--hota', action='store_true',
                        help='คำนวณ HOTA / DetA / AssA ด้วยกติกาของโหมด +ignore (ต้อง pip install TrackEval)')
    args = parser.parse_args()

    gt_dir = osp.abspath(osp.expanduser(args.gt_dir))
    runs = []
    for spec in args.run:
        name, sep, path = spec.partition('=')
        if not sep or not name or not path:
            parser.error('--run ต้องอยู่ในรูป NAME=DIR เช่น baseline=~/UAVdata/.../results/baseline')
        runs.append((name, osp.abspath(osp.expanduser(path))))

    mode_cfg = dict(MODES)
    modes = [m.strip() for m in args.modes.split(',') if m.strip()]
    for m in modes:
        if m not in mode_cfg:
            parser.error('ไม่รู้จักโหมด {} (มี: {})'.format(m, ', '.join(mode_cfg)))

    # sequence ที่มีครบทั้ง GT และผลทุก run
    def seq_names(d):
        return {osp.splitext(osp.basename(p))[0] for p in glob.glob(osp.join(d, '*.txt'))}

    gt_seqs = seq_names(gt_dir)
    if not gt_seqs:
        raise FileNotFoundError('ไม่พบไฟล์ GT ใน {}'.format(gt_dir))
    seqs = set(gt_seqs)
    for name, path in runs:
        have = seq_names(path)
        if not have:
            raise FileNotFoundError('ไม่พบไฟล์ผลใน {} ({})'.format(path, name))
        missing = sorted(have - gt_seqs)
        if missing:
            print('[Warning] {}: ไม่มี GT ของ {} — ข้าม'.format(name, ', '.join(missing)))
        seqs &= have
    if args.seqs:
        seqs &= set(s.strip() for s in args.seqs.split(','))
    seqs = sorted(seqs)
    print('วัดผล {} sequence | runs: {} | modes: {}\n'.format(
        len(seqs), ', '.join(n for n, _ in runs), ', '.join(modes)))

    hota_metric = None
    if args.hota:
        # TrackEval ยังใช้ np.float ซึ่งถูกลบไปใน NumPy 2.0 (แบบเดียวกับ np.asfarray ของ motmetrics ข้างบน)
        if not hasattr(np, 'float'):
            np.float = float
        from trackeval.metrics import HOTA   # import เฉพาะตอนใช้ คนที่ไม่ใส่ --hota ไม่ต้องติดตั้ง
        hota_metric = HOTA()
    hota_res = {r: {} for r, _ in runs}                      # run -> {"seq/คลาส": ผล HOTA}

    mh = mm.metrics.create()
    counts = {(r, m): {} for r, _ in runs for m in modes}   # (run, mode) -> {seq: count dict}
    dropped = {r: {} for r, _ in runs}                       # run -> {seq: จำนวนกล่องที่ถูกตัด}
    density = {}

    for i, seq in enumerate(seqs, 1):
        print('[{}/{}] {}'.format(i, len(seqs), seq), flush=True)
        gt_all = read_rows(osp.join(gt_dir, seq + '.txt'))
        eval_gt = gt_all[np.isin(gt_all[:, CLS], EVAL_CATEGORIES) & (gt_all[:, SCORE] == 1)]
        ign = gt_all[np.isin(gt_all[:, CLS], IGNORE_CATEGORIES)]
        results = [read_rows(osp.join(path, seq + '.txt')) for _, path in runs]
        density[seq] = gt_density(gt_all, eval_gt, args.K)

        keeps = ignore_keep_masks([eval_gt] + results, ign, args.ioa)
        eval_gt_kept = eval_gt[keeps[0]]

        for (name, _), res, keep in zip(runs, results, keeps[1:]):
            dropped[name][seq] = int((~keep).sum())
            for m in modes:
                cfg = mode_cfg[m]
                gt_m, res_m = (eval_gt_kept, res[keep]) if cfg['ignore'] else (eval_gt, res)
                res_frames = set(res[:, FRAME].astype(np.int64).tolist())
                if cfg['all_frames']:
                    frames = sorted(res_frames | set(eval_gt[:, FRAME].astype(np.int64).tolist()))
                else:
                    frames = sorted(f for f in res_frames if f >= 1)
                counts[(name, m)][seq] = eval_sequence(mh, gt_m, res_m, frames,
                                                       cfg['unique_id'], cfg['per_class'])
            if hota_metric is not None:
                # กติกาเดียวกับโหมด +ignore: ตัด ignored region แล้ว, แยกคลาส, นับทุกเฟรม
                res_kept = res[keep]
                frames = sorted(set(res[:, FRAME].astype(np.int64).tolist())
                                | set(eval_gt[:, FRAME].astype(np.int64).tolist()))
                for cls in EVAL_CATEGORIES:
                    g = eval_gt_kept[eval_gt_kept[:, CLS] == cls]
                    r = res_kept[res_kept[:, CLS] == cls]
                    if len(g) or len(r):
                        hota_res[name]['{}/{}'.format(seq, cls)] = hota_sequence(hota_metric, g, r, frames)

    # ----- 1. ภาพรวม: ตัวเลขเปลี่ยนไปเท่าไรเมื่อแก้กติกาทีละข้อ -----
    print('\n=== Overall ({} seq) — legacy ควรตรงกับตัวเลขเดิมจาก track_AMOT.py ==='.format(len(seqs)))
    overall_rows = []
    for name, _ in runs:
        for m in modes:
            s = summarize(add_counts(counts[(name, m)].values()))
            overall_rows.append([name, m, s['IDF1'], s['MOTA'], s['Rcll'], s['Prcn'],
                                 s['FP'], s['FN'], s['IDs'], s['GT']])
    print_table(['run', 'mode', 'IDF1', 'MOTA', 'Rcll', 'Prcn', 'FP', 'FN', 'IDs', 'GT'],
                [[r[0], r[1]] + ['{:.1f}'.format(v) for v in r[2:6]] + ['{:,}'.format(v) for v in r[6:]]
                 for r in overall_rows],
                [10, 8, 5, 5, 5, 5, 9, 9, 7, 9])

    # ----- 1b. HOTA: รวมทุก sequence × คลาส แบบถ่วงตามจำนวนกล่อง (= combine_classes_det_averaged ของ TrackEval) -----
    hota_rows = []
    if hota_metric is not None:
        print('\n=== HOTA (กติกาเดียวกับ +ignore) — DetA = ตา, AssA = การต่อ ID, HOTA = √(DetA × AssA) ===')
        for name, _ in runs:
            combined = hota_metric.combine_sequences(hota_res[name])
            hota_rows.append([name] + [100.0 * float(np.mean(combined[k])) for k in HOTA_FIELDS])
        print_table(['run'] + HOTA_FIELDS,
                    [[row[0]] + ['{:.1f}'.format(v) for v in row[1:]] for row in hota_rows],
                    [12] + [6] * len(HOTA_FIELDS))

    # ----- 2. ราย sequence: โหมดแรกเทียบโหมดสุดท้าย (ดูว่า FP ที่หายไปกระจุกที่ไหน) -----
    first, last = modes[0], modes[-1]
    seq_rows = []
    for name, _ in runs:
        print('\n=== {}: ราย sequence ({} → {}) ==='.format(name, first, last))
        rows = []
        for seq in seqs:
            a = summarize(counts[(name, first)][seq])
            b = summarize(counts[(name, last)][seq])
            rows.append([seq, a['FP'], b['FP'], dropped[name][seq], a['MOTA'], b['MOTA'], a['IDF1'], b['IDF1']])
            seq_rows.append([name] + rows[-1])
        print_table(['sequence', 'FP_' + first, 'FP_' + last, 'ign_drop',
                     'MOTA_' + first, 'MOTA_' + last, 'IDF1_' + first, 'IDF1_' + last],
                    [[r[0]] + ['{:,}'.format(v) for v in r[1:4]] + ['{:.1f}'.format(v) for v in r[4:]]
                     for r in rows],
                    [20, 10, 10, 9, 12, 12, 12, 12])

    # ----- 3. ความแน่นของฉากเทียบเพดาน K -----
    print('\n=== GT ต่อเฟรม (คลาส 1-10) เทียบเพดาน K ===')
    k_cols = []
    for k in args.K:
        k_cols += ['frames>{}'.format(k), 'evalGT_in>{}'.format(k)]
    dens_rows = [[seq, density[seq]['frames'], density[seq]['max_per_frame']]
                 + [density[seq][c] for c in k_cols] + [density[seq]['evalGT']] for seq in seqs]
    totals = ['TOTAL', sum(r[1] for r in dens_rows), max(r[2] for r in dens_rows)]
    totals += [sum(r[3 + j] for r in dens_rows) for j in range(len(k_cols))]
    totals += [sum(r[-1] for r in dens_rows)]
    print_table(['sequence', 'frames', 'max/frame'] + k_cols + ['evalGT'],
                [[r[0]] + ['{:,}'.format(v) for v in r[1:]] for r in dens_rows + [totals]],
                [20, 7, 9] + [max(len(c), 9) for c in k_cols] + [9])

    if args.out_dir:
        import pandas as pd
        out_dir = osp.abspath(osp.expanduser(args.out_dir))
        os.makedirs(out_dir, exist_ok=True)
        pd.DataFrame(overall_rows, columns=['run', 'mode', 'IDF1', 'MOTA', 'Rcll', 'Prcn', 'FP', 'FN', 'IDs', 'GT']
                     ).to_csv(osp.join(out_dir, 'overall.csv'), index=False)
        pd.DataFrame(seq_rows, columns=['run', 'sequence', 'FP_' + first, 'FP_' + last, 'ign_drop',
                                        'MOTA_' + first, 'MOTA_' + last, 'IDF1_' + first, 'IDF1_' + last]
                     ).to_csv(osp.join(out_dir, 'per_sequence.csv'), index=False)
        pd.DataFrame(dens_rows, columns=['sequence', 'frames', 'max_per_frame'] + k_cols + ['evalGT']
                     ).to_csv(osp.join(out_dir, 'gt_density.csv'), index=False)
        if hota_rows:
            pd.DataFrame(hota_rows, columns=['run'] + HOTA_FIELDS).to_csv(osp.join(out_dir, 'hota.csv'), index=False)
        print('\nบันทึก CSV ไว้ที่ {}'.format(out_dir))


if __name__ == '__main__':
    main()
