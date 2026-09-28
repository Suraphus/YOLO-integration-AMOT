from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from numpy.core._multiarray_umath import ndarray
import _init_paths
import os
import os.path as osp
import cv2
import logging
import motmetrics as mm
import numpy as np
import torch

from collections import defaultdict
from lib.tracker.multitracker import MCJDETracker, ASSOC_STAT_KEYS
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.evaluation import Evaluator
import lib.datasets.dataset.jde as datasets

from lib.tracking_utils.utils import mkdir_if_missing
from lib.opts import opts

def write_results_dict(file_name, results_dict, data_type, num_classes=10):
    if data_type == 'mot':
        save_format = '{frame},{id},{x1},{y1},{w},{h},{score}, {cls_id},-1, -1\n'
    else:
        raise ValueError(data_type)
    with open(file_name, 'w') as f:
        for cls_id in range(num_classes):
            if cls_id == 1 or cls_id == 2 or cls_id == 6 or cls_id == 7 or cls_id == 9:
                continue
            cls_results = results_dict[cls_id]
            for frame_id, tlwhs, track_ids, scores in cls_results:
                for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                    if track_id < 0:
                        continue
                    x1, y1, w, h = tlwh
                    re_cls_id = cls_id + 1
                    line = save_format.format(frame=frame_id,
                                              id=track_id,
                                              x1=x1, y1=y1, w=w, h=h,
                                              score=score,
                                              cls_id=re_cls_id)
                    f.write(line)
    logger.info('save results to {}'.format(file_name))

def write_dets(file_name, frame_dets):
    """กล่องดิบของ detector ต่อเฟรม: frame,-1,x,y,w,h,score,category (category = cls_id + 1 ตาม VisDrone)"""
    mkdir_if_missing(osp.dirname(file_name))
    with open(file_name, 'w') as f:
        for frame_id, dets in frame_dets:
            for x1, y1, x2, y2, score, cls_id in dets:
                f.write('{},-1,{:.2f},{:.2f},{:.2f},{:.2f},{:.4f},{}\n'.format(
                    frame_id, x1, y1, x2 - x1, y2 - y1, score, int(cls_id) + 1))
    logger.info('save raw detections to {}'.format(file_name))

def eval_seq(opt,
             data_loader,
             data_type,
             result_f_name,
             save_dir=None,
             show_image=True,
             frame_rate=30,
             oracle_file=None):
    if save_dir:
        mkdir_if_missing(save_dir)
    tracker = MCJDETracker(opt, frame_rate)
    if oracle_file:
        tracker.load_oracle(oracle_file)
    timer = Timer()
    results_dict = defaultdict(list)
    frame_dets = []
    frame_id = 0
    for path, img, img0 in data_loader:
        if frame_id % 30 == 0 and frame_id != 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(frame_id, 1.0 / max(1e-5, timer.average_time)))
        frame_id += 1


        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)
        timer.tic()
        online_targets_dict = tracker.update_tracking(blob, img0)
        timer.toc()
        if opt.dump_dets:
            frame_dets.append((frame_id, tracker.frame_dets))
        online_tlwhs_dict = defaultdict(list)
        online_ids_dict = defaultdict(list)
        online_scores_dict = defaultdict(list)
        for cls_id in range(opt.num_classes):
            online_targets = online_targets_dict[cls_id]
            for track in online_targets:
                tlwh = track.curr_tlwh
                t_id = track.track_id
                score = track.score
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    online_tlwhs_dict[cls_id].append(tlwh)
                    online_ids_dict[cls_id].append(t_id)
                    online_scores_dict[cls_id].append(score)
        for cls_id in range(opt.num_classes):
            results_dict[cls_id].append((frame_id,
                                         online_tlwhs_dict[cls_id],
                                         online_ids_dict[cls_id],
                                         online_scores_dict[cls_id]))
        if show_image or save_dir is not None:
            if frame_id > 0:
                online_im: ndarray = vis.plot_tracks(image=img0,
                                                     tlwhs_dict=online_tlwhs_dict,
                                                     obj_ids_dict=online_ids_dict,
                                                     num_classes=opt.num_classes,
                                                     scores=online_scores_dict,
                                                     frame_id=frame_id,
                                                     fps=1.0 / timer.average_time)
        if frame_id > 0:
            if show_image:
                cv2.imshow('online_im', online_im)
            if save_dir is not None:
                cv2.imwrite(os.path.join(save_dir, '{:05d}.jpg'.format(frame_id)), online_im)
    write_results_dict(result_f_name, results_dict, data_type)
    if opt.dump_dets:
        # ไว้ในโฟลเดอร์ย่อย dets/ เพื่อไม่ให้ reeval_visdrone.py (อ่าน results/<exp>/*.txt) เข้าใจผิดว่าเป็นผล tracking
        write_dets(osp.join(osp.dirname(result_f_name), 'dets', osp.basename(result_f_name)), frame_dets)
    return frame_id, timer.average_time, timer.calls, tracker.assoc_stats

def save_assoc_stats(file_name, seq_stats):
    """
    บันทึกตัวนับของ tracker ต่อ sequence (+ แถว TOTAL) — track ได้คู่ในขั้นไหน, MTC ถูกเรียก/สำเร็จกี่ครั้ง
    ความหมายของแต่ละคอลัมน์ดูที่ ASSOC_STAT_KEYS ใน lib/tracker/multitracker.py
    เป็นไฟล์ .csv เพื่อไม่ให้ reeval_visdrone.py (อ่าน *.txt) เข้าใจผิดว่าเป็นผลของ sequence
    """
    total = defaultdict(int)
    with open(file_name, 'w') as f:
        f.write(','.join(['sequence'] + ASSOC_STAT_KEYS) + '\n')
        for seq, stats in seq_stats:
            f.write(','.join([seq] + [str(stats[k]) for k in ASSOC_STAT_KEYS]) + '\n')
            for k in ASSOC_STAT_KEYS:
                total[k] += stats[k]
        f.write(','.join(['TOTAL'] + [str(total[k]) for k in ASSOC_STAT_KEYS]) + '\n')

    matched = total['stage1_match'] + total['stage2_match'] + total['stage3_low_match']
    logger.info('MTC: ส่งเข้า {} ครั้ง, กู้สำเร็จ {} ครั้ง ({:.1f}%) | '
                'ส่งเข้า MTC {:.1f} ครั้งต่อการจับคู่ 100 ครั้ง'.format(
                    total['mtc_candidates'], total['mtc_reactivated'],
                    100.0 * total['mtc_reactivated'] / max(total['mtc_candidates'], 1),
                    100.0 * total['mtc_candidates'] / max(matched, 1)))
    logger.info('save association stats to {}'.format(file_name))

def main(opt,
         data_root='',
         seqs=('',),
         exp_name='',
         save_images=False,
         save_videos=False,
         show_image=True):
    logger.setLevel(logging.INFO)
    result_root = os.path.join(data_root, '..', 'results', exp_name)
    mkdir_if_missing(result_root)
    data_type = 'mot'
    accs = []
    n_frame = 0
    timer_avgs, timer_calls = [], []
    seq_stats = []
    for seq in seqs:
        output_dir = os.path.join(
            data_root, '..', 'outputs', exp_name, seq) if save_images or save_videos else None
        logger.info('start seq: {}'.format(seq))
        dataloader = datasets.LoadImages(
            osp.join(data_root, seq), opt.img_size)
        result_filename = os.path.join(result_root, '{}.txt'.format(seq))
        frame_rate = 30
        # Oracle ใช้ GT ชุดเดียวกับที่ Evaluator ใช้วัดผล (setup_visdrone_eval.py กรองไว้แล้ว)
        oracle_file = osp.join(data_root, '..', 'annotations_eval', seq + '.txt') if opt.oracle else None
        nf, ta, tc, stats = eval_seq(opt, dataloader, data_type, result_filename,
                                     save_dir=output_dir, show_image=show_image, frame_rate=frame_rate,
                                     oracle_file=oracle_file)
        seq_stats.append((seq, stats))
        n_frame += nf
        timer_avgs.append(ta)
        timer_calls.append(tc)
        logger.info('Evaluate seq: {}'.format(seq))
        evaluator = Evaluator(data_root, seq, data_type)
        accs.append(evaluator.eval_file(result_filename))
    timer_avgs = np.asarray(timer_avgs)
    timer_calls = np.asarray(timer_calls)
    all_time = np.dot(timer_avgs, timer_calls)
    avg_time = all_time / np.sum(timer_calls)
    logger.info('Time elapsed: {:.2f} seconds, FPS: {:.2f}'.format(
        all_time, 1.0 / avg_time))
    metrics = mm.metrics.motchallenge_metrics
    mh = mm.metrics.create()
    summary = Evaluator.get_summary(accs, seqs, metrics)
    strsummary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names
    )
    print(strsummary)
    Evaluator.save_summary(summary, os.path.join(
        result_root, 'summary_{}.xlsx'.format(exp_name)))
    save_assoc_stats(os.path.join(result_root, 'assoc_stats.csv'), seq_stats)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    opt = opts().init()

    # รายชื่อ sequence อ่านจากโฟลเดอร์ที่ setup_visdrone_eval.py --split <split> เตรียมไว้
    # (test_dev มี 17, val มี 7) แทนการ hardcode ไว้เฉพาะ test_dev
    data_root = os.path.join(opt.data_dir, 'VisDrone2019', opt.split, 'sequences')
    if not osp.isdir(data_root):
        raise FileNotFoundError(
            'ไม่พบ {} — รัน setup_visdrone_eval.py --split {} ก่อน'.format(data_root, opt.split))
    seqs = sorted(d for d in os.listdir(data_root) if osp.isdir(osp.join(data_root, d)))
    print('split: {} | {} sequence | ผลจะอยู่ที่ {}'.format(
        opt.split, len(seqs), osp.join(data_root, '..', 'results', opt.exp_name)))
    opt.device = 'cuda:0'
    main(opt,
         data_root=data_root,
         seqs=seqs,
         exp_name=opt.exp_name,
         show_image=False,
         save_images=opt.save_images,
         save_videos=False)