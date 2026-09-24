"""
run_tracking.py

รัน AMOT (baseline หรือต่อ YOLO) กับวิดีโอไฟล์เดียวโดยตรง (.mov/.mp4)
พร้อมบันทึกผลลัพธ์เป็นวิดีโอ, เฟรมภาพ, และไฟล์ MOT format (.txt) สำหรับวิเคราะห์ต่อ

วิธีใช้:

  # Baseline (DLA-only)
  python run_tracking.py \
      --video ~/amot_workspace/AMOT/videos/clip.mp4 \
      --model ~/amot_workspace/AMOT/models/visdrone.pth \
      --output ~/amot_workspace/AMOT/output_videos/baseline.mp4 \
      --frames-dir ~/amot_workspace/AMOT/output_videos/baseline_frames

  # YOLO integration
  python run_tracking.py \
      --video ~/amot_workspace/AMOT/videos/clip.mp4 \
      --model ~/amot_workspace/AMOT/models/visdrone.pth \
      --output ~/amot_workspace/AMOT/output_videos/yolo.mp4 \
      --frames-dir ~/amot_workspace/AMOT/output_videos/yolo_frames \
      --use_yolo --yolo-model ~/amot_workspace/AMOT/models/yolo_best.pt \
      --yolo-conf 0.1

ผลลัพธ์:
  <output>.mp4           วิดีโอดิบ (mp4v)
  <output>_web.mp4       วิดีโอ H.264 web-optimized
  <output>_tracks.txt    MOT format สำหรับ compare_tracking_results.py / diagnose_fragmentation.py
  <frames-dir>/*.jpg     เฟรมภาพรายเฟรม (ถ้าระบุ --frames-dir)
"""

import argparse
import os
import os.path as osp
import subprocess
import sys

import cv2
import torch

# แก้ปัญหา ModuleNotFoundError: No module named 'dcn_v2' / '_ext'
# (setup.py develop บางเวอร์ชันไม่ลงทะเบียน path ให้ครบ — ยัด path ของ DCNv2/ เข้า sys.path ตรงๆ)
_here = os.path.dirname(os.path.abspath(__file__))
_dcnv2_path = os.path.abspath(os.path.join(_here, '..', 'DCNv2'))
sys.path.insert(0, _dcnv2_path)

import _init_paths
from lib.tracker.multitracker import MCJDETracker
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
from lib.tracking_utils.timer import Timer
from lib.tracking_utils.utils import mkdir_if_missing
import lib.datasets.dataset.jde as datasets
from lib.opts import opts


def parse_args():
    parser = argparse.ArgumentParser(description='รัน AMOT tracking บนวิดีโอไฟล์เดียว')
    parser.add_argument('--video', type=str, required=True,
                        help='path วิดีโอ input (.mp4/.mov)')
    parser.add_argument('--model', type=str, required=True,
                        help='path ของ DLA-34 weight (visdrone.pth)')
    parser.add_argument('--output', type=str, required=True,
                        help='path วิดีโอ output (.mp4)')
    parser.add_argument('--frames-dir', type=str, default='',
                        help='โฟลเดอร์สำหรับบันทึกเฟรมภาพรายเฟรม (ไม่ระบุ = ไม่บันทึก)')
    parser.add_argument('--conf-thres', type=float, default=0.4,
                        help='threshold ของ AMOT cascade (default 0.4)')

    # YOLO integration
    parser.add_argument('--use_yolo', action='store_true',
                        help='เปิดใช้ YOLO เป็น detector แทน mot_decode เดิม')
    parser.add_argument('--yolo-model', type=str, default='',
                        help='path ของ YOLO weight (best.pt)')
    parser.add_argument('--yolo-conf', type=float, default=0.1,
                        help='confidence threshold ของ YOLO (default 0.1)')

    return parser.parse_args()


def build_opt(args):
    """สร้าง opt object ของ AMOT แล้วเติมค่าจาก CLI args"""
    sys.argv = [sys.argv[0]]  # ป้องกัน opts().init() ไป parse args ของเราเอง
    opt = opts().init()

    opt.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    opt.load_model = args.model
    opt.conf_thres = args.conf_thres

    # ส่งค่า YOLO เข้า opt ให้ multitracker.py อ่านได้
    opt.use_yolo = args.use_yolo
    opt.yolo_model = args.yolo_model
    opt.yolo_conf = args.yolo_conf

    return opt


def run(opt, args):
    mkdir_if_missing(osp.dirname(osp.abspath(args.output)))
    if args.frames_dir:
        mkdir_if_missing(args.frames_dir)

    dataloader = datasets.LoadVideo(args.video, opt.img_size)
    frame_rate = dataloader.frame_rate if dataloader.frame_rate > 0 else 30

    logger.info('วิดีโอ: {} | frame_rate ต้นฉบับ: {:.2f} fps'.format(args.video, frame_rate))
    if opt.use_yolo:
        logger.info('โหมด: YOLO integration (conf={}) | weight: {}'.format(
            opt.yolo_conf, opt.yolo_model))
    else:
        logger.info('โหมด: ใช้ mot_decode เดิมของ AMOT (baseline, ไม่ใช้ YOLO)')

    tracker = MCJDETracker(opt, frame_rate)
    timer = Timer()

    out_w, out_h = dataloader.w, dataloader.h
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(args.output, fourcc, frame_rate, (out_w, out_h))

    results = []   # เก็บผลลัพธ์ MOT format
    frame_id = 0

    for path, img, img0 in dataloader:
        if frame_id % 30 == 0 and frame_id != 0:
            logger.info('Processing frame {} ({:.2f} fps)'.format(
                frame_id, 1.0 / max(1e-5, timer.average_time)))
        frame_id += 1

        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)

        timer.tic()
        online_targets_dict = tracker.update_tracking(blob, img0)
        timer.toc()

        online_tlwhs_dict, online_ids_dict, online_scores_dict = {}, {}, {}
        frame_results = []

        for cls_id in range(opt.num_classes):
            online_targets = online_targets_dict[cls_id]
            tlwhs, ids, scores = [], [], []
            for track in online_targets:
                tlwh = track.curr_tlwh
                tid = track.track_id
                if tlwh[2] * tlwh[3] > opt.min_box_area:
                    tlwhs.append(tlwh)
                    ids.append(tid)
                    scores.append(track.score)
                    # unique id ข้ามคลาส: cls_id*10000 + tid กัน ID ชนกัน
                    frame_results.append((tlwh, cls_id * 10000 + tid, track.score))
            online_tlwhs_dict[cls_id] = tlwhs
            online_ids_dict[cls_id] = ids
            online_scores_dict[cls_id] = scores

        # เก็บเป็น MOT format: frame, id, x, y, w, h, score, -1, -1, -1
        for tlwh, uid, score in frame_results:
            x, y, w, h = tlwh
            results.append('{},{},{:.2f},{:.2f},{:.2f},{:.2f},{:.4f},-1,-1,-1\n'.format(
                frame_id, uid, x, y, w, h, score))

        # เก็บสำเนาภาพดิบไว้ก่อนวาดกล่อง (กันเผื่อ plot_tracks วาดทับ img0 ตรงๆ แทนที่จะคืนภาพใหม่)
        # ใช้สำหรับ crop รูป object แบบไม่ติดกรอบใน dashboard ทีหลัง
        raw_frame = img0.copy()

        online_im = vis.plot_tracks(
            image=img0,
            tlwhs_dict=online_tlwhs_dict,
            obj_ids_dict=online_ids_dict,
            num_classes=opt.num_classes,
            scores=online_scores_dict,
            frame_id=frame_id,
            fps=1.0 / max(1e-5, timer.average_time),
        )

        video_writer.write(online_im)

        if args.frames_dir:
            cv2.imwrite(osp.join(args.frames_dir, '{:05d}.jpg'.format(frame_id)), online_im)
            cv2.imwrite(osp.join(args.frames_dir, '{:05d}_raw.jpg'.format(frame_id)), raw_frame)

    video_writer.release()

    # เขียนไฟล์ MOT format
    tracks_path = osp.splitext(args.output)[0] + '_tracks.txt'
    with open(tracks_path, 'w') as f:
        f.writelines(results)

    logger.info('เสร็จแล้ว')
    logger.info('  วิดีโอดิบ    : {}'.format(args.output))
    logger.info('  MOT tracks  : {}'.format(tracks_path))
    if args.frames_dir:
        logger.info('  เฟรมภาพ     : {}'.format(args.frames_dir))
    logger.info('  Total frames: {}, avg FPS: {:.2f}'.format(
        frame_id, 1.0 / max(1e-5, timer.average_time)))

    # แปลงเป็น H.264 web-optimized
    web_output_path = osp.splitext(args.output)[0] + '_web.mp4'
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', args.output, '-c:v', 'libx264', '-crf', '23',
             '-pix_fmt', 'yuv420p', web_output_path],
            check=True, capture_output=True
        )
        logger.info('  เวอร์ชัน web : {}'.format(web_output_path))
    except Exception as e:
        logger.warning('ffmpeg re-encode ไม่สำเร็จ (ไม่กระทบไฟล์ดิบ): {}'.format(e))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    args = parse_args()
    opt = build_opt(args)
    run(opt, args)
