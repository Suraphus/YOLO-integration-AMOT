#!/usr/bin/env python3
"""
run_tracking.py — รัน AMOT บนวิดีโอของตัวเอง (สำหรับเครื่อง local ไม่ใช่ Colab/Kaggle)

วิธีใช้:
    cd ~/amot_workspace/AMOT/src
    python run_tracking.py --video /path/to/video.mp4 --model /path/to/visdrone.pth

ทุก path เป็น argument ที่ต้องระบุเอง ไม่ต้องแก้ hardcode ในไฟล์เหมือนตอนอยู่บน Colab/Kaggle
"""
import sys
import os
import argparse

# แก้ปัญหา ModuleNotFoundError: No module named 'dcn_v2' / '_ext'
# (เจอปัญหานี้ตอนอยู่บน Colab/Kaggle เพราะ setup.py develop บางเวอร์ชันไม่ลงทะเบียน path ให้ครบ)
_here = os.path.dirname(os.path.abspath(__file__))
_dcnv2_path = os.path.abspath(os.path.join(_here, '..', 'DCNv2'))
sys.path.insert(0, _dcnv2_path)

import _init_paths
import cv2
import torch
from collections import defaultdict
from lib.opts import opts
from lib.tracker.multitracker import MCJDETracker
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.timer import Timer
import lib.datasets.dataset.jde as datasets
from lib.tracking_utils.io import write_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True, help='path ของวิดีโอที่จะ track')
    parser.add_argument('--model', required=True, help='path ของ visdrone.pth (หรือ checkpoint อื่น)')
    parser.add_argument('--output', default='output.mp4', help='path วิดีโอผลลัพธ์')
    parser.add_argument('--frames-dir', default='./demo_output_frames', help='โฟลเดอร์เก็บเฟรมรายภาพ')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='ลดจาก default 0.4 ถ้าเจอ missing detection')
    parser.add_argument('--det-thres', type=float, default=0.3, help='ลดจาก default 0.3 ถ้าเจอ missing detection')
    parser.add_argument('--track-buffer', type=int, default=30, help='เพิ่มถ้าเจอ ID switch บ่อยตอนวัตถุถูกบัง')
    parser.add_argument('--use_yolo', action='store_true')
    parser.add_argument('--yolo-model', type=str, default='') 
    parser.add_argument('--yolo-conf', type=float, default=0.1)
    args = parser.parse_args()

    if not os.path.isfile(args.video):
        raise FileNotFoundError(f"ไม่พบวิดีโอที่: {args.video}")
    if not os.path.isfile(args.model):
        raise FileNotFoundError(f"ไม่พบ checkpoint ที่: {args.model}")

    # ลบเฟรมเก่าทิ้งก่อน กันปัญหาเฟรมจากคลิปก่อนหน้าตกค้างปนกัน
    import shutil
    if os.path.isdir(args.frames_dir):
        shutil.rmtree(args.frames_dir)
    os.makedirs(args.frames_dir, exist_ok=True)

    opt = opts().init([])
    opt.use_yolo = args.use_yolo
    opt.yolo_model = args.yolo_model
    opt.yolo_conf = args.yolo_conf
    opt.conf_thres = args.conf_thres
    opt.det_thres = args.det_thres
    opt.track_buffer = args.track_buffer
    opt.load_model = args.model
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if opt.device.type == 'cpu':
        print("⚠️  ไม่พบ GPU — จะรันบน CPU (ช้ามาก และผลลัพธ์อาจไม่แม่นยำเท่า GPU ตามที่เคยทดสอบไว้)")

    dataloader = datasets.LoadVideo(args.video, opt.img_size)
    frame_rate = dataloader.frame_rate
    print(f"วิดีโอ: {args.video} | frame_rate ต้นฉบับ: {frame_rate:.2f} fps")

    tracker = MCJDETracker(opt, frame_rate)
    timer = Timer()

    video_writer = None  # สร้างตอนรู้ขนาดเฟรมแรก เขียนวิดีโอไปพร้อมกันเลย ไม่ต้องรอจบก่อน
    frame_id = 0
    results_dict = {}

    for path, img, img0 in dataloader:
        frame_id += 1
        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)
        timer.tic()
        online_targets_dict = tracker.update_tracking(blob, img0)
        timer.toc()

        online_tlwhs_dict = defaultdict(list)
        online_ids_dict = defaultdict(list)
        online_scores_dict = defaultdict(list)
        for cls_id in range(opt.num_classes):
            for t in online_targets_dict[cls_id]:
                if t.curr_tlwh[2] * t.curr_tlwh[3] > opt.min_box_area:
                    online_tlwhs_dict[cls_id].append(t.curr_tlwh)
                    online_ids_dict[cls_id].append(t.track_id)
                    online_scores_dict[cls_id].append(t.score)

        frame_results = []
        for cls_id in range(opt.num_classes):
            for tid, tlwh in zip(online_ids_dict[cls_id], online_tlwhs_dict[cls_id]):
                frame_results.append((tlwh, cls_id * 10000 + tid))
        results_dict[frame_id] = frame_results
        
        online_im = vis.plot_tracks(
            image=img0, tlwhs_dict=online_tlwhs_dict,
            obj_ids_dict=online_ids_dict, num_classes=opt.num_classes,
            scores=online_scores_dict, frame_id=frame_id,
            fps=1.0 / max(1e-5, timer.average_time)
        )

        cv2.imwrite(os.path.join(args.frames_dir, f'{frame_id:05d}.jpg'), online_im)

        if video_writer is None:
            h, w = online_im.shape[:2]
            video_writer = cv2.VideoWriter(
                args.output, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (w, h)
            )
        video_writer.write(online_im)

        if frame_id % 30 == 0:
            print(f'ประมวลผลแล้ว {frame_id} เฟรม | {1.0 / max(1e-5, timer.average_time):.1f} FPS')

    if video_writer is not None:
        video_writer.release()

        txt_output = args.output.replace('.mp4', '_tracks.txt')
        write_results(txt_output, results_dict, 'mot')

    print(f"\nเสร็จแล้ว! ประมวลผลไป {frame_id} เฟรม")
    print(f"วิดีโอผลลัพธ์ (mp4v codec): {args.output}")
    print(f"เฟรมรายภาพ: {args.frames_dir}")

    # แปลงเป็น H.264 อัตโนมัติให้เล่นได้ทุกที่ (เหมือนที่ทำตอนอยู่บน Colab/Kaggle)
    web_output = args.output.replace('.mp4', '_web.mp4')
    os.system(f'ffmpeg -y -i "{args.output}" -vcodec libx264 -pix_fmt yuv420p "{web_output}"')
    print(f"วิดีโอเวอร์ชัน web-compatible (H.264): {web_output}")


if __name__ == '__main__':
    main()

