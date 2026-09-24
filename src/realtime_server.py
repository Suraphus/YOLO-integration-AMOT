"""
realtime_server.py (v2)

รัน AMOT (baseline หรือ YOLO integration) เบื้องหลัง พร้อม 2 ฟีเจอร์ใหม่:
  1. เก็บผลทุกเฟรมลงดิสก์ -> dashboard ย้อนดูเฟรมเก่าได้ (ไม่ใช่แค่ live)
  2. อัปโหลดวิดีโอจากหน้าเว็บได้เลย ไม่ต้องสั่ง --video ทาง CLI ทุกรอบ

ติดตั้งก่อนใช้ (ครั้งเดียว):
  pip install fastapi "uvicorn[standard]" python-multipart --break-system-packages

วิธีใช้:

  # แบบไม่ระบุวิดีโอ (รอ upload จากหน้าเว็บ) — เหมาะกับการใช้งานทั่วไป
  python realtime_server.py \
      --model ~/test/YOLO-integration-AMOT/src/models/visdrone.pth \
      --use_yolo --yolo-model ~/test/YOLO-integration-AMOT/src/models/best.pt

  # แบบระบุวิดีโอทันที (เหมือนเดิม จะเริ่มประมวลผลทันทีที่ server start ด้วย)
  python realtime_server.py \
      --video ~/test/YOLO-integration-AMOT/src/videos/Input2.mp4 \
      --model ~/test/YOLO-integration-AMOT/src/models/visdrone.pth \
      --use_yolo --yolo-model ~/test/YOLO-integration-AMOT/src/models/best.pt

จากนั้นเปิด dashboard_live.html — อัปโหลดวิดีโอใหม่จากหน้าเว็บได้เลย
(อัปโหลดวิดีโอใหม่ระหว่างที่มีงานเก่ากำลังรันอยู่ = หยุดงานเก่าแล้วเริ่มงานใหม่ทันที)
"""

import argparse
import asyncio
import base64
import json
import os
import queue
import shutil
import sys
import threading

import cv2
import torch

# แก้ปัญหา ModuleNotFoundError: No module named 'dcn_v2' / '_ext'
# (เหมือนที่แก้ไว้ใน run_tracking.py — setup.py develop บางเวอร์ชันไม่ลงทะเบียน path ให้ครบ)
_here = os.path.dirname(os.path.abspath(__file__))
_dcnv2_path = os.path.abspath(os.path.join(_here, '..', 'DCNv2'))
sys.path.insert(0, _dcnv2_path)

import _init_paths
from lib.tracker.multitracker import MCJDETracker
from lib.tracking_utils import visualization as vis
from lib.tracking_utils.log import logger
import lib.datasets.dataset.jde as datasets
from lib.opts import opts

# import fastapi/uvicorn **หลัง** lib ของ AMOT เสมอ
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

CLASS_NAMES = {
    0: 'pedestrian', 1: 'people', 2: 'bicycle', 3: 'car', 4: 'van',
    5: 'truck', 6: 'tricycle', 7: 'awning-tricycle', 8: 'bus', 9: 'motor',
}

UPLOAD_DIR = os.path.join(_here, 'uploads')
RESULTS_DIR = os.path.join(_here, 'results')
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

app = FastAPI()
# อนุญาตให้ dashboard_live.html (เปิดจาก file:// หรือคนละพอร์ต) เรียก /upload, /frame, /status ได้
# (ปลอดภัยสำหรับใช้งานในเครื่องตัวเอง/เครือข่ายภายในเท่านั้น — ไม่ควรเปิดออกอินเทอร์เน็ตแบบนี้)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

clients = set()
frame_queue = queue.Queue(maxsize=2)  # สำหรับ push แบบ live เท่านั้น (ประวัติเก็บที่ดิสก์แยกต่างหาก)

GLOBAL_OPT = None  # ตั้งค่าตอน __main__ แล้ว /upload endpoint จะหยิบไปใช้ตอนเริ่มงานใหม่

job_lock = threading.Lock()
current_job = {
    "thread": None, "stop_event": None, "results_dir": None,
    "total_frames": 0, "latest_frame": 0, "running": False, "video_name": "",
}


def encode_jpeg(img, quality=60):
    if img is None or img.size == 0:
        return b""
    ok, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


def tracking_worker(video_path, opt, stop_event, results_dir):
    """รันในเธรดแยก — เขียนผลทุกเฟรมลงดิสก์ + push เฟรมล่าสุดผ่าน queue สำหรับ live"""
    dataloader = datasets.LoadVideo(video_path, opt.img_size)
    frame_rate = dataloader.frame_rate if dataloader.frame_rate > 0 else 30
    total_frames = getattr(dataloader, 'vn', 0)
    current_job["total_frames"] = total_frames

    logger.info('เริ่มประมวลผล: {} (total_frames~{})'.format(video_path, total_frames))
    if opt.use_yolo:
        logger.info('โหมด: YOLO integration (conf={})'.format(opt.yolo_conf))
    else:
        logger.info('โหมด: Baseline (DLA-only)')

    tracker = MCJDETracker(opt, frame_rate)
    frame_id = 0

    for path, img, img0 in dataloader:
        if stop_event.is_set():
            logger.info('หยุดงานเดิมตามคำสั่ง (มีวิดีโอใหม่เข้ามาแทน)')
            break

        frame_id += 1
        blob = torch.from_numpy(img).unsqueeze(0).to(opt.device)
        online_targets_dict = tracker.update_tracking(blob, img0)

        tlwhs_dict, ids_dict, scores_dict = {}, {}, {}
        objects = []

        for cls_id in range(opt.num_classes):
            tlwhs, ids, scores = [], [], []
            for track in online_targets_dict[cls_id]:
                tlwh = track.curr_tlwh
                if tlwh[2] * tlwh[3] <= opt.min_box_area:
                    continue
                tlwhs.append(tlwh)
                ids.append(track.track_id)
                scores.append(track.score)
                objects.append({
                    "id": int(track.track_id),
                    "class": CLASS_NAMES.get(cls_id, str(cls_id)),
                    "bbox": [round(float(v), 1) for v in tlwh],  # ฝั่งเว็บ crop รูปเองจาก bbox นี้ (ไม่ต้องส่ง thumbnail แยก)
                    "score": round(float(track.score), 2),
                })
            tlwhs_dict[cls_id] = tlwhs
            ids_dict[cls_id] = ids
            scores_dict[cls_id] = scores

        annotated = vis.plot_tracks(
            image=img0, tlwhs_dict=tlwhs_dict, obj_ids_dict=ids_dict,
            num_classes=opt.num_classes, scores=scores_dict,
            frame_id=frame_id, fps=0,
        )
        jpg_bytes = encode_jpeg(annotated, quality=70)

        payload = {"frame": frame_id, "total_frames": total_frames, "objects": objects}

        # เขียนลงดิสก์ (สำหรับย้อนดูทีหลังผ่าน GET /frame/{id})
        with open(os.path.join(results_dir, 'frame_{:05d}.json'.format(frame_id)), 'w') as f:
            json.dump(payload, f)
        with open(os.path.join(results_dir, 'frame_{:05d}.jpg'.format(frame_id)), 'wb') as f:
            f.write(jpg_bytes)

        current_job["latest_frame"] = frame_id

        # push แบบ live ผ่าน websocket (แนบรูปเป็น base64 เฉพาะตอนส่ง live เท่านั้น)
        live_payload = dict(payload)
        live_payload["annotated_frame_b64"] = base64.b64encode(jpg_bytes).decode()
        if frame_queue.full():
            try:
                frame_queue.get_nowait()  # ทิ้งเฟรมเก่าที่ยังไม่ถูกส่ง กัน backlog
            except queue.Empty:
                pass
        frame_queue.put(live_payload)

    current_job["running"] = False
    logger.info('ประมวลผลจบแล้ว ทั้งหมด {} เฟรม'.format(frame_id))


def start_job(video_path, opt):
    """เริ่มงานใหม่ — ถ้ามีงานเก่ารันอยู่ สั่งหยุดก่อนแล้วค่อยเริ่มใหม่"""
    with job_lock:
        old_thread = current_job["thread"]
        old_stop = current_job["stop_event"]
        if old_thread and old_thread.is_alive():
            old_stop.set()
            old_thread.join(timeout=5)

        stop_event = threading.Event()
        run_name = os.path.splitext(os.path.basename(video_path))[0]
        results_dir = os.path.join(RESULTS_DIR, run_name)
        os.makedirs(results_dir, exist_ok=True)

        current_job.update({
            "stop_event": stop_event, "results_dir": results_dir,
            "total_frames": 0, "latest_frame": 0, "running": True,
            "video_name": os.path.basename(video_path),
        })

        t = threading.Thread(target=tracking_worker, args=(video_path, opt, stop_event, results_dir), daemon=True)
        current_job["thread"] = t
        t.start()


async def broadcaster():
    """งานเดียวที่คอยดึงจาก queue แล้วส่งให้ client ทุกคนที่เชื่อมต่ออยู่ (กัน race กันตอนมีหลาย client)"""
    while True:
        await asyncio.sleep(0.03)
        try:
            payload = frame_queue.get_nowait()
        except queue.Empty:
            continue
        dead = []
        for ws in list(clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)


@app.on_event("startup")
async def on_startup():
    asyncio.create_task(broadcaster())


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    logger.info('Dashboard เชื่อมต่อแล้ว (ทั้งหมด {} client)'.format(len(clients)))
    try:
        while True:
            await websocket.receive_text()  # แค่รอจน client ตัดการเชื่อมต่อ ไม่ได้ใช้ข้อความที่ส่งมา
    except WebSocketDisconnect:
        clients.discard(websocket)
        logger.info('Dashboard ตัดการเชื่อมต่อ (เหลือ {} client)'.format(len(clients)))


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if GLOBAL_OPT is None:
        raise HTTPException(500, "เซิร์ฟเวอร์ยังไม่พร้อม (ไม่มี opt ที่ตั้งค่าไว้)")

    dest_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(dest_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    start_job(dest_path, GLOBAL_OPT)
    logger.info('อัปโหลดสำเร็จ เริ่มงานใหม่: {}'.format(file.filename))
    return {"status": "started", "video": file.filename}


@app.get("/status")
async def get_status():
    return {
        "latest_frame": current_job["latest_frame"],
        "total_frames": current_job["total_frames"],
        "running": current_job["running"],
        "video_name": current_job["video_name"],
    }


@app.get("/frame/{frame_id}")
async def get_frame(frame_id: int):
    results_dir = current_job["results_dir"]
    if not results_dir:
        raise HTTPException(404, "ยังไม่มีงานที่ประมวลผลเลย")

    json_path = os.path.join(results_dir, 'frame_{:05d}.json'.format(frame_id))
    jpg_path = os.path.join(results_dir, 'frame_{:05d}.jpg'.format(frame_id))
    if not os.path.exists(json_path) or not os.path.exists(jpg_path):
        raise HTTPException(404, "ยังไม่มีเฟรมนี้ (อาจยังประมวลผลไม่ถึง หรือเกินขอบเขต)")

    with open(json_path) as f:
        data = json.load(f)
    with open(jpg_path, 'rb') as f:
        data["annotated_frame_b64"] = base64.b64encode(f.read()).decode()
    return data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='',
                         help='ถ้าระบุ จะเริ่มประมวลผลทันทีตอน start server — ไม่ระบุก็ได้ แล้วค่อยอัปโหลดจากหน้าเว็บทีหลัง')
    parser.add_argument('--model', required=True)
    parser.add_argument('--conf-thres', type=float, default=0.4)
    parser.add_argument('--use_yolo', action='store_true')
    parser.add_argument('--yolo-model', type=str, default='')
    parser.add_argument('--yolo-conf', type=float, default=0.1)
    parser.add_argument('--port', type=int, default=8000)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    sys.argv = [sys.argv[0]]  # กัน opts().init() ไป parse arguments ของเราเอง
    opt = opts().init()
    opt.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    opt.load_model = args.model
    opt.conf_thres = args.conf_thres
    opt.use_yolo = args.use_yolo
    opt.yolo_model = args.yolo_model
    opt.yolo_conf = args.yolo_conf

    GLOBAL_OPT = opt

    if args.video:
        start_job(args.video, opt)
    else:
        logger.info('ยังไม่มีวิดีโอ — รอ upload จากหน้าเว็บ (dashboard_live.html)')

    logger.info('Server: http://0.0.0.0:{}  (WebSocket: /ws, Upload: /upload)'.format(args.port))
    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning")
