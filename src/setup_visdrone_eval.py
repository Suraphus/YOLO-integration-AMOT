"""
setup_visdrone_eval.py

เตรียม VisDrone2019-MOT-test-dev (ที่โหลดจาก GitHub ทางการ) ให้ใช้กับ track_AMOT.py ได้

สิ่งที่สคริปต์ทำ:
  1. สร้าง <data_dir>/VisDrone2019/test_dev/sequences เป็น symlink ไปยัง <src>/sequences
     (ไม่ copy ภาพ ประหยัดพื้นที่)
  2. กรอง GT จาก <src>/annotations ให้เหลือเฉพาะ 5 คลาสที่ track_AMOT.py เขียนผลออกมา
     (pedestrian, car, van, truck, bus) และตัด ignored region / กล่องที่ flag=0 ทิ้ง
     แล้วเขียนไปที่ <data_dir>/VisDrone2019/test_dev/annotations_eval/<seq>.txt
     ซึ่งเป็น path ที่ Evaluator (lib/tracking_utils/evaluation.py) อ่าน
  3. ตรวจว่าครบทั้ง 17 sequence ที่ track_AMOT.py ใช้ และจำนวนเฟรมภาพตรงกับ GT

วิธีใช้ (บน PC):
  python setup_visdrone_eval.py \
      --src ~/Downloads/VisDrone2019-MOT-test-dev \
      --data_dir ~/UAVdata

  จากนั้นรัน track_AMOT.py ด้วย --data_dir เดียวกัน
"""

import argparse
import os
import os.path as osp
import sys

# 17 sequence เดียวกับที่ hardcode ไว้ใน track_AMOT.py
SEQS = [
    'uav0000009_03358_v', 'uav0000073_00600_v', 'uav0000073_04464_v',
    'uav0000077_00720_v', 'uav0000088_00290_v', 'uav0000119_02301_v',
    'uav0000120_04775_v', 'uav0000161_00000_v', 'uav0000188_00000_v',
    'uav0000201_00000_v', 'uav0000249_00001_v', 'uav0000249_02688_v',
    'uav0000297_00000_v', 'uav0000297_02761_v', 'uav0000306_00230_v',
    'uav0000355_00001_v', 'uav0000370_00001_v',
]

# category ใน annotation ทางการของ VisDrone:
#   0 ignored, 1 pedestrian, 2 people, 3 bicycle, 4 car, 5 van, 6 truck,
#   7 tricycle, 8 awning-tricycle, 9 bus, 10 motor, 11 others
# track_AMOT.py (write_results_dict) เขียนเฉพาะ cls_id 0,3,4,5,8 ของ AMOT
# = pedestrian, car, van, truck, bus = category 1,4,5,6,9 ของ VisDrone
EVAL_CATEGORIES = {1, 4, 5, 6, 9}


def filter_gt(src_txt, dst_txt):
    kept, dropped = 0, 0
    frames = set()
    with open(src_txt, 'r') as f_r, open(dst_txt, 'w') as f_w:
        for line in f_r:
            cols = line.strip().split(',')
            if len(cols) < 8:
                continue
            flag = int(cols[6])      # 1 = นับในการวัดผล, 0 = ignore
            category = int(cols[7])
            if flag == 0 or category not in EVAL_CATEGORIES:
                dropped += 1
                continue
            frames.add(int(cols[0]))
            f_w.write(line if line.endswith('\n') else line + '\n')
            kept += 1
    return kept, dropped, frames


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--src', required=True,
                        help='โฟลเดอร์ VisDrone2019-MOT-test-dev ที่แตกไฟล์แล้ว (มี sequences/ และ annotations/)')
    parser.add_argument('--data_dir', required=True,
                        help='ค่าเดียวกับที่จะส่งให้ track_AMOT.py --data_dir')
    args = parser.parse_args()

    src = osp.abspath(osp.expanduser(args.src))
    data_dir = osp.abspath(osp.expanduser(args.data_dir))
    src_seq_dir = osp.join(src, 'sequences')
    src_ann_dir = osp.join(src, 'annotations')

    for d in (src_seq_dir, src_ann_dir):
        if not osp.isdir(d):
            sys.exit('[Error] ไม่พบ {} — ตรวจว่า --src ชี้ไปที่โฟลเดอร์ที่มี sequences/ และ annotations/'.format(d))

    test_dev = osp.join(data_dir, 'VisDrone2019', 'test_dev')
    dst_seq_dir = osp.join(test_dev, 'sequences')
    dst_ann_dir = osp.join(test_dev, 'annotations_eval')
    os.makedirs(test_dev, exist_ok=True)
    os.makedirs(dst_ann_dir, exist_ok=True)

    # ----- 1. symlink ภาพ -----
    if osp.islink(dst_seq_dir):
        if osp.realpath(dst_seq_dir) != osp.realpath(src_seq_dir):
            os.remove(dst_seq_dir)
            os.symlink(src_seq_dir, dst_seq_dir)
    elif osp.exists(dst_seq_dir):
        print('[Info] {} มีอยู่แล้ว (ไม่ใช่ symlink) — ใช้ของเดิม'.format(dst_seq_dir))
    else:
        os.symlink(src_seq_dir, dst_seq_dir)
    print('sequences -> {}'.format(osp.realpath(dst_seq_dir)))

    # ----- 2 & 3. กรอง GT และตรวจความครบถ้วน -----
    n_ok = 0
    print('\n{:<22} {:>7} {:>9} {:>9}  {}'.format('sequence', 'images', 'gt_kept', 'gt_drop', 'status'))
    for seq in SEQS:
        img_dir = osp.join(dst_seq_dir, seq)
        src_txt = osp.join(src_ann_dir, seq + '.txt')

        n_img = 0
        if osp.isdir(img_dir):
            n_img = len([f for f in os.listdir(img_dir)
                         if osp.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')])

        if n_img == 0 or not osp.isfile(src_txt):
            status = 'ไม่มีภาพ' if n_img == 0 else 'ไม่มี GT'
            print('{:<22} {:>7} {:>9} {:>9}  [Error] {}'.format(seq, n_img, '-', '-', status))
            continue

        kept, dropped, frames = filter_gt(src_txt, osp.join(dst_ann_dir, seq + '.txt'))
        status = 'OK'
        if frames and max(frames) > n_img:
            status = '[Warning] GT มีเฟรมถึง {} แต่มีภาพแค่ {}'.format(max(frames), n_img)
        else:
            n_ok += 1
        print('{:<22} {:>7} {:>9} {:>9}  {}'.format(seq, n_img, kept, dropped, status))

    print('\nพร้อมใช้ {}/{} sequence'.format(n_ok, len(SEQS)))
    print('GT ที่กรองแล้วอยู่ที่: {}'.format(dst_ann_dir))
    if n_ok == len(SEQS):
        print('\nรันต่อได้เลย:')
        print('  python track_AMOT.py --load_model /path/to/visdrone.pth --data_dir {}'.format(data_dir))
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
