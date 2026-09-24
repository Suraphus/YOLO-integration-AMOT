import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

# Add lib to PYTHONPATH
lib_path = osp.join(this_dir, 'lib')
add_path(lib_path)

# Add DCNv2 (setup.py develop บางเวอร์ชันไม่ลงทะเบียน path ให้ ทำให้ import dcn_v2 ไม่เจอ)
dcnv2_path = osp.abspath(osp.join(this_dir, '..', 'DCNv2'))
add_path(dcnv2_path)
