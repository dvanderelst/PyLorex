import os
import cv2 as cv
import multiprocessing as mp

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|max_delay;0|buffer_size;10240|stimeout;3000000")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
cv.ocl.setUseOpenCL(False)  # Disable OpenCL acceleration (stable behavior)
cv.setNumThreads(mp.cpu_count())

tracking_server_ip = '192.168.0.158'
tracking_server_port = 1234

lorex_ip = '192.168.1.19'

username = 'admin'
password = 'Bigb1984'
channels = {'tiger': 2,  'shark': 3}

temp_dir = 'Temp'

aruco_dict_name = 'DICT_4X4_1000'
aruco_size = 81
aruco_forward_axis = 'x'
aruco_yaw_offset_deg = 0

# Physical marker plane height above the floor (caliper-measured 2026-05-22).
# Used to sanity-check that PnP's reported height_mm matches reality.
marker_height_mm = 150.0
marker_height_tolerance_mm = 20.0

heading_draw_length = 5
axis_draw_length_mm = 10
# ArUco speed knobs
# To re-tune after hardware changes (different marker size, camera
# distance, sensor), run the two probe scripts in PyLorex/:
#   - script_probe_rtsp_rate.py        (measures DVR/RTSP frame arrival)
#   - script_probe_detection_rate.py   (measures detectMarkers cost at
#                                       multiple scales using this code path)
# 2026-05-10 measurements: at scale=1.0 detection on the 4K main stream
# took ~450 ms/frame (≈2 Hz) and bounded the whole pipeline. At scale=0.5
# detection drops to ~150 ms/frame (≈5 Hz, near the 17 fps grabber
# ceiling); per-frame yaw noise actually *decreased* (0.6° → 0.12°) —
# downsample lands the marker closer to refine_win=3's sweet spot.
aruco_detect_scale = 0.5
aruco_fast_refine = True      # keep subpix but lighter
aruco_refine_win = 3          # subpix window (3 good at ~46px markers)
aruco_refine_iters = 15       # fewer iterations than 30


calibration_dot_diameter_mm = 40.0
calibration_dot_rows = 5
calibration_dot_cols = 10
calibration_dot_spacing = calibration_dot_diameter_mm + 10.0

# Defines how finely the world plane is sampled in the rectified image.
# The homography already maps pixels → millimetres; this parameter just decides
# how many pixels to allocate per millimetre when generating the top-down view.
# Setting 1.0 makes the rectified image 1 px = 1 mm, i.e., a 1000 mm square becomes 1000 × 1000 px.
homography_mm_per_px = 1.0

# Camera-system geometry (translates shark's per-camera board frame into the
# unified tiger frame). Values are read at import time from a JSON config
# written by script_set_camera_center.py (which derives them from the clicked
# floor-mark positions + the user's tape-measured physical inter-camera
# distance). Hardcoded values below are the fallback if the config file is
# absent.
import json as _json
from pathlib import Path as _Path

shark2tiger_delta_x = 0
shark2tiger_delta_y = -1400  # fallback; overridden by camera_system.json

_camera_system_json = (
    _Path(__file__).resolve().parent.parent / "Calibration" / "Results" / "camera_system.json"
)
if _camera_system_json.exists():
    try:
        with open(_camera_system_json) as _f:
            _cs = _json.load(_f)
        shark2tiger_delta_x = float(_cs.get("shark2tiger_delta_x_mm", shark2tiger_delta_x))
        shark2tiger_delta_y = float(_cs.get("shark2tiger_delta_y_mm", shark2tiger_delta_y))
    except Exception as _e:
        print(f"[Settings] Could not parse {_camera_system_json}: {_e}; "
              f"using hardcoded fallback shark2tiger_delta_xy.")

# Environment capture (arena layout snapshots)
environment_root = 'Environment'
environment_arena_min_x_mm = -2500.0
environment_arena_max_x_mm = 3000.0
environment_arena_min_y_mm = -3500.0
environment_arena_max_y_mm = 2000.0
environment_map_mm_per_px = 5.0
environment_frame_samples = 3
