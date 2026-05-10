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

aruco_dict_name = 'DICT_4X4_100'
aruco_size = 60
aruco_forward_axis = 'x'
aruco_yaw_offset_deg = 0

heading_draw_length = 5
axis_draw_length_mm = 10
# ArUco speed knobs
# detect at 50% size, then scale corners back. With the 4K main stream
# (3840×2160), running detect at full size took ~450 ms/frame and bounded
# the whole pipeline to ~2 Hz — see `script_probe_detection_rate.py`
# measurements 2026-05-10. At scale=0.5 detection drops to ~150 ms/frame
# (~5 Hz, near the 17 fps grabber ceiling), per-frame yaw noise rises
# from ~0.6° to ~1° (still well under 3PiRobot's 2° settle tolerance).
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

# This converts the coordinates from the shark camera to the tiger camera frame.
# This assumes that the coordinates frames are only translated.
shark2tiger_delta_x = 0
shark2tiger_delta_y = -1840

# Environment capture (arena layout snapshots)
environment_root = 'Environment'
environment_arena_min_x_mm = -2500.0
environment_arena_max_x_mm = 3000.0
environment_arena_min_y_mm = -3000.0
environment_arena_max_y_mm = 1500.0
environment_map_mm_per_px = 5.0
environment_frame_samples = 3
