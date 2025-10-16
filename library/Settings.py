import os
import cv2 as cv

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|max_delay;0|buffer_size;10240|stimeout;3000000")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
cv.ocl.setUseOpenCL(False)  # Disable OpenCL acceleration (stable behavior)
cv.setNumThreads(1)         # Single-threaded (reproducible results)

temp_dir = 'Temp'

lorex_ip = '192.168.1.8'
username = 'admin'
password = 'Bigb1984'
channels = {'tiger': 2,  'shark': 3}
intrinsic_square_mm = 30.0  # size of printed chessboard squares for intrinsics calibration
intrinsic_inner_cols = 11
intrinsic_inner_rows = 7

homography_square_mm = 30.0  # size of printed chessboard squares for homography
homography_inner_cols = 11
homography_inner_rows = 7
homography_mm_per_px = 1.0

aruco_size = 60.0  # size of printed aruco markers in mm
aruco_dict = "DICT_5X5_100"
aruco_yaw_offset_deg = 0.0  # yaw offset to align aruco yaw with robot yaw
aruco_forward_axis = 'x'
