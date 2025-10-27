import os
import cv2 as cv

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|max_delay;0|buffer_size;10240|stimeout;3000000")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
cv.ocl.setUseOpenCL(False)  # Disable OpenCL acceleration (stable behavior)
cv.setNumThreads(1)         # Single-threaded (reproducible results)

temp_dir = 'Temp'

lorex_ip = '192.168.1.14'
username = 'admin'
password = 'Bigb1984'
channels = {'tiger': 2,  'shark': 3}
# intrinsic_square_mm = 30.0  # size of printed chessboard squares for intrinsics calibration
# intrinsic_inner_cols = 11
# intrinsic_inner_rows = 7

intrinsic_dot_diameter_mm = 40.0
intrinsic_dot_rows = 5
intrinsic_dot_cols = 10
intrinsic_dot_spacing = intrinsic_dot_diameter_mm + 10.0

homography_square_mm = 30.0  # size of printed chessboard squares for homography
homography_inner_cols = 11
homography_inner_rows = 7
homography_mm_per_px = 1.0

