import os
import cv2 as cv

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|max_delay;0|buffer_size;10240|stimeout;3000000")
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")
cv.ocl.setUseOpenCL(False)  # Disable OpenCL acceleration (stable behavior)
cv.setNumThreads(1)         # Single-threaded (reproducible results)


lorex_ip = '192.168.1.17'
username = 'admin'
password = 'Bigb1984'
channels = {'tiger': 2,  'shark': 3}

temp_dir = 'Temp'

aruco_dict = 'DICT_4X4_100'
aruco_size = 60
aruco_forward_axis = 'x'
aruco_yaw_offset_deg = 0

calibration_dot_diameter_mm = 40.0
calibration_dot_rows = 5
calibration_dot_cols = 10
calibration_dot_spacing = calibration_dot_diameter_mm + 10.0
homography_mm_per_px = 1.0

