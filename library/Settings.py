import os

key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
value = "rtsp_transport;tcp|max_delay;0|buffer_size;10240|stimeout;3000000"
os.environ.setdefault(key, value)

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