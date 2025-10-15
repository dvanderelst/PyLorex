from library import Lorex
from matplotlib import pyplot as plt
from library import Homography as hg
import os, cv2 as cv
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")  # prefer FFmpeg on Linux
cv.ocl.setUseOpenCL(False)
cv.setNumThreads(1)

cam = Lorex.LorexCamera("tiger")
# 1) Load the saved homography/pose bundle (from your earlier script)
cam.load_board_bundle()

Kb   = cam.bundle["K"]
dist = cam.bundle.get("dist", None)

# 1) choose ONE raw pixel
frame_raw = cam.get_frame(undistort=False)
h, w = frame_raw.shape[:2]
u, v = w//2, h//2

# 2) H-path (raw -> board)
xH, yH = cam.pixel_to_board_xy(u, v, use_raw=True)

# # 3) Ray-plane path using the *same* raw pixel and the *same* K/dist from bundle
# d_cam = hg.pixel_to_ray_cam(u, v, Kb, dist)
# P     = hg.intersect_ray_with_board(d_cam, cam.bundle["R"], cam.bundle["t"])
# xRT, yRT = float(P[0]), float(P[1])
#
# print(f"RAW pixel ({u},{v})  H:({xH:.2f},{yH:.2f})  RT:({xRT:.2f},{yRT:.2f})  Δ=({xRT-xH:.2f},{yRT-yH:.2f}) mm")
# cam.stop()