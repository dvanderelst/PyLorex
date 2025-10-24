from library import Lorex
from library import Settings
from matplotlib import pyplot as plt
import os, cv2 as cv
import time

cam = Lorex.LorexCamera("tiger")
# 1) Load the saved homography/pose bundle (from your earlier script)
cam.load_board_bundle()

Kb   = cam.bundle["K"]
dist = cam.bundle.get("dist", None)

# 1) choose ONE raw pixel
frame_raw = cam.get_frame(undistort=False)
h, w = frame_raw.shape[:2]
u, v = w//2, h//2

plt.figure()
plt.imshow(frame_raw)
plt.show()
print('frame shape:', frame_raw.shape)

# 2) H-path (raw -> board)
xH, yH = cam.pixel_to_board_xy(u, v, use_raw=True)

counter = 0
while True:
    start = time.time()
    detections = cam.get_aruco(draw=True)
    end = time.time()
    print(f"Detection time: {(end-start)*1000:.1f} ms")
    counter += 1
    print(counter)
    time.sleep(1)

# # 3) Ray-plane path using the *same* raw pixel and the *same* K/dist from bundle
# d_cam = hg.pixel_to_ray_cam(u, v, Kb, dist)
# P     = hg.intersect_ray_with_board(d_cam, cam.bundle["R"], cam.bundle["t"])
# xRT, yRT = float(P[0]), float(P[1])
#
# print(f"RAW pixel ({u},{v})  H:({xH:.2f},{yH:.2f})  RT:({xRT:.2f},{yRT:.2f})  Δ=({xRT-xH:.2f},{yRT-yH:.2f}) mm")
cam.stop()