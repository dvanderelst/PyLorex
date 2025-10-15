# PyLorex

PyLorex is a Python library designed to interface with Lorex security cameras, allowing users to access camera feeds, control camera settings, and manage recordings programmatically.

The library allows for **intrinsic camera calibration** and **homography estimation**, enabling users to correct lens distortion and map camera views to real-world coordinates: With intrinsics and a homography, every pixel can be mapped to an (X, Y) position on that plane. But if the object is not on that plane, the mapping doesn’t hold anymore. Also, this only holds as long **as the camera is not moved**.

# Scripts

+ `script_collect_calibration_images`: Captures a sequence of calibration images from a configured camera and saves them to the project's calibration image folder.
+ `script_run_calibration`: Calibrates a camera from a set of chessboard images and saves the camera intrinsics and an undistorted preview image.
+ `script_assess_calibration`: Assesses the quality of a camera calibration. Generates an HTML report.
+ `script_homohraphy`: Uses an image to estimate a homography from image pixels to a known planar target. Saves the homography and a visualization.
+ `script_test_lorex`: Tests getting images from a Lorex camera and undistorting them using previously saved camera intrinsics.

# Example

```python
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

# 3) Ray-plane path using the *same* raw pixel and the *same* K/dist from bundle
d_cam = hg.pixel_to_ray_cam(u, v, Kb, dist)
P     = hg.intersect_ray_with_board(d_cam, cam.bundle["R"], cam.bundle["t"])
xRT, yRT = float(P[0]), float(P[1])

print(f"RAW pixel ({u},{v})  H:({xH:.2f},{yH:.2f})  RT:({xRT:.2f},{yRT:.2f})  Δ=({xRT-xH:.2f},{yRT-yH:.2f}) mm")
cam.stop()
```