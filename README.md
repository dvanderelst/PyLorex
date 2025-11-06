# PyLorex

PyLorex is a Python library designed to interface with Lorex security cameras, allowing users to access camera feeds, control camera settings, and manage recordings programmatically.

The library allows for **intrinsic camera calibration** and **homography estimation**, enabling users to correct lens distortion and map camera views to real-world coordinates: With intrinsics and a homography, every pixel can be mapped to an (X, Y) position on that plane. But if the object is not on that plane, the mapping doesn’t hold anymore. Also, this only holds as long **as the camera is not moved**.

# Simple TCP telemetry server

The ``PyLorex.server.simple_tcp`` module offers a lightweight way to share the
latest ArUco detections with another machine. Run the server on the computer
that is directly connected to the cameras and poll it from the machine that
coordinates your robots.

For day-to-day usage edit ``script_start_server.py`` in the repository root and
adjust the configuration constants at the top of the file (camera list, host,
port, poll interval, etc.). By default it monitors the ``tiger`` and ``shark``
cameras::

    python script_start_server.py

When you need an ad-hoc configuration, you can launch the module directly and
repeat the ``--camera`` flag for each feed you want to track. Duplicate camera
names are ignored (the server logs a warning and keeps one worker per name)::

    python -m PyLorex.server.simple_tcp --camera tiger --camera panther

The repository also includes a compatibility shim that forwards to the module
entry point while still accepting all CLI flags::

    python run_server.py --camera tiger --camera panther

All entry points start a threaded TCP listener on ``0.0.0.0:9999`` and spawn
one worker per camera that continuously calls
:func:`library.Lorex.LorexCamera.get_aruco`. The server maintains the most
recent detections for each camera.
Clients connect via ``telnet``/``nc``/custom code and issue newline-terminated
commands:

* ``PING`` – sanity check, returns ``{"status": "ok"}``
* ``CAMERAS`` – list the camera names currently publishing telemetry
* ``GET <camera>`` – fetch the full snapshot (detections, timestamp, frame size)
* ``GET <camera> <marker_id>`` – retrieve a single marker dictionary

Each response is returned as one JSON object followed by a newline, making it
easy to consume from Python, Rust, or any language with basic socket support.

# Scripts

+ `script_collect_calibration_images`: Captures a sequence of calibration images from a configured camera and saves them to the project's calibration image folder.
+ `script_run_calibration`: Calibrates a camera from a set of chessboard images and saves the camera intrinsics and an undistorted preview image.
+ `script_assess_calibration`: Assesses the quality of a camera calibration. Generates an HTML report.
+ `script_homohraphy`: Uses an image to estimate a homography from image pixels to a known planar target. Saves the homography and a visualization.
+ `script_test_lorex`: Tests getting images from a Lorex camera and undistorting them using previously saved camera intrinsics.
+ `script_start_server`: Launches the TCP telemetry server with the lab's usual configuration (edit the script to change cameras or networking settings).

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
