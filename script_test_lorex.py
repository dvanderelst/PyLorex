from library import Lorex
from matplotlib import pyplot as plt
import os, cv2 as cv
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_VIDEOIO_PRIORITY_GSTREAMER", "0")  # prefer FFmpeg on Linux
cv.ocl.setUseOpenCL(False)
cv.setNumThreads(1)


cam = Lorex.LorexCamera("tiger")
cam.set_alpha(0.0)  # tightly cropped undistorted frame
f = cam.get_frame(undistort=True)
plt.imshow(f)
plt.show()

f = cam.get_frame(undistort=False)
plt.imshow(f)
plt.show()

cam.stop()