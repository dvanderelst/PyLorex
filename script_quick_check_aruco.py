# quick_check_aruco5x5.py
import cv2 as cv
import numpy as np
img = cv.imread("Image_screenshot_28.10.2025.png")  # full frame, not a tiny crop
aru = cv.aruco
params = cv.aruco.DetectorParameters()
params.cornerRefinementMethod = aru.CORNER_REFINE_SUBPIX
params.minMarkerPerimeterRate = 0.005
params.adaptiveThreshWinSizeMin = 5
params.adaptiveThreshWinSizeMax = 55
params.adaptiveThreshWinSizeStep = 5

for name in ("DICT_5X5_100", "DICT_5X5_250"):
    if not hasattr(aru, name):
        continue
    d = aru.getPredefinedDictionary(getattr(aru, name))
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    if min(gray.shape) < 700:
        gray = cv.resize(gray, None, fx=1.6, fy=1.6, interpolation=cv.INTER_CUBIC)
    if hasattr(aru, "ArucoDetector"):
        det = aru.ArucoDetector(d, params)
        corners, ids, rej = det.detectMarkers(gray)
    else:
        corners, ids, rej = aru.detectMarkers(gray, d, parameters=params)
    print(name, " -> ids:", None if ids is None else ids.ravel().tolist())
