import os
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from library import Settings, Lorex, Utils, Homography as hg

# --- stable OpenCV behaviour ---
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
cv.ocl.setUseOpenCL(False)
cv.setNumThreads(1)

# ---------------- USER SETTINGS ----------------
camera_name = "tiger"
alpha = 0.0
# ------------------------------------------------

# --- get a raw frame (no undistortion) ---
camera = Lorex.LorexCamera(camera_name)
camera.set_alpha(alpha)
frame = camera.get_frame(undistort=False)
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
plt.title("RAW frame")
plt.show()

# --- calibration info ---
K_raw, dist = hg.load_intrinsics(camera_name)
W_calib, H_calib = camera.calib_size
h_img, w_img = frame.shape[:2]

# scale K if stream size differs from calibration size
sx = w_img / float(W_calib)
sy = h_img / float(H_calib)
K_scaled = K_raw.copy()
K_scaled[0,0] *= sx
K_scaled[1,1] *= sy
K_scaled[0,2] *= sx
K_scaled[1,2] *= sy
print(f"calib size: {(W_calib,H_calib)}   stream: {(w_img,h_img)}   scale: {(sx,sy)}")

# --- board settings ---
square_size_mm = Settings.homography_square_mm   # side length of one checker square
inner_cols = Settings.homography_inner_cols
inner_rows = Settings.homography_inner_rows

# --- detect corners ---
corners, (cols, rows) = hg.detect_checkerboard(
    gray, [(inner_cols, inner_rows)], use_sb_first=True
)
print(f"Detected inner corners: {cols} x {rows}")

# --- 2D and 3D point sets ---
obj_mm = hg.board_points_mm(cols, rows, square_size_mm)
obj_xyz = np.concatenate(
    [obj_mm.reshape(-1,2),
     np.zeros((obj_mm.size//2,1), dtype=np.float32)],
    axis=1
).astype(np.float32)
img_pts = corners.reshape(-1,1,2).astype(np.float32)

# ==============================================================
# 1) Homography on raw pixels
# ==============================================================
H_raw, _ = hg.estimate_homography(corners, obj_mm)
R_raw, T_raw = hg.pose_from_homography(K_scaled, H_raw)
cam_center_raw = -R_raw.T @ T_raw
print("\n[1] Homography (raw pixels)")
print("distance (mm):", float(abs(cam_center_raw[2,0])))

# ==============================================================
# 2) Homography on undistorted points (removes distortion)
# ==============================================================
undist = cv.undistortPoints(img_pts, K_scaled, dist, P=np.eye(3, dtype=np.float32)).reshape(-1,2)
H_undist, _ = hg.estimate_homography(undist, obj_mm)
R_u, T_u = hg.pose_from_homography(np.eye(3, dtype=np.float32), H_undist)
cam_center_u = -R_u.T @ T_u
print("\n[2] Homography (undistorted points)")
print("distance (mm):", float(abs(cam_center_u[2,0])))

# ==============================================================
# 3) solvePnP ground truth (handles distortion directly)
# ==============================================================
ok, rvec, tvec = cv.solvePnP(obj_xyz, img_pts, K_scaled, dist, flags=cv.SOLVEPNP_EPNP)
if not ok:
    ok, rvec, tvec = cv.solvePnP(obj_xyz, img_pts, K_scaled, dist, flags=cv.SOLVEPNP_ITERATIVE)
R_pnp, _ = cv.Rodrigues(rvec)
cam_center_pnp = -R_pnp.T @ tvec
print("\n[3] solvePnP")
print("distance (mm):", float(abs(cam_center_pnp[2,0])))

camera.stop()
