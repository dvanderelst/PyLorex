#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from library import Settings, Lorex, Utils, Homography as hg

# -------- stable OpenCV behaviour --------
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
cv.ocl.setUseOpenCL(False)
cv.setNumThreads(1)

# ---------------- USER SETTINGS ----------------
camera_name = "tiger"
alpha = 0.0       # used by Lorex camera if/when you undistort, not needed here
show_preview = False  # set True to visualize the grabbed frame
# ------------------------------------------------

paths = Utils.get_calibration_paths(camera_name)
camera = Lorex.LorexCamera(camera_name)
camera.set_alpha(alpha)
frame = camera.get_frame(undistort=False)
if frame is None:
    camera.stop()
    raise RuntimeError("Failed to grab a frame.")
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
h_img, w_img = frame.shape[:2]

if show_preview:
    try:
        plt.imshow(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
        plt.title(f"RAW frame – {camera_name}")
        plt.show()
    except Exception:
        pass

# ---- intrinsics ----
K_raw, dist = hg.load_intrinsics(camera_name)
W_calib, H_calib = camera.calib_size

# scale K if stream size differs from calibration size
sx = w_img / float(W_calib)
sy = h_img / float(H_calib)
K_scaled = K_raw.copy()
K_scaled[0,0] *= sx
K_scaled[1,1] *= sy
K_scaled[0,2] *= sx
K_scaled[1,2] *= sy

print(f"[calib] {camera_name}: calib WH={(W_calib,H_calib)}  stream WH={(w_img,h_img)}  scale={(sx,sy)}")

# ---- checkerboard settings ----
# For a plain checkerboard, this is the **square side** in mm (i.e., inner-corner pitch)
square_size_mm = Settings.homography_square_mm
inner_cols = Settings.homography_inner_cols
inner_rows = Settings.homography_inner_rows
mm_per_px = Settings.homography_mm_per_px   # density used for rectified view only

# ---- detect inner corners (RAW pixels) ----
checker_shape = [(inner_cols, inner_rows)]
corners, (cols, rows) = hg.detect_checkerboard(gray, checker_shape, use_sb_first=True)
print(f"[detect] inner corners: {cols} x {rows}")
# ---- build model points (2D for H, 3D Z=0 for PnP) ----
obj_mm = hg.board_points_mm(cols, rows, square_size_mm)           # (N,2) in mm
obj_xyz = np.concatenate([obj_mm.reshape(-1,2), np.zeros((obj_mm.size//2,1), dtype=np.float32)],axis=1)
img_pts = corners.reshape(-1,1,2)

obj_xyz = obj_xyz.astype(np.float32)
img_pts = img_pts.astype(np.float32)
# ==============================================================
# Homography on RAW pixels (for rectification/overlays)
# ==============================================================
H_raw, _ = hg.estimate_homography(corners, obj_mm)
# Rectified visualization sized to inner-corner lattice (add margins if you want more)
width_mm  = (cols - 1) * square_size_mm
height_mm = (rows - 1) * square_size_mm
rectified = hg.rectify_image(frame, H_raw, width_mm, height_mm, mm_per_px=mm_per_px)

# ==============================================================
# PnP pose (uses intrinsics + distortion) — authoritative metric pose
# ==============================================================
ok, rvec, tvec = cv.solvePnP(obj_xyz, img_pts, K_scaled, dist, flags=cv.SOLVEPNP_EPNP)
if not ok:
    ok, rvec, tvec = cv.solvePnP(obj_xyz, img_pts, K_scaled, dist, flags=cv.SOLVEPNP_ITERATIVE)
    if not ok:
        camera.stop()
        raise RuntimeError("solvePnP failed.")
R_pnp, _ = cv.Rodrigues(rvec)
cam_center = -R_pnp.T @ tvec
dist_pnp = float(abs(cam_center[2,0]))

# reprojection error (px) — good quick sanity metric
proj, _ = cv.projectPoints(obj_xyz, rvec, tvec, K_scaled, dist)
rp_err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1,2) - corners)**2, axis=1))))

print(f"[PnP] distance to board (mm): {dist_pnp:.2f}  (~{dist_pnp/10:.1f} cm)")
print(f"[PnP] reprojection RMSE (px): {rp_err:.3f}")

# =========================
# SAVE + REPORT
# =========================
# images
cv.imwrite(paths['raw_image'], frame)
cv.imwrite(paths['rectified_image'], rectified)

# JSON summary
def nplist(a): return np.asarray(a).tolist()
summary = {
    "camera_name": camera_name,
    "image_size_WH": [int(w_img), int(h_img)],
    "calib_size_WH": [int(W_calib), int(H_calib)],
    "scale_xy": [float(sx), float(sy)],
    "checker_inner_cols_rows": [int(cols), int(rows)],
    "square_size_mm": float(square_size_mm),
    "mm_per_px_for_rectified": float(mm_per_px),
    "distances_mm": {
        "pnp": dist_pnp
    },
    "reprojection_rmse_px": rp_err,
    "K_scaled": nplist(K_scaled),
    "dist_coeffs": nplist(dist),
    "H_raw": nplist(H_raw),
    "R_pnp": nplist(R_pnp), "t_pnp": nplist(tvec),
    "raw_image": os.path.basename(paths['raw_image']),
    "rectified_image": os.path.basename(paths['rectified_image']),
    "report_path": paths['homography_report']
}
with open(paths['pose_json'], "w") as f:
    json.dump(summary, f, indent=2)

# NPZ dump for reproducibility/debugging
np.savez_compressed(
    paths['pose_npz'],
    K_scaled=K_scaled, dist=dist,
    H_raw=H_raw,
    R_pnp=R_pnp, t_pnp=tvec,
    obj_mm=obj_mm, obj_xyz=obj_xyz,
    img_pts=img_pts.reshape(-1,2),
    corners=corners
)

# HTML report: use PnP pose (truth) and H for rectification
report_path = hg.write_homography_report(
    camera_name=camera_name,
    input_bgr=frame,               # RAW frame for overlays
    H=H_raw,                       # for rectification/visualization
    cols=cols, rows=rows,          # detected counts
    square_size_mm=square_size_mm,
    mm_per_px=mm_per_px,
    corners=corners,
    obj_mm=obj_mm,
    rectified_bgr=rectified,
    K=K_scaled,                    # intrinsics consistent with this frame
    R=R_pnp, t=tvec                # authoritative metric pose
)

print("\n=== OUTPUTS ===")
print(f"JSON   : {paths['pose_json']}")
print(f"NPZ    : {paths['pose_npz']}")
print(f"Images : {os.path.basename(paths['raw_image'])}, {os.path.basename(paths['rectified_image'])}  in  {paths['result_folder']}")
print(f"Report : {report_path}")

camera.stop()

