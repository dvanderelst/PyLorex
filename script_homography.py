#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json
import numpy as np
import cv2 as cv

from library import Settings, Lorex, Utils, Homography as hg

# -------- stable OpenCV behaviour --------
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
cv.ocl.setUseOpenCL(False)
cv.setNumThreads(1)

# ---------------- USER SETTINGS ----------------
alpha = 0.0
camera_name = "tiger"
show_preview = False
origin_preset = "TR"
# ------------------------------------------------


paths = Utils.get_calibration_paths(camera_name)
camera = Lorex.LorexCamera(camera_name)
camera.set_alpha(alpha)
frame = camera.get_frame(undistort=False)
camera.stop()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
h_img, w_img = frame.shape[:2]

K_raw, dist = hg.load_intrinsics(camera_name)
W_calib, H_calib = camera.calib_size
sx = w_img / float(W_calib); sy = h_img / float(H_calib)
K_scaled = K_raw.copy()
K_scaled[0,0] *= sx; K_scaled[1,1] *= sy
K_scaled[0,2] *= sx; K_scaled[1,2] *= sy

square_size_mm = Settings.homography_square_mm
inner_cols = Settings.homography_inner_cols
inner_rows = Settings.homography_inner_rows
mm_per_px = Settings.homography_mm_per_px

corners, (cols, rows) = hg.detect_checkerboard(gray, [(inner_cols, inner_rows)], use_sb_first=True)
print(f"[detect] inner corners: {cols} x {rows}")

obj_mm = hg.board_points_mm(cols, rows, square_size_mm)
obj_mm = hg.reframe_obj_mm(obj_mm, cols, rows, square_size_mm, origin_preset)

obj_xyz = np.concatenate([obj_mm.reshape(-1,2), np.zeros((obj_mm.size//2,1), dtype=np.float32)], axis=1).astype(np.float32)
img_pts = corners.reshape(-1,1,2).astype(np.float32)

# Homography for rectification
H_raw, _ = hg.estimate_homography(corners, obj_mm)
width_mm  = (cols - 1) * square_size_mm
height_mm = (rows - 1) * square_size_mm
rectified = hg.rectify_image(frame, H_raw, width_mm, height_mm, mm_per_px=mm_per_px)

# PnP pose
ok, rvec, tvec = cv.solvePnP(obj_xyz, img_pts, K_scaled, dist, flags=cv.SOLVEPNP_EPNP)
R_pnp, _ = cv.Rodrigues(rvec)
cam_center = -R_pnp.T @ tvec
dist_pnp = float(abs(cam_center[2,0]))
proj, _ = cv.projectPoints(obj_xyz, rvec, tvec, K_scaled, dist)
rp_err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1,2) - corners)**2, axis=1))))
print(f"[PnP] distance to board (mm): {dist_pnp:.2f}  (~{dist_pnp/10:.1f} cm)")
print(f"[PnP] reprojection RMSE (px): {rp_err:.3f}")

overlay = hg.draw_board_axes_overlay(
    frame, rvec, tvec, K_scaled, dist,
    cols, rows, square_size_mm,
    label=origin_preset
)

raw_path       = paths['raw_image']
rectified_path = paths['rectified_image']
overlay_path   = paths['axes_overlay']

cv.imwrite(raw_path, frame)
cv.imwrite(rectified_path, rectified)
cv.imwrite(overlay_path, overlay)

# JSON + NPZ
pose_json_path = paths.get('pose_json', os.path.join(paths['result_folder'], f"pose_{camera_name}.json"))
pose_npz_path  = paths.get('pose_npz',  os.path.join(paths['result_folder'], f"pose_{camera_name}.npz"))

summary = {
    "camera_name": camera_name,
    "origin_preset": origin_preset,
    "image_size_WH": [int(w_img), int(h_img)],
    "calib_size_WH": [int(W_calib), int(H_calib)],
    "scale_xy": [float(sx), float(sy)],
    "checker_inner_cols_rows": [int(cols), int(rows)],
    "square_size_mm": float(square_size_mm),
    "mm_per_px_for_rectified": float(mm_per_px),
    "distances_mm": {"pnp": dist_pnp},
    "reprojection_rmse_px": rp_err,
    "K_scaled": np.asarray(K_scaled).tolist(),
    "dist_coeffs": np.asarray(dist).tolist(),
    "H_raw": np.asarray(H_raw).tolist(),
    "R_pnp": np.asarray(R_pnp).tolist(),
    "t_pnp": np.asarray(tvec).tolist(),
    "raw_image": os.path.basename(raw_path),
    "rectified_image": os.path.basename(rectified_path),
    "axes_overlay_image": os.path.basename(overlay_path),
    "report_path": paths["homography_report"],
}

with open(pose_json_path, "w") as f: json.dump(summary, f, indent=2)

np.savez_compressed(
    pose_npz_path,
    K_scaled=K_scaled, dist=dist, H_raw=H_raw,
    R_pnp=R_pnp, t_pnp=tvec,
    obj_mm=obj_mm, obj_xyz=obj_xyz,
    img_pts=img_pts.reshape(-1,2), corners=corners
)

report_path = hg.write_homography_report(
    camera_name=camera_name,
    input_bgr=frame,              # CLEAN RAW frame (no drawings)
    H=H_raw,
    cols=cols, rows=rows,
    square_size_mm=square_size_mm,
    mm_per_px=mm_per_px,
    corners=corners,
    obj_mm=obj_mm,
    rectified_bgr=rectified,
    K=K_scaled,
    R=R_pnp, t=tvec,
    axes_overlay_bgr=overlay,     # overlay shown only once
    origin_preset=origin_preset
)

print("\n=== OUTPUTS ===")
print(f"JSON   : {paths.get('pose_json')}")
print(f"NPZ    : {paths.get('pose_npz')}")
print(f"Report : {report_path}")


