import os, json
import numpy as np
import cv2 as cv
from library import Settings
from library import Lorex
from library import Utils
from library import Homography as hg

# ---------------- SETTINGS ----------------
alpha = 0.0
camera_name = "tiger"
origin_preset = "TR"   # board origin (bottom-left, top-left, etc.)
# ------------------------------------------------

paths = Utils.get_calibration_paths(camera_name)

# --- Capture current image from camera ---
camera = Lorex.LorexCamera(camera_name)
camera.set_alpha(alpha)
frame = camera.get_frame(undistort=False)
camera.stop()
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
h_img, w_img = frame.shape[:2]

# --- Load intrinsics and scale to full-res image size ---
K_raw, dist = hg.load_intrinsics(camera_name)
W_calib, H_calib = camera.calib_size
sx = w_img / float(W_calib)
sy = h_img / float(H_calib)
K_scaled = K_raw.copy()
K_scaled[0, 0] *= sx;  K_scaled[1, 1] *= sy
K_scaled[0, 2] *= sx;  K_scaled[1, 2] *= sy

# --- Dot pattern parameters ---
dot_rows = Settings.calibration_dot_rows        # e.g., 5
dot_cols = Settings.calibration_dot_cols        # e.g., 10
dot_spacing = Settings.calibration_dot_spacing  # center-to-center spacing (mm)
mm_per_px = Settings.homography_mm_per_px     # output rectification scale

# --- Detect circle grid ---
print("[detect] Looking for circle grid...")
params = cv.SimpleBlobDetector_Params()
params.filterByColor = True
params.blobColor = 0  # black dots
params.filterByArea = True
params.minArea = 300
params.maxArea = 20000
params.filterByCircularity = True
params.minCircularity = 0.7
params.filterByConvexity = True
params.minConvexity = 0.8
detector = cv.SimpleBlobDetector_create(params)

flags = cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING
found, corners = cv.findCirclesGrid(gray, (dot_cols, dot_rows),
                                    flags=flags, blobDetector=detector)

if not found:
    raise SystemExit("[error] No circle grid detected. Adjust lighting or blob parameters.")

print(f"[detect] Found grid of {dot_cols} x {dot_rows} dots.")

# --- Object coordinates in mm ---
obj_mm = np.zeros((dot_rows * dot_cols, 2), np.float32)
obj_mm[:, :] = np.mgrid[0:dot_cols, 0:dot_rows].T.reshape(-1, 2) * dot_spacing
obj_mm = hg.reframe_obj_mm(obj_mm, dot_cols, dot_rows, dot_spacing, origin_preset)

obj_xyz = np.concatenate(
    [obj_mm, np.zeros((obj_mm.shape[0], 1), dtype=np.float32)],
    axis=1
).astype(np.float32)
img_pts = corners.reshape(-1, 1, 2).astype(np.float32)

# --- Homography for rectification ---
H_raw, _ = hg.estimate_homography(corners, obj_mm)
width_mm = (dot_cols - 1) * dot_spacing
height_mm = (dot_rows - 1) * dot_spacing
rectified = hg.rectify_image(frame, H_raw, width_mm, height_mm, mm_per_px=mm_per_px)

# --- Pose estimation (PnP) ---
ok, rvec, tvec = cv.solvePnP(obj_xyz, img_pts, K_scaled, dist, flags=cv.SOLVEPNP_EPNP)
R_pnp, _ = cv.Rodrigues(rvec)
cam_center = -R_pnp.T @ tvec
dist_pnp = float(abs(cam_center[2, 0]))
proj, _ = cv.projectPoints(obj_xyz, rvec, tvec, K_scaled, dist)
rp_err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - corners) ** 2, axis=1))))
print(f"[PnP] distance to board (mm): {dist_pnp:.2f} (~{dist_pnp/10:.1f} cm)")
print(f"[PnP] reprojection RMSE (px): {rp_err:.3f}")

# --- Draw overlay with axes ---
overlay = hg.draw_board_axes_overlay(
    frame, rvec, tvec, K_scaled, dist,
    dot_cols, dot_rows, dot_spacing,
    label=origin_preset
)

# --- Save outputs ---
raw_path = paths['raw_image']
rectified_path = paths['rectified_image']
overlay_path = paths['axes_overlay']
cv.imwrite(raw_path, frame)
cv.imwrite(rectified_path, rectified)
cv.imwrite(overlay_path, overlay)

pose_json_path = paths.get('pose_json', os.path.join(paths['result_folder'], f"pose_{camera_name}.json"))
pose_npz_path = paths.get('pose_npz', os.path.join(paths['result_folder'], f"pose_{camera_name}.npz"))

summary = {
    "camera_name": camera_name,
    "origin_preset": origin_preset,
    "image_size_WH": [int(w_img), int(h_img)],
    "calib_size_WH": [int(W_calib), int(H_calib)],
    "scale_xy": [float(sx), float(sy)],
    "dot_pattern_cols_rows": [int(dot_cols), int(dot_rows)],
    "dot_spacing_mm": float(dot_spacing),
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

with open(pose_json_path, "w") as f:
    json.dump(summary, f, indent=2)

np.savez_compressed(
    pose_npz_path,
    K_scaled=K_scaled, dist=dist, H_raw=H_raw,
    R_pnp=R_pnp, t_pnp=tvec,
    obj_mm=obj_mm, obj_xyz=obj_xyz,
    img_pts=img_pts.reshape(-1, 2), corners=corners
)

report_path = hg.write_homography_report(
    camera_name=camera_name,
    input_bgr=frame,
    H=H_raw,
    cols=dot_cols, rows=dot_rows,
    square_size_mm=dot_spacing,  # reuses same param name in report
    mm_per_px=mm_per_px,
    corners=corners,
    obj_mm=obj_mm,
    rectified_bgr=rectified,
    K=K_scaled,
    R=R_pnp, t=tvec,
    axes_overlay_bgr=overlay,
    origin_preset=origin_preset
)

print("\n=== OUTPUTS ===")
print(f"JSON   : {pose_json_path}")
print(f"NPZ    : {pose_npz_path}")
print(f"Report : {report_path}")
