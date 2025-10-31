#!/usr/bin/env python3
"""
Reusable helpers for planar homography from a checkerboard pattern.

Key functions:
- load_intrinsics(camera_name)         -> K, dist
- detect_checkerboard(gray, grids)     -> (corners Nx1x2, (cols, rows))
- board_points_mm(cols, rows, s_mm)    -> (N,2) board coords (mm), Z=0
- estimate_homography(img_pts, obj_mm) -> H (image->mm) and inlier mask
- rectify_image(img, H, W_mm, H_mm, mm_per_px) -> top-down rectified image
- pose_from_points(K, dist, img_pts, obj_mm)   -> (R, t) of camera wrt plane
- image_to_plane_xy(H, u, v)           -> (X_mm, Y_mm) on the plane

Assumptions:
- Pattern is a checkerboard; detection returns *inner* corners.
- Board coordinates are metric (millimetres) in the plane Z=0.
"""

import base64
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional, Tuple
import cv2 as cv
import numpy as np
from library import Utils


# ----------------- Intrinsics -----------------

def load_intrinsics(camera_name: str):
    """Load camera intrinsics via Utils.get_calibration_paths(camera_name)."""
    from library import Utils  # lazy import to avoid hard dep if not needed
    paths = Utils.get_calibration_paths(camera_name)
    yml = str(paths["intrinsics_yml"])
    fs = cv.FileStorage(yml, cv.FILE_STORAGE_READ)
    if not fs.isOpened(): raise IOError(f"Cannot open intrinsics: {yml}")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    fs.release()
    return K, dist

def undistort_image(img_bgr, K, dist, alpha: float = 1.0):
    """Convenience undistort (one-off; for speed use precomputed remap maps)."""
    h, w = img_bgr.shape[:2]
    newK, _ = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
    return cv.undistort(img_bgr, K, dist, None, newK)

# ----------------- Detection -----------------

# def detect_checkerboard(img_gray: np.ndarray,
#                         candidate_grids: Iterable[Tuple[int, int]],
#                         use_sb_first: bool = True) -> Optional[Tuple[np.ndarray, Tuple[int,int]]]:
#     """
#     Try multiple inner-corner grid sizes. Return best (corners, (cols, rows)) or None.
#     Prefers findChessboardCornersSB (robust) then classic + subpix.
#     """
#     best = None
#     for (cols, rows) in candidate_grids:
#         pat = (cols, rows)
#         corners = None
#         if use_sb_first:
#             try:
#                 ok, corners = cv.findChessboardCornersSB(img_gray, pat,
# flags=cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY)
#                 if not ok:
#                     corners = None
#             except Exception:
#                 corners = None
#         if corners is None:
#             ok, corners = cv.findChessboardCorners(
#                 img_gray, pat,
#                 flags=cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
#             )
#             if ok:
#                 corners = cv.cornerSubPix(
#                     img_gray, corners,
#                     (5, 5), (-1, -1),
#                     (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
#                 )
#             else:
#                 continue
#
#         if best is None or corners.shape[0] > best[0].shape[0]:
#             best = (corners, pat)
#         if corners.shape[0] == cols * rows:
#             break
#     return best

# ----------------- Board model -----------------

def board_points_mm(cols: int, rows: int, square_size_mm: float) -> np.ndarray:
    """
    Build 2D board coordinates (Z=0) for inner corners in mm, origin at (0,0).
    OpenCV orders points row-major: x along columns, y along rows.
    """
    obj = np.zeros((rows * cols, 2), np.float32)
    xs, ys = np.meshgrid(np.arange(cols), np.arange(rows))
    obj[:, 0] = xs.flatten() * square_size_mm
    obj[:, 1] = ys.flatten() * square_size_mm
    return obj  # (N, 2)

# ----------------- Homography -----------------

def estimate_homography(img_pts: np.ndarray, obj_pts_mm: np.ndarray,
                        ransac_thresh_px: float = 2.0):
    """
    Compute H (3x3) that maps image (u,v,1) -> board (X_mm,Y_mm,1).
    Accepts corners as (N,1,2) or (N,2). Returns (H, inlier_mask).
    """
    ip = img_pts.reshape(-1, 1, 2).astype(np.float32)
    op = obj_pts_mm.reshape(-1, 1, 2).astype(np.float32)
    H, mask = cv.findHomography(ip, op, method=cv.RANSAC, ransacReprojThreshold=ransac_thresh_px)
    return H, mask

def image_to_plane_xy(H: np.ndarray, u: float, v: float) -> Tuple[float, float]:
    """Apply H (image->mm) to a single pixel coordinate (u,v)."""
    vec = np.array([u, v, 1.0], dtype=np.float64)
    w = H @ vec
    w /= (w[2] + 1e-12)
    return float(w[0]), float(w[1])

# ----------------- Rectification -----------------

def rectify_image(img_bgr: np.ndarray,
                  H_img2mm: np.ndarray,
                  width_mm: float,
                  height_mm: float,
                  mm_per_px: float = 1.0) -> np.ndarray:
    """
    Warp to a top-down metric view of a (width_mm x height_mm) rectangle.
    Result has size (round(width_mm/mm_per_px), round(height_mm/mm_per_px)).
    """
    W_px = int(round(width_mm / mm_per_px))
    H_px = int(round(height_mm / mm_per_px))
    S = np.array([[1.0 / mm_per_px, 0, 0],
                  [0, 1.0 / mm_per_px, 0],
                  [0, 0, 1]], dtype=np.float64)
    H_img2px = S @ H_img2mm
    rectified = cv.warpPerspective(img_bgr, H_img2px, (W_px, H_px),
                                   flags=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)
    return rectified


def pose_from_homography(k, h):
    """
    Decompose homography to pose: K^{-1} H = [r1 r2 t] up to scale.
    Returns R (3x3), t (3x1). Assumes right-handed camera frame.
    """
    k_inv = np.linalg.inv(k)
    b = k_inv @ h
    b1, b2, b3 = b[:,0], b[:,1], b[:,2]
    # Normalize to make r1, r2 unit and orthogonal
    s = 1.0 / np.linalg.norm(b1)
    r1 = s * b1
    r2 = s * b2
    r3 = np.cross(r1, r2)
    r_approx = np.stack([r1, r2, r3], axis=1)
    # Orthonormalize (SVD) to project to nearest rotation
    u, _, vt = np.linalg.svd(r_approx)
    r = u @ vt
    if np.linalg.det(r) < 0:r[:,2] *= -1  # fix handedness if needed
    t = (s * b3).reshape(3,1)
    return r, t

# ---------- HTML report helpers (self-contained, no extra deps) ----------

def _bgr_to_data_uri(bgr):
    """Encode BGR image as PNG and return data URI (single-file HTML)."""
    ok, buf = cv.imencode(".png", bgr)
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _image_with_corners(bgr, pattern, corners):
    vis = bgr.copy()
    try:
        cv.drawChessboardCorners(vis, pattern, corners, True)
    except Exception:
        pass
    return vis

def _homography_reproj_stats(H, img_pts, obj_pts_mm):
    """
    Map image points through H to plane and compare with obj_pts_mm.
    Returns (rms_mm, max_mm, per_point_errors_mm).
    """
    ip = img_pts.reshape(-1, 2).astype(np.float64)
    ones = np.ones((ip.shape[0], 1))
    P = np.hstack([ip, ones]) @ H.T      # (N,3)
    Z = P[:, 2:3] + 1e-12
    XY = P[:, :2] / Z                    # (N,2)
    diffs = XY - obj_pts_mm.reshape(-1, 2)
    errs = np.linalg.norm(diffs, axis=1)
    rms  = float(np.sqrt(np.mean(errs**2))) if errs.size else float("nan")
    mxx  = float(np.max(errs)) if errs.size else float("nan")
    return rms, mxx, errs

def write_homography_report(
    camera_name: str,
    input_bgr: np.ndarray,
    H: np.ndarray,
    cols: int,
    rows: int,
    square_size_mm: float,
    mm_per_px: float,
    corners: np.ndarray = None,
    obj_mm: np.ndarray = None,
    rectified_bgr: np.ndarray = None,
    K: np.ndarray = None,
    R: np.ndarray = None,
    t: np.ndarray = None,
    axes_overlay_bgr: np.ndarray = None,
    origin_preset="TL"# optional XY overlay image
) -> Path:
    """
    Write a single-file HTML report to Utils.get_calibration_paths(camera_name)["homography_report"].

    IMPORTANT: input_bgr must be from an undistorted image consistent with H.
    """
    paths = Utils.get_calibration_paths(camera_name)
    html_path = Path(paths["homography_report"])
    html_path.parent.mkdir(parents=True, exist_ok=True)

    width_mm  = (cols - 1) * square_size_mm
    height_mm = (rows - 1) * square_size_mm

    # Reprojection stats (in mm) if points are provided
    rms_mm = max_mm = float("nan")
    if corners is not None and obj_mm is not None:
        rms_mm, max_mm, _ = _homography_reproj_stats(H, corners, obj_mm)

    # Pose: use supplied R,t; else derive from H if K provided
    pose_note = ""
    if (R is None or t is None) and K is not None:
        try:
            R, t = pose_from_homography(K, H)
            pose_note = "(derived from H & K)"
        except Exception:
            R = t = None
            pose_note = "(pose unavailable)"

    # Render images to data URIs
    in_vis = _image_with_corners(input_bgr, (cols, rows), corners) if corners is not None else input_bgr
    in_uri = _bgr_to_data_uri(in_vis)
    rect_uri = _bgr_to_data_uri(rectified_bgr) if rectified_bgr is not None else ""
    axes_uri = _bgr_to_data_uri(axes_overlay_bgr) if axes_overlay_bgr is not None else ""

    # Image-center → plane (mm)
    h, w = input_bgr.shape[:2]
    center = np.array([w/2.0, h/2.0, 1.0], dtype=np.float64)
    c_map = H @ center
    c_map /= (c_map[2] + 1e-12)
    center_xy = (float(c_map[0]), float(c_map[1]))

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    H_pretty = "\n".join("[" + ", ".join(f"{x: .6f}" for x in row) + "]" for row in H)

    # Optional pose blocks
    R_block = t_block = cam_center_block = ""
    if R is not None and t is not None:
        cam_center = (-R.T @ t).ravel()
        z_axis = (R.T @ np.array([[0.0],[0.0],[1.0]])).ravel()
        R_block = "\n".join("[" + ", ".join(f"{x: .6f}" for x in row) + "]" for row in R)
        t_block = ", ".join(f"{x:.3f}" for x in t.ravel())
        cam_center_block = ", ".join(f"{x:.3f}" for x in cam_center)

    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Homography Report – {camera_name}</title>
<style>
body {{ font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; padding:24px; color:#222; }}
h1,h2,h3 {{ margin: .2em 0 .4em }}
small {{ color:#666 }}
.card {{ border:1px solid #eee; border-radius:12px; padding:16px; margin:14px 0; box-shadow:0 1px 2px rgba(0,0,0,.04) }}
.grid {{ width:100%; border-collapse:collapse }}
.grid th,.grid td {{ border:1px solid #eee; padding:8px 10px; text-align:left; vertical-align:top }}
.grid .num {{ text-align:right; font-variant-numeric: tabular-nums }}
.code {{ font-family: ui-monospace,SFMono-Regular,Menlo,Consolas,monospace; font-size:12px; white-space:pre }}
img {{ max-width:100%; height:auto; border-radius:10px; border:1px solid #eee }}
.kv th {{ width:260px; background:#fafafa }}
.badge {{ display:inline-block; padding:.15em .6em; border-radius:999px; font-size:12px; font-weight:600; background:#eef }}
</style>
</head>
<body>
  <h1>Homography Report</h1>
  <div><b>Camera:</b> <span class="badge">{camera_name}</span> &nbsp; <small>Generated: {now}</small></div>

  <div class="card">
    <h2>Summary</h2>
    <table class="grid kv">
      <tr><th>Inner corners (cols × rows)</th><td class="num">{cols} × {rows}</td></tr>
      <tr><th>Square size</th><td class="num">{square_size_mm:.3f} mm</td></tr>
      <tr><th>Plane span (inner area)</th><td class="num">{width_mm:.1f} mm × {height_mm:.1f} mm</td></tr>
      <tr><th>Sampling density</th><td class="num">{mm_per_px:.3f} mm/px</td></tr>
      <tr><th>Image center → plane</th><td class="num">({center_xy[0]:.1f} mm, {center_xy[1]:.1f} mm)</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Homography (image → plane mm)</h2>
    <div class="code">{H_pretty}</div>
  </div>

  <div class="card">
    <h2>Plane reprojection (from H)</h2>
    <table class="grid kv">
      <tr><th>Corner RMS error</th><td class="num">{rms_mm:.3f} mm</td></tr>
      <tr><th>Corner max error</th><td class="num">{max_mm:.3f} mm</td></tr>
      <tr><th>Notes</th><td>Errors compare known board points (mm) with image points mapped by H. Values near 0–1&nbsp;mm indicate a solid homography (printer scale permitting).</td></tr>
    </table>
  </div>

  <div class="card">
    <h2>Pose {(''+pose_note) if pose_note else ''}</h2>
    {"<div><b>R:</b><div class='code'>" + R_block + "</div></div>" if R_block else "<p>Not available.</p>"}
    {"<div><b>t (mm):</b> " + t_block + "</div>" if t_block else ""}
    {"<div><b>Camera center (board frame, mm):</b> " + cam_center_block + "</div>" if cam_center_block else ""}
  </div>

  <div class="card">
    <h2>Images</h2>
    <div><b>Input with detected corners</b></div>
    <img src="{in_uri}" alt="input-with-corners" />
    {"<div style='height:10px'></div><div><b>Axes overlay</b></div><img src='" + axes_uri + "' alt='axes-overlay' />" if axes_uri else ""}
    {"<div style='height:10px'></div><div><b>Rectified (top-down)</b></div><img src='" + rect_uri + "' alt='rectified' />" if rect_uri else ""}
  </div>
</body></html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path


# ---------- Geometry helpers for pixel <-> board ----------
def pixel_to_board_xy_raw(u, v, H):
    x = H @ np.array([u, v, 1.0], dtype=float)
    X = x[0]/x[2]; Y = x[1]/x[2]
    return float(X), float(Y)

def pixel_to_ray_cam(u, v, K, dist=None):
    p = np.array([[[u, v]]], dtype=np.float32)  # (1,1,2)
    if dist is not None:
        pn = cv.undistortPoints(p, K, dist, P=np.eye(3, dtype=np.float32)).reshape(2)
        x, y = float(pn[0]), float(pn[1])
        d_cam = np.array([x, y, 1.0], dtype=float)
    else:
        d_cam = np.linalg.inv(K) @ np.array([u, v, 1.0], dtype=float)
    return d_cam / np.linalg.norm(d_cam)

def intersect_ray_with_board(d_cam, R, t):
    Cb = -R.T @ t
    db = R.T @ d_cam.reshape(3,1)
    dz = float(db[2,0])
    if abs(dz) < 1e-9:
        raise RuntimeError("Ray is (near) parallel to the board plane.")
    s = -float(Cb[2,0]) / dz
    Pb = (Cb + s*db).reshape(3)
    Pb[2] = 0.0
    return Pb


def reframe_obj_mm(obj_mm, cols, rows, square_mm, origin="TL"):
    """
    Remap the 2D board model to use a different origin/sense.
    origin ∈ {"TL","TR","BL","BR"} (T=top, B=bottom, L=left, R=right in *image* sense)

    TL: default OpenCV (no change)
    TR: flip X around Xmax
    BL: flip Y around Ymax
    BR: flip both
    """
    Xmax = (cols - 1) * square_mm
    Ymax = (rows - 1) * square_mm
    pts = obj_mm.reshape(-1, 2).copy()

    if origin in ("TR", "BR"):  # flip X
        pts[:, 0] = Xmax - pts[:, 0]
    if origin in ("BL", "BR"):  # flip Y
        pts[:, 1] = Ymax - pts[:, 1]

    return pts.reshape(obj_mm.shape)


def draw_board_axes_overlay(frame_bgr, rvec, tvec, K, dist, cols, rows, square_mm, label):
    Xmax = (cols - 1) * square_mm
    Ymax = (rows - 1) * square_mm

    O  = np.float32([[0,0,0]])
    X1 = np.float32([[square_mm,0,0]])
    Y1 = np.float32([[0,square_mm,0]])


    axis_len = max(Xmax, Ymax) / 3.0
    def extend(A,B,L):
        v = (B-A); v = v/ (np.linalg.norm(v)+1e-9); return A + v*L

    Xtip = extend(O, X1, axis_len)
    Ytip = extend(O, Y1, axis_len)

    pts3 = np.vstack([O, Xtip, Ytip]).astype(np.float32)
    img, _ = cv.projectPoints(pts3, rvec, tvec, K, dist)
    p0, px, py = [tuple(np.round(p).astype(int)) for p in img.reshape(-1,2)]

    out = frame_bgr.copy()
    cv.circle(out, p0, 6, (255,255,255), -1)
    cv.line(out, p0, px, (0,0,255), 3); cv.putText(out, "X", px, cv.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2, cv.LINE_AA)
    cv.line(out, p0, py, (0,255,0), 3); cv.putText(out, "Y", py, cv.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2, cv.LINE_AA)
    cv.putText(out, f"origin={label}", (p0[0]+8, p0[1]-8), cv.FONT_HERSHEY_SIMPLEX, 0.6,(255,255,255),2, cv.LINE_AA)
    return out
