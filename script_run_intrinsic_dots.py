import cv2 as cv
import numpy as np
import os
from library import Utils
from library import Settings

###########################################################################
camera_name = 'tiger'
visual_check = True           # show windows during detection
save_visualizations = True    # write annotated PNGs to result folder
###########################################################################

def fit_for_display(img, max_w=1280, max_h=800):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv.resize(img, (int(w*s), int(h*s)), interpolation=cv.INTER_AREA) if s < 1.0 else img

def make_blob_detector(img_shape):
    p = cv.SimpleBlobDetector_Params()
    # We want BLACK dots on WHITE background
    p.filterByColor = True
    p.blobColor = 0

    # Size gate (tweak if needed)
    # Good defaults for dots roughly 25–80 px diameter in the image.
    p.filterByArea = True
    p.minArea = 300        # ≈ 20 px diameter
    p.maxArea = 20000      # ≈ 160 px diameter

    # Shape constraints to reject floor speckles etc.
    p.filterByCircularity = True
    p.minCircularity = 0.70
    p.filterByInertia = True
    p.minInertiaRatio = 0.20
    p.filterByConvexity = True
    p.minConvexity = 0.80

    # Keep blobs distinct
    p.minDistBetweenBlobs = 10

    return cv.SimpleBlobDetector_create(p)

def annotate(img_bgr, gray, detector, found, centers):
    """Return an image with green blob keypoints and, if found, red circle-grid corners."""
    vis = cv.drawKeypoints(img_bgr, detector.detect(gray), None, (0, 255, 0),
                           cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    if found:
        # drawChessboardCorners also works for circle grids
        cv.drawChessboardCorners(vis, (dot_cols, dot_rows), centers, found)
    return vis

# --------- Paths ----------
folders = Utils.get_calibration_paths(camera_name)
calibration_images_folder = folders['calibration_images_folder']
result_folder = folders['result_folder']
intrinsics_yml = folders['intrinsics_yml']
undistort_preview = folders['undistorted_preview']
Utils.create_folder(result_folder, clear=False)
if save_visualizations:
    Utils.create_folder(os.path.join(result_folder, "detect_vis"), clear=False)

# --------- Pattern settings (from Settings.py) ----------
dot_rows = Settings.intrinsic_dot_rows            # e.g., 5
dot_cols = Settings.intrinsic_dot_cols            # e.g., 10
dot_spacing = Settings.intrinsic_dot_spacing      # center-to-center spacing (mm)

# --------- Object points (planar, Z=0) ----------
objp = np.zeros((dot_rows * dot_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:dot_cols, 0:dot_rows].T.reshape(-1, 2) * float(dot_spacing)

# --------- Load images ----------
files = Utils.get_sorted_images(calibration_images_folder)
if not files:
    raise SystemExit("No images found.")
print(f"[info] Found {len(files)} candidate images in: {calibration_images_folder}")

objpoints, imgpoints, used_files = [], [], []
img_size = None
detector = None

flags = cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING

for idx, f in enumerate(files):
    img = cv.imread(f, cv.IMREAD_COLOR)
    if img is None:
        print(f"[warn] Cannot read: {f}")
        continue

    if img_size is None:
        img_size = (img.shape[1], img.shape[0])  # (w, h)
        detector = make_blob_detector(img.shape)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    found, centers = cv.findCirclesGrid(gray, (dot_cols, dot_rows),
                                        flags=flags, blobDetector=detector)

    if found:
        n_found = centers.shape[0]
        print(f"[info] {idx+1:03d}/{len(files):03d}  grid OK ({n_found} centers)  -> {os.path.basename(f)}")
        objpoints.append(objp.copy())
        imgpoints.append(centers)
        used_files.append(f)
    else:
        print(f"[skip] {idx+1:03d}/{len(files):03d}  grid NOT found          -> {os.path.basename(f)}")

    # ----- visualization per image (always shows blobs; centers overlay only if found) -----
    if visual_check or save_visualizations:
        vis = annotate(img, gray, detector, found, centers if found else None)
        if visual_check:
            cv.imshow("Detection (green=blobs, red=circle centers if found)", fit_for_display(vis))
            # small delay to ensure last frame is visible and not immediately overwritten
            cv.waitKey(150)
        if save_visualizations:
            out_path = os.path.join(result_folder, "detect_vis", f"vis_{idx:03d}.png")
            cv.imwrite(out_path, vis)

# Keep the last visualization visible until a keypress
if visual_check:
    cv.waitKey(0)
    try:
        cv.destroyAllWindows()
    except cv.error:
        pass

n = len(objpoints)
if n < 5:
    raise SystemExit(f"[error] Not enough valid views ({n}). Capture more varied angles/distances.")

# --------- Calibration ----------
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
calib_flags = 0

rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
    objectPoints=objpoints,
    imagePoints=imgpoints,
    imageSize=img_size,
    cameraMatrix=None,
    distCoeffs=None,
    flags=calib_flags,
    criteria=criteria
)

# --------- Per-image reprojection error ----------
per_err = []
for f, obj, ip, r, t in zip(used_files, objpoints, imgpoints, rvecs, tvecs):
    proj, _ = cv.projectPoints(obj, r, t, K, dist)
    e = cv.norm(ip, proj, cv.NORM_L2) / len(proj)
    per_err.append((f, float(e)))

# --------- Report ----------
print("\n=== Calibration (Dot Grid) ===")
print(f"Used images : {len(used_files)} / {len(files)}")
print(f"Image size  : {img_size}")
print(f"RMS error   : {rms:.4f} px")
print("Camera matrix K:\n", K)
print("Distortion coeffs:\n", dist.ravel())
print("\nWorst per-image errors (top 5):")
for f, e in sorted(per_err, key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {e:.3f} px  -  {f}")

# --------- Save .yml ---------
fs = cv.FileStorage(intrinsics_yml, cv.FILE_STORAGE_WRITE)
fs.write("image_width", int(img_size[0]))
fs.write("image_height", int(img_size[1]))
fs.write("camera_matrix", K)
fs.write("distortion_coefficients", dist)
fs.write("rms_reprojection_error", float(rms))
fs.write("pattern_rows", int(dot_rows))
fs.write("pattern_cols", int(dot_cols))
fs.write("pattern_center_spacing_mm", float(dot_spacing))
fs.startWriteStruct("per_image_error", cv.FileNode_SEQ)
for fname, e in per_err:
    fs.startWriteStruct("", cv.FileNode_MAP)
    fs.write("file", fname)
    fs.write("error_px", float(e))
    fs.endWriteStruct()
fs.endWriteStruct()
fs.release()
print(f"\n[ok] Saved calibration: {intrinsics_yml}")

# --------- Undistort preview (use best image) ---------
best_file = min(per_err, key=lambda x: x[1])[0] if per_err else used_files[0]
img = cv.imread(best_file, cv.IMREAD_COLOR)
h, w = img.shape[:2]
newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
und = cv.undistort(img, K, dist, None, newK)
x, y, rw, rh = roi
if rw > 0 and rh > 0:
    und = und[y:y+rh, x:x+rw]
cv.imwrite(undistort_preview, und)
print(f"[ok] Wrote undistorted preview: {undistort_preview}\nDone.")
