import cv2 as cv
import numpy as np
from library import Utils
from library import Settings

###########################################################################
camera_name = 'tiger'
visual_check = True
###########################################################################
def _fit_for_display(img, max_w=1280, max_h=800):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    if scale < 1.0:
        img = cv.resize(img, (int(w*scale), int(h*scale)), interpolation=cv.INTER_AREA)
    return img



folders = Utils.get_calibration_paths(camera_name)
calibration_images_folder = folders['calibration_images_folder']
result_folder = folders['result_folder']
intrinsics_yml = folders['intrinsics_yml']
undistort_preview = folders['undistorted_preview']
Utils.create_folder(result_folder, clear=False)

# === PREPARE OBJECT SPACE CORNERS ===
inner_cols = Settings.intrinsic_inner_cols
inner_rows = Settings.intrinsic_inner_rows
square_size = Settings.intrinsic_square_mm

objp = np.zeros((inner_rows * inner_cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:inner_cols, 0:inner_rows].T.reshape(-1, 2)
objp *= square_size

objpoints = []   # 3D points (one per valid image)
imgpoints = []   # 2D points (one per valid image)
used_files = []  # filenames used

# === LOAD IMAGES ===
print(calibration_images_folder)
files = Utils.get_sorted_images(calibration_images_folder)
if not files: raise SystemExit(f"No images")
print(f"[info] Found {len(files)} candidate images.")

img_size = None

# === CORNER DETECTION (SB detector with fallback) ===
for f in files:
    img = cv.imread(f, cv.IMREAD_COLOR)
    print(f'[info] Processing {f}')
    if img is None:
        print(f"[warn] Cannot read: {f}")
        continue

    if img_size is None:
        img_size = (img.shape[1], img.shape[0])  # (w, h)

    # Try the more robust SB detector first
    flags = cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY
    found, corners = cv.findChessboardCornersSB(img, (inner_cols, inner_rows), flags=flags)

    if not found:
        print('[warn] Using fallback')
        # Fallback to classical detector + subpix refinement
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        found, corners = cv.findChessboardCorners(
            gray, (inner_cols, inner_rows),
            flags=cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE
        )
        if found:
            corners = cv.cornerSubPix(
                gray, corners, (5, 5), (-1, -1),
                criteria=(cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 50, 1e-6)
            )

    if not found:
        print(f"[skip] No corners in: {f}")
        continue

    objpoints.append(objp.copy())
    imgpoints.append(corners)
    used_files.append(f)

    if visual_check:
        vis = img.copy()
        cv.drawChessboardCorners(vis, (inner_cols, inner_rows), corners, found)
        disp = _fit_for_display(vis, max_w=1280, max_h=800)  # tweak limits if you like
        cv.imshow("Corners", disp)
        cv.waitKey(150)

if visual_check:
    try:
        cv.destroyWindow("Corners")
    except cv.error:
        pass
    cv.waitKey(1)

if len(objpoints) < 5:
    raise SystemExit(f"[error] Not enough valid views ({len(objpoints)}) for stable calibration. Collect more images at varied angles/distances.")

# === CALIBRATION ===
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
flags = 0  # keep simple; no fixed params

rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
    objectPoints=objpoints,
    imagePoints=imgpoints,
    imageSize=img_size,
    cameraMatrix=None,
    distCoeffs=None,
    flags=flags,
    criteria=criteria
)

# === PER-IMAGE REPROJECTION ERROR ===
per_err = []
for f, obj, imgp, r, t in zip(used_files, objpoints, imgpoints, rvecs, tvecs):
    proj, _ = cv.projectPoints(obj, r, t, K, dist)
    err = cv.norm(imgp, proj, cv.NORM_L2) / len(proj)
    per_err.append((f, float(err)))

# === REPORT ===
print("\n=== Calibration Summary ===")
print(f"Images used : {len(used_files)} / {len(files)}")
print(f"Image size  : {img_size}  (w, h)")
print(f"RMS error   : {rms:.4f} px")
print("Camera matrix K:\n", K)
print("Distortion coeffs (k1 k2 p1 p2 k3 ...):\n", dist.ravel())

print("\nWorst per-image reprojection errors (top 5):")
for f, e in sorted(per_err, key=lambda x: x[1], reverse=True)[:5]:
    print(f"  {e:.3f} px  -  {f}")

# === SAVE TO .yml (OpenCV format) ===
fs = cv.FileStorage(intrinsics_yml, cv.FILE_STORAGE_WRITE)
fs.write("image_width", int(img_size[0]))
fs.write("image_height", int(img_size[1]))
fs.write("camera_matrix", K)
fs.write("distortion_coefficients", dist)
fs.write("rms_reprojection_error", float(rms))
fs.startWriteStruct("per_image_error", cv.FileNode_SEQ)
for fname, e in per_err:
    fs.startWriteStruct("", cv.FileNode_MAP)
    fs.write("file", fname)
    fs.write("error", float(e))
    fs.endWriteStruct()
fs.endWriteStruct()
fs.release()
print(f"\n[ok] Saved calibration to: {intrinsics_yml}")

# === UNDISTORT PREVIEW ===
sample = min(per_err, key=lambda x: x[1])[0] if per_err else used_files[0]
img = cv.imread(sample, cv.IMREAD_COLOR)
h, w = img.shape[:2]
newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
und = cv.undistort(img, K, dist, None, newK)
x, y, rw, rh = roi
if rw > 0 and rh > 0:
    und = und[y:y+rh, x:x+rw]
cv.imwrite(undistort_preview, und)
print(f"[ok] Wrote undistorted preview: {undistort_preview}")

print("\nDone.")
