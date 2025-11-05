import cv2 as cv
import numpy as np
import re
import os
from library import Utils

###########################################################################
camera_name = 'shark'
show_images = False
###########################################################################

folders = Utils.get_calibration_paths(camera_name)
intrinsics_yml = folders['intrinsics_yml']
report_base = os.path.join(folders['result_folder'], 'intrinsics_report')
report_txt = report_base + '.txt'
report_html = report_base + '.html'

print(f"[info] Reading: {intrinsics_yml}")

# --- Read YAML ---
fs = cv.FileStorage(intrinsics_yml, cv.FILE_STORAGE_READ)
if not fs.isOpened():
    raise SystemExit(f"[error] Cannot open file: {intrinsics_yml}")

K = fs.getNode("camera_matrix").mat()
dist = fs.getNode("distortion_coefficients").mat()
rms = fs.getNode("rms_reprojection_error").real()
img_w = int(fs.getNode("image_width").real())
img_h = int(fs.getNode("image_height").real())
fs.release()

# --- Parse per-image errors (error or error_px) ---
with open(intrinsics_yml, "r") as f:
    text = f.read()

pattern = r'file:\s*"([^"]+)"\s*error(?:_px)?:\s*([0-9.\-+Ee]+)'
matches = re.findall(pattern, text)
per_image_errors = [(m[0], float(m[1])) for m in matches]

if not per_image_errors:
    print("[warn] No per-image errors found in YAML file.")
mean_err = np.mean([e for _, e in per_image_errors]) if per_image_errors else np.nan
max_err = np.max([e for _, e in per_image_errors]) if per_image_errors else np.nan

# --- Qualitative evaluation ---
if rms < 0.3 and max_err < 0.6:
    quality = "Excellent"
elif rms < 0.5 and max_err < 1.0:
    quality = "Good"
elif rms < 1.0 and max_err < 2.0:
    quality = "Acceptable"
else:
    quality = "Poor"

print("\n=== Intrinsics Assessment ===")
print(f"File: {os.path.basename(intrinsics_yml)}")
print(f"Image size: {img_w} × {img_h}")
print(f"RMS reprojection error: {rms:.4f} px")
print(f"Mean per-image error:   {mean_err:.4f} px")
print(f"Max per-image error:    {max_err:.4f} px")
print(f"Overall quality:        {quality}")

# --- Write plain-text report ---
with open(report_txt, "w") as f:
    f.write("=== Intrinsics Assessment ===\n")
    f.write(f"File: {os.path.basename(intrinsics_yml)}\n")
    f.write(f"Image size: {img_w} × {img_h}\n")
    f.write(f"RMS reprojection error: {rms:.4f} px\n")
    f.write(f"Mean per-image error:   {mean_err:.4f} px\n")
    f.write(f"Max per-image error:    {max_err:.4f} px\n")
    f.write(f"Overall quality:        {quality}\n\n")
    f.write("Per-image errors (px):\n")
    for fname, e in sorted(per_image_errors, key=lambda x: x[1], reverse=True):
        f.write(f"{e:8.4f}   {fname}\n")

print(f"[ok] Text report written to: {report_txt}")

# --- Write HTML report ---
with open(report_html, "w") as f:
    f.write("<html><head><meta charset='utf-8'><style>")
    f.write("body{font-family:sans-serif;background:#f8f8f8;color:#222;padding:2em;}")
    f.write("table{border-collapse:collapse;margin-top:1em;}")
    f.write("th,td{border:1px solid #ccc;padding:4px 8px;text-align:left;}")
    f.write("th{background:#eee;}")
    f.write("</style></head><body>")
    f.write("<h2>Intrinsics Assessment</h2>")
    f.write(f"<p><b>File:</b> {os.path.basename(intrinsics_yml)}<br>")
    f.write(f"<b>Image size:</b> {img_w} × {img_h}<br>")
    f.write(f"<b>RMS reprojection error:</b> {rms:.4f} px<br>")
    f.write(f"<b>Mean per-image error:</b> {mean_err:.4f} px<br>")
    f.write(f"<b>Max per-image error:</b> {max_err:.4f} px<br>")
    f.write(f"<b>Overall quality:</b> <span style='font-weight:bold;color:#007700'>{quality}</span></p>")
    f.write("<table><tr><th>Error (px)</th><th>Image</th></tr>")
    for fname, e in sorted(per_image_errors, key=lambda x: x[1], reverse=True):
        f.write(f"<tr><td>{e:.4f}</td><td>{fname}</td></tr>")
    f.write("</table></body></html>")

print(f"[ok] HTML report written to: {report_html}")

# --- Optional visualization ---
if show_images and per_image_errors:
    test_file = per_image_errors[0][0]
    if os.path.exists(test_file):
        img = cv.imread(test_file)
        if img is not None:
            h, w = img.shape[:2]
            newK, roi = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
            und = cv.undistort(img, K, dist, None, newK)
            x, y, rw, rh = roi
            if rw > 0 and rh > 0:
                und = und[y:y+rh, x:x+rw]
            cv.imshow("Undistorted preview", und)
            cv.waitKey(0)
            cv.destroyAllWindows()
