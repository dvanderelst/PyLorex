"""
Live, hands-free intrinsic calibration with on-the-fly coverage feedback.

Why: vanilla intrinsic calibration is blind to *where* in the image plane the
board has been sampled. A fit with 0.18 px RMS at the centre can still
extrapolate badly at the corners, producing PnP-time errors downstream.
This script auto-captures on a timer (so you can be in the arena, hands on
the board), narrates progress aloud, and shows the coverage map on screen
so you can aim for the uncovered corners.

Workflow:
  1. Set `camera_name` below.
  2. Run. A live preview window opens. Audio announces each step.
  3. Stand in the arena holding the dot board. Every SECONDS_PER_CAPTURE
     seconds the script attempts a capture: 3 pips, then a shutter on
     success or a beep on failure. On success, the script announces sample
     count and current RMS.
  4. Watch the coverage overlay on screen fill in green; aim the board
     at the uncovered areas (corners especially).
  5. After TARGET_SAMPLES successful captures the script saves the
     calibration and the coverage PNG and exits. Ctrl-C also exits cleanly
     and still saves whatever has accumulated.

Keys (for use once you can reach the keyboard):
  Q      quit-and-save now
  D      delete most recent sample
"""

import os
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")
import re
import time

import cv2 as cv
import easygui
import numpy as np

from LorexLib import Grabber, Settings, Sound, Utils

# ---------------- SETTINGS ----------------
camera_name = 'shark'
PREVIEW_MAX_W = 1280
PREVIEW_MAX_H = 720
MIN_SAMPLES_FOR_CALIB = 10        # don't try calibrating until at least this many
TARGET_SAMPLES = 150               # auto-exit + save when this many successful captures
SECONDS_PER_CAPTURE = 8           # time between capture attempts (1 pip per second in last 3)
ANNOUNCE_RMS_EVERY = 10            # speak RMS every Nth successful capture
RECOMPUTE_EVERY_N = 10             # only run cv.calibrateCamera every Nth capture (keeps loop snappy)
# ------------------------------------------


def make_blob_detector():
    p = cv.SimpleBlobDetector_Params()
    p.filterByColor = True
    p.blobColor = 0
    p.filterByArea = True
    p.minArea = 300
    p.maxArea = 20000
    p.filterByCircularity = True
    p.minCircularity = 0.70
    p.filterByConvexity = True
    p.minConvexity = 0.80
    return cv.SimpleBlobDetector_create(p)


def fit_for_display(img, max_w=PREVIEW_MAX_W, max_h=PREVIEW_MAX_H):
    h, w = img.shape[:2]
    s = min(max_w / w, max_h / h, 1.0)
    return cv.resize(img, (int(w * s), int(h * s)), interpolation=cv.INTER_AREA) if s < 1.0 else img


def overlay_text(img, lines):
    y = 30
    for line in lines:
        cv.putText(img, line, (10, y), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        y += 30


def draw_coverage_patch(coverage_overlay, centers, color=(40, 200, 40)):
    pts = centers.reshape(-1, 2).astype(np.float32)
    hull = cv.convexHull(pts).reshape(-1, 1, 2).astype(np.int32)
    cv.fillPoly(coverage_overlay, [hull], color)
    cv.polylines(coverage_overlay, [hull], True, (0, 255, 0), 2)


def rebuild_coverage(coverage_overlay, imgpoints):
    coverage_overlay[:] = 0
    for ip in imgpoints:
        draw_coverage_patch(coverage_overlay, ip)


def recompute_calibration(objpoints, imgpoints, img_size):
    if len(objpoints) < MIN_SAMPLES_FOR_CALIB:
        return None, None, None, None
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-8)
    rms, K, dist, rvecs, tvecs = cv.calibrateCamera(
        objpoints, imgpoints, img_size,
        cameraMatrix=None, distCoeffs=None,
        flags=0, criteria=criteria,
    )
    per_err = []
    for obj, ip, r, t in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv.projectPoints(obj, r, t, K, dist)
        per_err.append(float(cv.norm(ip, proj, cv.NORM_L2) / len(proj)))
    return rms, K, dist, per_err


def save_intrinsics(intrinsics_yml, img_size, K, dist, rms,
                    dot_rows, dot_cols, dot_spacing):
    fs = cv.FileStorage(intrinsics_yml, cv.FILE_STORAGE_WRITE)
    fs.write("image_width", int(img_size[0]))
    fs.write("image_height", int(img_size[1]))
    fs.write("camera_matrix", K)
    fs.write("distortion_coefficients", dist)
    fs.write("rms_reprojection_error", float(rms))
    fs.write("pattern_rows", int(dot_rows))
    fs.write("pattern_cols", int(dot_cols))
    fs.write("pattern_center_spacing_mm", float(dot_spacing))
    fs.release()


def main():
    folders = Utils.get_calibration_paths(camera_name)
    calibration_images_folder = folders['calibration_images_folder']
    result_folder = folders['result_folder']
    intrinsics_yml = folders['intrinsics_yml']
    Utils.create_folder(result_folder, clear=False)
    Utils.create_folder(calibration_images_folder, clear=False)
    coverage_png = os.path.join(result_folder, f"coverage_{camera_name}.png")

    # Offer to wipe existing capture set
    existing = Utils.get_sorted_images(calibration_images_folder)
    keep_existing = False
    if existing:
        if easygui.ynbox(f"{len(existing)} existing calibration images for "
                         f"'{camera_name}'. Delete and start fresh?\n\n"
                         "(Choose No to keep them and add new samples on top — "
                         "the coverage map will reflect what's already there.)"):
            for f in existing:
                os.remove(f)
            existing = []
        else:
            keep_existing = True

    channel = Settings.channels[camera_name]
    grabber = Grabber.RTSPGrabber(channel=channel, auto_start=True)
    sounds = Sound.SoundPlayer()

    dot_rows = Settings.calibration_dot_rows
    dot_cols = Settings.calibration_dot_cols
    dot_spacing = Settings.calibration_dot_spacing
    objp = np.zeros((dot_rows * dot_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:dot_cols, 0:dot_rows].T.reshape(-1, 2) * float(dot_spacing)
    detector = make_blob_detector()
    grid_flags = cv.CALIB_CB_SYMMETRIC_GRID + cv.CALIB_CB_CLUSTERING

    objpoints, imgpoints, used_files = [], [], []
    K = dist = None
    rms = None
    per_err = None
    img_size = None
    coverage_overlay = None

    # If keeping existing images, re-detect and pre-populate the coverage
    # map + calibration estimate so the user can target only the gaps.
    if keep_existing and existing:
        print(f"[load] re-detecting {len(existing)} existing images...")
        for f in existing:
            img = cv.imread(f, cv.IMREAD_COLOR)
            if img is None:
                continue
            if img_size is None:
                img_size = (img.shape[1], img.shape[0])
                coverage_overlay = np.zeros_like(img)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            found, centers = cv.findCirclesGrid(
                gray, (dot_cols, dot_rows),
                flags=grid_flags, blobDetector=detector,
            )
            if found:
                objpoints.append(objp.copy())
                imgpoints.append(centers)
                used_files.append(f)
                draw_coverage_patch(coverage_overlay, centers)
        print(f"[load] {len(used_files)} of {len(existing)} loaded.")
        if len(objpoints) >= MIN_SAMPLES_FOR_CALIB:
            rms, K, dist, per_err = recompute_calibration(
                objpoints, imgpoints, img_size)
            if rms is not None:
                print(f"[load] starting RMS = {rms:.4f} px  worst = {max(per_err):.3f}")

    # Pick a new save index past any existing frameNN.jpg already on disk
    max_existing = -1
    for f in existing:
        m = re.match(r"frame(\d+)\.jpg", os.path.basename(f))
        if m:
            max_existing = max(max_existing, int(m.group(1)))
    save_counter = max_existing + 1

    print(f"[live] {camera_name}: auto-capture every {SECONDS_PER_CAPTURE}s, "
          f"target = {TARGET_SAMPLES} samples. Q=quit-and-save, D=delete last.")

    win = "intrinsic calibration (live)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)

    sounds.speak(
        f"Starting intrinsic calibration for {camera_name}. "
        f"{TARGET_SAMPLES} samples. {SECONDS_PER_CAPTURE} seconds between captures. "
        "Aim the board at the corners and edges.",
        volume=1.0, blocking=False,
    )

    now0 = time.time()
    next_capture_time = now0 + SECONDS_PER_CAPTURE
    next_pip_time = None
    pips_done = 0

    try:
        while True:
            frame = grabber.get_latest_bgr()
            if frame is None:
                cv.waitKey(30)
                continue
            if img_size is None:
                img_size = (frame.shape[1], frame.shape[0])
                coverage_overlay = np.zeros_like(frame)

            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            found, centers = cv.findCirclesGrid(
                gray, (dot_cols, dot_rows),
                flags=grid_flags, blobDetector=detector,
            )

            preview = frame.copy()
            preview = cv.addWeighted(preview, 1.0, coverage_overlay, 0.45, 0)
            if found:
                cv.drawChessboardCorners(preview, (dot_cols, dot_rows), centers, found)

            now = time.time()
            countdown_s = max(0.0, next_capture_time - now)
            lines = [
                f"camera   : {camera_name}",
                f"samples  : {len(objpoints)} / {TARGET_SAMPLES}",
                f"detect   : {'YES' if found else 'no'}",
                f"next cap : {countdown_s:4.1f} s",
                f"RMS      : {rms:.3f} px" if rms is not None else
                f"RMS      : n/a (>= {MIN_SAMPLES_FOR_CALIB} samples)",
            ]
            if per_err:
                lines.append(f"worst    : {max(per_err):.3f} px")
            overlay_text(preview, lines)

            cv.imshow(win, fit_for_display(preview))
            key = cv.waitKey(30) & 0xFF

            # 3-pip countdown in the last 3s before capture
            if next_pip_time is None and now >= next_capture_time - 3.0:
                next_pip_time = now
            if next_pip_time is not None and now >= next_pip_time and pips_done < 3:
                sounds.play('pips', volume=0.25, blocking=False)
                pips_done += 1
                next_pip_time = now + 1.0

            # Auto-capture
            if now >= next_capture_time:
                if found:
                    fname = os.path.join(
                        calibration_images_folder,
                        f"frame{save_counter:02d}.jpg",
                    )
                    cv.imwrite(fname, frame)
                    objpoints.append(objp.copy())
                    imgpoints.append(centers)
                    used_files.append(fname)
                    draw_coverage_patch(coverage_overlay, centers)
                    save_counter += 1
                    sounds.play('shutter', volume=1.0, blocking=False)
                    print(f"[capture] {len(used_files)} -> {os.path.basename(fname)}")
                    n = len(used_files)
                    # Only re-run calibration every Nth capture (or at TARGET_SAMPLES);
                    # cv.calibrateCamera gets slow with many samples and blocks the
                    # preview loop otherwise.
                    if n % RECOMPUTE_EVERY_N == 0 or n == TARGET_SAMPLES:
                        rms, K, dist, per_err = recompute_calibration(
                            objpoints, imgpoints, img_size)
                    if rms is not None:
                        print(f"          RMS={rms:.4f} px  worst={max(per_err):.3f}")
                        if n % ANNOUNCE_RMS_EVERY == 0 or n == TARGET_SAMPLES:
                            sounds.speak(
                                f"Sample {n}. R M S {rms:.2f}.",
                                volume=1.0, blocking=False,
                            )
                        else:
                            sounds.speak(f"Sample {n}", volume=1.0, blocking=False)
                    else:
                        sounds.speak(f"Sample {n}", volume=1.0, blocking=False)
                    if n >= TARGET_SAMPLES:
                        sounds.speak("Target reached. Saving.",
                                     volume=1.0, blocking=True)
                        break
                else:
                    sounds.speak("No marker. Reposition.",
                                 volume=1.0, blocking=False)
                    print("[skip] no grid detected at attempt time")
                next_capture_time = now + SECONDS_PER_CAPTURE
                next_pip_time = None
                pips_done = 0

            if key == ord('q'):
                break
            elif key == ord('d'):
                if not used_files:
                    print("[delete] nothing to delete")
                    sounds.speak("Nothing to delete.", volume=1.0, blocking=False)
                else:
                    last = used_files.pop()
                    objpoints.pop(); imgpoints.pop()
                    try:
                        os.remove(last)
                    except OSError:
                        pass
                    rebuild_coverage(coverage_overlay, imgpoints)
                    print(f"[delete] removed {os.path.basename(last)} "
                          f"({len(used_files)} remaining)")
                    sounds.speak(f"Deleted. {len(used_files)} remaining.",
                                 volume=1.0, blocking=False)
                    rms, K, dist, per_err = recompute_calibration(
                        objpoints, imgpoints, img_size)
    except KeyboardInterrupt:
        print("\n[interrupted] saving and exiting cleanly...")
    finally:
        # Final calibration pass over ALL accumulated samples (the in-loop
        # recompute is throttled, so what's in K/dist may be stale).
        if len(objpoints) >= MIN_SAMPLES_FOR_CALIB:
            print("\n[finalizing] running final calibration over all samples...")
            rms, K, dist, per_err = recompute_calibration(
                objpoints, imgpoints, img_size)
        if K is not None:
            save_intrinsics(intrinsics_yml, img_size, K, dist, rms,
                            dot_rows, dot_cols, dot_spacing)
            cv.imwrite(coverage_png, coverage_overlay)
            print(f"\n[saved] {intrinsics_yml}\n[saved] {coverage_png}")
            print(f"[final] {len(objpoints)} samples, RMS={rms:.4f} px, "
                  f"worst={max(per_err):.3f} px")
        else:
            print("\n[exit] not enough samples to calibrate; nothing saved.")
        grabber.stop()
        try:
            cv.destroyAllWindows()
        except cv.error:
            pass


if __name__ == "__main__":
    main()
