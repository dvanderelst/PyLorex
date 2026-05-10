#!/usr/bin/env python3
"""
script_probe_detection_rate.py

Measure how long aruco detection takes on a real DVR frame at several
downsample scales. Combined with `script_probe_rtsp_rate.py` (which
confirmed the DVR delivers ~17 fps on 4K main), this isolates which
side of the pipeline limits the end-to-end fresh-frame rate.

Hypothesis: with Settings.aruco_detect_scale = 1.0, detection runs on
the full 4K image, takes ~1+ second, and limits the system to ~0.8 Hz
of fresh outputs even though the camera is delivering at 17 fps.

Procedure (per camera, per scale):
  1. Grab one fresh frame via RTSPGrabber.
  2. Convert to gray (same path as Lorex.detect_markers).
  3. Time N back-to-back detectMarkers calls at the requested scale.
  4. Report min / median / max ms and the implied Hz.

A "good" result is detection time well under the inter-frame interval
(~60 ms at 17 fps). Anything longer means we'll stall on a frame and
miss N-1 of the frames that arrive while we're processing one.
"""

import time
from statistics import median

import cv2 as cv

from LorexLib import Settings
from LorexLib.Grabber import RTSPGrabber
from LorexLib.Lorex import LorexCamera


SCALES        = (1.0, 0.5, 0.25)
N_TIMES       = 20      # back-to-back detections on a single frame, per scale
WARMUP_DETECT = 3       # discard the first few to let allocator warm up


def probe_camera(camera_name: str) -> dict:
    print(f"\n=== {camera_name} ===")
    grabber = RTSPGrabber(channel=Settings.channels[camera_name], auto_start=True)
    frame = grabber.wait_latest_bgr(timeout=5.0)
    if frame is None:
        grabber.stop()
        print("  no frame within 5 s")
        return {}
    h, w = frame.shape[:2]
    print(f"  frame size: {w} × {h}")
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    cam = LorexCamera(camera_name)
    cam.ensure_aruco()

    rows = {}
    for s in SCALES:
        times_ms = []
        ids_first = None
        for i in range(N_TIMES + WARMUP_DETECT):
            t0 = time.time()
            corners, ids = cam.detect_markers(gray, scale_override=s)
            dt = (time.time() - t0) * 1000.0
            if i >= WARMUP_DETECT:
                times_ms.append(dt)
            if i == WARMUP_DETECT:
                ids_first = None if ids is None else sorted(int(x) for x in ids.ravel())
        times_ms.sort()
        med = median(times_ms)
        rows[s] = {
            "min_ms":    times_ms[0],
            "med_ms":    med,
            "max_ms":    times_ms[-1],
            "hz":        1000.0 / med if med > 0 else 0.0,
            "ids":       ids_first,
        }
    grabber.stop()
    return rows


def main():
    print(f"Probing detection time at scales {SCALES}")
    print(f"Settings.aruco_detect_scale (currently active) = "
          f"{Settings.aruco_detect_scale}")
    print(f"Settings.aruco_refine_win = {Settings.aruco_refine_win}  "
          f"iters = {Settings.aruco_refine_iters}  "
          f"fast_refine = {Settings.aruco_fast_refine}")

    headers = f"{'cam':>8}  {'scale':>5}  {'min ms':>7}  {'med ms':>7}  " \
              f"{'max ms':>7}  {'Hz':>6}  ids"
    print("\n" + headers)
    print("-" * len(headers))

    for cam_name in Settings.channels:
        rows = probe_camera(cam_name)
        for s, r in rows.items():
            print(f"{cam_name:>8}  {s:>5.2f}  "
                  f"{r['min_ms']:>7.1f}  {r['med_ms']:>7.1f}  "
                  f"{r['max_ms']:>7.1f}  {r['hz']:>6.1f}  {r['ids']}")

    print("\nInterpretation:")
    print("  • Hz at scale=1.0 ≪ 17 → detection is the bottleneck;")
    print("    bump Settings.aruco_detect_scale to 0.5 (or 0.25 if the")
    print("    marker is still big enough) and re-measure.")
    print("  • If marker IDs at scale=0.5 still match scale=1.0, the")
    print("    downsample doesn't lose detection robustness for this")
    print("    physical marker size + camera distance.")


if __name__ == "__main__":
    main()
