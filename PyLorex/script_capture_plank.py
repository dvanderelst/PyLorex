"""Capture synchronised tiger + shark snapshots of the calibration plank
for the arena-frame bundle-adjustment calibration.

Plank: three ArUco markers (DICT_4X4_1000, IDs 75/76/77) along the plank
axis, 500 mm centre-to-centre between adjacent markers, plank surface
25 mm above the floor. Marker side 90 mm.

Workflow:
  1. Lay the plank flat on the floor with the markers visible to at
     least one camera. Step out of frame.
  2. Watch the live preview (both cameras stacked) — detected target
     markers are highlighted green. Move the plank around until you see
     the markers detected reliably.
  3. Press SPACE to capture the current position. The script grabs
     several frames from each camera and saves the per-corner median +
     the raw frames.
  4. Move the plank to the next position, repeat. Target ~15 positions
     with mixed coverage:
        - 4-5 in tiger's solo zone (centre + periphery, varied orientation).
        - 4-5 in shark's solo zone (same).
        - 3-4 straddling the FOV overlap (extra valuable for cross-camera).
        - A couple of "rotate 90 degrees in place" pairs.
  5. Press Q or ESC when done.

Output layout under PyLorex/Calibration/PlankSnapshots/<session_dir>/:
    meta.json
    snapshot_001/
        frame_tiger.jpg
        frame_shark.jpg
        detections.json     - per-camera marker corner pixels
    snapshot_002/ ...

Controls (in the live preview window):
    SPACE   - capture the current position.
    R       - delete the most recent saved snapshot (redo).
    Q / ESC - finish and exit.

This data feeds two downstream scripts (to be written):
    - script_diagnose_plank.py  : compares plank distances reported by
      the current calibration to your tape-measured 500/500/1000 mm.
    - script_calibrate_plank.py : joint bundle adjustment that solves
      shark's (R, t) in tiger frame + each snapshot's plank pose.
"""

import json
import shutil
import time
from pathlib import Path

import cv2 as cv
import numpy as np

from LorexLib import Lorex, Sound


# ---------------- SETTINGS ----------------
TARGET_IDS = (75, 76, 77)
N_GRABS_PER_CAPTURE = 3            # median over N frames per camera per snapshot
GRAB_INTERVAL_S = 0.1              # spacing between the N grabs
CAMERA_NAMES = ("tiger", "shark")
PREVIEW_WIDTH = 800               # downscale for the on-screen preview only
SAVE_ROOT = Path(__file__).resolve().parent / "Calibration" / "PlankSnapshots"
COUNTDOWN_PIPS = 2                 # number of 1-Hz pips after SPACE before capture
STATUS_SPEAK_MIN_GAP_S = 1.5       # rate-limit detection-status announcements

# Auto-capture: script triggers a capture by itself when the plank has
# been stationary for AUTO_STABLE_FRAMES_REQUIRED preview frames in a row
# (with at least 2 target markers visible). Lets the user place + walk
# away + wait, without needing to come back to the keyboard each time.
AUTO_CAPTURE_ENABLED = True
AUTO_STABLE_FRAMES_REQUIRED = 20   # ~2 s at the preview loop's ~10 Hz
AUTO_STABLE_PIXELS = 1.5            # max per-frame corner motion to count "stable"
AUTO_MIN_TARGETS_SEEN = 2           # require >=N target markers visible in some cam
AUTO_COOLDOWN_S = 5.0               # min seconds between auto-captures
AUTO_HOLDING_ANNOUNCE_FRACTION = 0.5  # speak "Holding" at this fraction of stable period
# ------------------------------------------


def get_aruco_handles():
    """Return (dictionary, params, detector_or_None). Mirrors the
    new-vs-old OpenCV ArUco API selection from LorexLib.Lorex: prefer
    the ArucoDetector class when available; fall back to free-function
    cv.aruco.detectMarkers otherwise. DetectorParameters constructor
    also differs across versions."""
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    try:
        params = cv.aruco.DetectorParameters()
    except AttributeError:
        params = cv.aruco.DetectorParameters_create()
    detector = None
    if hasattr(cv.aruco, "ArucoDetector"):
        detector = cv.aruco.ArucoDetector(dictionary, params)
    return dictionary, params, detector


def detect_markers(frame_bgr, dictionary, params, detector):
    """Return {id: corners (4x2 in pixel coords)} for all detected markers."""
    if frame_bgr is None:
        return {}
    gray = cv.cvtColor(frame_bgr, cv.COLOR_BGR2GRAY)
    if detector is not None:
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        corners, ids, _ = cv.aruco.detectMarkers(
            gray, dictionary, parameters=params)
    if ids is None:
        return {}
    out = {}
    for c, i in zip(corners, ids.flatten()):
        out[int(i)] = c.reshape(-1, 2)
    return out


def is_frame_stable(detects, prev_detects, target_ids, threshold_px):
    """True iff (a) at least one target marker is visible in at least one
    camera, and (b) every currently-visible target marker was *also*
    visible in the previous frame and its 4 corner pixels haven't moved
    by more than threshold_px in any coordinate. Newly-appeared markers
    count as not-stable (something is moving)."""
    visible_now = set()
    for det in detects.values():
        for mid in target_ids:
            if mid in det:
                visible_now.add(mid)
    if not visible_now:
        return False
    for cam_name, det in detects.items():
        prev = prev_detects.get(cam_name, {})
        for mid in target_ids:
            if mid not in det:
                continue
            if mid not in prev:
                return False
            max_move = float(np.max(np.abs(det[mid] - prev[mid])))
            if max_move > threshold_px:
                return False
    return True


def median_corners(detection_lists, marker_id):
    """Median across frames of the 4 corner pixel positions for one marker."""
    coords = [d[marker_id] for d in detection_lists if marker_id in d]
    if not coords:
        return None
    return np.median(np.stack(coords, axis=0), axis=0)


def annotate(frame_bgr, detections, target_ids):
    """Draw detected markers; bright green for the target IDs, grey for
    anything else that happens to be in view."""
    out = frame_bgr.copy()
    for mid, corners in detections.items():
        col = (0, 255, 0) if mid in target_ids else (180, 180, 180)
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv.polylines(out, [pts], True, col, 2)
        cx, cy = corners.mean(axis=0).astype(int)
        cv.putText(out, str(mid), (int(cx) - 15, int(cy) + 8),
                   cv.FONT_HERSHEY_SIMPLEX, 1.2, col, 3, cv.LINE_AA)
    return out


def resize_for_preview(img, width):
    h, w = img.shape[:2]
    if w <= width:
        return img
    scale = width / w
    return cv.resize(img, (width, int(h * scale)), interpolation=cv.INTER_AREA)


def next_session_dir(root):
    root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    existing = {p.name for p in root.iterdir() if p.is_dir()}
    idx = 1
    while True:
        name = f"plank_{idx:04d}_{ts}"
        if name not in existing:
            return root / name
        idx += 1


def write_snapshot(snap_dir, snapshot_idx,
                   t_frame, s_frame, t_detect, s_detect):
    snap_dir.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(snap_dir / "frame_tiger.jpg"), t_frame)
    cv.imwrite(str(snap_dir / "frame_shark.jpg"), s_frame)
    payload = {
        "snapshot": int(snapshot_idx),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cameras": {
            "tiger": {
                "frame_size_WH": [int(t_frame.shape[1]), int(t_frame.shape[0])],
                "markers": [
                    {"id": int(k), "corners_px": v.tolist()}
                    for k, v in sorted(t_detect.items())
                ],
            },
            "shark": {
                "frame_size_WH": [int(s_frame.shape[1]), int(s_frame.shape[0])],
                "markers": [
                    {"id": int(k), "corners_px": v.tolist()}
                    for k, v in sorted(s_detect.items())
                ],
            },
        },
    }
    with open(snap_dir / "detections.json", "w") as f:
        json.dump(payload, f, indent=2)


def capture_one(cams, dictionary, params, detector):
    """Grab N frames from each camera, median frame + median per-corner
    detections per marker. Returns {cam_name: (median_frame, {id: 4x2})}
    or None on failure."""
    grabs = {n: [] for n in CAMERA_NAMES}
    detect_lists = {n: [] for n in CAMERA_NAMES}
    for _ in range(N_GRABS_PER_CAPTURE):
        for n in CAMERA_NAMES:
            f = cams[n].get_frame(undistort=False)
            if f is not None:
                grabs[n].append(f)
                detect_lists[n].append(detect_markers(f, dictionary, params, detector))
        time.sleep(GRAB_INTERVAL_S)

    final = {}
    for n in CAMERA_NAMES:
        if not grabs[n]:
            print(f"  [{n}] no frames grabbed; aborting this snapshot.")
            return None
        median_frame = np.median(np.stack(grabs[n], axis=0), axis=0).astype(np.uint8)
        ids_seen = set().union(*(d.keys() for d in detect_lists[n]))
        per_marker = {}
        for mid in ids_seen:
            mc = median_corners(detect_lists[n], mid)
            if mc is not None:
                per_marker[int(mid)] = mc
        final[n] = (median_frame, per_marker)
    return final


def main():
    session_dir = next_session_dir(SAVE_ROOT)
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"[session] {session_dir}")

    meta = {
        "session_dir": str(session_dir),
        "cameras": list(CAMERA_NAMES),
        "target_ids": list(TARGET_IDS),
        "n_grabs_per_capture": N_GRABS_PER_CAPTURE,
        "plank_geometry_mm": {
            "marker_ids": list(TARGET_IDS),
            "marker_offsets_x_mm": [-500.0, 0.0, +500.0],
            "marker_size_mm": 90.0,
            "marker_z_above_floor_mm": 25.0,
        },
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(session_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("[setup] opening cameras ...")
    cams = {n: Lorex.LorexCamera(n) for n in CAMERA_NAMES}
    for c in cams.values():
        try:
            c.wait_ready(timeout=5.0)
        except Exception as e:
            print(f"[warn] wait_ready raised {e}; continuing anyway.")

    dictionary, params, detector = get_aruco_handles()
    sounds = Sound.SoundPlayer()

    snapshot_idx = 0
    history = []  # list of (snapshot_idx, snap_dir) for redo
    last_status_spoken = (-1, -1)
    last_status_speak_time = 0.0
    prev_detects = {n: {} for n in CAMERA_NAMES}
    stable_count = 0
    halfway_announced = False
    cooldown_until = 0.0

    def do_capture(reason):
        """Run the capture sequence. reason is 'manual' (SPACE) or 'auto'.
        Mutates: snapshot_idx, history, last_status_spoken,
        last_status_speak_time, stable_count, halfway_announced, cooldown_until.
        Returns True on success, False on failure."""
        nonlocal snapshot_idx, last_status_spoken, last_status_speak_time
        nonlocal stable_count, halfway_announced, cooldown_until
        print(f"\n[capture] snapshot {snapshot_idx + 1} ({reason}) ...")
        if reason == "manual":
            sounds.speak("Move clear.", volume=1.0, blocking=False)
            time.sleep(0.8)
            for _ in range(COUNTDOWN_PIPS):
                sounds.play('pips', volume=0.4)
                time.sleep(1.0)
        else:  # auto
            sounds.play('pips', volume=0.5)
            time.sleep(0.4)
        final = capture_one(cams, dictionary, params, detector)
        sounds.play('shutter', volume=1.0)
        if final is None:
            sounds.speak("No frames. Try again.",
                         volume=1.0, blocking=False)
            return False
        snapshot_idx += 1
        snap_dir = session_dir / f"snapshot_{snapshot_idx:03d}"
        t_frame, t_detect = final["tiger"]
        s_frame, s_detect = final["shark"]
        write_snapshot(snap_dir, snapshot_idx,
                       t_frame, s_frame, t_detect, s_detect)
        history.append((snapshot_idx, snap_dir))
        t_tgt = sorted(i for i in t_detect if i in TARGET_IDS)
        s_tgt = sorted(i for i in s_detect if i in TARGET_IDS)
        t_n = len(t_tgt); s_n = len(s_tgt)
        print(f"  tiger target ids: {t_tgt}    "
              f"shark target ids: {s_tgt}")
        print(f"  saved -> {snap_dir.name}")
        suffix = " Move plank." if reason == "auto" else ""
        sounds.speak(
            f"Snapshot {snapshot_idx}. Tiger {t_n}, shark {s_n}.{suffix}",
            volume=1.0, blocking=False,
        )
        last_status_spoken = (t_n, s_n)
        last_status_speak_time = time.time()
        stable_count = 0
        halfway_announced = False
        cooldown_until = time.time() + AUTO_COOLDOWN_S
        return True

    print("[capture] live preview running. "
          + ("AUTO mode: place plank, walk away, wait for capture. "
             if AUTO_CAPTURE_ENABLED else "")
          + "SPACE = manual capture, R = redo last, Q/ESC = quit.")
    sounds.speak(
        ("Plank capture ready. " +
         ("Auto mode is on. Place the plank, step away, and wait. "
          if AUTO_CAPTURE_ENABLED else
          "Move the plank into view. Press space to capture. ") +
         "Press R to redo, Q to quit."),
        volume=1.0, blocking=False,
    )
    try:
        while True:
            # Live preview: one fresh frame per camera per loop iteration.
            frames = {}
            detects = {}
            for n in CAMERA_NAMES:
                f = cams[n].get_frame(undistort=False)
                if f is not None:
                    frames[n] = f
                    detects[n] = detect_markers(f, dictionary, params, detector)

            previews = []
            per_cam_target_count = {}
            for n in CAMERA_NAMES:
                if n not in frames:
                    per_cam_target_count[n] = 0
                    continue
                vis = annotate(frames[n], detects[n], TARGET_IDS)
                ids_str = sorted(detects[n].keys())
                tgt_seen = sum(1 for i in TARGET_IDS if i in detects[n])
                per_cam_target_count[n] = tgt_seen
                status = (f"{n}: ids {ids_str}    "
                          f"target {tgt_seen}/{len(TARGET_IDS)}")
                cv.putText(vis, status, (20, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1.2,
                           (255, 255, 255), 4, cv.LINE_AA)
                cv.putText(vis, status, (20, 50),
                           cv.FONT_HERSHEY_SIMPLEX, 1.2,
                           (0, 0, 0), 2, cv.LINE_AA)
                previews.append(resize_for_preview(vis, PREVIEW_WIDTH))

            # Audio: announce detection count when it changes, rate-limited.
            t_count = per_cam_target_count.get("tiger", 0)
            s_count = per_cam_target_count.get("shark", 0)
            status_now = (t_count, s_count)
            now = time.time()
            if (status_now != last_status_spoken
                    and (now - last_status_speak_time) >= STATUS_SPEAK_MIN_GAP_S):
                sounds.speak(f"Tiger {t_count}. Shark {s_count}.",
                             volume=0.6, blocking=False)
                last_status_spoken = status_now
                last_status_speak_time = now

            # Auto-capture: detect plank stillness, trigger capture when
            # the plank has been stationary long enough AND we're past the
            # post-capture cooldown.
            auto_triggered = False
            if AUTO_CAPTURE_ENABLED:
                sees_enough = (t_count >= AUTO_MIN_TARGETS_SEEN
                               or s_count >= AUTO_MIN_TARGETS_SEEN)
                stable_now = (sees_enough
                              and is_frame_stable(detects, prev_detects,
                                                  TARGET_IDS, AUTO_STABLE_PIXELS))
                if stable_now:
                    stable_count += 1
                else:
                    stable_count = 0
                    halfway_announced = False
                if (not halfway_announced
                        and stable_count >= int(AUTO_STABLE_FRAMES_REQUIRED
                                                * AUTO_HOLDING_ANNOUNCE_FRACTION)):
                    sounds.speak("Holding.", volume=0.6, blocking=False)
                    halfway_announced = True
                if (stable_count >= AUTO_STABLE_FRAMES_REQUIRED
                        and time.time() >= cooldown_until):
                    auto_triggered = True

            if previews:
                stacked = np.vstack(previews)
                footer = (f"snapshots saved: {snapshot_idx}    "
                          f"SPACE=capture  R=redo  Q/ESC=quit")
                cv.putText(stacked, footer,
                           (20, stacked.shape[0] - 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9,
                           (255, 255, 255), 3, cv.LINE_AA)
                cv.putText(stacked, footer,
                           (20, stacked.shape[0] - 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9,
                           (0, 0, 0), 1, cv.LINE_AA)
                cv.imshow("plank capture", stacked)

            key = cv.waitKey(80) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                if history:
                    last_idx, last_dir = history.pop()
                    shutil.rmtree(last_dir, ignore_errors=True)
                    snapshot_idx -= 1
                    print(f"[redo] removed snapshot {last_idx} ({last_dir.name})")
                    sounds.speak("Snapshot deleted.",
                                 volume=1.0, blocking=False)
                else:
                    print("[redo] nothing to undo.")
                    sounds.speak("Nothing to delete.",
                                 volume=1.0, blocking=False)
                # do_capture would have reset prev_detects implicitly via
                # cooldown; on a delete we don't need to do anything.
                prev_detects = detects
                continue
            if key == ord(' '):
                do_capture("manual")
                prev_detects = detects
                continue
            if auto_triggered:
                do_capture("auto")

            # Track detection corners between frames for the stability check.
            prev_detects = detects

    except KeyboardInterrupt:
        print("\n[interrupt] cleaning up ...")
    finally:
        try:
            sounds.speak(f"Done. {snapshot_idx} snapshots saved.",
                         volume=1.0, blocking=True)
        except Exception:
            pass
        for c in cams.values():
            try:
                c.stop()
            except Exception:
                pass
        cv.destroyAllWindows()

    print(f"\n[done] {snapshot_idx} snapshots saved under {session_dir}")
    print("       Next: run script_diagnose_plank.py against this session "
          "(once written).")


if __name__ == "__main__":
    main()
