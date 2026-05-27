"""Quantitative wall-probe diagnostic.

For each capture, the script:
  - Grabs synchronised tiger + shark frames.
  - Detects the robot's ArUco marker in both.
  - Via the current calibration (PyLorex K, dist, R, t + ray-plane
    intersection at z = robot marker height), projects each camera's
    detection to a world (X, Y) position.
  - Finds the nearest wall point from arena_features.npz (loaded from
    the 3PiRobot arena snapshot you point it at) and computes the
    perpendicular distance from the robot CENTRE to that wall point.
  - Expected distance = robot radius (48 mm) when the robot edge is
    flush against the wall. Bias = (distance - 48).

Auto-capture: when the robot's detection is stable for
AUTO_STABLE_FRAMES_REQUIRED preview frames in a row (corners not
moving by more than AUTO_STABLE_PIXELS px), the script triggers a
capture by itself — same pattern as script_capture_plank.py. Place
the robot edge-flush against a wall, walk away, wait for the audio
result.

After ~6-10 wall placements (varied positions and walls), the script
saves a CSV and a top-down plot showing per-camera robot positions
on the arena floor + walls, coloured by bias.

A/B testing different calibrations: each session's raw frames +
detection corners are saved, and the analysis path is wrapped in
diagnose_session(). To re-run diagnostics with a different
calibration (e.g., after swapping pose_shark files), call
diagnose_session(session_dir, save_suffix="...") from another script
or a Python REPL.

Usage:
    python script_capture_wall_probe.py [--arena ARENA_NAME]

Defaults: ARENA_NAME read from Settings.environment_arena... or the
most recent AcquisitionArenas/* directory.
"""

import argparse
import csv
import json
import shutil
import sys
import time
from pathlib import Path

import cv2 as cv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

from LorexLib import CalibIO, Lorex, Settings, Sound


# ---------------- SETTINGS ----------------
ROBOT_ARUCO_ID = 1            # Robot01's marker (per Control_code/Library/Settings.aruco_id)
ROBOT_RADIUS_MM = 48.0        # Pololu 3pi+ 2040: ~96 mm diameter / 2

ARENA_FEATURES_3PIROBOT_DIR = Path(
    "/home/dieter/Dropbox/PythonRepos/3PiRobot/Control_code/AcquisitionArenas")

CAMERA_NAMES = ("tiger", "shark")
N_GRABS_PER_CAPTURE = 3        # median over N frames per camera per capture
GRAB_INTERVAL_S = 0.1
PREVIEW_WIDTH = 800

SAVE_ROOT = Path(__file__).resolve().parent / "Calibration" / "WallProbeSnapshots"

# Auto-trigger settings (mirrors the plank-capture script).
AUTO_CAPTURE_ENABLED = True
AUTO_STABLE_FRAMES_REQUIRED = 20     # ~2 s at the preview loop's ~10 Hz
AUTO_STABLE_PIXELS = 1.5              # max per-frame corner motion to count "stable"
AUTO_COOLDOWN_S = 5.0                 # min seconds between auto-captures
AUTO_HOLDING_ANNOUNCE_FRACTION = 0.5

STATUS_SPEAK_MIN_GAP_S = 1.5
# ------------------------------------------


def get_aruco_handles():
    dictionary = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_4X4_1000)
    try:
        params = cv.aruco.DetectorParameters()
    except AttributeError:
        params = cv.aruco.DetectorParameters_create()
    detector = (cv.aruco.ArucoDetector(dictionary, params)
                if hasattr(cv.aruco, "ArucoDetector") else None)
    return dictionary, params, detector


def detect_markers(frame_bgr, dictionary, params, detector):
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


def project_corners_to_world(corners_px, K, dist, R, t, z_mm):
    """Same projection model as Lorex.get_aruco's ray-plane step."""
    pts = np.asarray(corners_px, dtype=np.float32).reshape(-1, 1, 2)
    undist = cv.undistortPoints(pts, K, dist).reshape(-1, 2)
    R_wfc = R.T
    C = -R.T @ np.asarray(t).reshape(3)
    world_xy = []
    for u, v in undist:
        d_world = R_wfc @ np.array([float(u), float(v), 1.0])
        if abs(d_world[2]) < 1e-9:
            world_xy.append([np.nan, np.nan])
            continue
        s = (z_mm - C[2]) / d_world[2]
        P = C + s * d_world
        world_xy.append([float(P[0]), float(P[1])])
    return np.array(world_xy)


def load_arena_walls(arena_dir):
    """Locate arena_features.npz under arena_dir (or its most recent
    env_* sub-dir). Returns (wall_xy Nx2, kdtree)."""
    arena_dir = Path(arena_dir)
    candidates = list(arena_dir.glob("env_*/arena_features.npz"))
    if not candidates:
        candidates = [arena_dir / "arena_features.npz"]
    candidates = [c for c in candidates if c.exists()]
    if not candidates:
        raise FileNotFoundError(
            f"No arena_features.npz under {arena_dir}")
    candidates.sort()
    feat_path = candidates[-1]
    print(f"[walls] loading {feat_path}")
    data = np.load(feat_path)
    kind = data["kind"]
    wx = data["x_mm"][kind == 0]
    wy = data["y_mm"][kind == 0]
    wall_xy = np.column_stack([wx, wy])
    return wall_xy, cKDTree(wall_xy), feat_path


def find_latest_arena(default_root=ARENA_FEATURES_3PIROBOT_DIR):
    """If no arena name given, pick the most-recently-modified one."""
    if not default_root.exists():
        raise FileNotFoundError(f"{default_root} does not exist")
    arenas = [p for p in default_root.iterdir() if p.is_dir()]
    if not arenas:
        raise FileNotFoundError(f"No arenas under {default_root}")
    arenas.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return arenas[0]


def nearest_wall(wall_kdtree, xy):
    """Return (distance_mm, wall_point_xy) for the wall point nearest xy."""
    d, idx = wall_kdtree.query(xy)
    return float(d), wall_kdtree.data[idx]


def annotate(frame_bgr, detections, target_id, robot_world_xy_per_cam,
             wall_distance_mm_per_cam, cam_name):
    """Draw the robot marker if detected + per-camera bias text."""
    out = frame_bgr.copy()
    for mid, corners in detections.items():
        col = (0, 255, 0) if mid == target_id else (180, 180, 180)
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv.polylines(out, [pts], True, col, 2)
        cx, cy = corners.mean(axis=0).astype(int)
        cv.putText(out, str(mid), (int(cx) - 15, int(cy) + 8),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, col, 2, cv.LINE_AA)
    # Status line.
    rxy = robot_world_xy_per_cam.get(cam_name)
    d = wall_distance_mm_per_cam.get(cam_name)
    if rxy is not None:
        bias = (d - ROBOT_RADIUS_MM) if d is not None else None
        if bias is not None:
            txt = (f"{cam_name}: ({rxy[0]:+.0f}, {rxy[1]:+.0f}) mm   "
                   f"wall_d={d:.0f}  bias={bias:+.0f} mm")
        else:
            txt = f"{cam_name}: ({rxy[0]:+.0f}, {rxy[1]:+.0f}) mm   wall?"
    else:
        txt = f"{cam_name}: robot not detected"
    cv.putText(out, txt, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0,
               (255, 255, 255), 4, cv.LINE_AA)
    cv.putText(out, txt, (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.0,
               (0, 0, 0), 2, cv.LINE_AA)
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
        name = f"wall_{idx:04d}_{ts}"
        if name not in existing:
            return root / name
        idx += 1


def is_frame_stable(detects_t, detects_s, prev_t, prev_s, target_id, threshold_px):
    """True iff the robot marker is visible in at least one camera and
    its corners haven't moved by > threshold_px in any coord from the
    previous frame."""
    cur_t = detects_t.get(target_id)
    cur_s = detects_s.get(target_id)
    if cur_t is None and cur_s is None:
        return False
    for cur, prev in [(cur_t, prev_t), (cur_s, prev_s)]:
        if cur is None:
            continue
        if prev is None:
            return False
        if float(np.max(np.abs(cur - prev))) > threshold_px:
            return False
    return True


def capture_one(cams, dictionary, params, detector, target_id):
    """Grab N frames per camera, median-frame and median per-corner of
    the target marker. Returns {cam: (frame, corners_or_None)}."""
    grabs = {n: [] for n in CAMERA_NAMES}
    detect_lists = {n: [] for n in CAMERA_NAMES}
    for _ in range(N_GRABS_PER_CAPTURE):
        for n in CAMERA_NAMES:
            f = cams[n].get_frame(undistort=False)
            if f is not None:
                grabs[n].append(f)
                detect_lists[n].append(detect_markers(f, dictionary, params, detector))
        time.sleep(GRAB_INTERVAL_S)
    out = {}
    for n in CAMERA_NAMES:
        if not grabs[n]:
            return None
        median_frame = np.median(np.stack(grabs[n], axis=0), axis=0).astype(np.uint8)
        all_corners = [d[target_id] for d in detect_lists[n] if target_id in d]
        if all_corners:
            median_corners = np.median(np.stack(all_corners, axis=0), axis=0)
        else:
            median_corners = None
        out[n] = (median_frame, median_corners)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arena", type=str, default=None,
                    help="path or name under 3PiRobot/AcquisitionArenas/ "
                         "to use for walls (default: most recent)")
    args = ap.parse_args()

    # Resolve arena.
    if args.arena is None:
        arena_dir = find_latest_arena()
    else:
        p = Path(args.arena)
        arena_dir = p if p.is_absolute() else ARENA_FEATURES_3PIROBOT_DIR / args.arena
    wall_xy, wall_kdtree, feat_path = load_arena_walls(arena_dir)
    print(f"[arena] {arena_dir.name}    walls: {len(wall_xy)} samples")

    # Session.
    session_dir = next_session_dir(SAVE_ROOT)
    session_dir.mkdir(parents=True, exist_ok=True)
    print(f"[session] {session_dir}")

    # Load camera calibrations (uses Settings.shark2tiger_delta_{x,y}).
    cal = {}
    for cam in CAMERA_NAMES:
        b = CalibIO.load_pose_bundle(cam)
        cal[cam] = {
            "K": np.asarray(b["K"], dtype=np.float64),
            "dist": np.asarray(b["dist"], dtype=np.float64),
            "R": np.asarray(b["R"], dtype=np.float64),
            "t": np.asarray(b["t"], dtype=np.float64).reshape(3),
        }
    marker_z = float(Settings.marker_height_mm)
    shark_delta = np.array([Settings.shark2tiger_delta_x,
                            Settings.shark2tiger_delta_y])
    print(f"[cal] marker_height_mm={marker_z}    "
          f"shark2tiger_delta={tuple(shark_delta)}")

    # Cameras + audio.
    print("[setup] opening cameras ...")
    cams = {n: Lorex.LorexCamera(n) for n in CAMERA_NAMES}
    for c in cams.values():
        try:
            c.wait_ready(timeout=5.0)
        except Exception:
            pass
    dictionary, params, detector = get_aruco_handles()
    sounds = Sound.SoundPlayer()

    # Save meta.
    meta = {
        "session_dir": str(session_dir),
        "arena_features_path": str(feat_path),
        "n_wall_samples": int(len(wall_xy)),
        "cameras": list(CAMERA_NAMES),
        "robot_aruco_id": int(ROBOT_ARUCO_ID),
        "robot_radius_mm": float(ROBOT_RADIUS_MM),
        "marker_height_mm": float(marker_z),
        "shark2tiger_delta": [float(shark_delta[0]), float(shark_delta[1])],
        "auto_capture_enabled": bool(AUTO_CAPTURE_ENABLED),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(session_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    # State.
    snapshot_idx = 0
    history = []  # (idx, path)
    prev_corners = {n: None for n in CAMERA_NAMES}
    stable_count = 0
    halfway_announced = False
    cooldown_until = 0.0
    last_status_spoken_t = (-99, -99, -99)
    last_status_speak_time = 0.0
    rows = []

    print("[probe] live preview running. "
          + ("AUTO mode: park robot edge-flush against wall, walk away, wait. "
             if AUTO_CAPTURE_ENABLED else "")
          + "SPACE = manual capture, R = redo last, Q/ESC = quit.")
    sounds.speak(
        "Wall probe ready. "
        + ("Auto mode is on. Park the robot against a wall, step away, and wait. "
           if AUTO_CAPTURE_ENABLED else
           "Press space to capture. ")
        + "Press R to redo, Q to quit.",
        volume=1.0, blocking=False,
    )

    def project_and_bias(corners_px, cam):
        """Project a single 4x2 corner set to world XY, find nearest wall,
        return (world_xy, wall_distance_mm) or (None, None)."""
        if corners_px is None:
            return None, None
        world_corners = project_corners_to_world(
            corners_px, cal[cam]["K"], cal[cam]["dist"],
            cal[cam]["R"], cal[cam]["t"], marker_z)
        if np.any(np.isnan(world_corners)):
            return None, None
        centre = world_corners.mean(axis=0)
        if cam == "shark":
            centre = centre + shark_delta
        d, _ = nearest_wall(wall_kdtree, centre)
        return centre, d

    def do_capture(reason):
        nonlocal snapshot_idx, stable_count, halfway_announced, cooldown_until
        nonlocal last_status_spoken_t, last_status_speak_time
        print(f"\n[capture] snapshot {snapshot_idx + 1} ({reason}) ...")
        if reason == "manual":
            sounds.speak("Move clear.", volume=1.0, blocking=False)
            time.sleep(0.8)
            for _ in range(2):
                sounds.play('pips', volume=0.4)
                time.sleep(1.0)
        else:
            sounds.play('pips', volume=0.5)
            time.sleep(0.4)
        final = capture_one(cams, dictionary, params, detector, ROBOT_ARUCO_ID)
        sounds.play('shutter', volume=1.0)
        if final is None:
            sounds.speak("No frames. Try again.", volume=1.0, blocking=False)
            return
        snapshot_idx += 1
        snap_dir = session_dir / f"snapshot_{snapshot_idx:03d}"
        snap_dir.mkdir(parents=True, exist_ok=True)
        # Save frames + detection.
        per_cam_payload = {}
        per_cam_bias = {}
        for cam in CAMERA_NAMES:
            frame, corners = final[cam]
            cv.imwrite(str(snap_dir / f"frame_{cam}.jpg"), frame)
            entry = {"frame_size_WH": [int(frame.shape[1]), int(frame.shape[0])]}
            if corners is not None:
                entry["robot_corners_px"] = corners.tolist()
                xy, d = project_and_bias(corners, cam)
                if xy is not None:
                    bias = d - ROBOT_RADIUS_MM
                    entry["world_xy_mm"] = [float(xy[0]), float(xy[1])]
                    entry["wall_distance_mm"] = float(d)
                    entry["bias_mm"] = float(bias)
                    per_cam_bias[cam] = bias
                    rows.append({
                        "snap_id": snapshot_idx,
                        "camera": cam,
                        "robot_x_mm": float(xy[0]),
                        "robot_y_mm": float(xy[1]),
                        "wall_distance_mm": float(d),
                        "bias_mm": float(bias),
                    })
            else:
                entry["robot_corners_px"] = None
            per_cam_payload[cam] = entry
        with open(snap_dir / "detections.json", "w") as f:
            json.dump({
                "snapshot": snapshot_idx,
                "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "cameras": per_cam_payload,
            }, f, indent=2)
        history.append((snapshot_idx, snap_dir))
        # Announce result.
        if per_cam_bias:
            parts = [f"{cam} {int(round(b)):+d}" for cam, b in per_cam_bias.items()]
            announce = f"Snapshot {snapshot_idx}. Bias " + ", ".join(parts) + ". Move robot."
        else:
            announce = f"Snapshot {snapshot_idx}. Robot not detected. Move robot."
        sounds.speak(announce, volume=1.0, blocking=False)
        print(f"  bias per camera: {per_cam_bias}")
        # Reset state.
        stable_count = 0
        halfway_announced = False
        cooldown_until = time.time() + AUTO_COOLDOWN_S
        last_status_speak_time = time.time()

    try:
        while True:
            frames = {}
            detects = {}
            for n in CAMERA_NAMES:
                f = cams[n].get_frame(undistort=False)
                if f is not None:
                    frames[n] = f
                    detects[n] = detect_markers(f, dictionary, params, detector)
                else:
                    detects[n] = {}

            # Per-camera live projection + nearest-wall bias.
            robot_xy = {}
            wall_d = {}
            for cam in CAMERA_NAMES:
                if frames.get(cam) is None:
                    continue
                corners = detects[cam].get(ROBOT_ARUCO_ID)
                xy, d = project_and_bias(corners, cam) if corners is not None else (None, None)
                if xy is not None:
                    robot_xy[cam] = xy
                    wall_d[cam] = d

            # Compose preview.
            previews = []
            for cam in CAMERA_NAMES:
                if cam not in frames:
                    continue
                vis = annotate(frames[cam], detects[cam], ROBOT_ARUCO_ID,
                               robot_xy, wall_d, cam)
                previews.append(resize_for_preview(vis, PREVIEW_WIDTH))

            # Live bias status (announce on change).
            t_b = (int(robot_xy["tiger"][0]) if "tiger" in robot_xy else None,
                   int(robot_xy["tiger"][1]) if "tiger" in robot_xy else None,
                   int(wall_d["tiger"] - ROBOT_RADIUS_MM) if "tiger" in wall_d else None)
            s_b = (int(robot_xy["shark"][0]) if "shark" in robot_xy else None,
                   int(robot_xy["shark"][1]) if "shark" in robot_xy else None,
                   int(wall_d["shark"] - ROBOT_RADIUS_MM) if "shark" in wall_d else None)
            status_now = (t_b[2], s_b[2], int(robot_xy.get("tiger", [0])[0]) // 100)
            now = time.time()
            if (status_now != last_status_spoken_t
                    and (now - last_status_speak_time) >= STATUS_SPEAK_MIN_GAP_S):
                if t_b[2] is not None or s_b[2] is not None:
                    parts = []
                    if t_b[2] is not None:
                        parts.append(f"tiger bias {t_b[2]:+d}")
                    if s_b[2] is not None:
                        parts.append(f"shark bias {s_b[2]:+d}")
                    sounds.speak(". ".join(parts), volume=0.55, blocking=False)
                last_status_spoken_t = status_now
                last_status_speak_time = now

            # Auto-capture.
            auto_triggered = False
            if AUTO_CAPTURE_ENABLED:
                stable_now = is_frame_stable(
                    detects.get("tiger", {}), detects.get("shark", {}),
                    prev_corners["tiger"], prev_corners["shark"],
                    ROBOT_ARUCO_ID, AUTO_STABLE_PIXELS)
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
                footer = (f"snapshots: {snapshot_idx}   "
                          f"SPACE=capture  R=redo  Q/ESC=quit")
                cv.putText(stacked, footer, (20, stacked.shape[0] - 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3, cv.LINE_AA)
                cv.putText(stacked, footer, (20, stacked.shape[0] - 25),
                           cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv.LINE_AA)
                cv.imshow("wall probe", stacked)

            key = cv.waitKey(80) & 0xFF
            if key in (27, ord('q'), ord('Q')):
                break
            if key in (ord('r'), ord('R')):
                if history:
                    last_idx, last_dir = history.pop()
                    shutil.rmtree(last_dir, ignore_errors=True)
                    rows[:] = [r for r in rows if r["snap_id"] != last_idx]
                    snapshot_idx -= 1
                    print(f"[redo] removed snapshot {last_idx}")
                    sounds.speak("Snapshot deleted.", volume=1.0, blocking=False)
                else:
                    sounds.speak("Nothing to delete.", volume=1.0, blocking=False)
                prev_corners = {n: detects.get(n, {}).get(ROBOT_ARUCO_ID)
                                for n in CAMERA_NAMES}
                continue
            if key == ord(' '):
                do_capture("manual")
            elif auto_triggered:
                do_capture("auto")

            prev_corners = {n: detects.get(n, {}).get(ROBOT_ARUCO_ID)
                            for n in CAMERA_NAMES}

    except KeyboardInterrupt:
        print("\n[interrupt] cleaning up ...")
    finally:
        try:
            sounds.speak(f"Done. {snapshot_idx} snapshots.", blocking=True)
        except Exception:
            pass
        for c in cams.values():
            try: c.stop()
            except Exception: pass
        cv.destroyAllWindows()

    # Final summary, CSV + plot.
    if not rows:
        print(f"\n[done] {snapshot_idx} snapshots saved, but no usable "
              "(robot-detected) measurements to summarise.")
        return
    csv_path = session_dir / "wall_probe.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {csv_path}")

    biases = np.array([r["bias_mm"] for r in rows])
    print(f"\n=== summary across {len(rows)} measurements ===")
    print(f"  bias overall: mean {biases.mean():+.1f}    "
          f"std {biases.std():.1f}    |max| {np.max(np.abs(biases)):.1f}")
    for cam in CAMERA_NAMES:
        sub = np.array([r["bias_mm"] for r in rows if r["camera"] == cam])
        if not len(sub):
            continue
        print(f"  [{cam}] n={len(sub):3d}    mean {sub.mean():+7.1f}    "
              f"std {sub.std():5.1f}    range [{sub.min():+7.1f}, {sub.max():+7.1f}]")

    # Plot.
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(wall_xy[:, 0], wall_xy[:, 1], s=2, c="0.7", alpha=0.5, label="walls")
    cmap = plt.get_cmap("RdBu_r")
    vmax = max(50.0, np.max(np.abs(biases)))
    for cam, marker_shape in zip(CAMERA_NAMES, ["o", "s"]):
        sub = [r for r in rows if r["camera"] == cam]
        if not sub:
            continue
        xs = [r["robot_x_mm"] for r in sub]
        ys = [r["robot_y_mm"] for r in sub]
        cs = [r["bias_mm"] for r in sub]
        sc = ax.scatter(xs, ys, c=cs, cmap=cmap, vmin=-vmax, vmax=+vmax,
                        s=140, marker=marker_shape, edgecolor="k", linewidth=0.5,
                        label=f"{cam} ({marker_shape})")
    plt.colorbar(sc, ax=ax, label="bias (mm)")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(f"Wall-probe bias — {session_dir.name}\n"
                 "robot position coloured by (wall_dist - robot_radius)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plot_path = session_dir / "wall_probe_plot.png"
    fig.savefig(plot_path, dpi=120)
    print(f"[plot] {plot_path}")


if __name__ == "__main__":
    main()
