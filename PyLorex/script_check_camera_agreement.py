"""
Diagnostic for two-camera floor-projection agreement.

Static section: prints, for tiger and shark, the homography pose quality
(reprojection RMSE, PnP camera height, optical-axis tilt off vertical,
camera nadir location). Large disagreements here flag a calibration issue
before any robot motion.

Live section: opens both cameras and walks you through N_POSITIONS
manual placements with audio cues (same style as the intrinsic
calibration script). For each position it speaks "Position k of N",
waits SECONDS_PER_POSITION while you place the robot, plays a pip,
samples (median over several frames), plays a shutter, then announces
the result. Ctrl-C stops early. On exit, writes a CSV and shows:
  - dy = y_tiger - y_shark vs x_mean
  - dx = x_tiger - x_shark vs x_mean
  - same two vs y_mean
  - spatial map (mean position) coloured by |dy|

A position-dependent dy(x) is the fingerprint of a tilt error in at least
one camera's extrinsics. A constant offset is a board-placement mismatch.
"""

import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from LorexLib import Lorex, Settings, Sound, Utils

TARGET_ID = 1                 # Robot01 marker; change if probing a different tag
N_POSITIONS = 15              # how many placements to walk through
SECONDS_PER_POSITION = 10     # time to lift + place + step away
FRAMES_PER_SAMPLE = 3         # frames to median over per placement
FRAME_INTERVAL_S = 0.2        # spacing between those frames (lets RTSP advance)
CSV_PATH = "Temp/camera_agreement.csv"
PLOT_PATH = "Temp/camera_agreement.png"


def print_pose_summary(camera_name):
    paths = Utils.get_calibration_paths(camera_name)
    pose_json = paths["pose_json"]
    if not os.path.exists(pose_json):
        print(f"  [{camera_name}] no pose_json at {pose_json}")
        return None

    with open(pose_json) as f:
        s = json.load(f)

    R = np.asarray(s["R_pnp"])
    t = np.asarray(s["t_pnp"]).reshape(3)
    C_pnp = -R.T @ t                          # PnP-derived centre in board frame
    # Camera optical axis in board frame (R^T @ +z_cam = third row of R)
    z_cam_in_board = R[2, :]
    cos_off_vertical = float(np.clip(abs(z_cam_in_board[2]), -1.0, 1.0))
    tilt_off_vertical_deg = float(np.degrees(np.arccos(cos_off_vertical)))

    # If c_measured_{cam}.json exists, it overrides the PnP C in Lorex.get_aruco
    # (per-camera board frame, same frame as PnP). Show both so the printed
    # summary matches what the live tracker is actually using.
    c_meas_path = paths.get("c_measured_json")
    C_meas = None
    if c_meas_path and os.path.exists(c_meas_path):
        with open(c_meas_path) as f:
            cj = json.load(f)
        C_meas = np.array([float(cj["Cx_mm"]),
                           float(cj["Cy_mm"]),
                           float(cj["Cz_mm"])])

    print(f"  [{camera_name}]")
    print(f"     reproj RMSE : {s['reprojection_rmse_px']:.3f} px")
    print(f"     PnP    height/nadir : {float(C_pnp[2]):7.1f} mm  "
          f"({float(C_pnp[0]):+7.1f}, {float(C_pnp[1]):+7.1f})")
    if C_meas is not None:
        d = C_meas - C_pnp
        print(f"     meas   height/nadir : {float(C_meas[2]):7.1f} mm  "
              f"({float(C_meas[0]):+7.1f}, {float(C_meas[1]):+7.1f})   "
              f"<-- used by live tracker")
        print(f"     meas - PnP          : "
              f"Δz {d[2]:+7.1f}    "
              f"Δxy ({d[0]:+7.1f}, {d[1]:+7.1f}) mm  "
              f"|Δxy|={float(np.hypot(d[0], d[1])):.1f}")
    else:
        print("     (no c_measured_{cam}.json — live tracker uses PnP C)")
    print(f"     optical-axis tilt off vertical: {tilt_off_vertical_deg:.2f} deg")
    C_active = C_meas if C_meas is not None else C_pnp
    return {"name": camera_name, "C": C_active, "C_pnp": C_pnp, "C_meas": C_meas,
            "rmse": s["reprojection_rmse_px"], "tilt_deg": tilt_off_vertical_deg}


def find_target(detections, target_id):
    for det in detections:
        if det.get("id") == target_id and det.get("floor_xy_mm") is not None:
            return det["floor_xy_mm"]
    return None


def sample_once(tiger, shark, target_id):
    """Median of FRAMES_PER_SAMPLE detections per camera, with shark's
    floor_xy_mm shifted into the unified (= tiger) frame via
    Settings.shark2tiger_delta_{x,y}. Returns
    (x_tiger, y_tiger, x_shark_unified, y_shark_unified, n_t, n_s).
    Cross-camera dx/dy is only meaningful when both coords are in the
    same frame; without this shift dy ≈ shark2tiger_delta_y ≈ -1400 mm
    swamps any real disagreement."""
    dx_unify = float(Settings.shark2tiger_delta_x)
    dy_unify = float(Settings.shark2tiger_delta_y)
    xs_t, ys_t, xs_s, ys_s = [], [], [], []
    for _ in range(FRAMES_PER_SAMPLE):
        dets_t, _ = tiger.get_aruco(draw=False, draw_world=False)
        dets_s, _ = shark.get_aruco(draw=False, draw_world=False)
        xy_t = find_target(dets_t, target_id)
        xy_s = find_target(dets_s, target_id)
        if xy_t is not None:
            xs_t.append(xy_t[0]); ys_t.append(xy_t[1])
        if xy_s is not None:
            xs_s.append(xy_s[0] + dx_unify)
            ys_s.append(xy_s[1] + dy_unify)
        time.sleep(FRAME_INTERVAL_S)
    n_t, n_s = len(xs_t), len(xs_s)
    if n_t == 0 or n_s == 0:
        return None
    return (float(np.median(xs_t)), float(np.median(ys_t)),
            float(np.median(xs_s)), float(np.median(ys_s)),
            n_t, n_s)


def live_loop(target_id):
    print(f"\n[manual] target_id = {target_id}, {N_POSITIONS} positions, "
          f"{SECONDS_PER_POSITION}s per position.\n")

    tiger = Lorex.LorexCamera("tiger")
    shark = Lorex.LorexCamera("shark")
    tiger.wait_ready(timeout=5.0)
    shark.wait_ready(timeout=5.0)
    sounds = Sound.SoundPlayer()

    sounds.speak(
        f"Starting two camera agreement check. {N_POSITIONS} positions, "
        f"{SECONDS_PER_POSITION} seconds each. Place the robot somewhere "
        "both cameras can see it and step away.",
        volume=1.0,
    )
    time.sleep(1.0)

    rows = []
    try:
        for k in range(1, N_POSITIONS + 1):
            sounds.speak(f"Position {k} of {N_POSITIONS}", volume=1.0)
            print(f"\n  [{k:02d}/{N_POSITIONS}] place the robot...")
            # Give the user time to place; final 3s as discrete pips.
            wait_total = max(0.0, SECONDS_PER_POSITION - 3.0)
            time.sleep(wait_total)
            for _ in range(3):
                sounds.play('pips', volume=0.25)
                time.sleep(1.0)
            result = sample_once(tiger, shark, target_id)
            sounds.play('shutter', volume=1.0)

            if result is None:
                sounds.speak("No marker detected. Skipping.", volume=1.0)
                print("        no marker detected in either camera; skipped.")
                continue
            x_t, y_t, x_s, y_s, n_t, n_s = result
            x_m = 0.5 * (x_t + x_s); y_m = 0.5 * (y_t + y_s)
            dx = x_t - x_s; dy = y_t - y_s
            rows.append((time.time(), x_t, y_t, x_s, y_s, x_m, y_m, dx, dy))
            print(f"        tiger=({x_t:+7.1f},{y_t:+7.1f}) n={n_t}/{FRAMES_PER_SAMPLE}  "
                  f"shark=({x_s:+7.1f},{y_s:+7.1f}) n={n_s}/{FRAMES_PER_SAMPLE}")
            print(f"        dx={dx:+6.1f} mm  dy={dy:+6.1f} mm  "
                  f"mean=({x_m:+7.1f},{y_m:+7.1f})")
            sounds.speak(
                f"Delta y {int(round(dy))} millimeters. "
                f"Delta x {int(round(dx))} millimeters.",
                volume=1.0,
            )
    except KeyboardInterrupt:
        print("\n[manual] interrupted by user.")
    finally:
        tiger.stop()
        shark.stop()

    if not rows:
        print("\n[manual] no samples recorded; nothing to save.")
        return

    sounds.speak(f"Done. {len(rows)} positions recorded.", volume=1.0)

    arr = np.array(rows, dtype=float)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    np.savetxt(
        CSV_PATH, arr, delimiter=",",
        header="t,x_tiger,y_tiger,x_shark,y_shark,x_mean,y_mean,dx,dy",
        comments="",
    )
    print(f"\n[live] {len(rows)} samples logged to {CSV_PATH}")

    # Summary
    dx = arr[:, 7]
    dy = arr[:, 8]
    print(f"  dx: mean {dx.mean():+.1f} mm  std {dx.std():.1f}  "
          f"range [{dx.min():+.1f}, {dx.max():+.1f}]")
    print(f"  dy: mean {dy.mean():+.1f} mm  std {dy.std():.1f}  "
          f"range [{dy.min():+.1f}, {dy.max():+.1f}]")

    plot(arr)


def plot(arr):
    x_m = arr[:, 5]; y_m = arr[:, 6]
    dx  = arr[:, 7]; dy  = arr[:, 8]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    ax = axes[0, 0]
    ax.scatter(x_m, dy, s=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x_mean (mm)"); ax.set_ylabel("dy = y_tiger - y_shark (mm)")
    ax.set_title("dy vs x  (tilt-error fingerprint)")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.scatter(y_m, dx, s=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("y_mean (mm)"); ax.set_ylabel("dx = x_tiger - x_shark (mm)")
    ax.set_title("dx vs y")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.scatter(x_m, dx, s=8)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("x_mean (mm)"); ax.set_ylabel("dx (mm)")
    ax.set_title("dx vs x")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    mag = np.hypot(dx, dy)
    sc = ax.scatter(x_m, y_m, c=mag, s=14, cmap="viridis")
    plt.colorbar(sc, ax=ax, label="|delta| (mm)")
    ax.set_xlabel("x_mean (mm)"); ax.set_ylabel("y_mean (mm)")
    ax.set_title("Spatial map of disagreement magnitude")
    ax.set_aspect("equal"); ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=120)
    print(f"[plot] saved {PLOT_PATH}")
    plt.show()


if __name__ == "__main__":
    print("=== Static pose summary ===")
    print_pose_summary("tiger")
    print_pose_summary("shark")

    live_loop(TARGET_ID)
