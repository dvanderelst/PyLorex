"""Diagnose the current calibration against the plank's known geometry.

For each snapshot captured by script_capture_plank.py:
  - Per camera that detected >= 2 of the target markers, project each
    detected marker's 4 image corners to world (X, Y) at z = 25 mm via
    the current K + dist + R + t calibration (ray-plane intersection,
    same projection model the runtime tracker now uses). Marker centre
    is the centroid of the four projected corners.
  - Compute pairwise distances between marker centres and compare to
    the tape-measured 500 / 500 / 1000 mm.

Outputs:
  - A CSV of every measurement (snap_id, camera, marker_pair,
    expected_mm, reported_mm, residual_mm, x_mean, y_mean).
  - A console summary per camera and per marker pair.
  - A four-panel plot (residual histogram, residual by pair/camera,
    residual vs arena position spatial map, residual vs plank
    orientation if recoverable).

This is intentionally non-invasive: it doesn't touch the calibration,
just reports what the current calibration says about a rigid plank we
measured by tape. If residuals are tiny (< ~30 mm), the current
calibration is already good and the planned bundle-adjustment fix
isn't necessary. If residuals are large and structured, the pattern
tells us which fix to pursue.

Usage:
    python script_diagnose_plank.py [session_dir]
If session_dir is omitted, uses the most recent under
PyLorex/Calibration/PlankSnapshots/.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from LorexLib import CalibIO, Settings


TARGET_IDS = (75, 76, 77)
KNOWN_DISTANCES_MM = {
    (75, 76): 500.0,
    (76, 77): 500.0,
    (75, 77): 1000.0,
}
PLANK_Z_MM = 25.0  # plank surface height above the floor (where markers sit)

PLANK_ROOT = Path(__file__).resolve().parent / "Calibration" / "PlankSnapshots"


def find_latest_session(root):
    if not root.exists():
        sys.exit(f"[error] no session root at {root}")
    sessions = sorted(p for p in root.iterdir() if p.is_dir())
    if not sessions:
        sys.exit(f"[error] no sessions under {root}")
    return sessions[-1]


def load_snapshot(snap_dir):
    with open(snap_dir / "detections.json") as f:
        return json.load(f)


def project_corners_to_world(corners_px, K, dist, R, t, z_mm):
    """Project an Nx2 set of pixel coordinates to world (X, Y) at z = z_mm
    via undistort + ray-plane intersection using extrinsics (R, t).

    Convention: R, t from cv.solvePnP, i.e. X_cam = R @ X_world + t.
    Camera centre in world: C = -R.T @ t. Ray direction in world is
    R.T @ d_cam_norm, where d_cam_norm is the undistorted normalised
    pixel ray (x_n, y_n, 1)."""
    pts = np.array(corners_px, dtype=np.float32).reshape(-1, 1, 2)
    undist = cv.undistortPoints(pts, K, dist).reshape(-1, 2)
    R_wfc = R.T
    C = -R.T @ np.asarray(t).reshape(3)
    world_xy = []
    for u, v in undist:
        d_cam = np.array([float(u), float(v), 1.0])
        d_world = R_wfc @ d_cam
        if abs(d_world[2]) < 1e-9:
            world_xy.append([float("nan"), float("nan")])
            continue
        s = (z_mm - C[2]) / d_world[2]
        P = C + s * d_world
        world_xy.append([float(P[0]), float(P[1])])
    return np.array(world_xy)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=str, nargs="?",
                    help="path to PlankSnapshots/<session> (default: latest)")
    args = ap.parse_args()

    if args.session_dir is None:
        session_dir = find_latest_session(PLANK_ROOT)
        print(f"[session] using latest: {session_dir.name}")
    else:
        session_dir = Path(args.session_dir).expanduser().resolve()
        if not session_dir.exists():
            sys.exit(f"[error] not found: {session_dir}")
    print(f"[session] {session_dir}")

    with open(session_dir / "meta.json") as f:
        meta = json.load(f)
    cams = meta.get("cameras") or ["tiger", "shark"]
    plank_z = float(meta.get("plank_geometry_mm", {}).get(
        "marker_z_above_floor_mm", PLANK_Z_MM))
    print(f"[session] cameras: {cams}    plank z above floor: {plank_z} mm")

    # Load calibration bundles for each camera.
    cal = {}
    for cam in cams:
        b = CalibIO.load_pose_bundle(cam)
        cal[cam] = {
            "K": np.asarray(b["K"], dtype=np.float64),
            "dist": np.asarray(b["dist"], dtype=np.float64),
            "R": np.asarray(b["R"], dtype=np.float64),
            "t": np.asarray(b["t"], dtype=np.float64).reshape(3),
        }

    snap_dirs = sorted(p for p in session_dir.iterdir()
                       if p.is_dir() and p.name.startswith("snapshot_"))
    print(f"[session] {len(snap_dirs)} snapshots")

    # For each snapshot, for each camera that saw enough markers, compute
    # reported pairwise distances and residuals.
    rows = []
    for sd in snap_dirs:
        snap = load_snapshot(sd)
        snap_id = int(snap["snapshot"])
        for cam in cams:
            cam_data = snap["cameras"].get(cam, {})
            markers = {int(m["id"]): np.array(m["corners_px"], dtype=np.float64)
                       for m in cam_data.get("markers", [])
                       if int(m["id"]) in TARGET_IDS}
            if len(markers) < 2:
                continue
            centres = {}
            for mid, corners_px in markers.items():
                world_corners = project_corners_to_world(
                    corners_px, cal[cam]["K"], cal[cam]["dist"],
                    cal[cam]["R"], cal[cam]["t"], plank_z)
                if np.any(np.isnan(world_corners)):
                    continue
                centres[mid] = world_corners.mean(axis=0)

            # If 75 and 77 are both detected, compute plank orientation
            # from their world positions (the bundle / fit script will
            # solve this rigorously; here we just record it for plotting).
            orientation_deg = float("nan")
            if 75 in centres and 77 in centres:
                v = centres[77] - centres[75]
                orientation_deg = float(np.degrees(np.arctan2(v[1], v[0])))

            for (id1, id2), expected in KNOWN_DISTANCES_MM.items():
                if id1 not in centres or id2 not in centres:
                    continue
                reported = float(np.linalg.norm(centres[id1] - centres[id2]))
                residual = reported - expected
                x_mid = 0.5 * (centres[id1][0] + centres[id2][0])
                y_mid = 0.5 * (centres[id1][1] + centres[id2][1])
                rows.append({
                    "snap_id": snap_id,
                    "camera": cam,
                    "pair": f"{id1}-{id2}",
                    "expected_mm": float(expected),
                    "reported_mm": reported,
                    "residual_mm": residual,
                    "x_mid_mm": float(x_mid),
                    "y_mid_mm": float(y_mid),
                    "plank_orientation_deg": orientation_deg,
                })

    if not rows:
        sys.exit("[error] no usable measurements (no snapshot had >=2 "
                 "target markers detected by either camera).")

    # Console summary.
    print(f"\n=== {len(rows)} measurements ===")
    arr = np.array([[r["residual_mm"] for r in rows]])
    print(f"  residual overall: mean {arr.mean():+.1f} mm    "
          f"std {arr.std():.1f}    |max| {np.max(np.abs(arr)):.1f}")

    print(f"\nBy camera:")
    for cam in cams:
        sub = [r["residual_mm"] for r in rows if r["camera"] == cam]
        if not sub:
            print(f"  [{cam}] no measurements")
            continue
        sub = np.array(sub)
        print(f"  [{cam}] n={len(sub):3d}    mean {sub.mean():+7.1f}    "
              f"std {sub.std():5.1f}    range [{sub.min():+7.1f}, {sub.max():+7.1f}]")

    print(f"\nBy marker pair (across cameras):")
    for pair in ("75-76", "76-77", "75-77"):
        sub = [r["residual_mm"] for r in rows if r["pair"] == pair]
        if not sub:
            print(f"  [{pair}] no measurements")
            continue
        sub = np.array(sub)
        exp = next(r["expected_mm"] for r in rows if r["pair"] == pair)
        rel_pct = 100.0 * sub.mean() / exp
        print(f"  [{pair}] expected {exp:.0f} mm    n={len(sub):3d}    "
              f"mean {sub.mean():+7.1f} ({rel_pct:+5.2f}%)    "
              f"std {sub.std():5.1f}    range [{sub.min():+7.1f}, {sub.max():+7.1f}]")

    # CSV output.
    csv_path = session_dir / "diagnose_results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[csv] {csv_path}")

    # Plots.
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # (0,0) histogram of residuals, split by camera.
    ax = axes[0, 0]
    for cam, col in zip(cams, ["tab:orange", "tab:blue"]):
        sub = [r["residual_mm"] for r in rows if r["camera"] == cam]
        if sub:
            ax.hist(sub, bins=20, alpha=0.6, label=f"{cam} n={len(sub)}",
                    color=col)
    ax.axvline(0, color="k", lw=0.5)
    ax.set_xlabel("residual (reported - expected), mm")
    ax.set_ylabel("count")
    ax.set_title("Residual histogram")
    ax.legend()
    ax.grid(alpha=0.3)

    # (0,1) residual by (pair, camera) — box / strip.
    ax = axes[0, 1]
    keys = []
    vals = []
    for cam in cams:
        for pair in ("75-76", "76-77", "75-77"):
            sub = [r["residual_mm"] for r in rows
                   if r["camera"] == cam and r["pair"] == pair]
            if sub:
                keys.append(f"{cam}\n{pair}")
                vals.append(sub)
    if vals:
        ax.boxplot(vals, labels=keys, showmeans=True)
        ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("residual (mm)")
    ax.set_title("By (camera, pair)")
    ax.grid(alpha=0.3, axis="y")

    # (1,0) spatial map of |residual| at plank midpoint position.
    ax = axes[1, 0]
    for cam, marker_shape in zip(cams, ["o", "s"]):
        xs = [r["x_mid_mm"] for r in rows if r["camera"] == cam]
        ys = [r["y_mid_mm"] for r in rows if r["camera"] == cam]
        cs = [r["residual_mm"] for r in rows if r["camera"] == cam]
        if xs:
            sc = ax.scatter(xs, ys, c=cs, cmap="RdBu_r",
                            vmin=-max(abs(np.array(cs)).max(), 50),
                            vmax=+max(abs(np.array(cs)).max(), 50),
                            s=40, marker=marker_shape, edgecolor="k",
                            linewidth=0.5, label=f"{cam} ({marker_shape})")
    if "sc" in dir():
        try:
            plt.colorbar(sc, ax=ax, label="residual (mm)")
        except Exception:
            pass
    ax.set_xlabel("x_mid (mm)")
    ax.set_ylabel("y_mid (mm)")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title("Residual vs arena position")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    # (1,1) residual vs plank orientation.
    ax = axes[1, 1]
    for cam, col in zip(cams, ["tab:orange", "tab:blue"]):
        sub = [(r["plank_orientation_deg"], r["residual_mm"])
               for r in rows if r["camera"] == cam
               and not np.isnan(r["plank_orientation_deg"])]
        if sub:
            xs, ys = zip(*sub)
            ax.scatter(xs, ys, s=20, c=col, alpha=0.7, label=cam)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("plank orientation (deg from world +X)")
    ax.set_ylabel("residual (mm)")
    ax.set_title("Residual vs plank orientation")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.suptitle(f"Plank distance diagnostic — {session_dir.name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plot_path = session_dir / "diagnose_plot.png"
    fig.savefig(plot_path, dpi=120)
    print(f"[plot] {plot_path}")
    plt.show()


if __name__ == "__main__":
    main()
