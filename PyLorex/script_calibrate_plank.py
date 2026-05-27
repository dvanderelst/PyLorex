"""Bundle-adjustment calibration from the plank-snapshot data.

Solves shark's (R, t) in tiger's frame, plus each snapshot's plank pose
(X, Y, yaw), by minimising the reprojection error of detected marker
corners across all snapshots and both cameras. Tiger's (R, t) is held
fixed (anchor), so the resulting world frame is identical to the
current arena coordinate system.

What this REPLACES vs what stays:
  - REPLACES: shark's PnP-derived (R, t) — re-solved against plank data
    that spans the full FOV, instead of the dot grid which only spans
    ~500 x 200 mm near image centre.
  - REPLACES: shark2tiger_delta_{x,y} — no longer needed; both cameras
    live in the same tiger-anchored world frame after the bundle.
  - KEEPS: K, dist for both cameras (intrinsics — unchanged).
  - KEEPS: tiger's (R, t) (anchor — the world frame stays the same as
    the current calibration).
  - KEEPS: H_raw for both cameras (no longer used by the runtime tracker
    after the 2026-05-26 ray-plane switch; left in place for the stitched
    arena image visualisation only).

Plank geometry (locked by construction):
  - Markers 75, 76, 77 at plank-frame X = -500, 0, +500 mm. Y = 0 for
    all. Z = +25 mm (plank surface above floor).
  - Each marker's 4 corners at ±45 mm in the marker's local frame
    (90 mm marker side).
  - All three markers' axes aligned with the plank's local axes (so
    plank yaw = marker yaw for all three markers).

Output (in the session folder):
  - pose_shark_arena_frame.npz / .json — shark's new (R, t) in the
    same format as Calibration/Results/pose_shark.*. To make this the
    canonical calibration, copy these over the existing files. The
    script does NOT overwrite anything in Calibration/Results/ on its
    own.
  - bundle_results.json — pre/post metrics, final plank poses, residuals.
  - frame_plot.png — top-down view of tiger nadir, shark nadir, the
    15 plank poses, and arena walls if available.

Usage:
    python script_calibrate_plank.py [session_dir]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from LorexLib import CalibIO, Settings, Utils


TARGET_IDS = (75, 76, 77)
MARKER_HALF_SIZE_MM = 45.0   # 90 mm marker / 2
MARKER_X_IN_PLANK = {75: -500.0, 76: 0.0, 77: +500.0}
KNOWN_PAIRS_MM = {(75, 76): 500.0, (76, 77): 500.0, (75, 77): 1000.0}

# ArUco corner order from cv.aruco.detectMarkers is TL, TR, BR, BL in
# the MARKER's canonical orientation (i.e., the printed pattern's frame,
# not the image frame). For a marker viewed face-up:
#   TL -> (-half, +half, 0)   TR -> (+half, +half, 0)
#   BR -> (+half, -half, 0)   BL -> (-half, -half, 0)
# This is the marker's local frame. The plank holds the markers in a row
# but the markers themselves can be rotated relative to the plank — we
# measured an empirical +90 deg offset on this physical plank (markers'
# +X axes are perpendicular to the plank's long axis). So when placing
# the corner offsets into the plank's frame, we rotate by this offset.
# If you remount markers with a different orientation, change this value.
MARKER_TO_PLANK_YAW_DEG = 90.0

# Intrinsics refinement: when True, the bundle also solves for K and the
# leading distortion coefficients per camera. Useful when the bundle's
# inter-camera distance disagrees with an external (tape) measurement —
# that disagreement is often intrinsic scale bias, and unfreezing K
# lets the data dictate the right values. When False, K + dist stay at
# their existing calibrated values and only extrinsics + plank poses
# move (the original behaviour).
REFINE_INTRINSICS = True
# Per-camera intrinsic params we let vary: fx, fy, cx, cy, k1, k2.
# (k3, p1, p2 left at their existing values — usually small and
# under-determined by a planar floor target.)
N_INTRINSIC_PARAMS = 6

# Soft regularization toward the original (checkerboard-calibrated)
# intrinsic values. Sigmas express "how far the bundle is allowed to
# move each intrinsic before incurring 1 sigma worth of cost". With
# only a single z-plane of markers, the focal-length / depth degeneracy
# is severe — without these priors the bundle drifts fx by 30%+ to a
# non-physical solution. Conservative sigmas keep it tethered while
# still allowing genuine percent-level corrections.
INTRINSIC_SIGMAS = {
    "fx": 25.0,     # ~1% of typical fx (~2400 px)
    "fy": 25.0,
    "cx": 30.0,     # principal point should be within ~30 px of nominal
    "cy": 30.0,
    "k1": 0.02,
    "k2": 0.02,
}

_MARKER_CORNERS_LOCAL = np.array([
    [-MARKER_HALF_SIZE_MM, +MARKER_HALF_SIZE_MM, 0.0],  # TL
    [+MARKER_HALF_SIZE_MM, +MARKER_HALF_SIZE_MM, 0.0],  # TR
    [+MARKER_HALF_SIZE_MM, -MARKER_HALF_SIZE_MM, 0.0],  # BR
    [-MARKER_HALF_SIZE_MM, -MARKER_HALF_SIZE_MM, 0.0],  # BL
], dtype=np.float64)


def _marker_to_plank_rotation():
    a = np.radians(MARKER_TO_PLANK_YAW_DEG)
    return np.array([[np.cos(a), -np.sin(a), 0.0],
                     [np.sin(a), +np.cos(a), 0.0],
                     [0.0,        0.0,        1.0]])


# Corner offsets pre-rotated into plank frame.
MARKER_CORNERS_IN_PLANK = (_marker_to_plank_rotation() @ _MARKER_CORNERS_LOCAL.T).T

PLANK_ROOT = Path(__file__).resolve().parent / "Calibration" / "PlankSnapshots"


def find_latest_session(root):
    if not root.exists():
        sys.exit(f"[error] no session root at {root}")
    sessions = sorted(p for p in root.iterdir() if p.is_dir())
    if not sessions:
        sys.exit(f"[error] no sessions under {root}")
    return sessions[-1]


def load_calibration(camera_name, to_tiger_frame=False):
    """Load (K, dist, R, t). If to_tiger_frame and camera_name=='shark',
    transform (R, t) from shark's per-camera board frame into the
    unified tiger frame using Settings.shark2tiger_delta_{x,y}. Tiger
    needs no transform. After this, all camera (R, t) live in the same
    world frame, which is what the bundle adjustment needs."""
    b = CalibIO.load_pose_bundle(camera_name)
    K = np.asarray(b["K"], dtype=np.float64)
    dist = np.asarray(b["dist"], dtype=np.float64)
    R = np.asarray(b["R"], dtype=np.float64)
    t = np.asarray(b["t"], dtype=np.float64).reshape(3)
    if to_tiger_frame and camera_name == "shark":
        # X_tiger = X_shark + delta, so X_shark = X_tiger - delta.
        # X_cam = R @ X_shark + t = R @ (X_tiger - delta) + t
        #       = R @ X_tiger + (t - R @ delta).
        delta = np.array([Settings.shark2tiger_delta_x,
                          Settings.shark2tiger_delta_y, 0.0])
        t = t - R @ delta
    return {"K": K, "dist": dist, "R": R, "t": t}


def project_corners_to_world(corners_px, K, dist, R, t, z_mm):
    """Same routine the diagnostic uses: undistort the pixels, intersect
    the camera ray with z = z_mm via the camera (R, t)."""
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


def estimate_plank_pose(detects_per_cam, cal, plank_z):
    """Initial guess of plank pose (X, Y, yaw_rad) for one snapshot.

    Use whichever cameras detect target markers. Per camera, project
    each marker's 4 corners to world, take the centroid as the marker
    centre. Then average across cameras (if both see the marker). Plank
    centre = marker 76 if visible, else mean of visible markers.
    Yaw = direction from marker 75 to marker 77, if both visible; else
    fall back to 0."""
    centres_world = {}
    for cam, detects in detects_per_cam.items():
        K, dist, R, t = cal[cam]["K"], cal[cam]["dist"], cal[cam]["R"], cal[cam]["t"]
        for mid in TARGET_IDS:
            if mid not in detects:
                continue
            world_corners = project_corners_to_world(
                detects[mid], K, dist, R, t, plank_z)
            if np.any(np.isnan(world_corners)):
                continue
            c = world_corners.mean(axis=0)
            centres_world.setdefault(mid, []).append(c)
    if not centres_world:
        return None
    centres = {mid: np.mean(np.stack(lst, axis=0), axis=0)
               for mid, lst in centres_world.items()}
    # Plank centre = marker 76 if visible, else mean of visible markers.
    if 76 in centres:
        plank_xy = centres[76].copy()
    else:
        plank_xy = np.mean(np.stack(list(centres.values()), axis=0), axis=0)
    # Yaw from 75 -> 77 if both visible.
    if 75 in centres and 77 in centres:
        d = centres[77] - centres[75]
        yaw = float(np.arctan2(d[1], d[0]))
    else:
        yaw = 0.0
    return np.array([plank_xy[0], plank_xy[1], yaw])


def plank_pose_to_world_corners(plank_xy_yaw, marker_id, plank_z):
    """Return the 4 marker corners (4x3) in world frame given the plank's
    (X, Y, yaw) pose and which marker we're looking at."""
    X, Y, yaw = plank_xy_yaw
    cos_y, sin_y = np.cos(yaw), np.sin(yaw)
    R_yaw = np.array([[cos_y, -sin_y, 0.0],
                      [sin_y, +cos_y, 0.0],
                      [0.0,   0.0,    1.0]])
    # Marker centre in plank frame.
    m_centre = np.array([MARKER_X_IN_PLANK[marker_id], 0.0, 0.0])
    # Corners in plank frame = marker centre + corner offsets that have
    # already been rotated from marker-local frame to plank frame via
    # the empirical MARKER_TO_PLANK_YAW_DEG offset.
    corners_plank = m_centre[None, :] + MARKER_CORNERS_IN_PLANK  # (4, 3)
    # Rotate by plank yaw, translate by (X, Y, plank_z).
    corners_world = (R_yaw @ corners_plank.T).T + np.array([X, Y, plank_z])
    return corners_world


def project_world_to_image(world_pts, K, dist, R, t):
    rvec, _ = cv.Rodrigues(R)
    proj, _ = cv.projectPoints(
        world_pts.astype(np.float64).reshape(-1, 1, 3),
        rvec, np.asarray(t, dtype=np.float64).reshape(3, 1),
        K, dist)
    return proj.reshape(-1, 2)


def unpack(params, n_snaps, refine_intrinsics, dist_full):
    """Unpack the optimizer's flat parameter vector.

    Layout when refine_intrinsics:
        [shark_rvec(3), shark_t(3),
         tiger_intrinsics(N_INTRINSIC_PARAMS),
         shark_intrinsics(N_INTRINSIC_PARAMS),
         plank_poses(3 per snap)]
    Layout otherwise:
        [shark_rvec(3), shark_t(3),
         plank_poses(3 per snap)]

    Intrinsics packing per camera: [fx, fy, cx, cy, k1, k2]. The full
    dist coefficient vector keeps k3, p1, p2 at their original values
    (from dist_full[cam])."""
    shark_rvec = params[:3]
    shark_t = params[3:6]
    R_shark, _ = cv.Rodrigues(shark_rvec.reshape(3, 1))
    if refine_intrinsics:
        ti = params[6:6 + N_INTRINSIC_PARAMS]
        si = params[6 + N_INTRINSIC_PARAMS:6 + 2 * N_INTRINSIC_PARAMS]
        plank_poses = params[6 + 2 * N_INTRINSIC_PARAMS:].reshape(n_snaps, 3)

        def build(intrin_vec, dist_orig):
            fx, fy, cx, cy, k1, k2 = intrin_vec
            K = np.array([[fx, 0.0, cx],
                          [0.0, fy, cy],
                          [0.0, 0.0, 1.0]])
            dist = np.array(dist_orig, dtype=np.float64).flatten()
            dist[0] = k1
            dist[1] = k2
            return K, dist

        K_t, dist_t = build(ti, dist_full["tiger"])
        K_s, dist_s = build(si, dist_full["shark"])
        K_per_cam = {"tiger": K_t, "shark": K_s}
        dist_per_cam = {"tiger": dist_t, "shark": dist_s}
    else:
        plank_poses = params[6:].reshape(n_snaps, 3)
        K_per_cam = None
        dist_per_cam = None
    return R_shark, shark_t, K_per_cam, dist_per_cam, plank_poses


def residuals(params, observations, intrinsics_fixed, dist_full,
              tiger_pose, n_snaps, plank_z, refine_intrinsics,
              intrinsic_priors=None):
    """Reprojection residuals across all observations, plus (when
    refine_intrinsics) soft regularization residuals tethering each
    intrinsic param to its initial value with INTRINSIC_SIGMAS scaling.

    intrinsics_fixed: {cam: (K, dist)} used when refine_intrinsics=False.
    dist_full: {cam: original full dist vector} -- seeds the unchanged
        higher-order distortion coefficients when refine_intrinsics=True.
    intrinsic_priors: {cam: (fx0, fy0, cx0, cy0, k10, k20)} initial
        values + INTRINSIC_SIGMAS -> regularization residuals."""
    R_shark, t_shark, K_per_cam_new, dist_per_cam_new, plank_poses = unpack(
        params, n_snaps, refine_intrinsics, dist_full)
    R_tiger, t_tiger = tiger_pose
    out = []
    for obs in observations:
        snap_i = obs["snap_idx"]
        mid = obs["marker_id"]
        cam = obs["camera"]
        corners_px = obs["corners_px"]
        plank_xyz = plank_poses[snap_i]
        world_corners = plank_pose_to_world_corners(plank_xyz, mid, plank_z)
        if refine_intrinsics:
            K = K_per_cam_new[cam]
            dist = dist_per_cam_new[cam]
        else:
            K, dist = intrinsics_fixed[cam]
        if cam == "tiger":
            proj = project_world_to_image(world_corners, K, dist, R_tiger, t_tiger)
        else:
            proj = project_world_to_image(world_corners, K, dist, R_shark, t_shark)
        out.append((proj - corners_px).flatten())

    # Intrinsic regularization residuals (one per varied param per camera).
    if refine_intrinsics and intrinsic_priors is not None:
        sigmas = np.array([INTRINSIC_SIGMAS["fx"], INTRINSIC_SIGMAS["fy"],
                           INTRINSIC_SIGMAS["cx"], INTRINSIC_SIGMAS["cy"],
                           INTRINSIC_SIGMAS["k1"], INTRINSIC_SIGMAS["k2"]])
        for c in ("tiger", "shark"):
            K_now = K_per_cam_new[c]; d_now = dist_per_cam_new[c]
            now = np.array([K_now[0, 0], K_now[1, 1], K_now[0, 2], K_now[1, 2],
                            d_now[0], d_now[1]])
            init = intrinsic_priors[c]
            out.append((now - init) / sigmas)
    return np.concatenate(out)


def save_camera_pose(out_dir, camera_name, R, t, K, dist,
                     src_bundle, original_pose_json):
    """Write pose_<camera>_arena_frame.{json,npz} carrying the bundle's
    refined (R, t, K, dist) plus the unchanged H_raw / H_undistorted
    from src_bundle. Format mirrors Calibration/Results/pose_<cam>.*
    so a copy-over is a drop-in replacement for the runtime tracker."""
    out_dir.mkdir(parents=True, exist_ok=True)
    H_raw = src_bundle.get("H_raw")
    H_undistorted = src_bundle.get("H_undistorted")
    payload = dict(original_pose_json)  # preserve metadata we don't touch
    payload["R_pnp"] = np.asarray(R).tolist()
    payload["t_pnp"] = np.asarray(t).reshape(3, 1).tolist()
    payload["K_scaled"] = np.asarray(K).tolist()
    payload["dist_coeffs"] = np.asarray(dist).flatten().tolist()
    payload["frame"] = "tiger-anchored, plank-bundle-adjustment"
    payload["source"] = "script_calibrate_plank.py"
    payload["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    with open(out_dir / f"pose_{camera_name}_arena_frame.json", "w") as f:
        json.dump(payload, f, indent=2)
    np.savez_compressed(
        out_dir / f"pose_{camera_name}_arena_frame.npz",
        K_scaled=K, dist=dist,
        H_raw=H_raw if H_raw is not None else np.eye(3),
        H_undistorted=(H_undistorted if H_undistorted is not None else np.eye(3)),
        R_pnp=np.asarray(R), t_pnp=np.asarray(t).reshape(3, 1),
        obj_mm=src_bundle.get("obj_mm", np.zeros((1, 2))),
        obj_xyz=np.zeros((1, 3)),
        img_pts=np.zeros((1, 2)),
        corners=np.zeros((1, 1, 2)),
    )


def save_camera_system(out_dir, dx_mm=0.0, dy_mm=0.0):
    """Write a zeroed camera_system.json next to the pose files. The
    plank bundle places shark's (R, t) directly in tiger's frame, so the
    legacy shark2tiger_delta_{x,y} offset should be zero after adoption."""
    payload = {
        "shark2tiger_delta_x_mm": float(dx_mm),
        "shark2tiger_delta_y_mm": float(dy_mm),
        "source": "script_calibrate_plank.py",
        "note": "Bundle outputs already in tiger frame; delta is zero by construction.",
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "camera_system_arena_frame.json", "w") as f:
        json.dump(payload, f, indent=2)
    return out_dir / "camera_system_arena_frame.json"




def plot_frame(out_path, tiger_cal, R_shark, t_shark, plank_poses, plank_z,
               session_name):
    """Top-down view of tiger + shark nadirs and the 15 plank poses."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Camera nadirs.
    C_tiger = -tiger_cal["R"].T @ tiger_cal["t"]
    C_shark = -R_shark.T @ t_shark

    # Optical axis projections at z = plank_z (the cameras' "looking
    # towards" direction on the plank's plane).
    def axis_arrow(R, t, length_mm=400.0):
        C = -R.T @ t
        cam_z_in_world = R.T @ np.array([0.0, 0.0, 1.0])
        # Project to plank plane: find lambda s.t. C + lam*cam_z is at z=plank_z
        if abs(cam_z_in_world[2]) < 1e-9:
            return C[:2], C[:2]
        lam = (plank_z - C[2]) / cam_z_in_world[2]
        tip = C + lam * cam_z_in_world
        return C[:2], tip[:2]

    for label, R, t, col in [
        ("tiger", tiger_cal["R"], tiger_cal["t"], "tab:orange"),
        ("shark", R_shark, t_shark, "tab:blue"),
    ]:
        nadir, axis_tip = axis_arrow(R, t)
        ax.scatter([nadir[0]], [nadir[1]], s=200, marker="*",
                   color=col, edgecolor="k", zorder=10, label=f"{label} nadir")
        ax.annotate("", xy=axis_tip, xytext=nadir,
                    arrowprops=dict(arrowstyle="->", color=col, lw=2))
        # Camera height label.
        Cz = (-R.T @ t)[2]
        ax.text(nadir[0] + 60, nadir[1] + 60,
                f"{label}\nh={Cz:.0f}",
                color=col, fontweight="bold")

    # Plank poses: each as a 1-m line segment 75->77 with a dot at 76.
    cmap = plt.get_cmap("viridis")
    n = len(plank_poses)
    for i, (X, Y, yaw) in enumerate(plank_poses):
        col = cmap(i / max(n - 1, 1))
        p75 = (X + MARKER_X_IN_PLANK[75] * np.cos(yaw),
               Y + MARKER_X_IN_PLANK[75] * np.sin(yaw))
        p77 = (X + MARKER_X_IN_PLANK[77] * np.cos(yaw),
               Y + MARKER_X_IN_PLANK[77] * np.sin(yaw))
        ax.plot([p75[0], p77[0]], [p75[1], p77[1]],
                color=col, lw=1.5, alpha=0.85)
        ax.scatter([X], [Y], s=20, color=col, zorder=5)
        ax.text(p77[0] + 30, p77[1] + 30, str(i + 1),
                fontsize=7, color=col)

    # Mark world origin and X axis.
    ax.scatter([0], [0], marker="+", s=200, color="k", lw=2)
    ax.annotate("", xy=(300, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="k"))
    ax.text(310, 5, "+X (tiger frame)", fontsize=9)

    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title(f"Arena-frame calibration — {session_name}\n"
                 "cameras (stars) + plank snapshots (lines, 75->77)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def reprojection_rmse(observations, params, intrinsics_fixed, dist_full,
                      tiger_pose, n_snaps, plank_z, refine_intrinsics,
                      intrinsic_priors=None):
    """Reprojection RMSE based ONLY on the image-pixel residuals (i.e.,
    excludes the intrinsic-regularization residuals so the number stays
    interpretable as 'mean pixel error per corner coord')."""
    res_all = residuals(params, observations, intrinsics_fixed, dist_full,
                        tiger_pose, n_snaps, plank_z, refine_intrinsics,
                        intrinsic_priors=intrinsic_priors)
    n_image_residuals = len(observations) * 4 * 2  # 4 corners * 2 dims each
    res_img = res_all[:n_image_residuals]
    return float(np.sqrt(np.mean(res_img ** 2)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session_dir", type=str, nargs="?",
                    help="path to PlankSnapshots/<session>; default: latest")
    args = ap.parse_args()

    if args.session_dir is None:
        session_dir = find_latest_session(PLANK_ROOT)
    else:
        session_dir = Path(args.session_dir).expanduser().resolve()
    print(f"[session] {session_dir}")

    with open(session_dir / "meta.json") as f:
        meta = json.load(f)
    cams = meta.get("cameras") or ["tiger", "shark"]
    plank_z = float(meta.get("plank_geometry_mm", {}).get(
        "marker_z_above_floor_mm", 25.0))

    cal = {c: load_calibration(c, to_tiger_frame=True) for c in cams}
    print(f"[cal] loaded; shark transformed into tiger frame "
          f"(shark2tiger_delta = ({Settings.shark2tiger_delta_x}, "
          f"{Settings.shark2tiger_delta_y}))")
    intrinsics = {c: (cal[c]["K"], cal[c]["dist"]) for c in cams}
    tiger_pose = (cal["tiger"]["R"], cal["tiger"]["t"])

    # Build observations + per-snapshot initial plank pose.
    snap_dirs = sorted(p for p in session_dir.iterdir()
                       if p.is_dir() and p.name.startswith("snapshot_"))
    observations = []
    plank_poses_init = []
    snap_index_map = {}  # snap_dir name -> idx in plank_poses_init
    skipped = []
    for sd in snap_dirs:
        with open(sd / "detections.json") as f:
            snap = json.load(f)
        detects_per_cam = {}
        for cam in cams:
            d = {int(m["id"]): np.asarray(m["corners_px"], dtype=np.float64)
                 for m in snap["cameras"].get(cam, {}).get("markers", [])
                 if int(m["id"]) in TARGET_IDS}
            if d:
                detects_per_cam[cam] = d
        if not detects_per_cam:
            skipped.append(sd.name)
            continue
        init = estimate_plank_pose(detects_per_cam, cal, plank_z)
        if init is None:
            skipped.append(sd.name)
            continue
        snap_idx = len(plank_poses_init)
        plank_poses_init.append(init)
        snap_index_map[sd.name] = snap_idx
        for cam, detects in detects_per_cam.items():
            for mid, corners_px in detects.items():
                observations.append({
                    "snap_idx": snap_idx, "camera": cam,
                    "marker_id": mid, "corners_px": corners_px,
                })

    n_snaps = len(plank_poses_init)
    print(f"[data] {n_snaps} usable snapshots, "
          f"{len(observations)} marker observations")
    if skipped:
        print(f"[data] skipped (no target markers): {skipped}")

    # Initial param vector.
    shark_rvec_init, _ = cv.Rodrigues(cal["shark"]["R"])
    dist_full = {c: np.asarray(cal[c]["dist"], dtype=np.float64).flatten()
                 for c in cams}

    parts = [shark_rvec_init.flatten(), cal["shark"]["t"].reshape(3)]
    intrinsic_priors = None
    if REFINE_INTRINSICS:
        intrinsic_priors = {}
        for c in ("tiger", "shark"):
            K_c = cal[c]["K"]; d_c = dist_full[c]
            init_vec = np.array([K_c[0, 0], K_c[1, 1], K_c[0, 2], K_c[1, 2],
                                 d_c[0], d_c[1]])
            parts.append(init_vec)
            intrinsic_priors[c] = init_vec.copy()
    parts.append(np.asarray(plank_poses_init).flatten())
    x0 = np.concatenate(parts)

    rmse_before = reprojection_rmse(observations, x0, intrinsics, dist_full,
                                    tiger_pose, n_snaps, plank_z,
                                    REFINE_INTRINSICS,
                                    intrinsic_priors=intrinsic_priors)
    print(f"[init ] reprojection RMSE: {rmse_before:.3f} px "
          f"(initial guess, before bundle)")
    if REFINE_INTRINSICS:
        print(f"[mode ] refining intrinsics with priors "
              f"(sigmas fx/fy={INTRINSIC_SIGMAS['fx']}, "
              f"cx/cy={INTRINSIC_SIGMAS['cx']}, k1/k2={INTRINSIC_SIGMAS['k1']})")
    else:
        print(f"[mode ] extrinsics only")

    # Bundle adjustment.
    print("[solve] running least_squares (TRF, huber loss) ...")
    result = least_squares(
        residuals, x0,
        args=(observations, intrinsics, dist_full,
              tiger_pose, n_snaps, plank_z, REFINE_INTRINSICS,
              intrinsic_priors),
        method="trf", loss="huber", f_scale=2.0,
        max_nfev=500, verbose=1, x_scale="jac",
    )
    rmse_after = reprojection_rmse(observations, result.x, intrinsics, dist_full,
                                   tiger_pose, n_snaps, plank_z,
                                   REFINE_INTRINSICS,
                                   intrinsic_priors=intrinsic_priors)
    print(f"[done ] reprojection RMSE: {rmse_after:.3f} px "
          f"(after bundle; was {rmse_before:.3f})")

    R_shark_new, t_shark_new, K_per_cam_new, dist_per_cam_new, plank_poses_new = \
        unpack(result.x, n_snaps, REFINE_INTRINSICS, dist_full)

    if REFINE_INTRINSICS:
        print(f"\n[intrinsics refinement]")
        for c in ("tiger", "shark"):
            K_old = cal[c]["K"]; d_old = dist_full[c]
            K_new = K_per_cam_new[c]; d_new = dist_per_cam_new[c]
            print(f"  [{c}]")
            print(f"    fx: {K_old[0,0]:8.2f}  ->  {K_new[0,0]:8.2f}  "
                  f"(delta {K_new[0,0]-K_old[0,0]:+7.2f}, "
                  f"{(K_new[0,0]/K_old[0,0]-1)*100:+5.2f}%)")
            print(f"    fy: {K_old[1,1]:8.2f}  ->  {K_new[1,1]:8.2f}  "
                  f"(delta {K_new[1,1]-K_old[1,1]:+7.2f}, "
                  f"{(K_new[1,1]/K_old[1,1]-1)*100:+5.2f}%)")
            print(f"    cx: {K_old[0,2]:8.2f}  ->  {K_new[0,2]:8.2f}  "
                  f"(delta {K_new[0,2]-K_old[0,2]:+7.2f})")
            print(f"    cy: {K_old[1,2]:8.2f}  ->  {K_new[1,2]:8.2f}  "
                  f"(delta {K_new[1,2]-K_old[1,2]:+7.2f})")
            print(f"    k1: {d_old[0]:+8.4f}  ->  {d_new[0]:+8.4f}  "
                  f"(delta {d_new[0]-d_old[0]:+7.4f})")
            print(f"    k2: {d_old[1]:+8.4f}  ->  {d_new[1]:+8.4f}  "
                  f"(delta {d_new[1]-d_old[1]:+7.4f})")

    # Sanity numbers in the new frame.
    C_tiger = -cal["tiger"]["R"].T @ cal["tiger"]["t"]
    C_shark_new = -R_shark_new.T @ t_shark_new
    print(f"\n[geometry]")
    print(f"  tiger nadir (fixed): ({C_tiger[0]:+8.1f}, {C_tiger[1]:+8.1f}, "
          f"{C_tiger[2]:8.1f}) mm")
    print(f"  shark nadir (new):   ({C_shark_new[0]:+8.1f}, {C_shark_new[1]:+8.1f}, "
          f"{C_shark_new[2]:8.1f}) mm")
    print(f"  inter-camera floor-plane distance: "
          f"{float(np.linalg.norm(C_tiger[:2] - C_shark_new[:2])):.1f} mm")

    # Plank-distance residuals using the new calibration (sanity check).
    plank_residuals = []
    for snap_idx in range(n_snaps):
        pp = plank_poses_new[snap_idx]
        centres = {mid: plank_pose_to_world_corners(pp, mid, plank_z).mean(axis=0)
                   for mid in TARGET_IDS}
        for (a, b), expected in KNOWN_PAIRS_MM.items():
            d = float(np.linalg.norm(centres[a][:2] - centres[b][:2]))
            plank_residuals.append(d - expected)
    plank_residuals = np.array(plank_residuals)
    print(f"\n[plank distance residuals after bundle]")
    print(f"  mean {plank_residuals.mean():+.2f} mm   "
          f"std {plank_residuals.std():.2f}   "
          f"|max| {np.max(np.abs(plank_residuals)):.2f}")

    # Save a complete adoption-ready calibration set: both cameras'
    # pose files (refined K, dist if REFINE_INTRINSICS was on) plus a
    # zeroed camera_system.json (since the bundle places shark in tiger's
    # frame by construction).
    out_files = []
    for c in cams:
        src_bundle = CalibIO.load_pose_bundle(c)
        with open(Path(Utils.get_calibration_paths(c)["pose_json"])) as f:
            orig_pose_json = json.load(f)
        if c == "tiger":
            R_save = cal["tiger"]["R"]; t_save = cal["tiger"]["t"]
        else:
            R_save = R_shark_new; t_save = t_shark_new
        if REFINE_INTRINSICS:
            K_save = K_per_cam_new[c]; d_save = dist_per_cam_new[c]
        else:
            K_save = cal[c]["K"]; d_save = cal[c]["dist"]
        save_camera_pose(session_dir, c, R_save, t_save, K_save, d_save,
                         src_bundle, orig_pose_json)
        out_files.append(session_dir / f"pose_{c}_arena_frame.json")
        out_files.append(session_dir / f"pose_{c}_arena_frame.npz")
    cs_path = save_camera_system(session_dir, dx_mm=0.0, dy_mm=0.0)
    out_files.append(cs_path)

    print(f"\n[output] adoption-ready files in {session_dir}:")
    for p in out_files:
        print(f"           {p.name}")
    print("\n[adopt] To make this calibration canonical:")
    canon = "PyLorex/PyLorex/Calibration/Results"
    relsess = session_dir.relative_to(Path(__file__).resolve().parent.parent.parent)
    print(f"           cd {canon}")
    print(f"           # back up the current calibration first")
    print(f"           for f in pose_tiger.{{json,npz}} pose_shark.{{json,npz}} camera_system.json; do "
          f"cp -n $f $f.pre-arena-frame.bak; done")
    print(f"           # copy the new calibration into place")
    print(f"           cp {relsess}/pose_tiger_arena_frame.json   pose_tiger.json")
    print(f"           cp {relsess}/pose_tiger_arena_frame.npz    pose_tiger.npz")
    print(f"           cp {relsess}/pose_shark_arena_frame.json   pose_shark.json")
    print(f"           cp {relsess}/pose_shark_arena_frame.npz    pose_shark.npz")
    print(f"           cp {relsess}/camera_system_arena_frame.json camera_system.json")

    # Save bundle metadata.
    bundle_info = {
        "session": session_dir.name,
        "n_snapshots_used": n_snaps,
        "n_observations": len(observations),
        "skipped": skipped,
        "reprojection_rmse_px_before": rmse_before,
        "reprojection_rmse_px_after": rmse_after,
        "plank_residuals_mm": {
            "mean": float(plank_residuals.mean()),
            "std": float(plank_residuals.std()),
            "abs_max": float(np.max(np.abs(plank_residuals))),
        },
        "tiger_nadir_mm": [float(v) for v in C_tiger],
        "shark_nadir_mm_new": [float(v) for v in C_shark_new],
        "frame": "tiger-anchored",
        "solver_status": int(result.status),
        "solver_message": str(result.message),
    }
    with open(session_dir / "bundle_results.json", "w") as f:
        json.dump(bundle_info, f, indent=2)

    # Plot the resulting coordinate frame.
    plot_frame(session_dir / "frame_plot.png",
               cal["tiger"], R_shark_new, t_shark_new,
               plank_poses_new, plank_z, session_dir.name)
    print(f"[plot]   frame_plot.png   -> {session_dir}")


if __name__ == "__main__":
    main()
