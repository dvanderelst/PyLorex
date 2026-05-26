"""Set per-camera centre C = (Cx, Cy, Cz) by clicking floor markers in a
freshly-captured stitched arena snapshot, then entering the tape-measured
height.

This is a calibration step you run *once* after physically marking each
camera's nadir on the floor with tape. Re-run only when a camera or its
mount moves.

Why:
    The marker-height correction (Lorex.get_aruco) needs the camera centre
    C in the board frame. Until now this came from PnP via -R.T @ t, but
    planar PnP on a single dot board is ambiguous; the resulting C can be
    off by tens of mm in any axis and shows up as a U-shaped cross-camera
    disagreement at the arena extremes (see handoff, 2026-05-22). With the
    nadirs physically marked, the stitched arena snapshot (whose XY axes
    are H_raw-correct at z=0) lets us read the true (Cx, Cy) directly.

Pipeline:
    1. Capture a fresh environment snapshot (cameras must be live; floor
       markers must be visible).
    2. For each camera in the snapshot, click its floor marker.
    3. Enter the tape-measured Cz in the terminal.
    4. Writes Calibration/Results/c_measured_{cam}.json — CalibIO.load_pose_bundle
       picks this up automatically next time the tracker runs.

Verify afterwards with SCRIPT_ProbeTrackerAtWall.py (robot circle should
sit flush against the green wall dots) and script_check_camera_agreement.py
(the dy(x) U-shape should collapse).

Controls (per camera):
    Left-click  : place / move the crosshair.
    SPACE       : accept current click, then enter Cz in the terminal.
    R           : redo (clears the click).
    S           : skip this camera (no file written).
    ESC / Q     : abort the whole run (no further files written).
"""
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2 as cv
import numpy as np

from LorexLib import Settings, Utils
from LorexLib.Environment import capture_environment_layout


# ---------------- SETTINGS ----------------
# Where to drop the captured snapshots. Resolved relative to PyLorex/
# package root (same convention as Utils.get_calibration_paths) so this
# script behaves the same regardless of the shell's cwd. Each run gets a
# fresh env_NNNN_<timestamp>/ subfolder under SAVE_ROOT.
SAVE_ROOT = Path(__file__).resolve().parent / "Calibration" / "CameraCenterSnapshots"
# ------------------------------------------


def pixel_to_world(col: float, row: float, bounds: dict, mm_per_px: float):
    """Match SCRIPT_BuildArenaGeometry._pixels_to_world_at_height (z=0)."""
    X = bounds["min_x"] + col * mm_per_px + 0.5 * mm_per_px
    Y = bounds["max_y"] - row * mm_per_px + 0.5 * mm_per_px
    return float(X), float(Y)


def click_camera_center(camera_name: str, arena_bgr: np.ndarray,
                        bounds: dict, mm_per_px: float):
    """Open a window for one camera and return (Cx, Cy, click_uv) in world mm
    + the clicked pixel, or None if skipped. Returns "ABORT" on ESC/Q."""
    window = f"set_camera_center [{camera_name}]"
    cv.namedWindow(window, cv.WINDOW_NORMAL)
    H, W = arena_bgr.shape[:2]
    cv.resizeWindow(window, min(W, 1600), min(H, 900))

    state = {"click": None}  # (col, row) in image pixels

    def on_mouse(event, x, y, flags, _userdata):
        if event == cv.EVENT_LBUTTONDOWN:
            state["click"] = (int(x), int(y))

    cv.setMouseCallback(window, on_mouse)

    prompt = (f"Click floor marker for '{camera_name}'.  "
              f"SPACE=accept  R=redo  S=skip  ESC/Q=abort")
    print(f"\n[{camera_name}] {prompt}")

    while True:
        vis = arena_bgr.copy()
        cv.putText(vis, prompt, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
        cv.putText(vis, prompt, (10, 30),
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv.LINE_AA)
        if state["click"] is not None:
            u, v = state["click"]
            Cx, Cy = pixel_to_world(u, v, bounds, mm_per_px)
            cv.drawMarker(vis, (u, v), (0, 0, 255),
                          markerType=cv.MARKER_CROSS,
                          markerSize=40, thickness=2)
            cv.circle(vis, (u, v), 8, (0, 0, 255), 2)
            label = f"({Cx:.0f}, {Cy:.0f}) mm"
            cv.putText(vis, label, (u + 14, v - 14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 3, cv.LINE_AA)
            cv.putText(vis, label, (u + 14, v - 14),
                       cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1, cv.LINE_AA)
        cv.imshow(window, vis)
        key = cv.waitKey(20) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            cv.destroyWindow(window)
            return "ABORT"
        if key == ord('s') or key == ord('S'):
            print(f"[{camera_name}] skipped.")
            cv.destroyWindow(window)
            return None
        if key == ord('r') or key == ord('R'):
            state["click"] = None
            print(f"[{camera_name}] cleared; click again.")
        if key == 32:  # SPACE
            if state["click"] is None:
                print(f"[{camera_name}] no click yet; click first or press S to skip.")
                continue
            u, v = state["click"]
            Cx, Cy = pixel_to_world(u, v, bounds, mm_per_px)
            cv.destroyWindow(window)
            return (Cx, Cy, (u, v))


def prompt_height(camera_name: str) -> float:
    while True:
        raw = input(f"[{camera_name}] tape-measured Cz (mm above floor): ").strip()
        try:
            Cz = float(raw)
        except ValueError:
            print("  not a number; try again.")
            continue
        if Cz <= 0:
            print("  must be positive; try again.")
            continue
        return Cz


def write_c_measured(camera_name: str, Cx_arena: float, Cy_arena: float, Cz: float,
                     env_dir: Path, click_uv: tuple):
    """Save C in each camera's PER-CAMERA board frame (same frame as R, t in
    pose_{cam}.npz, and as (X0, Y0) from H_raw). The stitched-snapshot click
    is in the UNIFIED arena frame, which equals tiger's board frame; shark's
    board frame is offset from it by Settings.shark2tiger_delta_{x,y}. Saving
    in per-camera frame means downstream code (Lorex.get_aruco) can use
    bundle["C_measured"] directly without any per-camera special-case logic."""
    if camera_name == "shark":
        dx_unified = float(Settings.shark2tiger_delta_x)
        dy_unified = float(Settings.shark2tiger_delta_y)
    else:
        dx_unified = 0.0
        dy_unified = 0.0
    Cx_board = Cx_arena - dx_unified
    Cy_board = Cy_arena - dy_unified

    paths = Utils.get_calibration_paths(camera_name)
    out_path = paths["c_measured_json"]
    os.makedirs(paths["result_folder"], exist_ok=True)
    payload = {
        "camera_name": camera_name,
        "Cx_mm": float(Cx_board),
        "Cy_mm": float(Cy_board),
        "Cz_mm": float(Cz),
        "frame": "per-camera board (same frame as pose_{cam}.npz R, t)",
        "click_arena_xy_mm": [float(Cx_arena), float(Cy_arena)],
        "applied_offset_to_unified_xy_mm": [dx_unified, dy_unified],
        "source": "clicked stitched snapshot + tape-measured height",
        "env_snapshot": str(env_dir.resolve()),
        "click_pixel_uv": [int(click_uv[0]), int(click_uv[1])],
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[{camera_name}] wrote {out_path}")
    print(f"  click (arena) = ({Cx_arena:.1f}, {Cy_arena:.1f}) mm")
    print(f"  C (board)     = ({Cx_board:.1f}, {Cy_board:.1f}, {Cz:.1f}) mm")


def prompt_intercam_distance() -> Optional[float]:
    """Prompt for the physical Euclidean distance (mm) between the two floor
    nadir marks (e.g., from a tape measure). Returns None if skipped."""
    print("\n[intercam] Physical inter-camera distance anchors the unified "
          "frame to reality. Enter the tape-measured distance between the "
          "two floor nadir marks (mm). Press ENTER to skip.")
    raw = input("[intercam] physical distance between tiger and shark nadirs (mm): ").strip()
    if not raw:
        print("[intercam] skipped; camera_system.json not written.")
        return None
    try:
        D = float(raw)
    except ValueError:
        print(f"[intercam] not a number ({raw!r}); skipped.")
        return None
    if D <= 0:
        print(f"[intercam] non-positive ({D}); skipped.")
        return None
    return D


def derive_and_write_camera_system(physical_distance_mm: float,
                                   rundir: Path) -> Optional[Path]:
    """Read both c_measured_{tiger,shark}.json, combine with the tape-measured
    physical inter-camera distance to derive shark2tiger_delta_{x,y}, and
    write camera_system.json. Assumes the two cameras are aligned in X (i.e.,
    the inter-camera vector is purely in Y in the unified frame). The sign of
    delta_y is taken from the current click ordering (whichever camera has
    the more negative Cy in its own board frame ends up more negative in the
    unified frame)."""
    tiger_paths = Utils.get_calibration_paths("tiger")
    shark_paths = Utils.get_calibration_paths("shark")
    if not (os.path.exists(tiger_paths["c_measured_json"])
            and os.path.exists(shark_paths["c_measured_json"])):
        print("[intercam] need both c_measured_tiger.json and "
              "c_measured_shark.json before deriving the offset; skipped.")
        return None

    with open(tiger_paths["c_measured_json"]) as f:
        ct = json.load(f)
    with open(shark_paths["c_measured_json"]) as f:
        cs = json.load(f)
    tiger_Cx = float(ct["Cx_mm"])
    tiger_Cy = float(ct["Cy_mm"])
    shark_Cx = float(cs["Cx_mm"])
    shark_Cy = float(cs["Cy_mm"])

    # delta_x: assume cameras are aligned in X (no physical X separation).
    # delta_y: |tiger_Cy - (shark_Cy + delta_y)| = physical_distance_mm.
    # Sign: match the current click ordering — whichever camera was clicked
    # more negative in unified Y stays more negative.
    delta_x = tiger_Cx - shark_Cx  # zeros out X disagreement at the nadirs
    shark_unified_old = shark_Cy + float(Settings.shark2tiger_delta_y)
    if shark_unified_old <= tiger_Cy:
        delta_y = (tiger_Cy - physical_distance_mm) - shark_Cy
    else:
        delta_y = (tiger_Cy + physical_distance_mm) - shark_Cy

    out_path = Path(tiger_paths["result_folder"]) / "camera_system.json"
    payload = {
        "shark2tiger_delta_x_mm": float(delta_x),
        "shark2tiger_delta_y_mm": float(delta_y),
        "physical_intercam_distance_mm": float(physical_distance_mm),
        "assumption": "cameras aligned in X; inter-camera vector is purely Y in the unified frame",
        "derived_from": {
            "tiger_C_in_tiger_frame_mm": [tiger_Cx, tiger_Cy],
            "shark_C_in_shark_frame_mm": [shark_Cx, shark_Cy],
        },
        "source": "script_set_camera_center.py",
        "env_snapshot": str(rundir.resolve()),
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[intercam] wrote {out_path}")
    print(f"           shark2tiger_delta_x = {delta_x:+.1f} mm")
    print(f"           shark2tiger_delta_y = {delta_y:+.1f} mm")
    print(f"           (previously in Settings.py: "
          f"x={Settings.shark2tiger_delta_x}, y={Settings.shark2tiger_delta_y})")
    return out_path


def main():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[snapshot] capturing environment snapshot under {SAVE_ROOT} ...")
    snapshot = capture_environment_layout(
        save=True,
        save_root=str(SAVE_ROOT),
    )
    arena = snapshot.get("arena_image")
    meta = snapshot.get("meta", {})
    rundir = Path(snapshot.get("rundir", SAVE_ROOT))
    if arena is None or not isinstance(arena, np.ndarray):
        sys.exit("[error] snapshot did not produce an arena image.")

    bounds = meta["arena_bounds_mm"]
    mm_per_px = float(meta["map_mm_per_px"])
    cameras = list(meta.get("cameras") or [])
    if not cameras:
        sys.exit("[error] snapshot meta has no 'cameras' list.")
    print(f"[snapshot] saved to {rundir}")
    print(f"[snapshot] arena {arena.shape[1]}x{arena.shape[0]} px, "
          f"bounds X[{bounds['min_x']:.0f},{bounds['max_x']:.0f}] "
          f"Y[{bounds['min_y']:.0f},{bounds['max_y']:.0f}] mm, "
          f"mm_per_px={mm_per_px:.2f}")
    print(f"[snapshot] cameras: {cameras}")

    for cam in cameras:
        res = click_camera_center(cam, arena, bounds, mm_per_px)
        if res == "ABORT":
            print("[abort] no further cameras processed.")
            return
        if res is None:
            continue
        Cx, Cy, click_uv = res
        Cz = prompt_height(cam)
        write_c_measured(cam, Cx, Cy, Cz, rundir, click_uv)

    # Anchor the cross-camera offset to a tape-measured physical distance.
    D = prompt_intercam_distance()
    if D is not None:
        derive_and_write_camera_system(D, rundir)

    print("\n[done] Settings.py will pick up camera_system.json automatically "
          "on next import (i.e. next process). CalibIO.load_pose_bundle picks "
          "up c_measured_{cam}.json automatically.")
    print("       Verify with SCRIPT_ProbeTrackerAtWall.py and "
          "script_check_camera_agreement.py.")


if __name__ == "__main__":
    main()
