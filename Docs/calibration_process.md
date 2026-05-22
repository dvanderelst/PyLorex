# Calibration Process

End-to-end procedure for calibrating a Lorex-camera tracking rig from scratch.
Three phases (intrinsics, homography, cross-camera alignment) plus an
end-to-end verification step. Phase 1 only needs to be redone when the lens
or sensor changes; phases 2-4 must be redone any time a camera is
physically moved.

All scripts live in `PyLorex/PyLorex/`. Edit the `camera_name` constant at
the top of each script and run once per camera (`'tiger'`, then `'shark'`).

## Phase 1: Intrinsic Calibration (per camera)

Solves for the camera matrix `K` and lens distortion coefficients
`dist`. These are properties of the lens + sensor and do not change when
the camera body is moved.

### 1a. Collect images

```
PyLorex/PyLorex/script_collect_intrinsic_images.py
```

- Set `camera_name`.
- Captures `number_of_images` (default 15) RTSP frames, with a beep
  between shots.
- Between shots, **physically move and tilt the calibration dot pattern
  through the camera's view**: vary position (corners, centre, edges),
  distance (close/far), and tilt (rotate, slant). Variety matters far
  more than the count — 15 well-varied frames beat 50 near-identical
  ones.
- Output: `Calibration/<camera>/calibration_images/frameNN.jpg`.

### 1b. Solve intrinsics

```
PyLorex/PyLorex/script_run_intrinsic_dots.py
```

- Set `camera_name`. Optional: bump `visual_check_ms` (default 1000) if
  you want longer per-image inspection during detection.
- Detects the circle grid in each image, runs `cv.calibrateCamera`,
  writes the result to `Calibration/<camera>/Results/intrinsics.yml`.
- Annotated detection PNGs are written to
  `Calibration/<camera>/Results/detect_vis/` for offline review.

**Acceptance criteria:**

- At least ~10 of the 15 frames should produce a valid grid detection.
- Final **RMS reprojection error**:
  - `< 0.5 px` → good.
  - `0.5 – 1.0 px` → acceptable; usable but not great.
  - `> 1.0 px` → poor coverage or noisy frames; recapture with more
    pose variety and rerun.
- The "Worst per-image errors (top 5)" list flags individual problem
  frames. If one frame's error is much higher than the others, look at
  its `vis_NNN.png` — usually the grid is partially occluded or
  the board was nearly edge-on.

### 1c. Validate intrinsics

```
PyLorex/PyLorex/script_assess_intrinsics.py
```

- Runs additional sanity checks on the saved intrinsics and writes
  diagnostic plots/images.

## Phase 2: Homography Calibration (per camera)

Solves for the camera's pose in the world frame: rotation `R_pnp`,
translation `t_pnp`, plus rectifying homographies `H_raw` (raw frame
→ board mm) and `H_undistorted` (undistorted frame → board mm).

**Precondition:** valid intrinsics from Phase 1.

### 2a. Place the calibration board

- Lay the calibration dot board flat on the floor inside the arena.
- **The board's position and orientation in the room defines the world
  frame's origin and axes for that camera.** Pick a spot you can
  recover later — taping the board outline onto the floor makes
  reproducing the placement trivial if the calibration has to be
  redone.
- For multi-camera setups (`tiger` + `shark`), see Phase 3 below for
  the parallel-axes precondition.

### 2b. Run the homography solver

```
PyLorex/PyLorex/script_run_homography.py
```

- Set `camera_name`.
- Set `origin_preset` (default `"TR"`) to choose which corner of the
  detected grid maps to the board's origin. Keep this consistent
  across cameras unless you have a specific reason not to.
- Captures a single frame from the camera, detects the dot grid,
  solves PnP, and saves:
  - `Calibration/Results/pose_<camera>.npz`
  - `Calibration/Results/pose_<camera>.json`
- Both files contain `K_scaled`, `dist`, `R_pnp`, `t_pnp`, `H_raw`,
  `H_undistorted`, plus debug fields.

**Acceptance criteria:**

- The script prints the camera-to-board distance from PnP
  (`distance to board (mm)`). Sanity-check it against your tape
  measure to the floor — should agree to a few mm.
- Reprojection error is printed. Should be a fraction of a px for a
  static board.

## Phase 3: Cross-Camera Alignment

When two or more cameras cover the arena, one is designated reference
(by convention: `tiger`) and the others are translated into the
reference's frame at runtime.

**Precondition for the simple translation-only model:** both
calibration boards must be placed with **parallel axes** (same
orientation in the room). If they're rotated relative to each other,
`shark2tiger_delta` cannot capture the rotation and `shark` data will
land in the unified frame with the wrong heading.

If the cameras cannot both see the same physical board (e.g. cameras
far apart), use two boards laid with the same orientation, and use a
tape measure to establish the offset between their two origins.

### 3a. Measure the offset

After Phase 2 has produced a pose bundle for each camera, measure (in
millimetres, in the room) the world-coordinate offset between
`shark`'s origin and `tiger`'s origin. Sign convention: the delta is
what to **add** to a shark-frame `(x, y)` to get a tiger-frame
`(x, y)`.

### 3b. Set the delta

Edit `PyLorex/LorexLib/Settings.py`:

```python
shark2tiger_delta_x = ...   # mm
shark2tiger_delta_y = ...   # mm
```

These are read at runtime by `ServerClient.get_trackers`, so any
process started after the edit (tracking server, scripts) will pick
them up.

## Phase 4: Verification

Quick end-to-end check that the calibration produces honest tracking.

1. **Restart the PyLorex tracking server** so it loads the new pose
   bundles and the new `shark2tiger_delta` values.
2. **Take a fresh environment snapshot** from the 3PiRobot side:
   `Control_code/SCRIPT_TakeEnvSnapshot.py`. Open the resulting
   `arena_tiger.png` / `arena_shark.png` and confirm the arena's foam
   wall structure is visible with margin around it. (If the wall is
   clipped or sits at the image edge, the cameras need to be
   physically re-aimed before continuing.)
3. **Park the robot in a position visible to both cameras** and run
   `Control_code/SCRIPT_ProbeTrackerAtWall.py`. Both cameras should
   report nearly the same marker `(x, y)`. A consistent offset means
   `shark2tiger_delta_{x,y}` needs adjusting; a position-dependent
   discrepancy points to one camera's extrinsics being off.
4. **Park the robot flush against a known wall** and confirm the
   robot circle in `TempOutput/probe.png` sits flush against the
   green wall dots from `arena_features.npz`. (Requires the new
   arena geometry to have been built — see below.)

If all four checks pass, the calibration chain is healthy. Then
proceed with:

- Re-annotate the new arena snapshot (green polylines along the foam
  top edge — see arena geometry docs).
- Re-run `Control_code/SCRIPT_BuildArenaGeometry.py` to regenerate
  `arena_features.npz`.
- Resume acquisitions / policy training.

## Notes

- The marker-height bias correction in `LorexLib/Lorex.py` relies on
  `R_pnp, t_pnp` and `Settings.marker_height_mm`. As long as the
  marker is physically mounted at the assumed height (150 mm by
  default), it produces correct marker `(X, Y)` automatically after
  a fresh calibration.
- `aruco_yaw_offset_deg` and `aruco_forward_axis` in
  `LorexLib/Settings.py` are implicitly calibrated against the
  marker's mounting on the robot, not against the camera pose. They
  do not need to change when the cameras move (only if the marker
  itself is remounted with a different orientation on the robot).
