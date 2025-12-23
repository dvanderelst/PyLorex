# Calibration Process

## Phase 1: Intrinsic Calibration

+ Collect intrinsic images per camera, one camera at a time:
   - `script_collect_intrinsic_images.py`
+ Run the intrinsic calibration from the collected images:
   - `script_run_intrinsic_dots.py`
+ Assess/validate the resulting intrinsics:
   - `script_assess_intrinsics.py`

## Phase 2: Homography Calibration

+ Collect one homography image per camera to define origin and axes:
   - `script_run_homography.py`
  
## Phase 3: Cross-Camera Alignment

**The `tiger` camera is the reference. Data from `shark` is translated into the `tiger` frame.**

+ After homography, measure the world-coordinate offset between the two
   camera frames (in millimetres).

+ Set the translation in `PyLorex/Library/Settings.py`:
  + `shark2tiger_delta_x`
  + `shark2tiger_delta_y`
