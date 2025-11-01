import cv2 as cv
import pandas as pd
from os import path
from pathlib import Path
import numpy as np
from math import atan2, degrees
from library import Grabber
from library import Settings
from library import Utils
from library import CalibIO
from library import Homography as hg

half = Settings.aruco_size / 2.0
obj_square = np.array([[-half, half, 0.0],[half, half, 0.0],[half, -half, 0.0],[-half, -half, 0.0]], np.float32)
red_color = (0, 0, 255)
green_color = (0, 255, 0)
blue_color = (255, 0, 0)
label_color = (10, 200, 10)
grey_color = (240, 240, 240)
dark_grey_color1 = (120, 120, 120)
dark_grey_color2 = (160, 160, 160)
cv_font = cv.FONT_HERSHEY_SIMPLEX

def parse_detections(detections):
    detected = detections[0]
    lines = []
    for x in detected:
        tag_id = x['id']
        x_mm = x['floor_xy_mm'][0]
        y_mm = x['floor_xy_mm'][1]
        yaw_deg = x['yaw_deg']
        line =[tag_id, x_mm, y_mm, yaw_deg]
        lines.append(line)
    header = ['id', 'x', 'y', 'yaw']
    if len(lines) > 0:
        lines = pd.DataFrame(lines)
        lines.columns = header
    else:
        lines = pd.DataFrame(columns = header)
    return lines


def compose_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T

def ipt(p):
    return tuple(np.rint(p).astype(int))

def pt2i(p):
    a = np.asarray(p).reshape(-1)[:2]
    if a.size < 2 or not np.all(np.isfinite(a)): return None
    return (int(round(float(a[0]))), int(round(float(a[1]))))

def cam_to_px(K, Xc):
    Xc = np.asarray(Xc, dtype=float).reshape(3)
    if not np.isfinite(Xc).all() or Xc[2] <= 1e-6: return None
    x = Xc[0] / Xc[2]; y = Xc[1] / Xc[2]
    u = K[0,0]*x + K[0,2]; v = K[1,1]*y + K[1,2]
    return pt2i([u, v])

def arrow(img, p0, p1, color, thick=2):
    p0 = pt2i(p0); p1 = pt2i(p1)
    if p0 is None or p1 is None: return  # skip bad/NaN points
    color = tuple(int(c) for c in color)
    cv.arrowedLine(img, p0, p1, color, thick, cv.LINE_AA, 0, 0.08)




def invert_extrinsics(R, t):
    """Inverse of [R|t] where X_cam = R*X_board + t -> returns board_from_cam."""
    R_inv = R.T
    t_inv = -R_inv @ t.reshape(3)
    return R_inv, t_inv

def yaw_from_R_board_tag(R_board_tag, forward_axis='x', offset_deg=0.0):
    ax = 0 if forward_axis.lower() == 'x' else 1  # 0->X, 1->Y in tag frame
    v = R_board_tag[:, ax].reshape(3)             # forward axis expressed in board frame
    yaw = degrees(atan2(float(v[1]), float(v[0])))  # atan2(y, x)
    yaw = (yaw + offset_deg + 360.0) % 360.0
    return yaw

def load_calibration(camera_name):
    """Return (K, dist, size) from a saved intrinsics YAML."""
    p = Utils.get_calibration_paths(camera_name)
    intrinsics_yml = p["intrinsics_yml"]
    fs = cv.FileStorage(str(intrinsics_yml), cv.FILE_STORAGE_READ)
    if not fs.isOpened(): raise IOError(f"Cannot open {intrinsics_yml}")
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    width = int(fs.getNode("image_width").real())
    height = int(fs.getNode("image_height").real())
    fs.release()
    print(f"[calib] {camera_name}: {width}x{height}")
    return K, dist, (width, height)

def undistort_image(img, K, dist, alpha=1.0):
    """One-off undistort (slower than remap)."""
    h, w = img.shape[:2]
    newK, _ = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
    return cv.undistort(img, K, dist, None, newK)

def make_aruco_params(aruco):
    # New or legacy API constructor
    if hasattr(aruco, "DetectorParameters"):
        p = aruco.DetectorParameters()
    elif hasattr(aruco, "DetectorParameters_create"):
        p = aruco.DetectorParameters_create()
    else:
        raise RuntimeError("[aruco] Could not create DetectorParameters")

    # --- Thresholding (scene is consistent; keep range tight & fast) ---
    # Smaller window range = less noise sensitivity, faster.
    if hasattr(p, "adaptiveThreshWinSizeMin"):  p.adaptiveThreshWinSizeMin  = 5
    if hasattr(p, "adaptiveThreshWinSizeMax"):  p.adaptiveThreshWinSizeMax  = 23
    if hasattr(p, "adaptiveThreshWinSizeStep"): p.adaptiveThreshWinSizeStep = 2
    if hasattr(p, "adaptiveThreshConstant"):    p.adaptiveThreshConstant    = 7  # try 6..9 if misses
    # --- Size gating (skip blobs that are way too small/big) ---
    # 46px side ≈ 184px perimeter; these rates suit 1080–4K frames well.
    if hasattr(p, "minMarkerPerimeterRate"):    p.minMarkerPerimeterRate    = 0.03
    if hasattr(p, "maxMarkerPerimeterRate"):    p.maxMarkerPerimeterRate    = 0.30

    # --- Geometry & corner precision ---
    if hasattr(p, "polygonalApproxAccuracyRate"): p.polygonalApproxAccuracyRate = 0.03
    if hasattr(p, "minCornerDistanceRate"):       p.minCornerDistanceRate       = 0.05
    if hasattr(p, "minDistanceToBorder"):         p.minDistanceToBorder         = 3

    # Subpixel corner refinement = less pose jitter
    if hasattr(aruco, "CORNER_REFINE_SUBPIX") and hasattr(p, "cornerRefinementMethod"):
        p.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(p, "cornerRefinementWinSize"):       p.cornerRefinementWinSize       = 5  # at 46px, 3–5 is good
    if hasattr(p, "cornerRefinementMaxIterations"): p.cornerRefinementMaxIterations = 30
    if hasattr(p, "cornerRefinementMinAccuracy"):   p.cornerRefinementMinAccuracy   = 0.01
    # --- Cell warping (match cell size to your tag in pixels) ---
    # 5x5 dict ⇒ 7×7 total with border; 46px/7 ≈ 6.6px per cell.
    if hasattr(p, "perspectiveRemovePixelPerCell"):         p.perspectiveRemovePixelPerCell = 8  # try 6–8
    if hasattr(p, "perspectiveRemoveIgnoredMarginPerCell"): p.perspectiveRemoveIgnoredMarginPerCell = 0.33
    # --- Decoding strictness ---
    if hasattr(p, "errorCorrectionRate"):                  p.errorCorrectionRate = 0.5  # slightly stricter than default
    if hasattr(p, "maxErroneousBitsInBorderRate"):         p.maxErroneousBitsInBorderRate = 0.5
    if hasattr(p, "minOtsuStdDev"):                        p.minOtsuStdDev = 5.0
    # --- Polarity safety ---
    if hasattr(p, "detectInvertedMarker"):  p.detectInvertedMarker = True
    return p

class LorexCamera:
    def __init__(self, camera_name, auto_start=True, alpha=1.0):
        self.camera_name = camera_name
        self.channel = Settings.channels[camera_name]
        self.paths = Utils.get_calibration_paths(camera_name)
        self.grabber = Grabber.RTSPGrabber(channel=self.channel, auto_start=auto_start)
        # Calibration state
        self.K = None
        self.dist = None
        self.calib_size = None  # (W, H) from YAML
        self.alpha = float(alpha)  # 0=crop edges, 1=keep FOV
        self.map1 = None
        self.map2 = None
        self.map_size = None  # (W, H) of maps
        self.newK = None      # <-- effective K for undistorted output at map_size
        # Pose/homography bundle (optional)
        self.bundle = None
        self.load_calibration()
        # Settings for grid plotting
        self.axis_len_mm = 100.0
        self.grid_step_mm = 100.0
        self.grid_extent_mm = (-2500, 2500, -2500, 2500)
        # --- ArUco detector cache / knobs ---
        self.aruco = None
        self.aruco_dict = None
        self.aruco_params = None
        self.aruco_detector = None
        self.aruco_detect_scale = Settings.aruco_detect_scale
        self.aruco_fast_refine = Settings.aruco_fast_refine
        self.aruco_refine_win = Settings.aruco_refine_win
        self.aruco_refine_iters = Settings.aruco_refine_iters

    def ensure_aruco(self):
        if self.aruco is None:
            try:
                import cv2.aruco as m
            except Exception:
                raise RuntimeError("[aruco] module not available")
            self.aruco = m

        # resolve dictionary object from Settings.aruco_dict_name
        if self.aruco_dict is None:
            raw = str(getattr(Settings, "aruco_dict_name", "DICT_5X5_100"))
            n = raw.split(".")[-1]  # allow "cv2.aruco.DICT_5X5_100"
            if not n.startswith("DICT_"): n = "DICT_" + n
            if hasattr(self.aruco, n):
                dict_id = getattr(self.aruco, n)
                self.aruco_dict = self.aruco.getPredefinedDictionary(int(dict_id))
            else:
                opts = [k for k in dir(self.aruco) if k.startswith("DICT_")]
                raise ValueError(f"[aruco] Unknown dictionary name: {raw}. Try: {', '.join(opts[:12])} ...")

        if self.aruco_params is None:
            p = make_aruco_params(self.aruco)
            if hasattr(p, "cornerRefinementWinSize"): p.cornerRefinementWinSize = self.aruco_refine_win
            if hasattr(p, "cornerRefinementMaxIterations"): p.cornerRefinementMaxIterations = self.aruco_refine_iters
            self.aruco_params = p

        if self.aruco_detector is None and hasattr(self.aruco, "ArucoDetector"):
            self.aruco_detector = self.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)

    def detect_markers(self, gray):
        self.ensure_aruco()
        s = float(self.aruco_detect_scale)
        if s != 1.0:
            small = cv.resize(gray, None, fx=s, fy=s, interpolation=cv.INTER_AREA)
            if self.aruco_detector is not None:
                corners, ids, _ = self.aruco_detector.detectMarkers(small)
            else:
                corners, ids, _ = self.aruco.detectMarkers(small, self.aruco_dict, parameters=self.aruco_params)
            if ids is None or len(ids) == 0: return None, None
            corners = [c / s for c in corners]
        else:
            if self.aruco_detector is not None:
                corners, ids, _ = self.aruco_detector.detectMarkers(gray)
            else:
                corners, ids, _ = self.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
            if ids is None or len(ids) == 0: return None, None
        return corners, ids

    def load_calibration(self):
        intrinsics_yml = self.paths.get("intrinsics_yml")
        if intrinsics_yml and path.exists(intrinsics_yml):
            try:
                self.K, self.dist, self.calib_size = load_calibration(self.camera_name)
                print(f"[calib] loaded for {self.camera_name}")
            except Exception as e:
                print(f"[calib] failed to load: {e}")
        else:
            print(f"[calib] not found for {self.camera_name} at {intrinsics_yml}")


    def start(self):
        self.grabber.start()

    def stop(self):
        self.grabber.stop()

    def wait_ready(self, timeout=5.0):
        return self.grabber.wait_latest_bgr(timeout=timeout)

    def ensure_maps(self, frame_shape):
        """Build rectification maps if missing or size changed."""
        if self.K is None or self.dist is None: return False
        h, w = frame_shape[:2]
        if self.map1 is not None and self.map_size == (w, h): return True
        # Build maps for current frame size (handles cameras that scale/letterbox)
        self.newK, _ = cv.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), self.alpha)
        self.map1, self.map2 = cv.initUndistortRectifyMap(self.K, self.dist, None, self.newK, (w, h), cv.CV_16SC2)
        self.map_size = (w, h)
        return True

    def get_frame(self, undistort=False):
        frame = self.grabber.get_latest_bgr()
        if frame is None: return None
        if undistort and self.K is not None and self.dist is not None:
            if self.ensure_maps(frame.shape):
                return cv.remap(frame, self.map1, self.map2, cv.INTER_LINEAR)
        return frame

    def save(self, p, undistort=True, encode_params=None):
        frame = self.get_frame(undistort=undistort)
        if frame is None:
            print("[save] No frame available")
            return False
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)
        if encode_params is None:
            ext = p.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                encode_params = [int(cv.IMWRITE_JPEG_QUALITY), 95]
            elif ext == ".png":
                encode_params = [int(cv.IMWRITE_PNG_COMPRESSION), 3]
            else:
                encode_params = []
        ok = cv.imwrite(str(p), frame, encode_params)
        if not ok: print(f"[save] Failed writing to {p}")
        return ok

    # ----- convenience helpers -----
    def reload_calibration(self):
        """Re-read YAML and rebuild maps on next frame."""
        intrinsics_yml = self.paths.get("intrinsics_yml")
        if not intrinsics_yml or not path.exists(intrinsics_yml):
            print(f"[calib] YAML missing at {intrinsics_yml}")
            return False
        self.K, self.dist, self.calib_size = load_calibration(self.camera_name)
        self.map1 = self.map2 = None
        self.map_size = None
        self.newK = None
        return True

    def set_alpha(self, alpha):
        """0=crop more (tighter), 1=keep more FOV (more edges)."""
        self.alpha = float(alpha)
        self.map1 = self.map2 = None
        self.map_size = None
        self.newK = None

    def get_current_intrinsics(self, frame_shape, undistort=False):
        """
        Return (K_used, dist_used) that match the pixel geometry the caller will use.
        - If undistort=False: scale K from YAML to this frame size; return dist from YAML.
        - If undistort=True : return the 'newK' for this frame size/alpha; return dist=None (already undistorted).
        """
        if self.K is None or self.dist is None: return None, None
        h, w = frame_shape[:2]
        if not undistort:
            if self.calib_size is None: raise RuntimeError("calib_size is None; cannot scale K")
            Wc, Hc = self.calib_size
            sx = w / float(Wc)
            sy = h / float(Hc)
            K_used = self.K.copy()
            K_used[0, 0] *= sx
            K_used[1, 1] *= sy
            K_used[0, 2] *= sx
            K_used[1, 2] *= sy
            return K_used, self.dist
        else:
            # Make sure newK matches this frame size
            if self.map1 is None or self.map_size != (w, h): self.ensure_maps(frame_shape)
            return self.newK, None  # undistorted pixels, so no dist

    # -------- homography/pose bundle --------
    def load_board_bundle(self):
        """Load H_raw, R, t, K, dist for this camera from saved files (pose_npz/json)."""
        self.bundle = CalibIO.load_pose_bundle(self.camera_name)
        # Basic key sanity
        for k in ("H_raw", "R", "t", "K"):
            if k not in self.bundle: raise KeyError(f"Pose bundle missing key: {k}")
        return True

    def has_bundle(self):
        return hasattr(self, "bundle") and self.bundle is not None

    def pixel_to_board_xy(self, u, v, use_raw=True, K_override=None, dist_override=None):
        """
        Return board-plane (X,Y) in mm for pixel (u,v).
        - If use_raw=True : assumes (u,v) is from a RAW frame; uses H_raw (fast).
        - If use_raw=False: assumes (u,v) is from an undistorted frame.
          * Tries H_undistorted first (fast, if available in bundle).
          * Falls back to ray-plane intersection with PnP pose.
          * If K_override/dist_override provided, uses those intrinsics.
          * Otherwise, derives intrinsics from current frame (assumes RAW).
        """
        if not self.has_bundle():
            raise RuntimeError("No board bundle loaded. Call load_board_bundle().")

        R = self.bundle["R"]
        t = self.bundle["t"]

        if use_raw:
            H = self.bundle["H_raw"]
            X, Y = hg.pixel_to_board_xy_raw(u, v, H)
            return float(X), float(Y)
        else:
            # Undistorted pixels: try fast H_undistorted first
            if "H_undistorted" in self.bundle:
                H = self.bundle["H_undistorted"]
                X, Y = hg.pixel_to_board_xy_raw(u, v, H)
                return float(X), float(Y)

            # Fallback: ray-plane intersection
            # Use override intrinsics if provided, otherwise get from current RAW frame
            if K_override is not None:
                K_used = K_override
                dist_used = dist_override
            else:
                frame = self.get_frame(undistort=False)
                if frame is None: raise RuntimeError("No frame available.")
                K_used, dist_used = self.get_current_intrinsics(frame.shape, undistort=False)

            d_cam = hg.pixel_to_ray_cam(u, v, K_used, dist_used)
            P = hg.intersect_ray_with_board(d_cam, R, t)
            return float(P[0]), float(P[1])

    def get_aruco(self, draw=False, draw_world=True, world_undistort=False, detection_scale=0.5, draw_grid=True):
        """
        Detect ArUco markers in the camera frame.

        Args:
            draw: Whether to draw detection results
            draw_world: Whether to draw world axes and grid
            world_undistort: Whether to undistort the frame for visualization
            detection_scale: Scale factor for detection (0.5 = half resolution, faster)
            draw_grid: Whether to draw the floor grid (can be slow at high res)
        """
        # --- Settings ---
        aruco_size = float(Settings.aruco_size)
        aruco_forward_axis = Settings.aruco_forward_axis
        aruco_yaw_offset_deg = float(Settings.aruco_yaw_offset_deg)
        # --- Ensure pose bundle & aruco handles ---
        if not self.has_bundle():
            try:
                self.load_board_bundle(); print(f"[bundle] Loaded pose bundle for {self.camera_name}")
            except Exception as e:
                print(f"[bundle] Could not load pose bundle: {e}"); self.bundle = None
        self.ensure_aruco()

        # --- Acquire RAW frame ---
        frame = self.get_frame(undistort=False)
        if frame is None: return [], None
        h, w = frame.shape[:2]

        # --- Intrinsics for RAW pixels ---
        K_used, dist_used = self.get_current_intrinsics(frame.shape, undistort=False)
        if K_used is None: return [], (frame.copy() if draw else None)

        # --- Optional rectified world (undistort once, then dist=None) ---
        if draw_world and world_undistort:
            # Use _ensure_maps to get consistent newK (respects self.alpha)
            if self._ensure_maps(frame.shape):
                frame = cv.remap(frame, self.map1, self.map2, cv.INTER_LINEAR)
                K_used, dist_used = self.newK, None  # <-- active intrinsics are now rectified
            else:
                # Fallback if maps cannot be built
                raise RuntimeError("Cannot build undistortion maps.")

        # --- Downsample for detection (major speed boost at high resolutions) ---
        frame_full = frame  # Keep original for drawing
        K_full = K_used     # Keep original for solvePnP
        dist_full = dist_used

        if detection_scale != 1.0:
            h_det = int(h * detection_scale)
            w_det = int(w * detection_scale)
            frame_detect = cv.resize(frame, (w_det, h_det), interpolation=cv.INTER_AREA)
            K_detect = K_used.copy()
            K_detect[0, 0] *= detection_scale  # fx
            K_detect[1, 1] *= detection_scale  # fy
            K_detect[0, 2] *= detection_scale  # cx
            K_detect[1, 2] *= detection_scale  # cy
        else:
            frame_detect = frame
            K_detect = K_used

        # --- Board (floor) extrinsics: guard access (for yaw & height) ---
        R_board_from_cam = None
        t_board_from_cam = None
        if self.bundle is not None and "R" in self.bundle and "t" in self.bundle:
            cam_R_from_board = self.bundle["R"];
            cam_t_from_board = self.bundle["t"].reshape(3)
            R_board_from_cam, t_board_from_cam = invert_extrinsics(cam_R_from_board, cam_t_from_board)

        # --- ArUco dict (robust string->enum) ---
        try:
            aruco_dict_id = getattr(cv.aruco, aruco_dict_name)
        except AttributeError:
            raise ValueError(f"[aruco] Unknown dictionary: {aruco_dict_name!r}.")
        aruco_dict = cv.aruco.getPredefinedDictionary(int(aruco_dict_id))

        # --- Detect on downsampled grayscale (major speedup) ---
        gray = cv.cvtColor(frame_detect, cv.COLOR_BGR2GRAY)
        corners, ids = detect_markers(cv.aruco, gray, aruco_dict)

        # --- Scale corners back to full resolution ---
        if detection_scale != 1.0 and corners is not None and len(corners) > 0:
            corners = [c / detection_scale for c in corners]

        vis = frame_full.copy() if draw else None

        # --- Draw world first (in the SAME geometry we're in now) -----------------
        if draw and draw_world:
            vis = self.draw_board_axes_and_grid(
                img=vis,
                undistort=False,  # <-- DO NOT undistort again
                K_used=K_full, dist_used=dist_full,  # <-- use full-res intrinsics (corners are scaled back)
                assume_img_rectified=(dist_full is None),  # True when world_undistort=True
                draw_grid=draw_grid  # <-- pass grid drawing preference
            )

        # --- Early out ---
        if ids is None or len(ids) == 0:
            if draw and vis is not None:
                Path(Settings.temp_dir).mkdir(parents=True, exist_ok=True)
                out = path.join(Settings.temp_dir, f"aruco_{self.camera_name}.jpg");
                cv.imwrite(out, vis)
            return [], vis

        # --- Prepare square model for IPPE_SQUARE ---
        s = float(aruco_size)
        obj_square = np.float32([[-s / 2, -s / 2, 0], [s / 2, -s / 2, 0], [s / 2, s / 2, 0], [-s / 2, s / 2, 0]])

        detections = []
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(-1, 2).astype(np.float32)
            pts = pts / self.aruco_detect_scale
            imgp = pts.reshape(-1, 1, 2)

            ok, rvec, tvec = cv.solvePnP(obj_square, imgp, K_full, dist_full, flags=cv.SOLVEPNP_IPPE_SQUARE)
            if not ok: continue

            u, v = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            # When dist_full is None, frame is undistorted; pass intrinsics to pixel_to_board_xy
            if dist_full is not None:
                # RAW frame: use fast H_raw homography
                X_floor, Y_floor = self.pixel_to_board_xy(u, v, use_raw=True)
            else:
                # Undistorted frame: use ray-plane intersection with correct intrinsics
                X_floor, Y_floor = self.pixel_to_board_xy(u, v, use_raw=False, K_override=K_full, dist_override=None)
            # Orientation & height if extrinsics known
            yaw_deg = None
            height_mm = None
            if R_board_from_cam is not None:
                R_cam_tag, _ = cv.Rodrigues(rvec);
                t_cam_tag = tvec.reshape(3)
                R_board_tag = R_board_from_cam @ R_cam_tag
                t_board_tag = R_board_from_cam @ t_cam_tag + t_board_from_cam
                fa = (aruco_forward_axis or "x").lower()
                if fa not in ("x", "y"): print(f"[warn] forward_axis={aruco_forward_axis!r} invalid; using 'x'"); fa = "x"
                yaw_deg = yaw_from_R_board_tag(R_board_tag, fa, aruco_yaw_offset_deg)
                height_mm = float(t_board_tag[2])

            det = {
                "id": int(tag_id),
                "floor_xy_mm": (float(X_floor), float(Y_floor)),
                "yaw_deg": None if yaw_deg is None else float(yaw_deg),
                "height_mm": None if height_mm is None else float(height_mm),
                "rvec_cam_tag": rvec, "tvec_cam_tag": tvec,
                "center_px": (u, v), "corners_px": pts.copy(),
            }
            detections.append(det)

            if draw and vis is not None:
                cv.aruco.drawDetectedMarkers(vis, [pts.reshape(1, 4, 2)], None)
                cv.drawFrameAxes(vis, K_full, dist_full, rvec, tvec, 0.25 * aruco_size)
                label = f"id={tag_id} "
                if det["yaw_deg"] is not None:
                    label += f"{det['yaw_deg']:.0f}dg"
                if det["floor_xy_mm"] is not None:
                    fx, fy = det['floor_xy_mm']
                    label += f" [{fx:.0f} {fy:.0f}]"
                font = cv.FONT_HERSHEY_SIMPLEX
                cv.putText(vis, label, (int(u) + 8, int(v) - 8), font, 0.6, label_color, 2, cv.LINE_AA)

                #----------------
                # how many pixels long are our X-axis and the tag side (measured vs predicted)?
                fx = float(K_used[0, 0]);
                Z = float(tvec[2]);
                L = float(Settings.axis_draw_length_mm)

                axis_px_meas = np.hypot(*(np.asarray(imgpts[1]).ravel()[:2] - np.asarray(imgpts[0]).ravel()[:2]))
                axis_px_pred = fx * L / max(Z, 1e-6)

                side_px_meas = np.hypot(*(pts[1] - pts[0]))  # marker edge in pixels from detections
                side_px_pred = fx * float(Settings.aruco_size) / max(Z, 1e-6)

                print(f"axis_px meas={axis_px_meas:.1f} pred={axis_px_pred:.1f}  |  side_px meas={side_px_meas:.1f} pred={side_px_pred:.1f}  Z={Z:.0f}mm")

                H, W = frame.shape[:2]
                fx = float(K_used[0, 0]);
                cx = float(K_used[0, 2])
                print(f"frame={W}x{H}  fx={fx:.1f}  cx={cx:.1f}")
                aruco_size = float(Settings.aruco_size)  # mm
                Z_hat = fx * aruco_size / max(side_px_meas, 1e-6)
                print(f"Z_from_image≈{Z_hat:.0f} mm  vs  Z_from_PnP={float(tvec[2]):.0f} mm")

                # --- robust forward/heading line in millimetres ---
                heading_draw_length = Settings.heading_draw_length  # physical length
                fa = (Settings.aruco_forward_axis or "x").lower()
                axis = np.array([1.0, 0.0, 0.0]) if fa == "x" else np.array([0.0, 1.0, 0.0])

                R, _ = cv.Rodrigues(rvec);
                t = tvec.reshape(3)

                # camera-frame start/end of the forward line
                p0_cam = t
                p1_cam = R.dot(axis * heading_draw_length) + t

                # if endpoint is behind/near camera, shorten to a safe length
                if not np.isfinite(p1_cam).all() or p1_cam[2] <= 10.0:
                    # pick a length proportional to distance, but keep it sane
                    axis_draw_length_mm = Settings.axis_draw_length_mm
                    p1_cam = R.dot(axis * axis_draw_length_mm) + t
                    # if still invalid, skip drawing
                    if not np.isfinite(p1_cam).all() or p1_cam[2] <= 1e-3:
                        p1_cam = None

                # project camera->pixels with our own math (avoids dtype/overflow issues)
                P0 = cam_to_px(K_used, p0_cam)
                P1 = cam_to_px(K_used, p1_cam) if p1_cam is not None else None

                # clamp to image bounds and draw
                if P0 is not None and P1 is not None:
                    H, W = vis.shape[:2]
                    if abs(P0[0]) <= 10 * W and abs(P0[1]) <= 10 * H and abs(P1[0]) <= 10 * W and abs(P1[1]) <= 10 * H:
                        cv.arrowedLine(vis, P0, P1, green_color, 2, cv.LINE_AA, 0, 0.08)

        if draw and vis is not None:
            Path(Settings.temp_dir).mkdir(parents=True, exist_ok=True)
            out = path.join(Settings.temp_dir, f"aruco_{self.camera_name}.jpg");
            cv.imwrite(out, vis)

        return detections, vis

    def draw_board_axes_and_grid(
        self,
        img=None,
        undistort=False,
        save_to=None,
        K_used=None,
        dist_used=None,
        assume_img_rectified=False,
        draw_grid=True
    ):
        """
        Draw world (board) XY axes and a Z=0 grid using ONLY the pose bundle.

        Modes:
        - If K_used/dist_used are provided, they are used directly (no internal scaling/undistort).
          Set assume_img_rectified=True when img is already undistorted and dist_used is None.
        - Else, function derives K/dist from the bundle (scaled to frame size) and:
            * undistort=False (default): draw on RAW frame, grid as polylines (follows distortion).
            * undistort=True: undistort the frame and draw with newK, dist=None.

        Args:
            draw_grid: Whether to draw the floor grid (can be expensive at high resolutions)

        Returns: the image (or None on failure).
        """
        draw_labels = True
        # --- Ensure pose bundle is available -------------------------------------
        if not self.has_bundle():
            try:
                self.load_board_bundle()
                print(f"[bundle] Loaded pose bundle for {self.camera_name}")
            except Exception as e:
                print(f"[axes] Could not load pose bundle: {e}")
                return None

        b = self.bundle
        if b is None:
            print("[axes] No bundle available.")
            return None

        for k in ("R", "t", "K"):
            if k not in b:
                print(f"[axes] Bundle missing '{k}'.")
                return None

        cam_R_from_board = np.asarray(b["R"], dtype=np.float64)
        cam_t_from_board = np.asarray(b["t"], dtype=np.float64).reshape(3)
        K_bundle = np.asarray(b["K"], dtype=np.float64)
        dist_bundle = None if b.get("dist") is None else np.asarray(b["dist"], dtype=np.float64)

        # --- Acquire image (RAW unless caller already passed an image) ------------
        created_img = False
        if img is None:
            img = self.get_frame(undistort=False)
            created_img = True
            if img is None:
                print("[axes] No image available.")
                return None
        h, w = img.shape[:2]

        # --- Determine intrinsics to use -----------------------------------------
        # Case A: caller supplied active intrinsics (recommended from get_aruco)
        if K_used is not None:
            # Use as-is; caller guarantees they match the 'img' geometry.
            pass
        else:
            # Case B: derive from bundle (scale to current frame and optionally undistort)
            # Get the frame size that K_bundle corresponds to
            bsize = b.get("frame_size")  # Now properly loaded from CalibIO
            if bsize is None:
                # Fallback: try other possible keys or use calibration size
                bsize = b.get("size") or b.get("calib_size") or self.calib_size

            if bsize is None:
                raise RuntimeError(
                    "[axes] Cannot determine bundle K frame size. "
                    "Bundle should contain 'frame_size' metadata."
                )

            bW, bH = int(bsize[0]), int(bsize[1])

            # Scale K_bundle from its native size (bW, bH) to current frame size (w, h)
            K_scaled = K_bundle.copy()
            if w != bW or h != bH:
                sx, sy = w / float(bW), h / float(bH)
                K_scaled[0, 0] *= sx  # fx
                K_scaled[0, 2] *= sx  # cx
                K_scaled[1, 1] *= sy  # fy
                K_scaled[1, 2] *= sy  # cy

            if undistort:
                alpha = float(getattr(self, "alpha", 1.0))  # keep FOV by default
                newK, _ = cv.getOptimalNewCameraMatrix(K_scaled, dist_bundle, (w, h), alpha)
                img = cv.undistort(img, K_scaled, dist_bundle, None, newK)
                K_used, dist_used = newK, None
            else:
                K_used, dist_used = K_scaled, dist_bundle

        # --- If caller says img is already rectified, ensure we won't undistort again
        if assume_img_rectified and dist_used is None:
            # Nothing to do; we trust K_used/dist_used from caller
            pass

        # --- Prepare projection (board -> camera) --------------------------------
        rvec, _ = cv.Rodrigues(cam_R_from_board)
        tvec = cam_t_from_board.reshape(3, 1)

        def pj(P):
            """Project Nx3 board points to image pixels using current K_used/dist_used."""
            P = np.asarray(P, dtype=np.float32).reshape(-1, 1, 3)
            uv, _ = cv.projectPoints(P, rvec, tvec, K_used, dist_used)
            return uv.reshape(-1, 2)

        # --- Draw axes on Z=0 (X red, Y green) -----------------------------------
        axis_len = float(self.axis_len_mm)
        O   = np.array([0.0,       0.0,      0.0 ], dtype=np.float32)
        Xpt = np.array([axis_len,  0.0,      0.0 ], dtype=np.float32)
        Ypt = np.array([0.0,       axis_len, 0.0 ], dtype=np.float32)
        Ouv, Xuv, Yuv = pj([O, Xpt, Ypt])

        cv.arrowedLine(img, ipt(Ouv), ipt(Xuv), red_color, 2, cv.LINE_AA)  # X
        cv.arrowedLine(img, ipt(Ouv), ipt(Yuv), green_color, 2, cv.LINE_AA)  # Y

        if draw_labels:
            cv.putText(img, "X", ipt(Xuv), cv_font, 0.6, red_color, 2, cv.LINE_AA)
            cv.putText(img, "Y", ipt(Yuv), cv_font, 0.6, green_color, 2, cv.LINE_AA)
            cv.putText(img, "Origin", ipt(Ouv + np.array([6, -6], dtype=np.float32)), cv_font , 0.5, grey_color, 1, cv.LINE_AA)

        # --- Draw Z=0 grid as polylines (follow distortion exactly when dist_used) -
        if draw_grid:
            xmin, xmax, ymin, ymax = self.grid_extent_mm
            step = float(self.grid_step_mm)
            grey1 = dark_grey_color1

            xi0 = int(np.ceil(xmin / step)); xi1 = int(np.floor(xmax / step))
            yi0 = int(np.ceil(ymin / step)); yi1 = int(np.floor(ymax / step))
            xs = (np.arange(xi0, xi1 + 1, dtype=int) * step).astype(np.float32)
            ys = (np.arange(yi0, yi1 + 1, dtype=int) * step).astype(np.float32)

            # Optimized sampling density: fewer points = faster drawing
            # 16-32 points sufficient for smooth curves at most resolutions
            npts = min(32, max(16, int(0.08 * (h + w) // 50)))

            def line_poly(world_start, world_end, n=npts):
                ws = np.array(world_start, dtype=np.float32)
                we = np.array(world_end, dtype=np.float32)
                ts = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                pts = ws[None, :] * (1.0 - ts) + we[None, :] * ts  # Nx3
                uv = pj(pts)                                        # Nx2
                return np.rint(uv).astype(np.int32).reshape(-1, 1, 2)

            # vertical lines: x = const
            for x in xs:
                poly = line_poly([x, ymin, 0.0], [x, ymax, 0.0])
                cv.polylines(img, [poly], False, grey1, 1, cv.LINE_AA)

            # horizontal lines: y = const
            for y in ys:
                poly = line_poly([xmin, y, 0.0], [xmax, y, 0.0])
                cv.polylines(img, [poly], False, grey1, 1, cv.LINE_AA)

            # Emphasize axes lines if in range
            grey2 = dark_grey_color2
            if xmin <= 0.0 <= xmax:
                cv.polylines(img, [line_poly([0.0, ymin, 0.0], [0.0, ymax, 0.0])], False, grey2, 1, cv.LINE_AA)
            if ymin <= 0.0 <= ymax:
                cv.polylines(img, [line_poly([xmin, 0.0, 0.0], [xmax, 0.0, 0.0])], False, grey2, 1, cv.LINE_AA)

        if save_to is not None:
            if not cv.imwrite(str(save_to), img): print(f"[axes] Failed to write {save_to}")

        return img
