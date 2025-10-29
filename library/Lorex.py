import cv2 as cv
from os import path
from pathlib import Path
import numpy as np
from math import atan2, degrees
from library import Grabber
from library import Settings
from library import Utils
from library import CalibIO
from library import Homography as hg

axis_len_mm = 100.0  # <- new: axis length
grid_step_mm = 100.0  # <- new: grid spacing
grid_extent_mm = (-2500, 2500, -2500, 2500)  # <- new: grid bounds

def compose_T(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3,:3] = R
    T[:3, 3] = t.reshape(3)
    return T


def ipt(p):
    return tuple(np.rint(p).astype(int))

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


def detect_markers(aruco, frame, dict_obj):
    # New API (4.7+)
    if hasattr(aruco, "ArucoDetector"):
        det_params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dict_obj, det_params)
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        # Old API fallback
        if hasattr(aruco, "DetectorParameters_create"):
            det_params = aruco.DetectorParameters_create()
        else:
            det_params = aruco.DetectorParameters()  # just in case
        corners, ids, _ = aruco.detectMarkers(frame, dict_obj, parameters=det_params)
    return corners, ids

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


    def _ensure_maps(self, frame_shape):
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
            if self._ensure_maps(frame.shape):
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
        if not ok:
            print(f"[save] Failed writing to {p}")
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
        if self.K is None or self.dist is None:
            return None, None
        h, w = frame_shape[:2]
        if not undistort:
            if self.calib_size is None:
                raise RuntimeError("calib_size is None; cannot scale K")
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
            if self.map1 is None or self.map_size != (w, h):
                self._ensure_maps(frame_shape)
            return self.newK, None  # undistorted pixels, so no dist

    # -------- homography/pose bundle --------
    def load_board_bundle(self):
        """Load H_raw, R, t, K, dist for this camera from saved files (pose_npz/json)."""
        self.bundle = CalibIO.load_pose_bundle(self.camera_name)
        # Basic key sanity
        for k in ("H_raw", "R", "t", "K"):
            if k not in self.bundle:
                raise KeyError(f"Pose bundle missing key: {k}")
        return True

    def has_bundle(self):
        return hasattr(self, "bundle") and self.bundle is not None

    def pixel_to_board_xy(self, u, v, use_raw=True):
        """
        Return board-plane (X,Y) in mm for pixel (u,v).
        - If use_raw=True : assumes (u,v) is from a RAW frame; uses H_raw (fast).
        - If use_raw=False: ray-plane intersection with PnP pose (works for raw or undistorted,
                            as long as K/dist match the pixels you queried).
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
            # Match intrinsics to *whatever* pixels you are using for (u,v).
            # Here we assume RAW for simplicity; switch undistort=True if you query an undistorted frame.
            frame = self.get_frame(undistort=False)
            if frame is None: raise RuntimeError("No frame available.")
            K_used, dist_used = self.get_current_intrinsics(frame.shape, undistort=False)
            d_cam = hg.pixel_to_ray_cam(u, v, K_used, dist_used)
            P = hg.intersect_ray_with_board(d_cam, R, t)
            return float(P[0]), float(P[1])

    def get_aruco(self, draw=False, draw_world=True, world_undistort=False):
        # --- Settings ---
        aruco_dict = Settings.aruco_dict  # e.g. "DICT_5X5_100"
        aruco_size = float(Settings.aruco_size)  # mm
        aruco_forward_axis = Settings.aruco_forward_axis  # 'x' or 'y'
        aruco_yaw_offset_deg = float(Settings.aruco_yaw_offset_deg)

        # --- Ensure bundle is loaded ---
        if not self.has_bundle():
            try:
                self.load_board_bundle()
                print(f"[bundle] Loaded pose bundle for {self.camera_name}")
            except Exception as e:
                print(f"[bundle] Could not load pose bundle: {e}")
                self.bundle = None

        # --- Acquire RAW frame (detection expects RAW to match K/dist & H_raw) ---
        frame = self.get_frame(undistort=False)
        if frame is None: return [], None

        # --- Intrinsics for RAW pixels (scaled to current frame) ---
        K_used, dist_used = self.get_current_intrinsics(frame.shape, undistort=False)
        if K_used is None: return [], (frame.copy() if draw else None)

        # --- Board (floor) extrinsics: guard access ---
        R_board_from_cam = None
        t_board_from_cam = None
        if self.bundle is not None and "R" in self.bundle and "t" in self.bundle:
            cam_R_from_board = self.bundle["R"]
            cam_t_from_board = self.bundle["t"].reshape(3)
            R_board_from_cam, t_board_from_cam = invert_extrinsics(cam_R_from_board, cam_t_from_board)

        # --- ArUco dict ---
        aruco = cv.aruco
        if not hasattr(aruco, aruco_dict):
            raise ValueError(f"ArUco dictionary '{aruco_dict}' not found in cv.aruco.")
        dict_id = getattr(aruco, aruco_dict)
        aruco_dict_loaded = aruco.getPredefinedDictionary(dict_id)

        # --- Detect (handles both API versions) ---
        corners, ids = detect_markers(aruco, frame, aruco_dict_loaded)
        vis = frame.copy() if draw else None

        # --- Optional: draw world axes/grid FIRST so markers/axes both sit on RAW ---
        if draw and draw_world:
            if world_undistort:
                print("[get_aruco] world_undistort=True not supported here; drawing on RAW to match marker overlays.")
            vis = self.draw_board_axes_and_grid(img=vis, undistort=False, save_to=None)

        if ids is None or len(ids) == 0:
            # Save (if draw) and return early
            if draw and vis is not None:
                temp_dir = Settings.temp_dir
                camera_name = self.camera_name
                output_file = path.join(temp_dir, f"aruco_{camera_name}.jpg")
                cv.imwrite(output_file, vis)
            return [], vis

        # --- Square object points (TL,TR,BR,BL) in mm ---
        half = aruco_size / 2.0
        obj_square = np.array(
            [[-half, half, 0.0],
             [half, half, 0.0],
             [half, -half, 0.0],
             [-half, -half, 0.0]], dtype=np.float32
        )

        detections = []
        for c, tag_id in zip(corners, ids.flatten()):
            pts = c.reshape(-1, 2).astype(np.float32)  # (4,2)
            imgp = pts.reshape(-1, 1, 2)

            ok, rvec, tvec = cv.solvePnP(
                obj_square, imgp, K_used, dist_used, flags=cv.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue

            # Floor (X,Y) at tag center via homography (RAW path)
            u, v = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            X_floor, Y_floor = self.pixel_to_board_xy(u, v, use_raw=True)

            # Orientation & height (if extrinsics known)
            yaw_deg = None
            height_mm = None
            if R_board_from_cam is not None:
                R_cam_tag, _ = cv.Rodrigues(rvec)
                t_cam_tag = tvec.reshape(3)
                R_board_tag = R_board_from_cam @ R_cam_tag
                t_board_tag = R_board_from_cam @ t_cam_tag + t_board_from_cam
                yaw_deg = yaw_from_R_board_tag(R_board_tag, aruco_forward_axis, aruco_yaw_offset_deg)
                height_mm = float(t_board_tag[2])

            det = {
                "id": int(tag_id),
                "floor_xy_mm": (float(X_floor), float(Y_floor)),
                "yaw_deg": None if yaw_deg is None else float(yaw_deg),
                "height_mm": None if height_mm is None else float(height_mm),
                "rvec_cam_tag": rvec,
                "tvec_cam_tag": tvec,
                "center_px": (u, v),
                "corners_px": pts.copy(),
            }
            detections.append(det)

            if draw and vis is not None:
                # draw markers/axes on RAW geometry
                cv.aruco.drawDetectedMarkers(vis, [pts.reshape(1, 4, 2)], None)
                cv.drawFrameAxes(vis, K_used, dist_used, rvec, tvec, 0.25 * aruco_size)
                label = f"id={tag_id} "
                if det["yaw_deg"] is not None:
                    label += f"{det['yaw_deg']:.0f}dg"
                if det["floor_xy_mm"] is not None:
                    floor_x = det['floor_xy_mm'][0]
                    floor_y = det['floor_xy_mm'][1]
                    label += f" [{floor_x:.0f} {floor_y:.0f}]"
                cv.putText(vis, label, (int(u) + 8, int(v) - 8),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (25, 225, 25), 2, cv.LINE_AA)

        # --- Save debug image if requested ---
        if draw and vis is not None:
            temp_dir = Settings.temp_dir
            camera_name = self.camera_name
            output_file = path.join(temp_dir, f"aruco_{camera_name}.jpg")
            cv.imwrite(output_file, vis)
        return detections, vis

    def draw_board_axes_and_grid(self, img=None, undistort=False, save_to=None):
        """
        Draw world (board) XY axes and a Z=0 grid using ONLY the pose bundle.
        - If undistort=True: undistorts the frame and projects with newK + dist=None,
          so world straight lines appear straight.
        - If undistort=False (default): draws on RAW frame; grid is rendered as
          polylines that follow distortion exactly.

        Returns the resulting image, or None on failure.
        """
        draw_grid = True
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

        # Try to get calibration frame size to scale intrinsics to current frame
        bsize = b.get("frame_size") or b.get("size") or b.get("calib_size")
        bW = bH = None
        if isinstance(bsize, (tuple, list)) and len(bsize) == 2: bW, bH = int(bsize[0]), int(bsize[1])
        # --- Acquire image (always RAW); we handle undistort below ----------------
        if img is None:
            img = self.get_frame(undistort=False)
            if img is None:
                print("[axes] No image available.")
                return None
        h, w = img.shape[:2]

        # --- Scale K from bundle to current frame size ----------------------------
        K_scaled = K_bundle.copy()
        if bW and bH and (w != bW or h != bH):
            sx, sy = w / float(bW), h / float(bH)
            K_scaled[0, 0] *= sx  # fx
            K_scaled[0, 2] *= sx  # cx
            K_scaled[1, 1] *= sy  # fy
            K_scaled[1, 2] *= sy  # cy

        # --- Optional undistort: use newK + dist=None for all projections --------
        if undistort:
            alpha = float(getattr(self, "alpha", 1.0))  # keep FOV by default
            newK, _ = cv.getOptimalNewCameraMatrix(K_scaled, dist_bundle, (w, h), alpha)
            img = cv.undistort(img, K_scaled, dist_bundle, None, newK)
            K_used, dist_used = newK, None
        else:
            K_used, dist_used = K_scaled, dist_bundle

        # --- Prepare projection (board -> camera) --------------------------------
        rvec, _ = cv.Rodrigues(cam_R_from_board)
        tvec = cam_t_from_board.reshape(3, 1)

        def pj(P):
            """Project Nx3 board points to image pixels using current K/dist."""
            P = np.asarray(P, dtype=np.float32).reshape(-1, 1, 3)
            uv, _ = cv.projectPoints(P, rvec, tvec, K_used, dist_used)
            return uv.reshape(-1, 2)

        def ipt(p):
            """Round to nearest pixel for consistent rasterization everywhere."""
            return tuple(np.rint(np.asarray(p, dtype=np.float32)).astype(int))

        # --- Draw axes on Z=0 (X red, Y green) -----------------------------------
        axis_len = float(getattr(self, "axis_len_mm", 100.0))
        O = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        Xpt = np.array([axis_len, 0.0, 0.0], dtype=np.float32)
        Ypt = np.array([0.0, axis_len, 0.0], dtype=np.float32)

        Ouv, Xuv, Yuv = pj([O, Xpt, Ypt])

        cv.arrowedLine(img, ipt(Ouv), ipt(Xuv), (0, 0, 255), 2, cv.LINE_AA, tipLength=0.08)  # X
        cv.arrowedLine(img, ipt(Ouv), ipt(Yuv), (0, 255, 0), 2, cv.LINE_AA, tipLength=0.08)  # Y

        if draw_labels:
            cv.putText(img, "X", ipt(Xuv), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
            cv.putText(img, "Y", ipt(Yuv), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
            cv.putText(img, "Origin", ipt(Ouv + np.array([6, -6], dtype=np.float32)), cv.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv.LINE_AA)

        # --- Draw Z=0 grid as polylines (follow distortion exactly) --------------
        if draw_grid:
            xmin, xmax, ymin, ymax = getattr(self, "grid_extent_mm", (-2500, 2500, -2500, 2500))
            S = float(getattr(self, "grid_step_mm", 100.0))
            col = (120, 120, 120)

            # grid coordinates
            xi0 = int(np.ceil(xmin / S));
            xi1 = int(np.floor(xmax / S))
            yi0 = int(np.ceil(ymin / S));
            yi1 = int(np.floor(ymax / S))
            xs = (np.arange(xi0, xi1 + 1, dtype=int) * S).astype(np.float32)
            ys = (np.arange(yi0, yi1 + 1, dtype=int) * S).astype(np.float32)

            # sampling density ~32–128 points per line depending on image size
            npts = max(32, int(0.25 * (h + w) // 50))

            def line_poly(world_start, world_end, n=npts):
                ws = np.array(world_start, dtype=np.float32)
                we = np.array(world_end, dtype=np.float32)
                ts = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                pts = ws[None, :] * (1.0 - ts) + we[None, :] * ts  # Nx3
                uv = pj(pts)  # Nx2
                return np.rint(uv).astype(np.int32).reshape(-1, 1, 2)

            # vertical lines: x = const
            for x in xs:
                poly = line_poly([x, ymin, 0.0], [x, ymax, 0.0])
                cv.polylines(img, [poly], False, col, 1, cv.LINE_AA)

            # horizontal lines: y = const
            for y in ys:
                poly = line_poly([xmin, y, 0.0], [xmax, y, 0.0])
                cv.polylines(img, [poly], False, col, 1, cv.LINE_AA)

            # emphasize the axes lines (x=0 and y=0) if they fall within the extent
            hi = (160, 160, 160)
            if xmin <= 0.0 <= xmax:
                poly = line_poly([0.0, ymin, 0.0], [0.0, ymax, 0.0])
                cv.polylines(img, [poly], False, hi, 1, cv.LINE_AA)
            if ymin <= 0.0 <= ymax:
                poly = line_poly([xmin, 0.0, 0.0], [xmax, 0.0, 0.0])
                cv.polylines(img, [poly], False, hi, 1, cv.LINE_AA)

        if save_to is not None:
            if not cv.imwrite(str(save_to), img):
                print(f"[axes] Failed to write {save_to}")

        return img
