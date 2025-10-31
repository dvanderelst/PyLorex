"""
Lorex camera module for ArUco marker detection and tracking.

This module provides camera calibration, frame capture, marker detection,
and coordinate transformation capabilities for the Lorex camera system.
"""

import logging
from math import atan2, degrees
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import cv2 as cv
import numpy as np
import pandas as pd

from library import Grabber
from library import Settings
from library import Utils
from library import CalibIO
from library import Homography as hg

# Configure logging
logger = logging.getLogger(__name__)

# Visualization constants
COLOR_RED_BGR = (0, 0, 255)
COLOR_GREEN_BGR = (0, 255, 0)
COLOR_LIGHT_GREEN_BGR = (25, 225, 25)
COLOR_GRID_GRAY = (120, 120, 120)
COLOR_GRID_HIGHLIGHT = (160, 160, 160)
COLOR_WHITE = (240, 240, 240)
LABEL_FONT = cv.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SIZE = 0.6
LABEL_FONT_SIZE_SMALL = 0.5
LABEL_OFFSET = 8
LABEL_THICKNESS = 2
GRID_LINE_THICKNESS = 1
AXIS_LINE_THICKNESS = 2

def parse_detections(detections: List[List[Dict]]) -> pd.DataFrame:
    """
    Parse ArUco detection results into a DataFrame.

    Args:
        detections: List containing detected markers with id, floor_xy_mm, and yaw_deg

    Returns:
        DataFrame with columns: id, x, y, yaw
    """
    detected = detections[0]
    lines = []
    for marker in detected:
        marker_id = marker['id']
        x_mm = marker['floor_xy_mm'][0]
        y_mm = marker['floor_xy_mm'][1]
        yaw_deg = marker['yaw_deg']
        line = [marker_id, x_mm, y_mm, yaw_deg]
        lines.append(line)

    header = ['id', 'x', 'y', 'yaw']
    if len(lines) > 0:
        df = pd.DataFrame(lines, columns=header)
    else:
        df = pd.DataFrame(columns=header)
    return df


def compose_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """
    Compose a 4x4 transformation matrix from rotation and translation.

    Args:
        R: 3x3 rotation matrix
        t: 3x1 or (3,) translation vector

    Returns:
        4x4 transformation matrix
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def int_point(point: np.ndarray) -> Tuple[int, int]:
    """
    Convert a point to integer pixel coordinates.

    Args:
        point: 2D point as numpy array

    Returns:
        Tuple of (x, y) as integers
    """
    return tuple(np.rint(point).astype(int))


def invert_extrinsics(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Invert extrinsic camera parameters.

    Given X_cam = R*X_board + t, compute the inverse transformation
    to get board_from_cam.

    Args:
        R: 3x3 rotation matrix (cam from board)
        t: 3x1 translation vector (cam from board)

    Returns:
        Tuple of (R_inv, t_inv) for board from cam transformation
    """
    R_inv = R.T
    t_inv = -R_inv @ t.reshape(3)
    return R_inv, t_inv


def yaw_from_R_board_tag(
    R_board_tag: np.ndarray,
    forward_axis: str = 'x',
    offset_deg: float = 0.0
) -> float:
    """
    Compute yaw angle from rotation matrix in board frame.

    Args:
        R_board_tag: 3x3 rotation matrix from tag to board frame
        forward_axis: Which tag axis points forward ('x' or 'y')
        offset_deg: Offset to add to computed yaw in degrees

    Returns:
        Yaw angle in degrees [0, 360)
    """
    ax = 0 if forward_axis.lower() == 'x' else 1  # 0->X, 1->Y in tag frame
    v = R_board_tag[:, ax].reshape(3)  # forward axis expressed in board frame
    yaw = degrees(atan2(float(v[1]), float(v[0])))  # atan2(y, x)
    yaw = (yaw + offset_deg + 360.0) % 360.0
    return yaw

def load_calibration(camera_name: str) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """
    Load camera calibration from saved intrinsics YAML file.

    Args:
        camera_name: Name/identifier of the camera

    Returns:
        Tuple of (K, dist, size) where:
            K: 3x3 camera matrix
            dist: Distortion coefficients
            size: (width, height) of calibrated image

    Raises:
        IOError: If calibration file cannot be opened
    """
    paths = Utils.get_calibration_paths(camera_name)
    intrinsics_yml = paths["intrinsics_yml"]
    file_storage = cv.FileStorage(str(intrinsics_yml), cv.FILE_STORAGE_READ)

    if not file_storage.isOpened():
        raise IOError(f"Cannot open calibration file: {intrinsics_yml}")

    K = file_storage.getNode("camera_matrix").mat()
    dist = file_storage.getNode("distortion_coefficients").mat()
    width = int(file_storage.getNode("image_width").real())
    height = int(file_storage.getNode("image_height").real())
    file_storage.release()

    logger.info(f"Loaded calibration for {camera_name}: {width}x{height}")
    return K, dist, (width, height)


def undistort_image(
    img: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    alpha: float = 1.0
) -> np.ndarray:
    """
    Undistort an image using camera calibration (slower than remap).

    Args:
        img: Input image to undistort
        K: 3x3 camera matrix
        dist: Distortion coefficients
        alpha: Free scaling parameter (0=crop edges, 1=keep all pixels)

    Returns:
        Undistorted image
    """
    h, w = img.shape[:2]
    newK, _ = cv.getOptimalNewCameraMatrix(K, dist, (w, h), alpha)
    return cv.undistort(img, K, dist, None, newK)

def make_aruco_params(aruco: Any) -> Any:
    """
    Create optimized ArUco detector parameters for consistent scenes.

    Configures parameters for reliable detection with tight thresholding
    and subpixel corner refinement to minimize pose jitter.

    Args:
        aruco: cv2.aruco module reference

    Returns:
        DetectorParameters object (type varies by OpenCV version)

    Raises:
        RuntimeError: If DetectorParameters cannot be created
    """
    # New or legacy API constructor
    if hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
    elif hasattr(aruco, "DetectorParameters_create"):
        params = aruco.DetectorParameters_create()
    else:
        raise RuntimeError("Could not create ArUco DetectorParameters")

    # Thresholding (scene is consistent; keep range tight & fast)
    # Smaller window range = less noise sensitivity, faster
    if hasattr(params, "adaptiveThreshWinSizeMin"):
        params.adaptiveThreshWinSizeMin = 5
    if hasattr(params, "adaptiveThreshWinSizeMax"):
        params.adaptiveThreshWinSizeMax = 23
    if hasattr(params, "adaptiveThreshWinSizeStep"):
        params.adaptiveThreshWinSizeStep = 2
    if hasattr(params, "adaptiveThreshConstant"):
        params.adaptiveThreshConstant = 7  # Adjust 6-9 if detection misses

    # Size gating (skip blobs that are way too small/big)
    # 46px side ≈ 184px perimeter; these rates suit 1080-4K frames
    if hasattr(params, "minMarkerPerimeterRate"):
        params.minMarkerPerimeterRate = 0.03
    if hasattr(params, "maxMarkerPerimeterRate"):
        params.maxMarkerPerimeterRate = 0.30

    # Geometry & corner precision
    if hasattr(params, "polygonalApproxAccuracyRate"):
        params.polygonalApproxAccuracyRate = 0.03
    if hasattr(params, "minCornerDistanceRate"):
        params.minCornerDistanceRate = 0.05
    if hasattr(params, "minDistanceToBorder"):
        params.minDistanceToBorder = 3

    # Subpixel corner refinement = less pose jitter
    if hasattr(aruco, "CORNER_REFINE_SUBPIX") and hasattr(params, "cornerRefinementMethod"):
        params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
    if hasattr(params, "cornerRefinementWinSize"):
        params.cornerRefinementWinSize = 5  # at 46px, 3-5 is good
    if hasattr(params, "cornerRefinementMaxIterations"):
        params.cornerRefinementMaxIterations = 30
    if hasattr(params, "cornerRefinementMinAccuracy"):
        params.cornerRefinementMinAccuracy = 0.01

    # Cell warping (match cell size to your tag in pixels)
    # 5x5 dict => 7x7 total with border; 46px/7 ≈ 6.6px per cell
    if hasattr(params, "perspectiveRemovePixelPerCell"):
        params.perspectiveRemovePixelPerCell = 8  # Try 6-8
    if hasattr(params, "perspectiveRemoveIgnoredMarginPerCell"):
        params.perspectiveRemoveIgnoredMarginPerCell = 0.33

    # Decoding strictness
    if hasattr(params, "errorCorrectionRate"):
        params.errorCorrectionRate = 0.5  # Slightly stricter than default
    if hasattr(params, "maxErroneousBitsInBorderRate"):
        params.maxErroneousBitsInBorderRate = 0.5
    if hasattr(params, "minOtsuStdDev"):
        params.minOtsuStdDev = 5.0

    # Polarity safety
    if hasattr(params, "detectInvertedMarker"):
        params.detectInvertedMarker = True

    return params


def detect_markers(
    aruco: Any,
    frame: np.ndarray,
    dict_obj: Any
) -> Tuple[Optional[List], Optional[np.ndarray]]:
    """
    Detect ArUco markers in an image with version-tolerant API handling.

    Args:
        aruco: cv2.aruco module reference
        frame: Input image (grayscale or BGR)
        dict_obj: ArUco dictionary object

    Returns:
        Tuple of (corners, ids):
            corners: List of corner arrays for each detected marker
            ids: Array of marker IDs, or None if no markers found
    """
    # Ensure image is grayscale
    if frame.ndim == 3 and frame.shape[2] == 3:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    else:
        gray = frame.copy()

    # Build detector parameters
    params = make_aruco_params(aruco)

    # API compatibility handling
    if hasattr(aruco, "ArucoDetector"):
        # New API (OpenCV >= 4.7)
        detector = aruco.ArucoDetector(dict_obj, params)
        corners, ids, _ = detector.detectMarkers(gray)
    else:
        # Legacy API fallback
        corners, ids, _ = aruco.detectMarkers(gray, dict_obj, parameters=params)

    return corners, ids

class LorexCamera:
    """
    Camera interface for ArUco marker detection and tracking.

    Provides camera calibration, frame capture, marker detection, and coordinate
    transformation capabilities for the Lorex camera system. Supports both raw
    and undistorted image processing with cached remap operations for performance.

    Attributes:
        camera_name: Identifier for the camera
        channel: RTSP channel number for this camera
        grabber: RTSPGrabber instance for frame acquisition
        K: 3x3 camera intrinsic matrix (from calibration)
        dist: Distortion coefficients (from calibration)
        calib_size: (width, height) from calibration file
        alpha: Undistortion free scaling (0=crop edges, 1=keep FOV)
        newK: Effective camera matrix for undistorted images
        bundle: Pose bundle containing homography and extrinsics (optional)
        axis_len_mm: Length of coordinate axes for visualization
        grid_step_mm: Grid spacing for floor visualization
        grid_extent_mm: (x_min, x_max, y_min, y_max) for grid bounds

    Example:
        >>> with LorexCamera("camera1") as cam:
        ...     detections, vis = cam.get_aruco(draw=True)
        ...     for det in detections:
        ...         print(f"Marker {det['id']} at {det['floor_xy_mm']}")
    """

    def __init__(self, camera_name: str, auto_start: bool = True, alpha: float = 1.0):
        """
        Initialize a Lorex camera interface.

        Args:
            camera_name: Name/identifier of the camera
            auto_start: Whether to automatically start the frame grabber
            alpha: Undistortion scaling parameter (0=crop edges, 1=keep FOV)
        """
        self.camera_name = camera_name
        self.channel = Settings.channels[camera_name]
        self.paths = Utils.get_calibration_paths(camera_name)
        self.grabber = Grabber.RTSPGrabber(channel=self.channel, auto_start=auto_start)

        # Calibration state
        self.K: Optional[np.ndarray] = None
        self.dist: Optional[np.ndarray] = None
        self.calib_size: Optional[Tuple[int, int]] = None  # (W, H) from YAML
        self.alpha = float(alpha)  # 0=crop edges, 1=keep FOV
        self.map1: Optional[np.ndarray] = None
        self.map2: Optional[np.ndarray] = None
        self.map_size: Optional[Tuple[int, int]] = None  # (W, H) of maps
        self.newK: Optional[np.ndarray] = None  # Effective K for undistorted output

        # Pose/homography bundle (optional)
        self.bundle: Optional[Dict[str, Any]] = None

        self.load_calibration()

        # Settings for grid plotting
        self.axis_len_mm = 100.0
        self.grid_step_mm = 100.0
        self.grid_extent_mm = (-2500, 2500, -2500, 2500)

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures grabber is stopped."""
        self.stop()
        return False

    def load_calibration(self) -> bool:
        """
        Load camera calibration from YAML file.

        Returns:
            True if calibration loaded successfully, False otherwise
        """
        intrinsics_yml = self.paths.get("intrinsics_yml")
        if intrinsics_yml and Path(intrinsics_yml).exists():
            try:
                self.K, self.dist, self.calib_size = load_calibration(self.camera_name)
                logger.info(f"Calibration loaded for {self.camera_name}")
                return True
            except Exception as e:
                logger.error(f"Failed to load calibration for {self.camera_name}: {e}")
                return False
        else:
            logger.warning(f"Calibration not found for {self.camera_name} at {intrinsics_yml}")
            return False

    def start(self):
        """Start the frame grabber."""
        self.grabber.start()

    def stop(self):
        """Stop the frame grabber and release resources."""
        self.grabber.stop()

    def wait_ready(self, timeout: float = 5.0) -> bool:
        """
        Wait for frame grabber to have a frame ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if frame is ready, False on timeout
        """
        return self.grabber.wait_latest_bgr(timeout=timeout)


    def _ensure_maps(self, frame_shape: Tuple[int, int, int]) -> bool:
        """
        Build rectification maps if missing or size changed.

        Args:
            frame_shape: Shape of the frame (height, width, channels)

        Returns:
            True if maps are ready, False if calibration not available
        """
        if self.K is None or self.dist is None:
            return False

        h, w = frame_shape[:2]
        if self.map1 is not None and self.map_size == (w, h):
            return True

        # Build maps for current frame size (handles cameras that scale/letterbox)
        self.newK, _ = cv.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), self.alpha)
        self.map1, self.map2 = cv.initUndistortRectifyMap(
            self.K, self.dist, None, self.newK, (w, h), cv.CV_16SC2
        )
        self.map_size = (w, h)
        return True

    def get_frame(self, undistort: bool = False) -> Optional[np.ndarray]:
        """
        Get the latest frame from the camera.

        Args:
            undistort: If True, return undistorted frame using cached remap

        Returns:
            Frame as numpy array (BGR), or None if unavailable
        """
        frame = self.grabber.get_latest_bgr()
        if frame is None:
            return None

        if undistort and self.K is not None and self.dist is not None:
            if self._ensure_maps(frame.shape):
                return cv.remap(frame, self.map1, self.map2, cv.INTER_LINEAR)

        return frame

    def save(
        self,
        output_path: Path,
        undistort: bool = True,
        encode_params: Optional[List[int]] = None
    ) -> bool:
        """
        Save the current frame to a file.

        Args:
            output_path: Path to save the image
            undistort: Whether to save undistorted frame
            encode_params: OpenCV encoding parameters (auto-detected if None)

        Returns:
            True if saved successfully, False otherwise
        """
        frame = self.get_frame(undistort=undistort)
        if frame is None:
            logger.warning("Cannot save frame - no frame available from grabber")
            return False

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if encode_params is None:
            ext = output_path.suffix.lower()
            if ext in (".jpg", ".jpeg"):
                encode_params = [int(cv.IMWRITE_JPEG_QUALITY), 95]
            elif ext == ".png":
                encode_params = [int(cv.IMWRITE_PNG_COMPRESSION), 3]
            else:
                encode_params = []

        ok = cv.imwrite(str(output_path), frame, encode_params)
        if not ok:
            logger.error(f"Failed to write frame to {output_path}")
        else:
            logger.info(f"Saved frame to {output_path}")

        return ok

    def reload_calibration(self) -> bool:
        """
        Re-read calibration YAML and rebuild maps on next frame.

        Returns:
            True if reload successful, False otherwise
        """
        intrinsics_yml = self.paths.get("intrinsics_yml")
        if not intrinsics_yml or not Path(intrinsics_yml).exists():
            logger.error(f"Calibration YAML missing at {intrinsics_yml}")
            return False

        try:
            self.K, self.dist, self.calib_size = load_calibration(self.camera_name)
            self.map1 = self.map2 = None
            self.map_size = None
            self.newK = None
            logger.info(f"Reloaded calibration for {self.camera_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reload calibration: {e}")
            return False

    def set_alpha(self, alpha: float):
        """
        Set undistortion alpha parameter and invalidate cached maps.

        Args:
            alpha: Free scaling parameter (0=crop more/tighter, 1=keep FOV/more edges)
        """
        self.alpha = float(alpha)
        self.map1 = self.map2 = None
        self.map_size = None
        self.newK = None
        logger.debug(f"Set alpha to {alpha}, maps will be rebuilt on next frame")

    def get_current_intrinsics(
        self,
        frame_shape: Tuple[int, int, int],
        undistort: bool = False
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get camera intrinsics matching the pixel geometry being used.

        Args:
            frame_shape: Shape of the frame (height, width, channels)
            undistort: Whether the frame will be undistorted

        Returns:
            Tuple of (K_used, dist_used):
                - If undistort=False: scaled K from YAML, original dist
                - If undistort=True: newK for this frame size/alpha, dist=None

        Raises:
            RuntimeError: If calib_size is None and scaling is needed
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

    def load_board_bundle(self) -> bool:
        """
        Load homography and pose bundle for this camera.

        Loads H_raw, R, t, K, dist from saved pose files.

        Returns:
            True if loaded successfully

        Raises:
            KeyError: If bundle is missing required keys
        """
        self.bundle = CalibIO.load_pose_bundle(self.camera_name)

        # Basic key sanity check
        for key in ("H_raw", "R", "t", "K"):
            if key not in self.bundle:
                raise KeyError(f"Pose bundle missing required key: {key}")

        logger.info(f"Loaded board bundle for {self.camera_name}")
        return True

    def has_bundle(self) -> bool:
        """
        Check if a pose bundle is loaded.

        Returns:
            True if bundle is loaded, False otherwise
        """
        return hasattr(self, "bundle") and self.bundle is not None

    def pixel_to_board_xy(
        self,
        u: float,
        v: float,
        use_raw: bool = True
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to board-plane (X, Y) in mm.

        Args:
            u: Pixel x-coordinate
            v: Pixel y-coordinate
            use_raw: If True, use H_raw homography (fast, assumes raw frame).
                     If False, use ray-plane intersection with PnP pose.

        Returns:
            Tuple of (X_mm, Y_mm) on the board plane

        Raises:
            RuntimeError: If no bundle is loaded or no frame is available
        """
        if not self.has_bundle():
            raise RuntimeError("No board bundle loaded. Call load_board_bundle() first.")

        R = self.bundle["R"]
        t = self.bundle["t"]

        if use_raw:
            H = self.bundle["H_raw"]
            X, Y = hg.pixel_to_board_xy_raw(u, v, H)
            return float(X), float(Y)
        else:
            # Match intrinsics to the pixels being queried
            # Assumes RAW for simplicity; switch undistort=True for undistorted frames
            frame = self.get_frame(undistort=False)
            if frame is None:
                raise RuntimeError("No frame available for intrinsics computation")

            K_used, dist_used = self.get_current_intrinsics(frame.shape, undistort=False)
            d_cam = hg.pixel_to_ray_cam(u, v, K_used, dist_used)
            P = hg.intersect_ray_with_board(d_cam, R, t)
            return float(P[0]), float(P[1])

    def get_aruco(
        self,
        draw: bool = False,
        draw_world: bool = True,
        world_undistort: bool = False
    ) -> Tuple[List[Dict[str, Any]], Optional[np.ndarray]]:
        """
        Detect ArUco markers in the current frame.

        Args:
            draw: Whether to draw detection results on the frame
            draw_world: Whether to draw world coordinate axes and grid
            world_undistort: Whether to undistort the frame for visualization

        Returns:
            Tuple of (detections, visualization):
                detections: List of detection dictionaries with keys:
                    - id: Marker ID
                    - floor_xy_mm: (X, Y) position on floor in mm
                    - yaw_deg: Orientation in degrees [0, 360)
                    - height_mm: Height above floor in mm
                    - rvec_cam_tag: Rotation vector from camera to tag
                    - tvec_cam_tag: Translation vector from camera to tag
                    - center_px: (u, v) pixel coordinates of marker center
                    - corners_px: 4x2 array of corner pixel coordinates
                visualization: Annotated frame if draw=True, else None
        """
        # Settings from config
        aruco_dict_name = Settings.aruco_dict_name  # e.g. "DICT_5X5_100"
        aruco_size = float(Settings.aruco_size)  # mm
        aruco_forward_axis = Settings.aruco_forward_axis  # 'x' or 'y'
        aruco_yaw_offset_deg = float(Settings.aruco_yaw_offset_deg)

        # Ensure bundle is loaded (for board/world drawing and height/yaw)
        if not self.has_bundle():
            try:
                self.load_board_bundle()
                logger.info(f"Loaded pose bundle for {self.camera_name}")
            except Exception as e:
                logger.warning(f"Could not load pose bundle: {e}")
                self.bundle = None

        # Acquire RAW frame (may be undistorted below)
        frame = self.get_frame(undistort=False)
        if frame is None:
            logger.warning("No frame available for ArUco detection")
            return [], None
        h, w = frame.shape[:2]

        # Intrinsics for RAW pixels (scaled to current frame)
        K_used, dist_used = self.get_current_intrinsics(frame.shape, undistort=False)
        if K_used is None:
            logger.warning("No calibration available for ArUco detection")
            return [], (frame.copy() if draw else None)

        # If we want rectified world: undistort the frame & switch to newK/dist=None
        if draw_world and world_undistort:
            alpha = float(getattr(self, "alpha", 1.0))
            newK, _ = cv.getOptimalNewCameraMatrix(K_used, dist_used, (w, h), alpha)
            frame = cv.undistort(frame, K_used, dist_used, None, newK)
            K_used, dist_used = newK, None  # Active intrinsics are now rectified

        # Board (floor) extrinsics: guard access (for yaw & height)
        R_board_from_cam = None
        t_board_from_cam = None
        if self.bundle is not None and "R" in self.bundle and "t" in self.bundle:
            cam_R_from_board = self.bundle["R"]
            cam_t_from_board = self.bundle["t"].reshape(3)
            R_board_from_cam, t_board_from_cam = invert_extrinsics(cam_R_from_board, cam_t_from_board)

        # ArUco dictionary (robust string->enum)
        try:
            aruco_dict_id = getattr(cv.aruco, aruco_dict_name)
        except AttributeError:
            raise ValueError(
                f"Unknown ArUco dictionary: {aruco_dict_name!r}. "
                f"Examples: 'DICT_4X4_50', 'DICT_5X5_100', 'DICT_6X6_250'."
            )
        aruco_dict = cv.aruco.getPredefinedDictionary(int(aruco_dict_id))

        # Detect markers on grayscale (version-tolerant via detect_markers helper)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        corners, ids = detect_markers(cv.aruco, gray, aruco_dict)
        vis = frame.copy() if draw else None

        # Draw world first (in the SAME geometry we're in now)
        if draw and draw_world:
            # We already picked the geometry above by possibly rectifying the frame
            vis = self.draw_board_axes_and_grid(
                img=vis,
                undistort=False,  # DO NOT undistort again
                K_used=K_used,
                dist_used=dist_used,  # Use active intrinsics
                assume_img_rectified=(dist_used is None)  # True when world_undistort=True
            )

        # Early out if no detections
        if ids is None or len(ids) == 0:
            if draw and vis is not None:
                output_dir = Path(Settings.temp_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"aruco_{self.camera_name}.jpg"
                cv.imwrite(str(output_path), vis)
                logger.debug(f"Saved no-detection image to {output_path}")
            return [], vis

        # Prepare solvePnP object points (square corners in tag plane, mm)
        half = aruco_size / 2.0
        obj_square = np.array(
            [[-half, half, 0.0],
             [half, half, 0.0],
             [half, -half, 0.0],
             [-half, -half, 0.0]],
            dtype=np.float32
        )

        detections = []
        for corner, tag_id in zip(corners, ids.flatten()):
            pts = corner.reshape(-1, 2).astype(np.float32)  # (4,2)
            img_points = pts.reshape(-1, 1, 2)

            # Solve for pose
            ok, rvec, tvec = cv.solvePnP(
                obj_square, img_points, K_used, dist_used, flags=cv.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                logger.debug(f"solvePnP failed for marker {tag_id}")
                continue

            # Floor XY at tag center (use RAW-vs-UNDISTORTED consistently)
            u, v = float(pts[:, 0].mean()), float(pts[:, 1].mean())
            # When frame is rectified (dist_used is None), pixel_to_board_xy uses rectified homography
            X_floor, Y_floor = self.pixel_to_board_xy(u, v, use_raw=(dist_used is not None))

            # Orientation & height if extrinsics known
            yaw_deg = None
            height_mm = None
            if R_board_from_cam is not None:
                R_cam_tag, _ = cv.Rodrigues(rvec)
                t_cam_tag = tvec.reshape(3)
                R_board_tag = R_board_from_cam @ R_cam_tag
                t_board_tag = R_board_from_cam @ t_cam_tag + t_board_from_cam

                # Safe forward-axis handling
                forward_axis = (aruco_forward_axis or "x").lower()
                if forward_axis not in ("x", "y"):
                    logger.warning(f"Invalid forward_axis={aruco_forward_axis!r}, using 'x'")
                    forward_axis = "x"

                yaw_deg = yaw_from_R_board_tag(R_board_tag, forward_axis, aruco_yaw_offset_deg)
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

            # Draw detection on visualization frame
            if draw and vis is not None:
                cv.aruco.drawDetectedMarkers(vis, [pts.reshape(1, 4, 2)], None)
                cv.drawFrameAxes(vis, K_used, dist_used, rvec, tvec, 0.25 * aruco_size)

                # Build label with ID, yaw, and position
                label = f"id={tag_id} "
                if det["yaw_deg"] is not None:
                    label += f"{det['yaw_deg']:.0f}dg"
                if det["floor_xy_mm"] is not None:
                    fx, fy = det['floor_xy_mm']
                    label += f" [{fx:.0f} {fy:.0f}]"

                cv.putText(
                    vis, label,
                    (int(u) + LABEL_OFFSET, int(v) - LABEL_OFFSET),
                    LABEL_FONT, LABEL_FONT_SIZE, COLOR_LIGHT_GREEN_BGR,
                    LABEL_THICKNESS, cv.LINE_AA
                )

        # Save debug image if requested
        if draw and vis is not None:
            output_dir = Path(Settings.temp_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"aruco_{self.camera_name}.jpg"
            cv.imwrite(str(output_path), vis)
            logger.debug(f"Saved detection image to {output_path}")

        logger.info(f"Detected {len(detections)} ArUco markers")
        return detections, vis

    def draw_board_axes_and_grid(
        self,
        img: Optional[np.ndarray] = None,
        undistort: bool = False,
        save_to: Optional[Path] = None,
        K_used: Optional[np.ndarray] = None,
        dist_used: Optional[np.ndarray] = None,
        assume_img_rectified: bool = False
    ) -> Optional[np.ndarray]:
        """
        Draw world (board) XY axes and a Z=0 grid using the pose bundle.

        Args:
            img: Image to draw on (if None, acquires from grabber)
            undistort: Whether to undistort the frame before drawing
            save_to: Optional path to save the result
            K_used: Camera matrix to use (if None, derives from bundle)
            dist_used: Distortion coefficients to use (if None, derives from bundle)
            assume_img_rectified: Set True when img is already undistorted and dist_used is None

        Modes:
            - If K_used/dist_used provided: used directly (no internal scaling/undistort)
            - Else, derives K/dist from bundle (scaled to frame size) and:
                * undistort=False: draw on RAW frame, grid as polylines (follows distortion)
                * undistort=True: undistort the frame and draw with newK, dist=None

        Returns:
            Annotated image, or None on failure
        """
        draw_grid = True
        draw_labels = True

        # Ensure pose bundle is available
        if not self.has_bundle():
            try:
                self.load_board_bundle()
                logger.info(f"Loaded pose bundle for {self.camera_name}")
            except Exception as e:
                logger.error(f"Could not load pose bundle: {e}")
                return None

        bundle = self.bundle
        if bundle is None:
            logger.error("No bundle available for drawing axes and grid")
            return None

        # Validate required keys
        for key in ("R", "t", "K"):
            if key not in bundle:
                logger.error(f"Bundle missing required key: '{key}'")
                return None

        cam_R_from_board = np.asarray(bundle["R"], dtype=np.float64)
        cam_t_from_board = np.asarray(bundle["t"], dtype=np.float64).reshape(3)
        K_bundle = np.asarray(bundle["K"], dtype=np.float64)
        dist_bundle = None if bundle.get("dist") is None else np.asarray(bundle["dist"], dtype=np.float64)

        # Acquire image (RAW unless caller already passed an image)
        if img is None:
            img = self.get_frame(undistort=False)
            if img is None:
                logger.error("No image available for drawing axes and grid")
                return None
        h, w = img.shape[:2]

        # Determine intrinsics to use
        # Case A: caller supplied active intrinsics (recommended from get_aruco)
        if K_used is not None:
            # Use as-is; caller guarantees they match the 'img' geometry
            pass
        else:
            # Case B: derive from bundle (scale to current frame and optionally undistort)
            # Try to get calibration frame size to scale intrinsics to current frame
            bundle_size = bundle.get("frame_size") or bundle.get("size") or bundle.get("calib_size")
            bundle_width = bundle_height = None
            if isinstance(bundle_size, (tuple, list)) and len(bundle_size) == 2:
                bundle_width, bundle_height = int(bundle_size[0]), int(bundle_size[1])

            K_scaled = K_bundle.copy()
            if bundle_width and bundle_height and (w != bundle_width or h != bundle_height):
                sx, sy = w / float(bundle_width), h / float(bundle_height)
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

        # If caller says img is already rectified, ensure we won't undistort again
        if assume_img_rectified and dist_used is None:
            # Nothing to do; we trust K_used/dist_used from caller
            pass

        # Prepare projection (board -> camera)
        rvec, _ = cv.Rodrigues(cam_R_from_board)
        tvec = cam_t_from_board.reshape(3, 1)

        def project_points(points_3d):
            """Project Nx3 board points to image pixels using current K_used/dist_used."""
            points_3d = np.asarray(points_3d, dtype=np.float32).reshape(-1, 1, 3)
            points_2d, _ = cv.projectPoints(points_3d, rvec, tvec, K_used, dist_used)
            return points_2d.reshape(-1, 2)

        # Draw axes on Z=0 (X red, Y green)
        axis_len = float(self.axis_len_mm)
        origin = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        x_point = np.array([axis_len, 0.0, 0.0], dtype=np.float32)
        y_point = np.array([0.0, axis_len, 0.0], dtype=np.float32)

        origin_px, x_px, y_px = project_points([origin, x_point, y_point])

        cv.arrowedLine(img, int_point(origin_px), int_point(x_px),
                       COLOR_RED_BGR, AXIS_LINE_THICKNESS, cv.LINE_AA)
        cv.arrowedLine(img, int_point(origin_px), int_point(y_px),
                       COLOR_GREEN_BGR, AXIS_LINE_THICKNESS, cv.LINE_AA)

        if draw_labels:
            cv.putText(img, "X", int_point(x_px),
                       LABEL_FONT, LABEL_FONT_SIZE, COLOR_RED_BGR,
                       LABEL_THICKNESS, cv.LINE_AA)
            cv.putText(img, "Y", int_point(y_px),
                       LABEL_FONT, LABEL_FONT_SIZE, COLOR_GREEN_BGR,
                       LABEL_THICKNESS, cv.LINE_AA)
            cv.putText(img, "Origin",
                       int_point(origin_px + np.array([6, -6], dtype=np.float32)),
                       LABEL_FONT, LABEL_FONT_SIZE_SMALL, COLOR_WHITE,
                       1, cv.LINE_AA)

        # Draw Z=0 grid as polylines (follows distortion exactly when dist_used)
        if draw_grid:
            x_min, x_max, y_min, y_max = self.grid_extent_mm
            step = float(self.grid_step_mm)

            x_idx_min = int(np.ceil(x_min / step))
            x_idx_max = int(np.floor(x_max / step))
            y_idx_min = int(np.ceil(y_min / step))
            y_idx_max = int(np.floor(y_max / step))

            x_values = (np.arange(x_idx_min, x_idx_max + 1, dtype=int) * step).astype(np.float32)
            y_values = (np.arange(y_idx_min, y_idx_max + 1, dtype=int) * step).astype(np.float32)

            # Sampling density ~32-128 points per line depending on image size
            num_points = max(32, int(0.25 * (h + w) // 50))

            def create_line_polyline(world_start, world_end, n=num_points):
                """Create polyline from 3D world line for curved projection."""
                start = np.array(world_start, dtype=np.float32)
                end = np.array(world_end, dtype=np.float32)
                t = np.linspace(0.0, 1.0, n, dtype=np.float32)[:, None]
                points_3d = start[None, :] * (1.0 - t) + end[None, :] * t  # Nx3
                points_2d = project_points(points_3d)  # Nx2
                return np.rint(points_2d).astype(np.int32).reshape(-1, 1, 2)

            # Vertical grid lines: x = const
            for x in x_values:
                poly = create_line_polyline([x, y_min, 0.0], [x, y_max, 0.0])
                cv.polylines(img, [poly], False, COLOR_GRID_GRAY, GRID_LINE_THICKNESS, cv.LINE_AA)

            # Horizontal grid lines: y = const
            for y in y_values:
                poly = create_line_polyline([x_min, y, 0.0], [x_max, y, 0.0])
                cv.polylines(img, [poly], False, COLOR_GRID_GRAY, GRID_LINE_THICKNESS, cv.LINE_AA)

            # Emphasize axes lines if in range
            if x_min <= 0.0 <= x_max:
                poly = create_line_polyline([0.0, y_min, 0.0], [0.0, y_max, 0.0])
                cv.polylines(img, [poly], False, COLOR_GRID_HIGHLIGHT, GRID_LINE_THICKNESS, cv.LINE_AA)
            if y_min <= 0.0 <= y_max:
                poly = create_line_polyline([x_min, 0.0, 0.0], [x_max, 0.0, 0.0])
                cv.polylines(img, [poly], False, COLOR_GRID_HIGHLIGHT, GRID_LINE_THICKNESS, cv.LINE_AA)

        # Save to file if requested
        if save_to is not None:
            save_path = Path(save_to)
            if not cv.imwrite(str(save_path), img):
                logger.error(f"Failed to write axes/grid image to {save_path}")
            else:
                logger.debug(f"Saved axes/grid image to {save_path}")

        return img
