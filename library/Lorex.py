import cv2 as cv
from os import path
from library import Grabber
from library import Settings
from library import Utils
from pathlib import Path


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

        intrinsics_yml = self.paths.get("intrinsics_yml")
        if intrinsics_yml and path.exists(intrinsics_yml):
            try:
                self.K, self.dist, self.calib_size = load_calibration(camera_name)
                print(f"[calib] loaded for {camera_name}")
            except Exception as e:
                print(f"[calib] failed to load: {e}")
        else:
            print(f"[calib] not found for {camera_name} at {intrinsics_yml}")

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
        newK, _ = cv.getOptimalNewCameraMatrix(self.K, self.dist, (w, h), self.alpha)
        self.map1, self.map2 = cv.initUndistortRectifyMap(self.K, self.dist, None, newK, (w, h), cv.CV_16SC2)
        self.map_size = (w, h)
        return True

    def get_frame(self, undistort=True):
        frame = self.grabber.get_latest_bgr()
        if frame is None: return None
        if undistort and self.K is not None and self.dist is not None:
            if self._ensure_maps(frame.shape): return cv.remap(frame, self.map1, self.map2, cv.INTER_LINEAR)
        return frame

    def save(self, p, undistort=True, encode_params=None):
        frame = self.get_frame(undistort=undistort)
        if frame is None:
            print("[save] No frame available")
            return False

        # Ensure destination exists
        p = Path(p)
        p.parent.mkdir(parents=True, exist_ok=True)

        # Default encode params based on extension (optional)
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
        return True

    def set_alpha(self, alpha):
        """0=crop more (tighter), 1=keep more FOV (more edges)."""
        self.alpha = float(alpha)
        self.map1 = self.map2 = None
        self.map_size = None


