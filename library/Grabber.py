import os, cv2, threading, time
from library import Settings

class RTSPGrabber:
    def __init__(self, channel, auto_start=True):
        self.channel = channel
        self.ip = Settings.lorex_ip
        self.username = Settings.username
        self.password = Settings.password
        self.subtype = 0
        self.port = 554
        self.warmup_frames = 15

        self.cap = None
        self.frame = None
        self.frame_lock = threading.Lock()

        self._stop = threading.Event()
        self._ready = threading.Event()
        self._started = threading.Event()
        self.thread = None

        if auto_start:
            print('[grabber] starting ...')
            self.start()
            self.wait_latest_bgr(timeout=5.0)
            if self._ready.is_set(): print('[grabber] ready.')

    # -------- context manager --------
    def __enter__(self): self.start(); return self
    def __exit__(self, exc_type, exc, tb): self.close()
    def close(self): self.stop()
    # Avoid heavy logic in __del__ at interpreter shutdown
    def __del__(self):
        try: self.stop()
        except Exception: pass

    # -------- internal helpers --------
    def _rtsp_url(self):
        return (f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}"
                f"/cam/realmonitor?channel={self.channel}&subtype={self.subtype}")

    def _open(self):
        # Open inside worker thread
        self.cap = cv2.VideoCapture(self._rtsp_url(), cv2.CAP_FFMPEG)
        try:
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

    def _warmup(self):
        # Flush decoder/buffer; bail early if stopping
        for _ in range(self.warmup_frames):
            if self._stop.is_set(): return
            self.cap.read()

    def _reopen(self):
        # Called by worker thread only
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        if self._stop.is_set():  # do not reopen if stopping
            return
        time.sleep(0.1)
        self._open()
        if self.cap and self.cap.isOpened():
            self._warmup()

    def _loop(self):
        try:
            self._open()
            if self.cap and self.cap.isOpened():
                self._warmup()
            self._started.set()

            while not self._stop.is_set():
                if not self.cap or not self.cap.isOpened():
                    if self._stop.is_set():
                        break
                    self._reopen()
                    continue

                ret, frame = self.cap.read()
                if not ret:
                    # transient glitch: try to reopen
                    self._reopen()
                    continue

                with self.frame_lock:
                    self.frame = frame
                self._ready.set()  # first valid frame arrived
        finally:
            # Release capture in the SAME thread that uses it
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
            self.cap = None
            self._ready.clear()
            self._started.clear()

    # -------- public API --------
    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._loop, name="RTSPGrabber", daemon=False)
        self.thread.start()
        # Optional: wait until the worker has opened the device (not necessarily ready)
        self._started.wait(timeout=2.0)

    def stop(self):
        if not (self.thread and self.thread.is_alive()):
            return
        self._stop.set()
        # Do NOT release self.cap here — worker will release it
        self.thread.join(timeout=3.0)
        self.thread = None
        # Clear frame after join so readers don't see stale frames
        with self.frame_lock:
            self.frame = None
        self._ready.clear()

    def wait_latest_bgr(self, timeout: float = 3.0):
        """Blocking: waits until first valid frame (or timeout), then returns a copy."""
        if not self._ready.wait(timeout=timeout):
            return None
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()

    def get_latest_bgr(self):
        """Non-blocking: returns newest frame (or None if not ready yet)."""
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()

    def save(self, path: str, block: bool = False, timeout: float = 3.0) -> str:
        f = self.wait_latest_bgr(timeout) if block else self.get_latest_bgr()
        if f is None:
            raise RuntimeError("No frame available yet (try block=True or increase timeout/warmup).")
        cv2.imwrite(path, f)
        return path

    def set_subtype(self, subtype: int):
        if subtype == self.subtype:
            return
        self.subtype = subtype
        self._ready.clear()
        # Request reopen by closing cap in worker loop
        if self.thread and self.thread.is_alive():
            # Signal reopen by forcing failure
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def set_channel(self, channel: int):
        if channel == self.channel:
            return
        self.channel = channel
        self._ready.clear()
        # Same reopen trick
        if self.thread and self.thread.is_alive():
            if self.cap:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None

    def show_latest_bgr(self, window_name="Full Resolution"):
        """
        Display the latest frame at full resolution using OpenCV.
        Returns True if a frame was displayed, False otherwise.
        """
        frame = self.get_latest_bgr()
        if frame is None:
            print("[grabber] No frame available to display.")
            return False

        # Create a resizable window
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Display the frame
        cv2.imshow(window_name, frame)

        # Resize the window to match the frame's dimensions
        height, width = frame.shape[:2]
        cv2.resizeWindow(window_name, width, height)

        # Wait for a key press to close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return True


