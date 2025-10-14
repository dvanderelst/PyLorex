import os, cv2, threading, time
from library import Settings

# Optional: nudge OpenCV/FFmpeg to keep latency low (works when using CAP_FFMPEG builds)
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|max_delay;0|buffer_size;10240|stimeout;3000000")

class LiveRTSPGrabber:
    def __init__(self, channel, auto_start=True):
        self.channel = channel
        self.ip = Settings.lorex_ip
        self.username = Settings.username
        self.password = Settings.password
        self.subtype = 1

        self.port = 554
        self.warmup_frames = 15

        self.cap = None
        self.frame = None
        self.frame_lock = threading.Lock()
        self.running = False
        self.thread = None
        self.ready = threading.Event()   # <- set after first *valid* frame is stored

        if auto_start:
            self.start()
            self.wait_latest_bgr(timeout=5.0)

    def _rtsp_url(self):
        return f"rtsp://{self.username}:{self.password}@{self.ip}:{self.port}/cam/realmonitor?channel={self.channel}&subtype={self.subtype}"

    def start(self):
        if self.running: return
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread: self.thread.join(timeout=1.0)
        if self.cap: self.cap.release(); self.cap = None
        self.ready.clear()

    def _open(self):
        # If your OpenCV is built with FFMPEG, this hint enables the env options above
        self.cap = cv2.VideoCapture(self._rtsp_url(), cv2.CAP_FFMPEG)
        try: self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception: pass

    def _reopen(self):
        if self.cap: self.cap.release()
        time.sleep(0.1)
        self._open()
        # flush decoder/buffer
        for _ in range(self.warmup_frames):
            self.cap.read()

    def _loop(self):
        self._open()
        for _ in range(self.warmup_frames):
            self.cap.read()
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self._reopen()
                continue
            with self.frame_lock:
                self.frame = frame
            self.ready.set()  # first good frame has arrived

    # ---- Public API ----
    def get_latest_bgr(self):
        """Non-blocking: returns newest frame (or None if not ready yet)."""
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()

    def wait_latest_bgr(self, timeout:float=3.0):
        """Blocking: waits until first valid frame (up to timeout), then returns a copy (or None)."""
        if not self.ready.wait(timeout=timeout):
            return None
        with self.frame_lock:
            return None if self.frame is None else self.frame.copy()

    def save(self, path:str, block:bool=False, timeout:float=3.0)->str:
        f = self.wait_latest_bgr(timeout) if block else self.get_latest_bgr()
        if f is None: raise RuntimeError("No frame available yet (try block=True or increase timeout/warmup).")
        cv2.imwrite(path, f)
        return path

    def set_subtype(self, subtype:int):
        if subtype == self.subtype: return
        self.subtype = subtype
        self.ready.clear()
        if self.running: self._reopen()

    def set_channel(self, channel:int):
        if channel == self.channel: return
        self.channel = channel
        self.ready.clear()
        if self.running: self._reopen()


# -------------------------------
# Quick test: grab a few frames
# -------------------------------
if __name__ == "__main__":
    g = LiveRTSPGrabber(channel=2)
    g.start()

    # 1) Block for first valid frame so you don’t miss the first 2–3
    f0 = g.wait_latest_bgr(timeout=5.0)
    if f0 is None:
        g.stop()
        raise SystemExit("Failed to receive first frame (increase timeout or check RTSP URL).")
    g.save("snap_0.jpg", block=False)
    print("Saved snap_0.jpg")

    # 2) Then grab a few more, non-blocking (should advance clock quickly)
    for i in range(1, 5):
        time.sleep(1)  # half-second cadence
        g.save(f"snap_{i}.jpg", block=False)
        print(f"Saved snap_{i}.jpg")

    g.stop()
