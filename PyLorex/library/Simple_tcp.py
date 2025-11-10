"""Simple TCP server to expose Lorex camera marker detections.

The server maintains one :class:`LorexCamera` worker per configured camera. Each
worker keeps the most recent ArUco detections in a shared telemetry store. A
threaded TCP server answers very small ASCII commands so that another machine
can query the latest snapshot.

Protocol (newline terminated ASCII commands)::

    PING\n                          -> {"status": "ok"}\n
    CAMERAS\n                       -> {"cameras": ["name", ...]}\n
    GET <camera>\n                 -> latest snapshot for camera\n
    GET <camera> <id>\n            -> single marker entry (or error)\n
    GETALL\n                      -> snapshots for every known camera\n
Each response is a single JSON object followed by a newline.

Run ``python -m PyLorex.library.simple_tcp --camera tiger`` to start a
server that tracks the camera named ``tiger`` using the settings from
:mod:`PyLorex.library.Settings`. You can also use the convenience wrappers in
``PyLorex/script_start_server.py`` or ``PyLorex/run_server.py`` if you prefer a
simpler command line.
"""

from __future__ import annotations
import sys
import argparse
import errno
import json
import logging
import socketserver
import threading
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from library import Settings
from library.Lorex import LorexCamera
import numpy as np

LOGGER = logging.getLogger("pylorex.simple_tcp")


@dataclass
class CameraSnapshot:
    """Latest detection payload for a camera."""

    camera: str
    captured_at: float
    detections: List[dict]
    frame_size: Optional[Tuple[int, int]]
    error: Optional[str] = None

    def to_payload(self) -> dict:
        payload = {
            "camera": self.camera,
            "captured_at": self.captured_at,
            "detections": self.detections,
            "frame_size": self.frame_size,
        }
        if self.error is not None:
            payload["error"] = self.error
        return _json_safe(payload)


def _json_safe(value):
    """Return *value* converted to plain Python types for JSON encoding."""

    if isinstance(value, dict):
        return {str(key): _json_safe(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]


    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return value.item()

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _json_safe(tolist())
        except Exception:  # pragma: no cover - fall back if conversion fails
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # pragma: no cover - fall back if conversion fails
            pass

    return value


class TelemetryStore:
    """Thread-safe container for the latest camera snapshots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshots: Dict[str, CameraSnapshot] = {}

    def update(self, snapshot: CameraSnapshot) -> None:
        with self._lock:
            self._snapshots[snapshot.camera] = snapshot

    def update_error(self, camera: str, message: str) -> None:
        snapshot = CameraSnapshot(
            camera=camera,
            captured_at=time.time(),
            detections=[],
            frame_size=None,
            error=message,
        )
        self.update(snapshot)

    def list_cameras(self) -> List[str]:
        with self._lock:
            return sorted(self._snapshots.keys())

    def get(self, camera: str) -> Optional[CameraSnapshot]:
        with self._lock:
            return self._snapshots.get(camera)

    def get_all(self) -> List[CameraSnapshot]:
        with self._lock:
            return [self._snapshots[name] for name in sorted(self._snapshots.keys())]

    def get_marker(self, camera: str, marker_id: int) -> Optional[dict]:
        snapshot = self.get(camera)
        if snapshot is None:
            return None
        for det in snapshot.detections:
            if int(det.get("id")) == marker_id:
                return det
        return None


class CameraWorker(threading.Thread):
    """Background acquisition thread that keeps the latest detections fresh."""

    def __init__(
        self,
        camera_name: str,
        store: TelemetryStore,
        poll_interval: float = 0.1,
        detection_scale: Optional[float] = None,
        draw: bool = False,
    ) -> None:
        super().__init__(name=f"CameraWorker[{camera_name}]", daemon=True)
        self.camera_name = camera_name
        self.store = store
        self.poll_interval = max(poll_interval, 0.01)
        self.detection_scale = detection_scale
        self.draw = draw
        self._stopevt = threading.Event()
        self._camera: Optional[LorexCamera] = None

    def stop(self) -> None:
        self._stopevt.set()
        self.join(timeout=5.0)

    def _ensure_camera(self) -> LorexCamera:
        if self._camera is None:
            LOGGER.info("Starting LorexCamera for %s", self.camera_name)
            self._camera = LorexCamera(self.camera_name, auto_start=True)
        return self._camera

    def run(self) -> None:  # noqa: D401 - short description inherited
        cam: Optional[LorexCamera] = None
        try:
            cam = self._ensure_camera()
        except Exception as exc:  # noqa: BLE001 - surface failure to the store
            LOGGER.exception("Failed to start camera %s", self.camera_name)
            self.store.update_error(self.camera_name, str(exc))
            return
        try:
            while not self._stopevt.is_set():
                start = time.time()
                try:
                    detections, _ = cam.get_aruco(
                        draw=self.draw,
                        detection_scale=self.detection_scale,
                    )
                    snapshot = CameraSnapshot(
                        camera=self.camera_name,
                        captured_at=start,
                        detections=detections,
                        frame_size=cam.calib_size,
                    )
                    self.store.update(snapshot)
                except Exception as exc:  # noqa: BLE001 - worker must stay alive
                    LOGGER.exception("Detection loop for camera %s failed", self.camera_name)
                    self.store.update_error(self.camera_name, str(exc))
                    if self._stopevt.wait(timeout=1.0):
                        break
                    continue

                if self._stopevt.wait(timeout=self.poll_interval):
                    break
        finally:
            if cam is not None and cam.grabber is not None:
                try:
                    cam.grabber.stop()
                except Exception:  # noqa: BLE001 - best effort shutdown
                    LOGGER.debug("Failed to stop grabber for %s", self.camera_name, exc_info=True)


class SimpleTCPHandler(socketserver.StreamRequestHandler):
    """Handle line-oriented TCP commands."""

    server: "SimpleTCPServer"

    def handle(self) -> None:  # noqa: D401
        peer = self.client_address
        LOGGER.info("Connection from %s:%s", peer[0], peer[1])
        try:
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                command = line.decode("utf-8", errors="ignore").strip()
                if not command:
                    continue
                response = self.dispatch(command)
                self.wfile.write(json.dumps(response).encode("utf-8") + b"\n")
        finally:
            LOGGER.info("Connection closed %s:%s", peer[0], peer[1])

    def dispatch(self, command: str) -> dict:
        parts = command.split()
        if not parts:
            return {"error": "empty command"}

        keyword = parts[0].upper()
        if keyword == "PING":
            return {"status": "ok", "time": time.time()}
        if keyword == "CAMERAS":
            return {"cameras": self.server.store.list_cameras()}
        if keyword == "GETALL":
            snapshots = [snap.to_payload() for snap in self.server.store.get_all()]
            return {"snapshots": snapshots}
        if keyword == "GET":
            if len(parts) < 2:
                return {"error": "usage: GET <camera> [marker_id]"}
            camera = parts[1]
            if len(parts) == 2:
                snapshot = self.server.store.get(camera)
                if snapshot is None:
                    return {"error": f"camera '{camera}' not found"}
                return snapshot.to_payload()
            try:
                marker_id = int(parts[2])
            except ValueError:
                return {"error": "marker_id must be an integer"}
            marker = self.server.store.get_marker(camera, marker_id)
            if marker is None:
                return {"error": f"marker {marker_id} not found for camera '{camera}'"}
            return {
                "camera": camera,
                "marker_id": marker_id,
                "detection": marker,
            }
        if keyword == "HELP":
            return {
                "commands": [
                    "PING",
                    "CAMERAS",
                    "GETALL",
                    "GET <camera>",
                    "GET <camera> <marker_id>",
                ]
            }
        return {"error": f"unknown command '{command}'"}


class SimpleTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    """Threaded TCP server carrying a shared telemetry store."""

    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address: Tuple[str, int], store: TelemetryStore) -> None:
        super().__init__(server_address, SimpleTCPHandler)
        self.store = store


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple TCP telemetry relay for Lorex cameras")
    parser.add_argument(
        "--camera",
        action="append",
        dest="cameras",
        required=True,
        help="Camera name to track (can repeat)",
    )
    parser.add_argument(
        "--host",
        default=Settings.tracking_server_ip,
        help=(
            "Bind host (default: "
            f"{Settings.tracking_server_ip}). Use 0.0.0.0 to listen on all interfaces "
            "if this machine should accept remote clients."
        ),
    )
    parser.add_argument(
        "--port",
        type=int,
        default=Settings.lorex_server_port,
        help=f"Bind port (default: {Settings.lorex_server_port})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.1,
        help="Seconds between detection polls (default: 0.1)",
    )
    parser.add_argument(
        "--detection-scale",
        type=float,
        default=None,
        help="Optional scale override for marker detection",
    )
    parser.add_argument(
        "--draw",
        action="store_true",
        help="Enable drawing for debugging (writes frames to temp dir)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING...)",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


def start_workers(
    cameras: Iterable[str],
    store: TelemetryStore,
    poll_interval: float,
    detection_scale: Optional[float],
    draw: bool,
) -> List[CameraWorker]:
    workers: List[CameraWorker] = []
    for name in cameras:
        worker = CameraWorker(
            camera_name=name,
            store=store,
            poll_interval=poll_interval,
            detection_scale=detection_scale,
            draw=draw,
        )
        worker.start()
        workers.append(worker)
    return workers


def stop_workers(workers: Iterable[CameraWorker]) -> None:
    for worker in workers:
        worker.stop()


def run_server(
    cameras: Sequence[str],
    host: str = Settings.tracking_server_ip,
    port: int = Settings.lorex_server_port,
    poll_interval: float = 0.1,
    detection_scale: Optional[float] = None,
    draw: bool = False,
    log_level: Optional[str] = "INFO",
) -> None:
    """Start the telemetry server with an explicit configuration."""

    # ``cameras`` is often provided via CLI flags or a config tuple. Convert to a
    # list while preserving order but de-duplicating entries so we don't spin up
    # multiple workers for the same feed if the caller repeats a name.
    original_cameras = list(cameras)
    camera_list = list(dict.fromkeys(original_cameras))
    if len(camera_list) < len(original_cameras):
        duplicates: List[str] = []
        seen: set[str] = set()
        for name in original_cameras:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            else:
                seen.add(name)
        LOGGER.warning(
            "Duplicate camera names requested; only launching one worker per name: %s",
            ", ".join(duplicates),
        )
    if not camera_list:
        raise ValueError("at least one camera name must be provided")

    if detection_scale is not None and detection_scale <= 0:
        raise ValueError("detection_scale must be positive")

    if log_level is not None:
        configure_logging(log_level)

    store = TelemetryStore()
    workers = start_workers(
        cameras=camera_list,
        store=store,
        poll_interval=poll_interval,
        detection_scale=detection_scale,
        draw=draw,
    )

    # Bind and create server
    try:
        # Bind to '0.0.0.0' to accept connections from all interfaces
        server = SimpleTCPServer(('0.0.0.0', port), store)
        LOGGER.info("Serving on 0.0.0.0:%s (accessible via %s)", port, host)
    except OSError as exc:
        if exc.errno == errno.EADDRNOTAVAIL:
            raise RuntimeError(
                "The telemetry server could not bind to "
                f"0.0.0.0:{port}. The server is configured to listen on all interfaces. "
                f"Clients should connect using {host}:{port}. Update Settings.tracking_server_ip or pass --host "
                "when starting the server to choose a reachable interface."
            ) from exc
        raise  # re-raise other OS errors

    # Start the server (clean Ctrl-C handling)
    try:
        LOGGER.info("Serving on port %s", port)
        server.serve_forever()  # Ctrl-C -> KeyboardInterrupt here
    except KeyboardInterrupt:
        LOGGER.info("KeyboardInterrupt (Ctrl-C) received; shutting down")
    except Exception as e:
        LOGGER.error("Server error: %s", e, exc_info=True)
    finally:
        # Orderly shutdown: stop accepting, close socket, stop workers
        try:
            server.shutdown()
        except Exception:
            pass
        try:
            server.server_close()
        except Exception:
            pass
        stop_workers(workers)
        LOGGER.info("Server and workers stopped")

def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_server(
        cameras=args.cameras,
        host=args.host,
        port=args.port,
        poll_interval=args.interval,
        detection_scale=args.detection_scale,
        draw=args.draw,
        log_level=args.log_level,
    )


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()