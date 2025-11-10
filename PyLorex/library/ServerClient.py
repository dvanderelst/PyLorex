"""Client utilities for interacting with the PyLorex TCP telemetry server."""

from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from library import Settings

__all__ = ["TelemetryClient", "TelemetryError", "MarkerDetection", "CameraSnapshot"]


class TelemetryError(RuntimeError):
    """Raised when the telemetry server reports an error."""


@dataclass(frozen=True)
class MarkerDetection:
    """Lightweight container for a marker detection returned by the server."""

    data: Dict[str, Any]

    @property
    def id(self) -> int:
        """Return the marker identifier as an integer."""

        return int(self.data["id"])


@dataclass(frozen=True)
class CameraSnapshot:
    """Snapshot of the latest detections for a camera."""

    camera: str
    captured_at: float
    detections: List[MarkerDetection]
    frame_size: Optional[tuple[int, int]]
    error: Optional[str] = None


class TelemetryClient:
    """Simple polling client for the :mod:`PyLorex.library.simple_tcp` service."""

    def __init__(
        self,
        host: str = Settings.tracking_server_ip,
        port: int = Settings.lorex_server_port,
        timeout: float = 5.0,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Public API
    def ping(self) -> Dict[str, Any]:
        """Return the server status payload."""

        return self._request("PING")

    def list_cameras(self) -> List[str]:
        """Return the list of camera names known to the server."""

        payload = self._request("CAMERAS")
        cameras = payload.get("cameras")
        if not isinstance(cameras, list):
            raise TelemetryError("response missing 'cameras' list")
        return [str(name) for name in cameras]

    def get_snapshot(self, camera: str) -> CameraSnapshot:
        """Fetch the latest snapshot for *camera*."""

        payload = self._request(f"GET {camera}")
        if "camera" not in payload:
            # Server returned an error payload instead of a snapshot.
            raise TelemetryError(payload.get("error", "unexpected response"))

        detections = [MarkerDetection(det) for det in payload.get("detections", [])]
        frame_size = payload.get("frame_size")
        if frame_size is not None:
            frame_size = tuple(frame_size)
        return CameraSnapshot(
            camera=str(payload.get("camera", camera)),
            captured_at=float(payload.get("captured_at", 0.0)),
            detections=detections,
            frame_size=frame_size,
            error=payload.get("error"),
        )

    def get_trackers(self):
        processed = {}
        info = self.get_raw_trackers()
        shark2tiger_delta_x = Settings.shark2tiger_delta_x
        shark2tiger_delta_y = Settings.shark2tiger_delta_y
        for camera_data in info:
            camera_name = camera_data.camera
            for detection in camera_data.detections:
                id = detection.data['id']
                new_raw_x = detection.data['floor_xy_mm'][0]
                new_raw_y = detection.data['floor_xy_mm'][1]
                new_yaw = detection.data['yaw_deg']
                new_entry = not id in processed.keys()
                if new_entry:
                    processed[id] = [camera_name, new_raw_x, new_raw_y, new_yaw]
                else:
                    existing_raw_x = processed[id][1]
                    existing_raw_y = processed[id][2]
                    existing_center_distance = (existing_raw_x ** 2 + existing_raw_y ** 2) ** 0.5
                    new_center_distance = (new_raw_x ** 2 + new_raw_y ** 2) ** 0.5
                    if new_center_distance < existing_center_distance:
                        processed[id] = [camera_name, new_raw_x, new_raw_y, new_yaw]
        for id in processed.keys():
            camera_name, x, y, yaw = processed[id]
            if camera_name == 'shark':
                x += shark2tiger_delta_x
                y += shark2tiger_delta_y
            processed[id] = [camera_name, x, y, yaw]
        processed['raw_tracker_data'] = info
        return processed

    def get_raw_trackers(self) -> List[CameraSnapshot]:
        """Return the latest snapshots for every camera."""

        payload = self._request("GETALL")
        snapshots = payload.get("snapshots")
        if not isinstance(snapshots, list):
            raise TelemetryError("response missing 'snapshots' list")

        results: List[CameraSnapshot] = []
        for entry in snapshots:
            if not isinstance(entry, dict):
                raise TelemetryError("snapshot payload must be an object")

            camera = entry.get("camera")
            if camera is None:
                raise TelemetryError("snapshot missing 'camera'")

            raw_detections = entry.get("detections", [])
            if not isinstance(raw_detections, list):
                raise TelemetryError("snapshot contains invalid 'detections'")
            detections = [MarkerDetection(det) for det in raw_detections]

            frame_size_value = entry.get("frame_size")
            frame_size: Optional[tuple[int, int]] = None
            if frame_size_value is not None:
                if (
                    not isinstance(frame_size_value, (list, tuple))
                    or len(frame_size_value) != 2
                ):
                    raise TelemetryError("snapshot contains invalid 'frame_size'")
                frame_size = (int(frame_size_value[0]), int(frame_size_value[1]))

            results.append(
                CameraSnapshot(
                    camera=str(camera),
                    captured_at=float(entry.get("captured_at", 0.0)),
                    detections=detections,
                    frame_size=frame_size,
                    error=entry.get("error"),
                )
            )

        return results

    def get_marker(self, camera: str, marker_id: int) -> MarkerDetection:
        """Return a single marker detection."""

        payload = self._request(f"GET {camera} {int(marker_id)}")
        if "detection" not in payload:
            raise TelemetryError(payload.get("error", "marker not found"))
        detection = payload["detection"]
        if not isinstance(detection, dict):
            raise TelemetryError("response contains invalid detection payload")
        return MarkerDetection(detection)

    # ------------------------------------------------------------------
    # Internal helpers
    def _request(self, command: str) -> Dict[str, Any]:
        print('here', self.host, self.port)
        with socket.create_connection((self.host, self.port), timeout=self.timeout) as sock:
            sock.sendall(command.encode("utf-8") + b"\n")
            data = self._readline(sock)
        try:
            return json.loads(data)
        except json.JSONDecodeError as exc:
            raise TelemetryError(f"invalid JSON response: {data!r}") from exc

    def _readline(self, sock: socket.socket) -> str:
        chunks = []
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                break
            chunks.append(chunk)
            if b"\n" in chunk:
                break
        if not chunks:
            raise TelemetryError("connection closed by server")
        return b"".join(chunks).decode("utf-8").split("\n", 1)[0].strip()
