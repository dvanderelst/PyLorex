"""Convenience launcher for the Lorex TCP telemetry server.

Edit the module-level constants below to match the usual lab setup. This keeps
the configuration in one obvious place while still allowing you to run the
script directly via ``python script_start_server.py``.
"""

from __future__ import annotations
from server.simple_tcp import run_server
from library import Settings


# --- Default lab configuration -------------------------------------------------
CAMERAS = ("tiger", "shark")
HOST = "0.0.0.0"
PORT = 1234
POLL_INTERVAL = 0.1  # seconds between detection polls
DETECTION_SCALE = None  # ``None`` -> use camera default
DRAW_DEBUG = False
LOG_LEVEL = "INFO"


def main() -> None:
    """Start the telemetry server with the lab's usual defaults."""

    run_server(
        cameras=CAMERAS,
        host=HOST,
        port=PORT,
        poll_interval=POLL_INTERVAL,
        detection_scale=DETECTION_SCALE,
        draw=DRAW_DEBUG,
        log_level=LOG_LEVEL,
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
