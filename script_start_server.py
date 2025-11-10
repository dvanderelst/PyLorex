"""Convenience launcher for the Lorex TCP telemetry server.

Edit the module-level constants below to match the usual lab setup. This keeps
the configuration in one obvious place while still allowing you to run the
script directly via ``python script_start_server.py``.
"""

from __future__ import annotations

from PyLorex.library.server.simple_tcp import run_server


# --- Default lab configuration -------------------------------------------------
CAMERAS = ("tiger", "shark")
# Bind address for the telemetry service. Leave at "0.0.0.0" to listen on all
# interfaces of the computer that runs this script. Cameras are still selected
# by name via ``CAMERAS`` above.
HOST = "0.0.0.0"
PORT = 9999
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
