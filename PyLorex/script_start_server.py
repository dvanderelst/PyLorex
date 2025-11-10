"""Convenience launcher for the Lorex TCP telemetry server.

Edit the module-level constants below to match the usual lab setup. This keeps
the configuration in one obvious place while still allowing you to run the
script directly via ``python PyLorex/script_start_server.py``.
"""

from __future__ import annotations

import sys

from library import Settings
from library import Utils
from library.Simple_tcp import run_server


# --- Default lab configuration -------------------------------------------------
CAMERAS = ("tiger", "shark")
# Bind address for the telemetry service. Pull the defaults from ``Settings`` so
# CLI wrappers stay in sync. Cameras are still selected by name via
# ``CAMERAS`` above.
PORT = Settings.lorex_server_port
POLL_INTERVAL = 0.1  # seconds between detection polls
DETECTION_SCALE = None  # ``None`` -> use camera default
DRAW_DEBUG = False
LOG_LEVEL = "INFO"


def main() -> None:
    """Start the telemetry server with the lab's usual defaults."""

    try:
        run_server(
            cameras=CAMERAS,
            host='0.0.0.0', #<-- Listen on all interfaces
            port=PORT,
            poll_interval=POLL_INTERVAL,
            detection_scale=DETECTION_SCALE,
            draw=DRAW_DEBUG,
            log_level=LOG_LEVEL,
        )
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover - script entry point
    current_ip = Utils.get_local_ip()
    print(f"Starting Lorex telemetry server on {current_ip}:{PORT} ...")
    main()
