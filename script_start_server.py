"""Convenience launcher for the Lorex TCP telemetry server.

Adjust the configuration values in :func:`main` to suit the typical lab
setup. By default the script starts the tracker on host ``0.0.0.0`` and
monitors the ``tiger`` and ``shark`` cameras.
"""

from __future__ import annotations

from PyLorex.server.simple_tcp import run_server


def main() -> None:
    """Start the telemetry server with the lab's usual defaults."""

    run_server(
        cameras=("tiger", "shark"),
        host="0.0.0.0",
        port=9999,
        poll_interval=0.1,
        detection_scale=None,
        draw=False,
        log_level="INFO",
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()
