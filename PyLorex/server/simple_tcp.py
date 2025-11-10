"""Compatibility shim for the telemetry server module.

The implementation moved to :mod:`PyLorex.library.server.simple_tcp`, but this
module keeps the historical import path working so existing scripts that use
``PyLorex.server.simple_tcp`` continue to run without modification.
"""

from PyLorex.library.server.simple_tcp import (
    CameraSnapshot,
    CameraWorker,
    SimpleTCPServer,
    TelemetryStore,
    main,
    run_server,
)

__all__ = [
    "CameraSnapshot",
    "CameraWorker",
    "SimpleTCPServer",
    "TelemetryStore",
    "main",
    "run_server",
]


if __name__ == "__main__":  # pragma: no cover - CLI shim
    main()
