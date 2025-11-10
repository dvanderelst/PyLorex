"""Compatibility wrapper around the simple TCP telemetry server.

This module makes it easy to launch the tracker service via
``python PyLorex/run_server.py`` while keeping the main implementation inside
``PyLorex.library.simple_tcp``. All command line options supported by the
original module are available here as well. To monitor more than one
camera, repeat the ``--camera`` flag::

    python PyLorex/run_server.py --camera tiger --camera panther

This mirrors the ``python -m PyLorex.library.simple_tcp`` entry point
but can be easier to remember when deploying onto a lab machine.
"""

from PyLorex.library.simple_tcp import main as _main


def main() -> None:
    """Delegate to :func:`PyLorex.library.simple_tcp.main`."""

    _main()


if __name__ == "__main__":  # pragma: no cover - CLI shim
    main()
