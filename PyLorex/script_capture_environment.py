import time

from LorexLib.Environment import capture_environment_layout

# Edit these values in PyCharm before running for a quick interactive workflow.
cameras = None  # e.g. ["tiger", "shark"] or None for Settings.channels
run_name = "test"
samples = None  # None uses Settings.environment_frame_samples
latest_snapshot = capture_environment_layout(run_name=run_name)
