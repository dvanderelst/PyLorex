#!/usr/bin/env python3
"""
script_probe_rtsp_rate.py

Measure the **actual frame arrival rate** delivered by the DVR over RTSP,
independent of PyLorex's detection pipeline. Opens a fresh
cv2.VideoCapture for each (channel, subtype) combination, reads frames
back-to-back for a fixed duration, and reports effective FPS, frame
size, and inter-frame timing percentiles.

Why: the 3PiRobot client sees ~0.8 Hz of distinct tracker snapshots
even though PyLorex's detection loop logs 7-8 Hz. The hypothesis (see
TODO.md, top item) is that cap.read() itself is blocking for ~1.25 s
per frame because the DVR is delivering at low fps — i.e., the
bottleneck is upstream of PyLorex's code. This script confirms or
refutes that hypothesis at the RTSP layer.

Reads:
  - subtype=0 ("main" stream — full resolution; what Grabber uses today)
  - subtype=1 ("sub" stream — lower res, typically lighter)
for each configured channel in Settings.channels.

Caveat: if the PyLorex server is currently running and holding an RTSP
session for a channel, the DVR may refuse a second concurrent session
(Lorex/Dahua DVRs commonly allow 4-8 concurrent sessions per user, but
configured limits vary). If a stream fails to open here, stop the
server and re-run.
"""

import time
from statistics import median

import cv2

from LorexLib import Settings


DURATION_S = 10.0
WARMUP_FRAMES = 5
TIMEOUT_S = 6.0   # bail if we get nothing within this long after warmup


def _rtsp_url(channel: int, subtype: int) -> str:
    return (f"rtsp://{Settings.username}:{Settings.password}"
            f"@{Settings.lorex_ip}:554/cam/realmonitor"
            f"?channel={channel}&subtype={subtype}")


def probe(channel_name: str, channel: int, subtype: int,
          duration_s: float = DURATION_S) -> dict:
    url = _rtsp_url(channel, subtype)
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if not cap.isOpened():
        return {"channel": channel_name, "channel_id": channel,
                "subtype": subtype, "error": "failed to open"}

    # Warm up so the decoder has flushed initial keyframe-wait latency.
    t_warmup = time.time()
    for _ in range(WARMUP_FRAMES):
        cap.read()
        if time.time() - t_warmup > TIMEOUT_S:
            cap.release()
            return {"channel": channel_name, "channel_id": channel,
                    "subtype": subtype,
                    "error": f"no frames within {TIMEOUT_S}s of opening"}

    sizes = set()
    inter_frame: list = []
    frames = 0
    last_t = time.time()
    t_start = last_t
    while time.time() - t_start < duration_s:
        ret, frame = cap.read()
        now = time.time()
        if not ret:
            break
        frames += 1
        if frame is not None:
            sizes.add(frame.shape)
        inter_frame.append(now - last_t)
        last_t = now
    elapsed = time.time() - t_start
    cap.release()

    if not inter_frame:
        return {"channel": channel_name, "channel_id": channel,
                "subtype": subtype, "error": "0 frames captured"}

    inter_ms = [x * 1000.0 for x in inter_frame]
    inter_sorted = sorted(inter_ms)
    p95_idx = max(0, int(len(inter_sorted) * 0.95) - 1)
    return {
        "channel":           channel_name,
        "channel_id":        channel,
        "subtype":           subtype,
        "frames":            frames,
        "elapsed_s":         elapsed,
        "fps":               frames / elapsed if elapsed > 0 else 0.0,
        "frame_sizes":       sorted(sizes),
        "interframe_ms_min": inter_sorted[0],
        "interframe_ms_med": median(inter_ms),
        "interframe_ms_p95": inter_sorted[p95_idx],
        "interframe_ms_max": inter_sorted[-1],
    }


def main():
    print(f"Probing DVR at {Settings.lorex_ip} for "
          f"{DURATION_S:.0f} s per (channel, subtype) combination.\n"
          f"Channels: {Settings.channels}\n")

    print(f"{'name':>8}  {'ch':>2}  {'sub':>3}  "
          f"{'fps':>6}  {'frames':>6}  {'min':>6}  {'med':>6}  "
          f"{'p95':>6}  {'max':>6}  {'size':>15}")
    print("-" * 100)

    for ch_name, ch_id in Settings.channels.items():
        for subtype in (0, 1):
            r = probe(ch_name, ch_id, subtype)
            if "error" in r:
                print(f"{ch_name:>8}  {ch_id:>2}  {subtype:>3}  "
                      f"ERROR: {r['error']}")
                continue
            sz = r["frame_sizes"][0] if r["frame_sizes"] else (0, 0, 0)
            sz_str = f"{sz[1]}x{sz[0]}"
            print(f"{r['channel']:>8}  {r['channel_id']:>2}  "
                  f"{r['subtype']:>3}  "
                  f"{r['fps']:>6.2f}  {r['frames']:>6d}  "
                  f"{r['interframe_ms_min']:>6.1f}  "
                  f"{r['interframe_ms_med']:>6.1f}  "
                  f"{r['interframe_ms_p95']:>6.1f}  "
                  f"{r['interframe_ms_max']:>6.1f}  "
                  f"{sz_str:>15}")

    print()
    print("Interpretation:")
    print("  • fps ≪ 25 → DVR is delivering at low fps; PyLorex can't go")
    print("    faster than this. Check DVR web UI for that channel/stream's")
    print("    'Frame Rate' / 'FPS' / 'Encoding' setting.")
    print("  • fps ≈ 25-30 → DVR is fine; bottleneck is in PyLorex code")
    print("    (detection thread, cap.read() pacing, store update logic).")
    print("  • Large gap between med and max interframe → bursty delivery,")
    print("    likely RTSP keyframe interval or network jitter.")


if __name__ == "__main__":
    main()
