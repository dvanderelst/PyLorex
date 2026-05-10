# PyLorex — open items

## Surface `captured_at` through `get_position` so clients can filter stale snapshots

**Status:** open, low priority. Worked around downstream in
`3PiRobot/Control_code/Library/TrackerNav.py:wait_for_stable_pose`
(bit-equality filter on consecutive reads), which works fine at the
current fresh-frame rate (3.3 Hz after the 2026-05-10
`aruco_detect_scale` fix — see "Resolved" below).

### Problem

Clients that poll `tracker.get_position(...)` at ~10 Hz frequently receive
back-to-back **bit-identical snapshots** — same `(x, y, yaw)` floats — even
though they expect each call to return a freshly-captured pose. Downstream
code that uses spread-based stability detection (e.g., a "is the robot done
moving?" check) is fooled by these duplicates: 4 consecutive identical reads
trivially satisfy any spread tolerance, so the check declares "settled" while
the robot is still in motion.

Symptoms observed in `3PiRobot/Control_code/SCRIPT_DiagnoseDriveCurl.py`
(2026-05-08):

- Phase 2 step 8: `client.step(angle=-30°)` registered as Δyaw = −4.5°.
  The robot did rotate ~30° physically; the post-rotate settle locked onto a
  cluster of identical mid-rotation snapshots.
- Phase 2 step 14: same rotate command registered as Δyaw = +2.0°.
- Phase 1 std blew up to 4° (vs 1.5° on a cleaner earlier run) for the same
  reason: drives ended while the robot was still physically moving, so
  successive steps inherited each other's residual motion.

### Root cause

Two ~10 Hz loops in the chain, neither synchronized to the other:

1. **Server detection loop** (`LorexLib/Simple_tcp.py:178-213`):
   ```
   while running:
       cam.get_aruco(...)                # detection work
       store.update(snapshot)
       self._stopevt.wait(self.poll_interval)   # default 0.1 s
   ```

2. **Client poll loop** (downstream, e.g. `wait_for_stable_pose`):
   `get_position()` → TCP `GETALL` → server returns latest snapshot from the
   store. Polled at ~10 Hz.

Whenever the client polls between two store updates, the server returns the
most recent snapshot **a second time**. The client has no way to distinguish
"freshly captured" from "re-served from the buffer" — the snapshot's
`captured_at` is not exposed in `get_tracker(...)`'s return.

The relevant shape today (`LorexLib/ServerClient.py:get_tracker`):
```
returns (camera_name, x, y, yaw)
```
The underlying `CameraSnapshot.captured_at` (set at `Simple_tcp.py:187`) is
already populated server-side, just not propagated through `get_tracker`.

### Recommended fix

Surface `captured_at` (a monotonic-ish float seconds, e.g.
`time.time()` at the moment of detection) on `get_tracker(...)`:

```
returns (camera_name, x, y, yaw, captured_at)   # or a small dataclass
```

`LorexTracker.get_position` in the consumer repo can then either:
- Pass `captured_at` through to clients so they can dedupe on
  `captured_at == last_captured_at` (no false matches when the robot is at
  rest with sub-pixel jitter — the camera will still emit fresh
  `captured_at` values even if `(x, y, yaw)` floats happen to repeat).
- Optionally add a server-side `WAIT_NEW <since>` request that blocks until a
  snapshot newer than `since` is available (eliminates the "wasted poll"
  problem entirely).

### Why the downstream bit-equality filter is only a stopgap

Filtering on `(x, y, yaw)` equality assumes aruco sub-pixel jitter is enough
to distinguish a fresh "stationary" frame from a re-served one. That's true
in practice today — aruco corner-detection rarely produces bit-identical
floats — but it's a heuristic, not a guarantee. If detection settles into a
sub-pixel-stable regime (e.g., after refinement converges), the filter could
block legitimate convergence. `captured_at` is the principled signal.

### Reference

- Diagnostic that exposed this: `3PiRobot/Control_code/SCRIPT_DiagnoseDriveCurl.py`
  output at `Diagnostics/drive_curl_20260508T162749/`.
- Downstream workaround: `3PiRobot/Control_code/Library/TrackerNav.py:wait_for_stable_pose`
  (the `last_read` / repeat-frame filter, plus the `prior_pose` motion-required gate).

---

# Resolved

## ✅ Fresh-frame rate bottleneck — `aruco_detect_scale = 1.0` was running detection on 4K (2026-05-10)

**Fix:** flipped `Settings.aruco_detect_scale` 1.0 → 0.5
(commit `ab8e817`).

### What was happening

The 3PiRobot client measured only **0.8 Hz of distinct tracker snapshots**
despite the server's `process_loop` reporting detections at 7–8 Hz. The
mystery resolved into a chain measurable with two probe scripts (kept in
`PyLorex/`):

1. `script_probe_rtsp_rate.py` — DVR delivers ~17 fps on both main
   (3840×2160) and sub (704×480) streams. RTSP/network was *not* the
   bottleneck.
2. `script_probe_detection_rate.py` — at `aruco_detect_scale = 1.0`,
   `aruco.detectMarkers` on the 4K main-stream frame took **~450 ms /
   call (≈ 2.2 Hz)**, which (combined with the per-camera worker
   serialization and the 100 ms `poll_interval`) explained the 0.8 Hz
   client-side rate.

### What the fix changed

Setting `aruco_detect_scale = 0.5` downsamples each frame to 1920×1080
before running aruco detection, then scales the detected corners back.
Effects measured on the 3PiRobot side:

| | scale=1.0 (before) | scale=0.5 (after) |
|---|---|---|
| Distinct reads / 300 polls | 24 (8 %) | 100 (33 %) |
| Fresh-frame rate at client | 0.8 Hz | **3.3 Hz** |
| Per-frame σ_yaw | 0.613° | **0.117°** |
| Per-frame σ_x / σ_y | 0.16 / 0.26 mm | 0.11 / 0.08 mm |

The noise drop was unexpected; likely the downsample puts the marker
closer to `aruco_refine_win = 3`'s "good at ~46 px markers" sweet spot,
and the downsample's anti-aliasing low-pass-filters sensor noise.

### Retuning if hardware changes

If marker size, camera distance, or sensor changes, re-run the two probe
scripts and adjust `aruco_detect_scale`. The 17 fps grabber ceiling is
the practical upper bound — any detection rate above ~15 Hz just burns
CPU re-running on the same buffered frame.
