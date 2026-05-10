# PyLorex — open items

## Investigate why fresh-frame rate is ~0.8 Hz when detection logs say 7–8 Hz

**Status:** open, **HIGH priority** — bottlenecks every settle in 3PiRobot
(calibration, policy deploy, closed-loop nav). Worked around for now by
relaxing tolerances and timeouts in `3PiRobot/Library/TrackerNav.py`, but
that costs ~5 s per settled tracker read across the entire stack.

### Problem (measured 2026-05-10)

Empirical `SCRIPT_MeasureTrackerNoise.py` results from the 3PiRobot client:
- 300 polls in ~30 s → only **24 distinct snapshots** (8% distinct rate).
- 92% of polls return bit-identical re-serves of the previous snapshot.
- Per-call latency to PyLorex (`SCRIPT_TimeTrackerRequest.py`): **median
  1 ms, max 26 ms** — TCP round-trip is not the bottleneck.
- Effective fresh-frame rate at the client: **0.8 Hz**.

PyLorex's own `Simple_tcp.py:178-213` `process_loop` reports
"processed frame in X ms (Y Hz)" at 7–8 Hz. So the *detection step* is
running at the expected rate, but the **frame being detected isn't
actually changing** between most iterations — the detector spins on the
same image bytes 10× before the Grabber's `frame` variable refreshes.

### Hypothesis

The pipeline is:
```
DVR → RTSP → Grabber thread (cv2.VideoCapture + cap.read in a loop)
                └─ updates self.frame
Detection thread → reads self.frame → aruco detect → store update
TCP handler      → reads store on GETALL
```

If `Grabber.frame` only refreshes at ~0.8 Hz, the detection thread spins
at 7–8 Hz on stale bytes and produces the same result repeatedly.

Most plausible single cause: **the DVR sub-stream is configured to a low
fps in the DVR's web UI** (Lorex DVRs commonly default sub-streams to
1–7 fps). If `Grabber` is on the sub-stream, that's the bound regardless
of how fast detection runs.

### What to investigate

1. **Check the DVR's sub-stream FPS setting via its web UI.** If it's
   1 fps, raise to 10–15 fps. Single biggest expected win, no code change.
2. **Confirm which RTSP URL `Grabber.py` is opening** (sub vs. main).
   Switching to the main stream is an alternative if the DVR allows.
3. **Instrument `Grabber._loop`** with a per-frame timestamp / counter so
   the actual frame-arrival rate is logged (not just the detection rate).
   Even cheaper than (1)/(2): tells us if the bottleneck is RTSP-side or
   somewhere else in the chain (decoder, codec keyframe interval, etc.).
4. **RTSP keyframe interval (GOP size).** Some DVRs default to GOP = fps,
   so 1 keyframe per second; combined with packet loss the decoder may
   wait for keyframes. Settable in DVR config.

### Why it matters

Every `wait_for_stable_pose` settle in 3PiRobot has to wait for `n_consec`
(default 4) distinct fresh reads. At 0.8 Hz that's 5+ seconds *minimum*
per settle, on top of the rotation/drive itself. A typical calibration
session has 60+ settles → a clean 0.8 Hz → 8 Hz speedup would convert a
~10 minute calibration run into ~1 minute.

Same applies to closed-loop nav (`TrackerNav.go_to_pose`) and the policy
deployment loop (`SCRIPT_RunPolicy.py`).

---

## Surface `captured_at` through `get_position` so clients can filter stale snapshots

**Status:** open, low priority. Worked around downstream in `3PiRobot/Control_code/Library/TrackerNav.py:wait_for_stable_pose` (bit-equality filter on consecutive reads).

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
       cam.get_aruco(...)                # ~20–50 ms of detection work
       store.update(snapshot)
       self._stopevt.wait(self.poll_interval)   # default 0.1 s
   ```
   Effective store-update rate ≈ 1 / (detection_ms + 100 ms) ≈ **7–8 Hz**.

2. **Client poll loop** (downstream, e.g. `wait_for_stable_pose`):
   `get_position()` → TCP `GETALL` → server returns latest snapshot from the
   store. Polled at ~10 Hz.

Whenever the client polls between two store updates (which happens often,
since 10 > 8), the server returns the most recent snapshot **a second time**.
The client has no way to distinguish "freshly captured" from "re-served from
the buffer" — the snapshot's `captured_at` is not exposed in
`get_tracker(...)`'s return.

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
