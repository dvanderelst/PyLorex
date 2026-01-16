"""
Click-to-sample HSV ranges from a static image.

Usage:
  python script_hsv_sampler.py --image path/to/frame.jpg [--pad 0]
  python script_hsv_sampler.py --camera tiger [--pad 0]

Controls:
  - Left click: add a sample point (records HSV at that pixel).
  - Right click: remove the nearest sample to the click (if any within the pick radius).
  - Middle click or Shift+Left click: remove the nearest sample within the pick radius.
  - 'p': print the current min/max HSV box.
  - 'c': clear all samples.
  - 'q' or Esc: quit (prints the final range if any points were sampled).
"""

import argparse
import cv2 as cv
import numpy as np
from typing import List, Tuple


Point = Tuple[int, int]


def _compute_bounds(samples_hsv: List[np.ndarray], pad: int) -> Tuple[np.ndarray, np.ndarray]:
    stack = np.stack(samples_hsv, axis=0).astype(np.int32)
    lo = np.maximum(stack.min(axis=0) - pad, 0)
    hi = np.array([179, 255, 255], dtype=np.int32)
    hi = np.minimum(stack.max(axis=0) + pad, hi)
    return lo.astype(np.uint8), hi.astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample HSV values by clicking on an image.")
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--image", help="Path to a BGR image file.")
    source_group.add_argument("--camera", help="Camera name to capture a snapshot from.")
    parser.add_argument("--pad", type=int, default=0, help="Pad the min/max HSV by this many units.")
    parser.add_argument(
        "--pick-radius",
        type=int,
        default=12,
        help="Maximum pixel distance to remove a sample on right click.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.25,
        help="Scale factor for display only (sampling uses the original image).",
    )
    args = parser.parse_args()

    if args.image:
        frame_bgr = cv.imread(args.image)
        if frame_bgr is None:
            raise SystemExit(f"Could not read image: {args.image}")
    else:
        from LorexLib.Lorex import LorexCamera

        camera = LorexCamera(args.camera, auto_start=True)
        try:
            camera.wait_ready(timeout=5.0)
            frame_bgr = camera.get_frame(undistort=False)
            if frame_bgr is None:
                raise SystemExit(f"Could not capture a frame from camera: {args.camera}")
        finally:
            camera.stop()
    if args.display_scale <= 0:
        raise SystemExit("--display-scale must be > 0.")
    frame_hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    if args.display_scale == 1.0:
        display_bgr = frame_bgr
    else:
        display_bgr = cv.resize(
            frame_bgr,
            None,
            fx=args.display_scale,
            fy=args.display_scale,
            interpolation=cv.INTER_AREA,
        )

    samples_xy: List[Point] = []
    samples_hsv: List[np.ndarray] = []
    window = "HSV Sampler"
    mask_window = "Mask Preview"
    cv.namedWindow(window, cv.WINDOW_AUTOSIZE)
    cv.namedWindow(mask_window, cv.WINDOW_AUTOSIZE)

    def _remove_nearest(x_img: int, y_img: int, reason: str) -> None:
        if not samples_xy:
            return
        pts = np.array(samples_xy, dtype=np.int32)
        dists = np.sum((pts - np.array([x_img, y_img])) ** 2, axis=1)
        idx = int(np.argmin(dists))
        if dists[idx] <= args.pick_radius * args.pick_radius:
            removed_xy = samples_xy.pop(idx)
            removed_hsv = samples_hsv.pop(idx)
            print(
                f"Removed sample at {removed_xy} HSV={tuple(int(v) for v in removed_hsv)} "
                f"(nearest to click at ({x_img}, {y_img})) [{reason}]"
            )
        else:
            print(f"No sample within {args.pick_radius}px of click at ({x_img}, {y_img}).")

    def on_mouse(event: int, x: int, y: int, flags: int, *_: int) -> None:
        x_img = int(round(x / args.display_scale))
        y_img = int(round(y / args.display_scale))
        x_img = int(np.clip(x_img, 0, frame_bgr.shape[1] - 1))
        y_img = int(np.clip(y_img, 0, frame_bgr.shape[0] - 1))
        if event == cv.EVENT_LBUTTONDOWN:
            if flags & cv.EVENT_FLAG_SHIFTKEY:
                _remove_nearest(x_img, y_img, "shift+left")
                return
            samples_xy.append((x_img, y_img))
            samples_hsv.append(frame_hsv[y_img, x_img].copy())
            print(
                f"Added sample at ({x_img}, {y_img}) HSV="
                f"{tuple(int(v) for v in frame_hsv[y_img, x_img])}"
            )
        elif event == cv.EVENT_RBUTTONDOWN:
            _remove_nearest(x_img, y_img, "right")
        elif event == cv.EVENT_MBUTTONDOWN:
            _remove_nearest(x_img, y_img, "middle")

    cv.setMouseCallback(window, on_mouse)

    print(
        "\n".join(
            [
                "HSV Sampler controls",
                "--------------------",
                "Mouse:",
                "  Left click           Add a sample at the cursor",
                "  Right click          Remove nearest sample (within pick radius)",
                "  Middle click         Remove nearest sample (within pick radius)",
                "  Shift + Left click   Remove nearest sample (within pick radius)",
                "Keyboard:",
                "  p    Print current HSV min/max",
                "  c    Clear all samples",
                "  q    Quit (also prints final HSV min/max)",
                "  Esc  Quit",
            ]
        )
    )

    while True:
        display = display_bgr.copy()
        for (x, y) in samples_xy:
            dx = int(round(x * args.display_scale))
            dy = int(round(y * args.display_scale))
            cv.circle(display, (dx, dy), 4, (0, 255, 255), thickness=2)
        if samples_hsv:
            lo, hi = _compute_bounds(samples_hsv, args.pad)
            mask = cv.inRange(frame_hsv, lo, hi)
            masked = cv.bitwise_and(frame_bgr, frame_bgr, mask=mask)
            if masked.shape[:2] != display.shape[:2]:
                masked = cv.resize(
                    masked,
                    (display.shape[1], display.shape[0]),
                    interpolation=cv.INTER_AREA,
                )
            cv.imshow(mask_window, masked)
            cv.putText(
                display,
                f"Samples: {len(samples_hsv)}  HSV range: {tuple(int(v) for v in lo)} - {tuple(int(v) for v in hi)}",
                (10, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )
        else:
            cv.imshow(mask_window, np.zeros_like(display))
            cv.putText(
                display,
                "Click to add samples",
                (10, 25),
                cv.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv.LINE_AA,
            )

        cv.imshow(window, display)
        key = cv.waitKey(30) & 0xFF
        if key in (ord("q"), 27):  # q or ESC
            break
        if key == ord("p") and samples_hsv:
            lo, hi = _compute_bounds(samples_hsv, args.pad)
            print(f"Current HSV range: ({tuple(int(v) for v in lo)}, {tuple(int(v) for v in hi)})")
        if key == ord("c"):
            samples_xy.clear()
            samples_hsv.clear()
            print("Cleared samples.")

    cv.destroyAllWindows()
    if samples_hsv:
        lo, hi = _compute_bounds(samples_hsv, args.pad)
        print(f"Final HSV range: ({tuple(int(v) for v in lo)}, {tuple(int(v) for v in hi)})")
    else:
        print("No samples collected.")


if __name__ == "__main__":
    main()
