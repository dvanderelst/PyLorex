"""Capture environment layout (e.g., wall tape) into a unified arena frame."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2 as cv
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from LorexLib import Settings
from LorexLib.Lorex import LorexCamera

__all__ = ["capture_environment_layout"]


def capture_environment_layout(
    cameras: Optional[Iterable[str]] = None,
    *,
    run_name: Optional[str] = None,
    save: bool = True,
    save_root: Optional[os.PathLike | str] = None,
    sample_count: Optional[int] = None,
    hsv_ranges: Optional[List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]] = None,
    map_mm_per_px: Optional[float] = None,
    arena_bounds_mm: Optional[Tuple[float, float, float, float]] = None,
    mask_stride: Optional[int] = None,
    morph_kernel: Optional[int] = None,
    morph_iterations: Optional[int] = None,
    min_blob_area_px: Optional[int] = None,
    use_undistorted: Optional[bool] = None,
) -> Dict[str, Any]:
    """Capture a snapshot of wall tape locations in board coordinates.

    Returns a dict with:
      - mask: unified uint8 mask (255=wall) in arena/board coordinates
      - meta: metadata about the capture and processing settings
      - per_camera_masks: per-camera masks in the same coordinate frame
    """

    camera_list = _resolve_cameras(cameras)
    if not camera_list:
        raise ValueError("No cameras provided for environment capture.")

    bounds = arena_bounds_mm or (
        float(Settings.environment_arena_min_x_mm),
        float(Settings.environment_arena_max_x_mm),
        float(Settings.environment_arena_min_y_mm),
        float(Settings.environment_arena_max_y_mm),
    )
    min_x, max_x, min_y, max_y = bounds
    mm_per_px = float(map_mm_per_px or Settings.environment_map_mm_per_px)
    stride = int(mask_stride or Settings.environment_mask_stride)
    kernel_size = int(morph_kernel or Settings.environment_morph_kernel)
    kernel_iters = int(morph_iterations or Settings.environment_morph_iterations)
    min_area = int(min_blob_area_px or Settings.environment_min_blob_area_px)
    sample_count = int(sample_count or Settings.environment_frame_samples)
    hsv_ranges = hsv_ranges or Settings.environment_wall_hsv_ranges

    height_px = int(np.floor((max_y - min_y) / mm_per_px)) + 1
    width_px = int(np.floor((max_x - min_x) / mm_per_px)) + 1
    unified = np.zeros((height_px, width_px), dtype=np.uint8)
    per_camera_masks: Dict[str, np.ndarray] = {}
    per_camera_frames: Dict[str, np.ndarray] = {}
    per_camera_arena_frames: Dict[str, np.ndarray] = {}
    camera_meta: Dict[str, Any] = {}
    arena_sum = np.zeros((height_px, width_px, 3), dtype=np.float32)
    arena_count = np.zeros((height_px, width_px, 1), dtype=np.float32)

    for camera_name in camera_list:
        cam = LorexCamera(camera_name, auto_start=True)
        try:
            cam.load_board_bundle()
            H_raw = cam.bundle.get("H_raw")
            H_undistorted = cam.bundle.get("H_undistorted")
            if use_undistorted is None:
                want_undistorted = H_undistorted is not None
            else:
                want_undistorted = bool(use_undistorted) and H_undistorted is not None
            # Use undistorted frames with H_undistorted when available to avoid
            # distortion-induced misalignment between cameras at the image edges.
            if want_undistorted:
                alpha = cam.bundle.get("alpha")
                if alpha is not None:
                    # Match calibration undistort FOV so the homography stays consistent.
                    cam.set_alpha(alpha)
            frame = _capture_median_frame(cam, sample_count, undistort=want_undistorted)
            if frame is None:
                camera_meta[camera_name] = {"status": "no_frame"}
                continue
            per_camera_frames[camera_name] = frame
            mask = _segment_tape(frame, hsv_ranges, kernel_size, kernel_iters, min_area)
            if mask is None or np.count_nonzero(mask) == 0:
                camera_meta[camera_name] = {"status": "no_mask"}
                continue

            H_use = H_undistorted if want_undistorted else H_raw
            if H_use is None:
                camera_meta[camera_name] = {"status": "missing_homography"}
                continue

            offset_xy = None
            if camera_name == "shark":
                offset_xy = (Settings.shark2tiger_delta_x, Settings.shark2tiger_delta_y)
            projected = _project_mask_to_arena(
                mask,
                np.asarray(H_use, dtype=np.float64),
                min_x,
                max_x,
                min_y,
                max_y,
                mm_per_px,
                stride,
                offset_xy=offset_xy,
            )
            if projected is None:
                camera_meta[camera_name] = {"status": "no_projected_points"}
                continue
            per_camera_masks[camera_name] = projected
            unified = np.maximum(unified, projected)
            warped = _warp_frame_to_arena(
                frame,
                np.asarray(H_use, dtype=np.float64),
                min_x,
                max_x,
                min_y,
                max_y,
                mm_per_px,
                offset_xy=offset_xy,
            )
            if warped is not None:
                per_camera_arena_frames[camera_name] = warped
                arena_sum += warped.astype(np.float32)
                arena_count += (warped.sum(axis=2, keepdims=True) > 0).astype(np.float32)
            camera_meta[camera_name] = {
                "status": "ok",
                "mask_pixels": int(np.count_nonzero(mask)),
                "projected_pixels": int(np.count_nonzero(projected)),
                "homography": "H_undistorted" if want_undistorted else "H_raw",
                "undistort_frame": bool(want_undistorted),
            }
        finally:
            cam.stop()

    meta = _build_meta(
        camera_list=camera_list,
        bounds=bounds,
        mm_per_px=mm_per_px,
        sample_count=sample_count,
        stride=stride,
        hsv_ranges=hsv_ranges,
        morph_kernel=kernel_size,
        morph_iterations=kernel_iters,
        min_blob_area_px=min_area,
        use_undistorted=bool(use_undistorted) if use_undistorted is not None else None,
        per_camera=camera_meta,
    )
    if run_name:
        meta["run_name"] = run_name

    if np.any(arena_count > 0):
        arena_image = (arena_sum / np.maximum(arena_count, 1.0)).astype(np.uint8)
        # Stretch contrast so low-contrast floors remain visible in the stitched overview.
        arena_image = _stretch_contrast(arena_image, arena_count[:, :, 0] > 0)
    else:
        arena_image = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    snapshot = {
        "mask": unified,
        "meta": meta,
        "per_camera_masks": per_camera_masks,
        "per_camera_frames": per_camera_frames,
        "per_camera_arena_frames": per_camera_arena_frames,
        "arena_image": arena_image,
    }

    if save:
        _save_snapshot(snapshot, run_name=run_name, save_root=save_root)

    return snapshot


def _resolve_cameras(cameras: Optional[Iterable[str]]) -> List[str]:
    if cameras is None:
        return list(Settings.channels.keys())
    return [str(name) for name in cameras]


def _capture_median_frame(
    cam: LorexCamera,
    sample_count: int,
    *,
    undistort: bool = False,
) -> Optional[np.ndarray]:
    frames = []
    for _ in range(max(sample_count, 1)):
        frame = cam.get_frame(undistort=undistort)
        if frame is None:
            continue
        frames.append(frame)
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]
    stack = np.stack(frames, axis=0)
    median = np.median(stack, axis=0)
    return median.astype(np.uint8)


def _segment_tape(
    frame_bgr: np.ndarray,
    hsv_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
    kernel_size: int,
    kernel_iters: int,
    min_blob_area_px: int,
) -> Optional[np.ndarray]:
    hsv = cv.cvtColor(frame_bgr, cv.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for low, high in hsv_ranges:
        lo = np.array(low, dtype=np.uint8)
        hi = np.array(high, dtype=np.uint8)
        mask = cv.bitwise_or(mask, cv.inRange(hsv, lo, hi))

    if kernel_size > 1 and kernel_iters > 0:
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=kernel_iters)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=kernel_iters)

    if min_blob_area_px > 1:
        num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask, connectivity=8)
        filtered = np.zeros_like(mask)
        for label in range(1, num_labels):
            area = stats[label, cv.CC_STAT_AREA]
            if area >= min_blob_area_px:
                filtered[labels == label] = 255
        mask = filtered

    return mask


def _project_mask_to_arena(
    mask: np.ndarray,
    H_raw: np.ndarray,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    mm_per_px: float,
    stride: int,
    offset_xy: Optional[Tuple[float, float]] = None,
) -> Optional[np.ndarray]:
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return None
    if stride > 1:
        idx = np.arange(0, ys.size, stride)
        ys = ys[idx]
        xs = xs[idx]
    pts = np.stack([xs, ys], axis=1).astype(np.float32).reshape(-1, 1, 2)
    mapped = cv.perspectiveTransform(pts, H_raw).reshape(-1, 2)
    x = mapped[:, 0]
    y = mapped[:, 1]
    if offset_xy is not None:
        x = x + float(offset_xy[0])
        y = y + float(offset_xy[1])
    width_px = int(np.floor((max_x - min_x) / mm_per_px)) + 1
    height_px = int(np.floor((max_y - min_y) / mm_per_px)) + 1
    cols = np.rint((x - min_x) / mm_per_px).astype(int)
    rows = np.rint((max_y - y) / mm_per_px).astype(int)
    valid = (
        np.isfinite(x) & np.isfinite(y) &
        (cols >= 0) & (cols < width_px) &
        (rows >= 0) & (rows < height_px)
    )
    if not np.any(valid):
        return None
    projected = np.zeros((height_px, width_px), dtype=np.uint8)
    projected[rows[valid], cols[valid]] = 255
    return projected


def _warp_frame_to_arena(
    frame: np.ndarray,
    H_img2mm: np.ndarray,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
    mm_per_px: float,
    offset_xy: Optional[Tuple[float, float]] = None,
) -> Optional[np.ndarray]:
    # Rough stitch: map camera pixels into the arena grid using the same mm->px mapping as the mask.
    height_px = int(np.floor((max_y - min_y) / mm_per_px)) + 1
    width_px = int(np.floor((max_x - min_x) / mm_per_px)) + 1
    if width_px <= 0 or height_px <= 0:
        return None
    scale = 1.0 / mm_per_px
    H_mm2arena = np.array(
        [
            [scale, 0.0, -min_x * scale],
            [0.0, -scale, max_y * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    if offset_xy is not None:
        dx, dy = float(offset_xy[0]), float(offset_xy[1])
        H_offset = np.array(
            [
                [1.0, 0.0, dx],
                [0.0, 1.0, dy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        H_img2arena = H_mm2arena @ (H_offset @ H_img2mm)
    else:
        H_img2arena = H_mm2arena @ H_img2mm
    warped = cv.warpPerspective(
        frame,
        H_img2arena,
        (width_px, height_px),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped


def _stretch_contrast(image: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 3:
        return image
    if valid_mask is None or not np.any(valid_mask):
        return image
    out = image.astype(np.float32)
    for ch in range(3):
        vals = out[:, :, ch][valid_mask]
        if vals.size < 10:
            continue
        lo, hi = np.percentile(vals, [2.0, 98.0])
        if hi <= lo:
            continue
        scale = 255.0 / (hi - lo)
        out[:, :, ch] = (out[:, :, ch] - lo) * scale
    out = np.clip(out, 0, 255).astype(np.uint8)
    out[~valid_mask] = 0
    return out


def _build_meta(
    *,
    camera_list: List[str],
    bounds: Tuple[float, float, float, float],
    mm_per_px: float,
    sample_count: int,
    stride: int,
    hsv_ranges: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]],
    morph_kernel: int,
    morph_iterations: int,
    min_blob_area_px: int,
    use_undistorted: Optional[bool],
    per_camera: Dict[str, Any],
) -> Dict[str, Any]:
    min_x, max_x, min_y, max_y = bounds
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cameras": camera_list,
        "arena_bounds_mm": {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
        },
        "map_mm_per_px": mm_per_px,
        "sample_count": sample_count,
        "mask_stride": stride,
        "hsv_ranges": hsv_ranges,
        "morph_kernel": morph_kernel,
        "morph_iterations": morph_iterations,
        "min_blob_area_px": min_blob_area_px,
        "use_undistorted": use_undistorted,
        "per_camera": per_camera,
    }


def _save_snapshot(
    snapshot: Dict[str, Any],
    run_name: Optional[str] = None,
    save_root: Optional[os.PathLike | str] = None,
) -> str:
    if save_root is None:
        root = Path(Settings.environment_root)
    else:
        root = Path(save_root)
        if not root.is_absolute():
            root = Path.cwd() / root
    root.mkdir(parents=True, exist_ok=True)
    basename = _snapshot_basename(run_name=run_name, root=root)
    run_dir = root / basename
    if run_name and run_dir.exists():
        raise FileExistsError(f"Environment run folder already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    mask = snapshot["mask"]
    meta = snapshot["meta"]
    per_camera = snapshot["per_camera_masks"]
    per_camera_frames = snapshot.get("per_camera_frames", {})
    arena_image = snapshot.get("arena_image")

    mask_path = run_dir / "mask.png"
    mask_npy_path = run_dir / "mask.npy"
    meta_path = run_dir / "meta.json"

    cv.imwrite(str(mask_path), mask)
    np.save(str(mask_npy_path), mask)
    if arena_image is not None:
        arena_path = run_dir / "arena.png"
        cv.imwrite(str(arena_path), arena_image)
        arena_gray = cv.cvtColor(arena_image, cv.COLOR_BGR2GRAY)
        arena_gray_rgb = cv.cvtColor(arena_gray, cv.COLOR_GRAY2BGR)
        arena_mask_path = run_dir / "arena_mask.png"
        cv.imwrite(str(arena_mask_path), arena_gray_rgb)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    timestamp_path = run_dir / "timestamp.txt"
    with open(timestamp_path, "w", encoding="utf-8") as f:
        f.write(meta.get("timestamp", ""))
    plot_path = run_dir / "plot.png"
    _save_point_plot(mask, meta, plot_path)

    for camera_name, cam_mask in per_camera.items():
        cam_path = run_dir / f"mask_{camera_name}.png"
        cv.imwrite(str(cam_path), cam_mask)
    for camera_name, cam_frame in per_camera_frames.items():
        frame_path = run_dir / f"frame_{camera_name}.jpg"
        cv.imwrite(str(frame_path), cam_frame)
    for camera_name, cam_arena in snapshot.get("per_camera_arena_frames", {}).items():
        arena_cam_path = run_dir / f"arena_{camera_name}.png"
        cv.imwrite(str(arena_cam_path), cam_arena)

    return basename


def _snapshot_basename(run_name: Optional[str], root: Path) -> str:
    timestamp = time.strftime("%Y-%m-%dT%H-%M-%S")
    safe_name = _sanitize_label(run_name)
    if safe_name:
        return safe_name
    index = _next_snapshot_index(root)
    return f"env_{index:04d}_{timestamp}"


def _next_snapshot_index(root: Path) -> int:
    pattern = re.compile(r"^env_(\\d{4})_.*$")
    indices = []
    if root.exists():
        for name in os.listdir(root):
            match = pattern.match(name)
            if match:
                indices.append(int(match.group(1)))
    return max(indices, default=0) + 1


def _sanitize_label(label: Optional[str]) -> str:
    if not label:
        return ""
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", label.strip())
    return cleaned.strip("-")


def _save_point_plot(mask: np.ndarray, meta: Dict[str, Any], plot_path: Path) -> None:
    arena = meta.get("arena_bounds_mm", {})
    try:
        min_x = float(arena.get("min_x"))
        max_x = float(arena.get("max_x"))
        min_y = float(arena.get("min_y"))
        max_y = float(arena.get("max_y"))
        mm_per_px = float(meta.get("map_mm_per_px"))
    except (TypeError, ValueError):
        return
    if mask.ndim != 2:
        return

    rows, cols = np.where(mask > 0)
    if rows.size == 0:
        return

    # Convert mask pixel indices back to board coordinates.
    x_mm = min_x + cols * mm_per_px
    y_mm = max_y - rows * mm_per_px

    # Downsample for plotting if needed.
    if x_mm.size > 20000:
        step = int(np.ceil(x_mm.size / 20000))
        x_mm = x_mm[::step]
        y_mm = y_mm[::step]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=160)
    ax.scatter(x_mm, y_mm, s=1, c="black", alpha=0.9, linewidths=0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(min_y, max_y)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(True, linewidth=0.4, alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(plot_path), dpi=160)
    plt.close(fig)
