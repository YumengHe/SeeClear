from pathlib import Path

import cv2
import numpy as np
from PIL import Image


DEPTH_COLORMAP_COLORS = np.array(
    [
        [0xDB, 0x4F, 0x55],
        [0xFA, 0xFC, 0xBB],
        [0xA3, 0xDA, 0xA7],
        [0x3E, 0x89, 0xBE],
        [0x64, 0x59, 0xA6],
    ],
    dtype=np.float32,
)


def _normalize_depth(depth: np.ndarray, p_low: float = 2.0, p_high: float = 98.0) -> np.ndarray:
    d = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(d)
    if not np.any(valid):
        return np.zeros(d.shape, dtype=np.float32)
    dv = d[valid]
    lo = np.percentile(dv, p_low)
    hi = np.percentile(dv, p_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        lo = float(np.min(dv))
        hi = float(np.max(dv))
        if hi <= lo + 1e-12:
            return np.zeros(d.shape, dtype=np.float32)
    x = (d - lo) / (hi - lo)
    return np.clip(x, 0.0, 1.0).astype(np.float32)


def depth_to_gray_near_black(depth: np.ndarray) -> np.ndarray:
    return (_normalize_depth(depth) * 255.0 + 0.5).astype(np.uint8)


def depth_to_soft_colormap(depth: np.ndarray) -> np.ndarray:
    x = _normalize_depth(depth)
    positions = np.linspace(0.0, 1.0, len(DEPTH_COLORMAP_COLORS), dtype=np.float32)
    channels = [
        np.interp(x, positions, DEPTH_COLORMAP_COLORS[:, channel])
        for channel in range(3)
    ]
    rgb = np.stack(channels, axis=-1)
    return np.clip(rgb + 0.5, 0, 255).astype(np.uint8)


def save_depth_color(depth_npy: Path, color_path: Path) -> None:
    depth = np.load(depth_npy)
    color_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_to_soft_colormap(depth), mode="RGB").save(color_path)


def save_depth_visualizations(depth_npy: Path, gray_path: Path, color_path: Path) -> None:
    depth = np.load(depth_npy)
    gray_path.parent.mkdir(parents=True, exist_ok=True)
    color_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(depth_to_gray_near_black(depth), mode="L").save(gray_path)
    Image.fromarray(depth_to_soft_colormap(depth), mode="RGB").save(color_path)


def _clean_mask(mask: np.ndarray) -> np.ndarray:
    """Largest CC → close (fill holes) → open (remove speckles/smooth edges)."""
    m = mask.astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n_labels <= 1:
        return mask
    largest = int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
    m = (labels == largest).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k_close)
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k_open)
    return m.astype(bool)


def save_depth_per_instance_vis(
    depth_npy: Path,
    mask_paths: list,
    out_path: Path,
) -> None:
    """Per-instance normalized depth gray visualization.

    Background (no mask) = white (255).
    Each instance mask region = depth normalized independently within that mask,
    near=black far=white. Overlapping masks: later overwrites earlier.
    """
    depth = np.load(depth_npy).astype(np.float32)
    H, W = depth.shape[:2]
    canvas = np.full((H, W), 255, dtype=np.uint8)

    for mask_path in mask_paths:
        mask_img = Image.open(mask_path).convert("L")
        if mask_img.size != (W, H):
            mask_img = mask_img.resize((W, H), Image.NEAREST)
        mask = _clean_mask(np.array(mask_img) > 0)
        if not np.any(mask):
            continue
        d_in = depth[mask]
        lo = float(np.percentile(d_in, 2))
        hi = float(np.percentile(d_in, 98))
        if hi <= lo + 1e-12:
            lo, hi = float(np.min(d_in)), float(np.max(d_in))
        if hi <= lo + 1e-12:
            canvas[mask] = 128
            continue
        norm = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)
        canvas[mask] = (norm[mask] * 255.0 + 0.5).astype(np.uint8)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(canvas, mode="L").save(out_path)
