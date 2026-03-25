"""
inpaint.py — Erase original text from document images before overlaying translation.

Strategy:
  1. For each bounding box, sample the background color from a border strip
     around the box (avoids sampling the text itself).
  2. Fill the box with that sampled color.
  3. Optionally apply a light median blur to soften edges.

This is a fast, lightweight approach that works well for clean document scans.
For complex backgrounds (photos, heavy gradients), a full neural inpainting model
(e.g. LaMa) can be swapped in as a drop-in replacement for _fill_region().
"""
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image


BORDER_SAMPLE_PX = 4   # pixels around the box to sample for background color
BLUR_KERNEL = 3         # median blur kernel; set to 0 to disable


def _pil_to_cv(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def _cv_to_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _sample_background(arr: np.ndarray, bbox: Tuple[int, int, int, int]) -> Tuple[int, int, int]:
    """Sample median color from a border strip around the bounding box."""
    x1, y1, x2, y2 = bbox
    h, w = arr.shape[:2]
    pad = BORDER_SAMPLE_PX

    # Clamp border region to image bounds
    bx1 = max(0, x1 - pad)
    by1 = max(0, y1 - pad)
    bx2 = min(w, x2 + pad)
    by2 = min(h, y2 + pad)

    border_pixels = []
    full_region = arr[by1:by2, bx1:bx2]
    mask = np.zeros(full_region.shape[:2], dtype=bool)

    # Top/bottom strips
    strip_h = min(pad, full_region.shape[0])
    mask[:strip_h, :] = True
    mask[-strip_h:, :] = True
    # Left/right strips
    strip_w = min(pad, full_region.shape[1])
    mask[:, :strip_w] = True
    mask[:, -strip_w:] = True

    border_pixels = full_region[mask]
    if len(border_pixels) == 0:
        return (255, 255, 255)

    median = np.median(border_pixels, axis=0).astype(int)
    return (int(median[0]), int(median[1]), int(median[2]))  # BGR


def _fill_region(arr: np.ndarray, bbox: Tuple[int, int, int, int], color: Tuple[int, int, int]):
    """Fill a bounding box region with a solid color, then soften edges."""
    x1, y1, x2, y2 = bbox
    h, w = arr.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    arr[y1:y2, x1:x2] = color

    if BLUR_KERNEL > 0:
        # Blur a slightly expanded region to blend fill edges
        pad = BLUR_KERNEL * 2
        bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
        bx2, by2 = min(w, x2 + pad), min(h, y2 + pad)
        roi = arr[by1:by2, bx1:bx2]
        arr[by1:by2, bx1:bx2] = cv2.medianBlur(roi, BLUR_KERNEL if BLUR_KERNEL % 2 == 1 else BLUR_KERNEL + 1)


def erase_text_blocks(image: Image.Image, blocks: List[Dict[str, Any]]) -> Image.Image:
    """
    Remove original text from the image at each block's bounding box.
    Returns a new PIL Image with text erased.
    """
    arr = _pil_to_cv(image)

    for block in blocks:
        bbox = block.get("bbox")
        if not bbox:
            continue
        bg_color = _sample_background(arr, bbox)
        _fill_region(arr, bbox, bg_color)

    return _cv_to_pil(arr)
