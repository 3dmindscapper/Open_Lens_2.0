"""
renderer.py — Overlay translated text onto the inpainted document image.

For each OCR block, we:
  1. Determine the bounding box size.
  2. Estimate the font size from the *original* text's line count and box height
     (so the translated text matches the original visual density).
  3. Respect explicit line breaks (\\n) from the OCR/translation output.
  4. Word-wrap only within each line when it exceeds the box width.
  5. Sample the dominant text color from the original image.
  6. Draw text with bold / alignment matching based on the OCR category.
"""
from typing import List, Dict, Any, Tuple
import os

from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ── Font lookup tables ────────────────────────────────────────────────────────
_FONT_REGULAR = [
    r"C:\Windows\Fonts\arial.ttf",
    r"C:\Windows\Fonts\calibri.ttf",
    r"C:\Windows\Fonts\segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/Library/Fonts/Arial.ttf",
]

_FONT_BOLD = [
    r"C:\Windows\Fonts\arialbd.ttf",
    r"C:\Windows\Fonts\calibrib.ttf",
    r"C:\Windows\Fonts\segoeuib.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/Library/Fonts/Arial Bold.ttf",
]

_BOLD_CATEGORIES = {"Title", "Section-header"}
_CENTER_CATEGORIES = {"Title"}

_font_cache: Dict[Tuple[str, int], ImageFont.FreeTypeFont] = {}


def _resolve_font_path(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


_regular_path = _resolve_font_path(_FONT_REGULAR)
_bold_path = _resolve_font_path(_FONT_BOLD)


def _get_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    variant = "bold" if bold else "regular"
    key = (variant, size)
    if key in _font_cache:
        return _font_cache[key]

    path = (_bold_path if bold else _regular_path) or _regular_path
    if path:
        try:
            f = ImageFont.truetype(path, size)
            _font_cache[key] = f
            return f
        except Exception:
            pass

    f = ImageFont.load_default()
    _font_cache[key] = f
    return f


# ── Text wrapping (respects existing newlines) ───────────────────────────────

def _wrap_line(draw: ImageDraw.ImageDraw, line: str,
               font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Word-wrap a single line (no \\n) to fit within *max_width* pixels."""
    if not line.strip():
        return [line]

    words = line.split()
    if not words:
        return [line]

    result: List[str] = []
    current = words[0]

    for word in words[1:]:
        test = current + " " + word
        bbox = draw.textbbox((0, 0), test, font=font)
        if (bbox[2] - bbox[0]) <= max_width:
            current = test
        else:
            result.append(current)
            current = word

    result.append(current)
    return result


def _wrap_text_structured(draw: ImageDraw.ImageDraw, text: str,
                          font: ImageFont.FreeTypeFont,
                          max_width: int) -> List[str]:
    """
    Split on explicit \\n first (preserving document structure),
    then word-wrap within each logical line.
    """
    logical_lines = text.split("\n")
    output: List[str] = []
    for ll in logical_lines:
        output.extend(_wrap_line(draw, ll, font, max_width))
    return output


# ── Font size estimation ──────────────────────────────────────────────────────

def _estimate_font_size(original_text: str, box_h: int, box_w: int,
                        draw: ImageDraw.ImageDraw, bold: bool = False,
                        min_size: int = 6, max_size: int = 200) -> int:
    """
    Estimate font size from the *original* text so the translated text
    matches the original's visual density.

    Strategy:
      - Count logical lines in the original text.
      - Derive a target font size = box_h / (num_lines * 1.35)
        (the 1.35 factor accounts for line spacing / ascent-descent).
      - Clamp to [min_size, max_size].
    """
    orig_lines = [l for l in original_text.split("\n") if l.strip()]
    num_lines = max(1, len(orig_lines))

    # target_size so that num_lines of text at that size fill the box height
    target = int(box_h / (num_lines * 1.35))
    target = max(min_size, min(target, max_size))

    # Verify the widest original line fits at this size;
    # shrink if any single line is still wider than the box.
    font = _get_font(target, bold)
    for line in orig_lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        if lw > box_w and box_w > 0:
            # Scale down proportionally
            ratio = box_w / lw
            target = max(min_size, int(target * ratio))
            font = _get_font(target, bold)

    return target


def _fit_translated_text(draw: ImageDraw.ImageDraw, translated: str,
                         box_w: int, box_h: int, base_size: int,
                         bold: bool = False,
                         min_size: int = 6) -> Tuple[ImageFont.FreeTypeFont, int, List[str]]:
    """
    Starting from *base_size* (estimated from original text), shrink only if
    the wrapped translated text overflows the box height.
    Never grow larger than base_size — this preserves original proportions.
    """
    size = base_size
    while size >= min_size:
        font = _get_font(size, bold)
        lines = _wrap_text_structured(draw, translated, font, box_w)
        total_h = _lines_height(draw, lines, font)
        if total_h <= box_h:
            return font, size, lines
        size -= 1

    # At minimum size, just return whatever we can
    font = _get_font(min_size, bold)
    lines = _wrap_text_structured(draw, translated, font, box_w)
    return font, min_size, lines


def _lines_height(draw: ImageDraw.ImageDraw, lines: List[str],
                  font: ImageFont.FreeTypeFont, spacing: int = 2) -> int:
    if not lines:
        return 0
    total = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font)
        total += (bbox[3] - bbox[1]) + spacing
    return total - spacing


# ── Color sampling ────────────────────────────────────────────────────────────

def _sample_text_color(
    original_image: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Tuple[int, int, int]:
    """
    Sample the dominant text color from the original image at the text region.
    Uses the 5th percentile of brightness to avoid picking up watermarks/stamps.
    Clamps to max brightness 80 so text is always clearly readable.
    """
    x1, y1, x2, y2 = bbox
    w, h = original_image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    if x2 <= x1 or y2 <= y1:
        return (20, 20, 20)

    crop = original_image.crop((x1, y1, x2, y2)).convert("L")
    arr = np.array(crop)

    flat = arr.flatten()
    if flat.size == 0:
        return (20, 20, 20)

    # Use 5th percentile — captures actual text strokes, not watermarks
    p5 = int(np.percentile(flat, 5))
    # Clamp: text should never be lighter than mid-gray
    brightness = min(p5, 80)
    return (brightness, brightness, brightness)


# ── Main rendering ────────────────────────────────────────────────────────────

def render_translations(
    inpainted_image: Image.Image,
    original_image: Image.Image,
    blocks: List[Dict[str, Any]],
    padding: int = 4,
) -> Image.Image:
    """
    Draw translated text onto the inpainted image at each block's bounding box.

    Estimates font size from the original text's line count so titles stay large
    and multi-line body blocks don't inflate.  Preserves \\n line breaks from
    the OCR output to maintain form/table structure.
    """
    result = inpainted_image.copy()
    draw = ImageDraw.Draw(result)

    for block in blocks:
        translated = block.get("translated", "").strip()
        original_text = block.get("text", "")
        bbox = block.get("bbox")
        if not translated or not bbox:
            continue

        category = block.get("category", "Text")
        bold = category in _BOLD_CATEGORIES
        center = category in _CENTER_CATEGORIES

        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1 - padding * 2)
        box_h = max(1, y2 - y1 - padding * 2)

        # Estimate font size from original text, then fit translated text
        base_size = _estimate_font_size(original_text, box_h, box_w, draw, bold)
        font, _, lines = _fit_translated_text(
            draw, translated, box_w, box_h, base_size, bold=bold
        )
        color = _sample_text_color(original_image, bbox)

        # Draw lines top-aligned (matching original document flow)
        line_spacing = 2
        y_offset = y1 + padding

        for line in lines:
            lbbox = draw.textbbox((0, 0), line or " ", font=font)
            lw = lbbox[2] - lbbox[0]
            lh = lbbox[3] - lbbox[1]

            if center:
                x_offset = x1 + padding + max(0, (box_w - lw) // 2)
            else:
                x_offset = x1 + padding

            if line.strip():
                draw.text((x_offset, y_offset), line, font=font, fill=color)
            y_offset += lh + line_spacing

    return result
