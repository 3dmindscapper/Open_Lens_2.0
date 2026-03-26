"""
renderer.py — Overlay translated text onto the inpainted document image.

Strategy for matching the original layout:
  1. Strip markdown formatting from OCR text (**, *, #, etc.).
  2. Binary-search the font size where the *original* text (with word wrap)
     fits the bounding box — this recovers the actual font size used in the
     original document.
  3. Render the translated text at that same font size.  If the translation
     is longer, allow the font to shrink down to 70% of the original size
     but no further — readability over pixel-perfection.
  4. Preserve explicit \\n line breaks from the OCR (form field structure).
  5. Bold / centered rendering for Title and Section-header categories.
"""
import re
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


# ── Markdown stripping ───────────────────────────────────────────────────────

def _strip_markdown(text: str) -> str:
    """Remove markdown formatting markers that inflate character counts.
    Preserves content inside the markers."""
    # Headers: # Title, ## Section  (only at line starts)
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # Bold+italic combos: ***text***, ___text___
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text)
    text = re.sub(r'_{3}(.+?)_{3}', r'\1', text)
    # Bold: **text**, __text__
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text)
    text = re.sub(r'_{2}(.+?)_{2}', r'\1', text)
    # Italic: *text* (but not ***Redacted*** which was already handled)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    return text


# ── Text wrapping (respects existing newlines) ───────────────────────────────

def _wrap_line(draw: ImageDraw.ImageDraw, line: str,
               font: ImageFont.FreeTypeFont, max_width: int) -> List[str]:
    """Word-wrap a single line (no \\n) to fit within *max_width* pixels."""
    if not line.strip():
        return [""]

    words = line.split()
    if not words:
        return [""]

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


def _wrap_text(draw: ImageDraw.ImageDraw, text: str,
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


def _lines_height(draw: ImageDraw.ImageDraw, lines: List[str],
                  font: ImageFont.FreeTypeFont, spacing: int = 2) -> int:
    """Total pixel height of a list of text lines."""
    if not lines:
        return 0
    total = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line or " ", font=font)
        total += (bbox[3] - bbox[1]) + spacing
    return total - spacing


# ── Font size recovery ────────────────────────────────────────────────────────

def _find_original_font_size(draw: ImageDraw.ImageDraw, original_text: str,
                             box_w: int, box_h: int, bold: bool = False,
                             min_size: int = 6, max_size: int = 120) -> int:
    """
    Binary-search the largest font size where the wrapped *original* text
    fits inside (box_w, box_h).  This recovers the actual font size from
    the original document.
    """
    clean = _strip_markdown(original_text)
    lo, hi = min_size, max_size
    best = min_size

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(mid, bold)
        lines = _wrap_text(draw, clean, font, box_w)
        h = _lines_height(draw, lines, font)
        if h <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def _fit_translated_text(draw: ImageDraw.ImageDraw, translated: str,
                         box_w: int, box_h: int, original_size: int,
                         bold: bool = False,
                         min_size: int = 6) -> Tuple[ImageFont.FreeTypeFont, int, List[str]]:
    """
    Render translated text starting at *original_size*.  If it overflows,
    shrink down to max(70% of original_size, min_size).
    """
    floor_size = max(min_size, int(original_size * 0.7))
    size = original_size

    while size >= floor_size:
        font = _get_font(size, bold)
        lines = _wrap_text(draw, translated, font, box_w)
        total_h = _lines_height(draw, lines, font)
        if total_h <= box_h:
            return font, size, lines
        size -= 1

    # At floor, just accept overflow
    font = _get_font(floor_size, bold)
    lines = _wrap_text(draw, translated, font, box_w)
    return font, floor_size, lines


# ── Color sampling ────────────────────────────────────────────────────────────

def _sample_text_color(
    original_image: Image.Image,
    bbox: Tuple[int, int, int, int],
) -> Tuple[int, int, int]:
    """
    Sample the dominant text color from the original image at the text region.
    Uses the 5th percentile of brightness to target actual text strokes.
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

    p5 = int(np.percentile(flat, 5))
    brightness = min(p5, 80)
    return (brightness, brightness, brightness)


# ── Form layout detection ─────────────────────────────────────────────────────

def _detect_form_pairs(orig_text: str, trans_text: str):
    """
    Detect alternating label/value lines that represent a two-column form.

    Returns (orig_pairs, trans_pairs, header_or_None) where each pair is
    (label, value) and header is (orig_header, trans_header) if the block
    starts with a standalone header line (odd total count).
    Returns None if the pattern is not detected.
    """
    orig_clean = _strip_markdown(orig_text)
    trans_clean = _strip_markdown(trans_text)

    orig_lines = [l for l in orig_clean.split("\n") if l.strip()]
    trans_lines = [l for l in trans_clean.split("\n") if l.strip()]

    if len(orig_lines) != len(trans_lines) or len(orig_lines) < 4:
        return None

    header = None
    o_lines, t_lines = orig_lines, trans_lines

    if len(o_lines) % 2 != 0:
        # Odd count: treat the first line as a standalone header
        if len(o_lines) >= 5:
            header = (o_lines[0], t_lines[0])
            o_lines = o_lines[1:]
            t_lines = t_lines[1:]
        else:
            return None

    orig_pairs = [(o_lines[i], o_lines[i + 1])
                  for i in range(0, len(o_lines), 2)]
    trans_pairs = [(t_lines[i], t_lines[i + 1])
                   for i in range(0, len(t_lines), 2)]
    return orig_pairs, trans_pairs, header


def _find_form_font_size(draw: ImageDraw.ImageDraw,
                         labels: List[str], values: List[str],
                         box_w: int, box_h: int, bold: bool = False,
                         min_size: int = 6, max_size: int = 120) -> int:
    """
    Binary-search the largest font size where side-by-side label–value rows
    fit inside (box_w, box_h).  Values are allowed to word-wrap within their
    column, and the resulting extra height is accounted for.
    """
    lo, hi = min_size, max_size
    best = min_size

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(mid, bold)

        ref = draw.textbbox((0, 0), "Ay", font=font)
        line_h = (ref[3] - ref[1]) + 2

        # Label column width: longest label + gap
        gap = max(int(mid * 0.5), 4)
        max_lw = 0
        for lbl in labels:
            if lbl.strip():
                lb = draw.textbbox((0, 0), lbl, font=font)
                max_lw = max(max_lw, lb[2] - lb[0])

        tab_stop = max_lw + gap
        # Labels must not exceed 60% of box_w
        if tab_stop > box_w * 0.6:
            hi = mid - 1
            continue

        value_space = max(1, box_w - tab_stop)

        # Total height: label is 1 line, value may wrap
        total_h = 0
        for val in values:
            if val.strip():
                val_lines = _wrap_line(draw, val, font, value_space)
                total_h += max(1, len(val_lines)) * line_h
            else:
                total_h += line_h
        total_h -= 2  # remove last spacing

        if total_h <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def _render_form(draw: ImageDraw.ImageDraw,
                 pairs: List[Tuple[str, str]],
                 x: int, y: int, box_w: int,
                 font_size: int, bold: bool,
                 color: Tuple[int, int, int]):
    """Render form label-value pairs side-by-side with tab-stop alignment."""
    font = _get_font(font_size, bold)
    spacing = 2

    # Tab stop: longest label + gap
    max_label_w = 0
    for label, _ in pairs:
        lb = draw.textbbox((0, 0), label, font=font)
        max_label_w = max(max_label_w, lb[2] - lb[0])

    gap = max(int(font_size * 0.5), 4)
    tab_stop = min(max_label_w + gap, int(box_w * 0.55))
    value_space = max(1, box_w - tab_stop)

    y_off = y
    for label, value in pairs:
        ref = draw.textbbox((0, 0), label or " ", font=font)
        line_h = ref[3] - ref[1]

        if label.strip():
            draw.text((x, y_off), label, font=font, fill=color)

        if value.strip():
            val_lines = _wrap_line(draw, value, font, value_space)
            vy = y_off
            for vl in val_lines:
                if vl.strip():
                    draw.text((x + tab_stop, vy), vl, font=font, fill=color)
                vlb = draw.textbbox((0, 0), vl or " ", font=font)
                vy += (vlb[3] - vlb[1]) + spacing
            val_total = vy - y_off
            y_off += max(line_h + spacing, val_total)
        else:
            y_off += line_h + spacing


# ── Main rendering ────────────────────────────────────────────────────────────

def render_translations(
    inpainted_image: Image.Image,
    original_image: Image.Image,
    blocks: List[Dict[str, Any]],
    padding: int = 2,
) -> Image.Image:
    """
    Draw translated text onto the inpainted image at each block's bounding box.

    Recovers the original font size by binary-searching what fits the original
    text in the box, then renders the translated text at the same size (shrinking
    by at most 30% if the translation is longer).

    For form-like blocks (alternating label/value lines), renders with tab-stop
    alignment so labels and values appear side-by-side as in the original.
    """
    result = inpainted_image.copy()
    draw = ImageDraw.Draw(result)

    for block in blocks:
        translated_raw = block.get("translated", "").strip()
        original_raw = block.get("text", "")
        bbox = block.get("bbox")
        if not translated_raw or not bbox:
            continue

        # Strip markdown for rendering (but keep \n structure)
        translated = _strip_markdown(translated_raw)
        category = block.get("category", "Text")
        bold = category in _BOLD_CATEGORIES
        center = category in _CENTER_CATEGORIES

        x1, y1, x2, y2 = bbox
        box_w = max(1, x2 - x1 - padding * 2)
        box_h = max(1, y2 - y1 - padding * 2)

        color = _sample_text_color(original_image, bbox)

        # ── Form layout: detect alternating label/value pairs ─────────
        if not center:
            form = _detect_form_pairs(original_raw, translated_raw)
            if form:
                orig_pairs, trans_pairs, header = form
                orig_labels = [p[0] for p in orig_pairs]
                orig_values = [p[1] for p in orig_pairs]

                # Include header row in sizing so it gets height budget
                sz_labels = ([header[0]] + orig_labels) if header else orig_labels
                sz_values = (["x"] + orig_values) if header else orig_values

                stacked_size = _find_original_font_size(
                    draw, original_raw, box_w, box_h, bold)
                form_size = _find_form_font_size(
                    draw, sz_labels, sz_values, box_w, box_h, bold)

                # Only use form layout if it gives a clearly larger font,
                # meaning the original must have had side-by-side layout.
                if form_size >= stacked_size * 1.3:
                    # Find size for translated pairs (may need to shrink)
                    trans_labels = [p[0] for p in trans_pairs]
                    trans_values = [p[1] for p in trans_pairs]
                    tsz_labels = ([header[1]] + trans_labels) if header else trans_labels
                    tsz_values = (["x"] + trans_values) if header else trans_values
                    render_size = _find_form_font_size(
                        draw, tsz_labels, tsz_values, box_w, box_h, bold)
                    render_size = max(render_size, int(form_size * 0.7))

                    y_start = y1 + padding

                    # Render header line first (if present)
                    if header:
                        hdr_font = _get_font(render_size, bold)
                        _, trans_hdr = header
                        if trans_hdr.strip():
                            draw.text((x1 + padding, y_start),
                                      trans_hdr, font=hdr_font, fill=color)
                        ref = draw.textbbox((0, 0), trans_hdr or " ", font=hdr_font)
                        y_start += (ref[3] - ref[1]) + 2

                    _render_form(draw, trans_pairs,
                                 x1 + padding, y_start,
                                 box_w, render_size, bold, color)
                    continue

        # ── Normal rendering ──────────────────────────────────────────
        orig_size = _find_original_font_size(
            draw, original_raw, box_w, box_h, bold=bold
        )

        font, used_size, lines = _fit_translated_text(
            draw, translated, box_w, box_h, orig_size, bold=bold
        )

        # Draw lines top-aligned
        spacing = 2
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
            y_offset += lh + spacing

    return result
