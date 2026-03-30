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
    # HTML table structure → newlines (dots.mocr emits HTML for Table blocks)
    text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</th>', '\t', text, flags=re.IGNORECASE)
    text = re.sub(r'</td>', '\t', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\t+', '  ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


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

def _is_data_like(text: str) -> bool:
    """Return True if text looks like a data value rather than a field label.

    Labels are descriptive noun phrases (e.g. "Nationalité", "Date de naissance").
    Data values contain digits, dates, or look like proper names.
    """
    if re.search(r'\d', text):
        return True
    words = text.split()
    if len(words) >= 2:
        upper = sum(1 for w in words if len(w) > 1 and w[0].isupper())
        if upper >= 2:
            return True
    return False


def _detect_form_entries(orig_text: str, trans_text: str):
    """
    Detect alternating label/value lines that represent a two-column form.

    Handles blocks with an odd line count by finding the best standalone line
    to exclude.  The "best" standalone is the one whose removal produces pairs
    where the longest label is minimised — correct pairings have short noun-
    phrase labels, wrong pairings put long data values in the label column.

    Returns a list of entry tuples:
        ("pair", orig_label, orig_value, trans_label, trans_value)
        ("full", orig_text, trans_text)
    Returns None if the block doesn't look like a form.
    """
    orig_clean = _strip_markdown(orig_text)
    trans_clean = _strip_markdown(trans_text)

    orig_lines = [l for l in orig_clean.split("\n") if l.strip()]
    trans_lines = [l for l in trans_clean.split("\n") if l.strip()]

    n = len(orig_lines)
    if n != len(trans_lines) or n < 4:
        return None

    standalone_idx = -1  # -1 means none (even count)

    if n % 2 != 0:
        if n - 1 < 4:
            return None

        # Try every candidate position; pick the one that produces the
        # most sensible pairing.  Scoring: (data_like_count, max_label_len).
        # data_like_count: labels shouldn't look like data values (contain
        # digits, dates, or proper-name patterns).  max_label_len: shorter
        # labels indicate a better split.
        best_k = 0
        best_score = (float('inf'), float('inf'), float('inf'))
        for k in range(n):
            remaining = [orig_lines[i] for i in range(n) if i != k]
            labels = [remaining[i] for i in range(0, len(remaining), 2)]
            values = [remaining[i] for i in range(1, len(remaining), 2)]
            data_count = sum(1 for l in labels if _is_data_like(l))
            # More data-like values = better pairing; use negative so
            # minimisation favours higher value-quality.
            value_quality = -sum(1 for v in values if _is_data_like(v))
            max_ll = max(len(l) for l in labels)
            score = (data_count, value_quality, max_ll)
            if score < best_score:
                best_score = score
                best_k = k
        standalone_idx = best_k

    # Build pair map: label_index → value_index
    indices = [i for i in range(n) if i != standalone_idx]
    pair_map: Dict[int, int] = {}
    value_set: set = set()
    for k in range(0, len(indices), 2):
        pair_map[indices[k]] = indices[k + 1]
        value_set.add(indices[k + 1])

    # Walk original indices in order to preserve document flow
    entries = []
    for i in range(n):
        if i == standalone_idx:
            entries.append(("full", orig_lines[i], trans_lines[i]))
        elif i in pair_map:
            j = pair_map[i]
            entries.append(("pair",
                            orig_lines[i], orig_lines[j],
                            trans_lines[i], trans_lines[j]))
        elif i in value_set:
            pass  # already consumed by its label
        else:
            entries.append(("full", orig_lines[i], trans_lines[i]))

    pair_count = sum(1 for e in entries if e[0] == "pair")
    if pair_count < 2:
        return None

    return entries


def _find_form_font_size(draw: ImageDraw.ImageDraw,
                         entries: list,
                         box_w: int, box_h: int, bold: bool = False,
                         which: str = "orig",
                         min_size: int = 6, max_size: int = 120) -> int:
    """
    Binary-search the largest font size where form entries (pairs + full-width)
    fit inside (box_w, box_h).  *which* selects original text ("orig") or
    translated text ("trans") for sizing.
    """
    lo, hi = min_size, max_size
    best = min_size

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(mid, bold)

        ref = draw.textbbox((0, 0), "Ay", font=font)
        line_h = (ref[3] - ref[1]) + 2

        # Compute label column width from pair entries
        gap = max(int(mid * 0.5), 4)
        max_lw = 0
        for entry in entries:
            if entry[0] == "pair":
                lbl = entry[1] if which == "orig" else entry[3]
                if lbl.strip():
                    lb = draw.textbbox((0, 0), lbl, font=font)
                    max_lw = max(max_lw, lb[2] - lb[0])

        tab_stop = max_lw + gap
        if tab_stop > box_w * 0.55:
            hi = mid - 1
            continue

        value_space = max(1, box_w - tab_stop)

        # Total height (account for both label and value wrapping)
        label_col_w = max(1, tab_stop - gap)
        total_h = 0
        for entry in entries:
            if entry[0] == "pair":
                lbl = entry[1] if which == "orig" else entry[3]
                val = entry[2] if which == "orig" else entry[4]
                lbl_rows = max(1, len(_wrap_line(draw, lbl, font, label_col_w))) if lbl.strip() else 1
                val_rows = max(1, len(_wrap_line(draw, val, font, value_space))) if val.strip() else 1
                total_h += max(lbl_rows, val_rows) * line_h
            else:  # "full"
                txt = entry[1] if which == "orig" else entry[2]
                full_lines = _wrap_line(draw, txt, font, box_w)
                total_h += max(1, len(full_lines)) * line_h
        total_h -= 2  # remove last spacing

        if total_h <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def _render_form(draw: ImageDraw.ImageDraw,
                 entries: list,
                 x: int, y: int, box_w: int,
                 font_size: int, bold: bool,
                 color: Tuple[int, int, int]):
    """Render form entries: pairs side-by-side, full-width lines spanning box."""
    font = _get_font(font_size, bold)
    spacing = 2

    # Tab stop: longest translated label + gap
    max_label_w = 0
    for entry in entries:
        if entry[0] == "pair":
            label = entry[3]  # translated label
            lb = draw.textbbox((0, 0), label, font=font)
            max_label_w = max(max_label_w, lb[2] - lb[0])

    gap = max(int(font_size * 0.5), 4)
    tab_stop = min(max_label_w + gap, int(box_w * 0.55))
    value_space = max(1, box_w - tab_stop)

    y_off = y
    for entry in entries:
        if entry[0] == "pair":
            _, _, _, label, value = entry
            ref = draw.textbbox((0, 0), label or " ", font=font)
            line_h = ref[3] - ref[1]

            # Wrap label if it exceeds the tab stop area
            if label.strip():
                label_lines = _wrap_line(draw, label, font, tab_stop - gap)
                ly = y_off
                for ll in label_lines:
                    if ll.strip():
                        draw.text((x, ly), ll, font=font, fill=color)
                    llb = draw.textbbox((0, 0), ll or " ", font=font)
                    ly += (llb[3] - llb[1]) + spacing
                label_total = ly - y_off
            else:
                label_total = line_h + spacing

            if value.strip():
                val_lines = _wrap_line(draw, value, font, value_space)
                vy = y_off
                for vl in val_lines:
                    if vl.strip():
                        draw.text((x + tab_stop, vy), vl, font=font, fill=color)
                    vlb = draw.textbbox((0, 0), vl or " ", font=font)
                    vy += (vlb[3] - vlb[1]) + spacing
                val_total = vy - y_off
                y_off += max(label_total, val_total)
            else:
                y_off += max(label_total, line_h + spacing)
        else:
            # Full-width standalone line
            _, _, text = entry
            if text.strip():
                full_lines = _wrap_line(draw, text, font, box_w)
                for fl in full_lines:
                    if fl.strip():
                        draw.text((x, y_off), fl, font=font, fill=color)
                    flb = draw.textbbox((0, 0), fl or " ", font=font)
                    y_off += (flb[3] - flb[1]) + spacing


# ── HTML table parsing and rendering ──────────────────────────────────────────

def _parse_html_table(html: str) -> List[List[str]]:
    """Parse an HTML table into a list of rows, each a list of cell strings.

    Handles malformed OCR HTML where <thead> may contain <th> cells
    without a wrapping <tr> tag.
    """
    rows: List[List[str]] = []

    # 1) Extract header row from <thead> (may lack <tr> wrapper)
    thead_m = re.search(r'<thead[^>]*>(.*?)</thead>', html,
                        re.DOTALL | re.IGNORECASE)
    if thead_m:
        header_cells = re.findall(r'<th[^>]*>(.*?)</th>', thead_m.group(1),
                                  re.DOTALL | re.IGNORECASE)
        header = [re.sub(r'<[^>]+>', '', c).strip() for c in header_cells]
        if header:
            rows.append(header)

    # 2) Extract body rows from <tr>...</tr>
    #    Skip rows already captured from <thead>
    thead_span = (thead_m.start(), thead_m.end()) if thead_m else None
    for m in re.finditer(r'<tr[^>]*>(.*?)</tr>', html,
                         re.DOTALL | re.IGNORECASE):
        if thead_span and m.start() >= thead_span[0] and m.end() <= thead_span[1]:
            continue  # already captured as header
        row_html = m.group(1)
        cells = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', row_html,
                           re.DOTALL | re.IGNORECASE)
        row = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if row:
            rows.append(row)
    return rows


def _is_html_table(text: str) -> bool:
    """Check whether the block text contains HTML table markup."""
    return bool(re.search(r'<tr[^>]*>', text, re.IGNORECASE))


def _find_table_font_size(draw: ImageDraw.ImageDraw,
                          rows: List[List[str]],
                          col_widths: List[int],
                          box_h: int, bold: bool = False,
                          min_size: int = 6, max_size: int = 60) -> int:
    """Binary-search the largest font where all table rows fit vertically,
    accounting for cell text wrapping within assigned column widths."""
    lo, hi = min_size, max_size
    best = min_size

    while lo <= hi:
        mid = (lo + hi) // 2
        font = _get_font(mid, bold)
        ref = draw.textbbox((0, 0), "Ay", font=font)
        line_h = (ref[3] - ref[1]) + 2

        total_h = 0
        for row in rows:
            max_lines_in_row = 1
            for ci, cell in enumerate(row):
                cw = col_widths[ci] if ci < len(col_widths) else col_widths[-1]
                wrapped = _wrap_line(draw, cell, font, cw)
                max_lines_in_row = max(max_lines_in_row, len(wrapped))
            total_h += max_lines_in_row * line_h + 1

        if total_h <= box_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1

    return best


def _is_numeric_cell(text: str) -> bool:
    """True if cell content is numeric (amounts, percentages, etc.)."""
    cleaned = text.replace(',', '').replace('.', '').replace(' ', '')
    alpha = sum(1 for c in cleaned if c.isalpha())
    return alpha <= 1 and any(c.isdigit() for c in cleaned)


def _render_table(draw: ImageDraw.ImageDraw,
                  rows: List[List[str]],
                  x: int, y: int, box_w: int, box_h: int,
                  bold: bool,
                  color: Tuple[int, int, int]):
    """Render parsed HTML table rows as a grid within the bounding box."""
    if not rows:
        return

    # Normalise rows so every row has the same number of columns
    n_cols = max(len(r) for r in rows)
    if n_cols == 0:
        return
    for row in rows:
        while len(row) < n_cols:
            row.append("")

    # Distribute column widths proportionally based on content
    col_gap = 4
    available_w = box_w - col_gap * (n_cols - 1)

    # Measure max content width per column at a reference size
    ref_font = _get_font(10, bold)
    col_max_w = [0] * n_cols
    for row in rows:
        for ci, cell in enumerate(row):
            if ci < n_cols and cell.strip():
                cb = draw.textbbox((0, 0), cell, font=ref_font)
                col_max_w[ci] = max(col_max_w[ci], cb[2] - cb[0])

    # Give empty columns a minimum width so they don't collapse to 20px
    for ci in range(n_cols):
        if col_max_w[ci] == 0:
            col_max_w[ci] = 10  # minimal weight for empty cols

    total_content = sum(col_max_w) or 1
    col_widths = [max(20, int(available_w * (cw / total_content)))
                  for cw in col_max_w]

    # Find font size
    font_size = _find_table_font_size(draw, rows, col_widths, box_h, bold)
    font = _get_font(font_size, bold)
    ref = draw.textbbox((0, 0), "Ay", font=font)
    line_h = (ref[3] - ref[1]) + 2

    # Detect which columns are numeric (right-align them)
    numeric_col = [False] * n_cols
    for ci in range(n_cols):
        num_count = sum(1 for r in rows[1:] if ci < len(r)
                        and r[ci].strip() and _is_numeric_cell(r[ci]))
        total_count = sum(1 for r in rows[1:] if ci < len(r) and r[ci].strip())
        if total_count > 0 and num_count / total_count >= 0.5:
            numeric_col[ci] = True

    # Render rows
    y_off = y
    for ri, row in enumerate(rows):
        is_header = (ri == 0 and rows[0] != rows[-1])  # treat first row as header
        max_lines = 1
        row_cells: List[List[str]] = []
        for ci, cell in enumerate(row):
            cw = col_widths[ci] if ci < len(col_widths) else col_widths[-1]
            wrapped = _wrap_line(draw, cell, font, cw)
            row_cells.append(wrapped)
            max_lines = max(max_lines, len(wrapped))

        # Draw each cell
        x_off = x
        for ci, cell_lines in enumerate(row_cells):
            cw = col_widths[ci] if ci < len(col_widths) else col_widths[-1]
            for li, line in enumerate(cell_lines):
                if line.strip():
                    # Right-align numeric columns
                    if numeric_col[ci] and not is_header:
                        tb = draw.textbbox((0, 0), line.strip(), font=font)
                        text_w = tb[2] - tb[0]
                        tx = x_off + cw - text_w
                    else:
                        tx = x_off
                    cell_font = _get_font(font_size, True) if is_header else font
                    draw.text((tx, y_off + li * line_h), line.strip(),
                              font=cell_font, fill=color)
            x_off += cw + col_gap

        y_off += max_lines * line_h + 1


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

        # ── HTML table rendering ──────────────────────────────────────
        if _is_html_table(translated_raw) or _is_html_table(original_raw):
            # Use translated text if it has HTML, otherwise translate was
            # already stripped — parse from original and render translated rows
            html_src = translated_raw if _is_html_table(translated_raw) else original_raw
            rows = _parse_html_table(html_src)
            if rows and any(any(c.strip() for c in r) for r in rows):
                _render_table(draw, rows, x1 + padding, y1 + padding,
                              box_w, box_h, bold, color)
                continue

        # ── Form layout: detect alternating label/value pairs ─────────
        if not center:
            entries = _detect_form_entries(original_raw, translated_raw)
            if entries:
                stacked_size = _find_original_font_size(
                    draw, original_raw, box_w, box_h, bold)
                form_size = _find_form_font_size(
                    draw, entries, box_w, box_h, bold, which="orig")

                # Only use form layout if it gives a clearly larger font,
                # meaning the original must have had side-by-side layout.
                if form_size >= stacked_size * 1.3:
                    render_size = _find_form_font_size(
                        draw, entries, box_w, box_h, bold, which="trans")
                    render_size = max(render_size, int(form_size * 0.7))

                    _render_form(draw, entries,
                                 x1 + padding, y1 + padding,
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
