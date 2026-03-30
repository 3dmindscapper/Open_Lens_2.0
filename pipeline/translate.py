"""
translate.py — Translation via M2M-100 (MIT license, 100 languages, any-to-any).

Model sizes selectable from the UI:
  - m2m100_418m (default): ~1.7 GB, faster, good for CPU or limited VRAM
  - m2m100_1.2b: ~4.9 GB, higher quality, needs more memory
"""
import re
from typing import List, Dict, Any


def set_model(key: str):
    """Switch M2M-100 model size. Valid: 'm2m100_418m', 'm2m100_1.2b'."""
    from pipeline.translate_m2m import set_model as _set
    _set(key)


def get_model() -> str:
    from pipeline.translate_m2m import get_model as _get
    return _get()


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting before translation."""
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text)
    text = re.sub(r'_{3}(.+?)_{3}', r'\1', text)
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text)
    text = re.sub(r'_{2}(.+?)_{2}', r'\1', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    return text


def _strip_html(text: str) -> str:
    """Convert HTML table structure to plain text with line breaks.

    dots.mocr outputs Table blocks as HTML.  We convert row/cell boundaries
    to newlines so the text preserves structure instead of becoming a blob.
    """
    text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</th>', '\t', text, flags=re.IGNORECASE)
    text = re.sub(r'</td>', '\t', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\t+', '  ', text)
    text = re.sub(r'[^\S\n]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate a string via M2M-100. Returns original text on failure."""
    if not text.strip():
        return text
    if src_lang == tgt_lang:
        return text

    src = src_lang.lower().split("-")[0].split("_")[0]
    tgt = tgt_lang.lower().split("-")[0].split("_")[0]

    if src == "unknown":
        return text

    from pipeline.translate_m2m import supports_pair, translate, translate_lines

    if not supports_pair(src, tgt):
        return _strip_html(_strip_markdown(text))

    try:
        # HTML table: translate cell contents, keep structure
        if re.search(r'<tr[^>]*>', text, re.IGNORECASE):
            return _translate_html_table(text, src, tgt)

        clean = _strip_html(_strip_markdown(text))

        # Form-like: translate each line individually
        if _looks_like_form(clean):
            return _translate_form_lines(clean, src, tgt)

        # Regular text: group short lines for context
        return translate_lines(clean, src, tgt)

    except Exception as e:
        print(f"[Translate] Warning: {e}")
        return _strip_html(_strip_markdown(text))


def _translate_html_table(html: str, src: str, tgt: str) -> str:
    """Translate HTML table cell contents via M2M-100."""
    from pipeline.translate_m2m import translate

    def _translate_cell(match):
        tag = match.group(1)
        attrs = match.group(2)
        content = match.group(3)
        close = match.group(4)
        inner = re.sub(r'<[^>]+>', '', content).strip()
        if not inner or _is_numeric_line(inner):
            translated = inner
        else:
            translated = translate(inner, src, tgt)
        return f"<{tag}{attrs}>{translated}</{close}>"

    return re.sub(
        r'<(td|th)([^>]*)>(.*?)</(td|th)>',
        _translate_cell, html,
        flags=re.DOTALL | re.IGNORECASE,
    )


def _translate_form_lines(clean: str, src: str, tgt: str) -> str:
    """Translate form-like text line-by-line via M2M-100."""
    from pipeline.translate_m2m import translate

    lines = clean.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
        elif _is_numeric_line(stripped):
            result.append(stripped)
        else:
            result.append(translate(stripped, src, tgt))
    return "\n".join(result)


def _looks_like_form(text: str) -> bool:
    """Detect alternating label/value structure (>=4 lines with short/data pattern)."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return False

    pairs = 0
    i = 0
    while i < len(lines) - 1:
        label = lines[i]
        value = lines[i + 1]
        if (len(label) < 50 and not _is_numeric_line(label)
                and (_is_numeric_line(value) or _is_data_value(value))):
            pairs += 1
            i += 2
        else:
            i += 1
    return pairs >= 3


def _is_data_value(text: str) -> bool:
    """Heuristic: looks like a data value rather than a label."""
    stripped = text.strip()
    if re.match(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', stripped):
        return True
    digits = sum(1 for c in stripped if c.isdigit())
    if digits >= 2 and digits / max(len(stripped), 1) > 0.15:
        return True
    words = stripped.split()
    if words and words[0][0:1].isupper() and len(words) <= 8:
        return True
    if len(stripped) <= 20:
        return True
    return False


def _is_numeric_line(text: str) -> bool:
    """True if the line is mostly numbers/dates/amounts (shouldn't be translated)."""
    alpha = sum(1 for c in text if c.isalpha())
    return alpha <= 2


def translate_blocks(
    blocks: List[Dict[str, Any]],
    tgt_lang: str = "en",
) -> List[Dict[str, Any]]:
    """
    Translate all OCR blocks in-place (adds 'translated' key).
    Returns the same list with translations added.
    """
    lang_counts: Dict[str, int] = {}
    for block in blocks:
        lang = block.get("lang", "unknown")
        if lang != "unknown":
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    majority_lang = max(lang_counts, key=lang_counts.get) if lang_counts else "unknown"

    for block in blocks:
        src = block.get("lang", "unknown")
        if src == "unknown" and majority_lang != "unknown":
            src = majority_lang
            block["lang"] = majority_lang
        block["translated"] = translate_text(block["text"], src, tgt_lang)
    return blocks
