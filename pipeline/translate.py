"""
translate.py — Local translation using Argos Translate.

Language packs are downloaded automatically on first use.
No internet connection required after initial pack download.
"""
import re
from typing import List, Dict, Any
import argostranslate.package
import argostranslate.translate

# Cache installed language pairs to avoid repeated lookups
_installed_pairs: set = set()
_initialized = False


def _init():
    global _initialized
    if _initialized:
        return
    argostranslate.package.update_package_index()
    _initialized = True


def _ensure_lang_pack(src: str, tgt: str):
    """Download and install the language pack if not already present."""
    pair_key = f"{src}->{tgt}"
    if pair_key in _installed_pairs:
        return

    installed = argostranslate.translate.get_installed_languages()
    installed_codes = {lang.code for lang in installed}

    if src in installed_codes and tgt in installed_codes:
        _installed_pairs.add(pair_key)
        return

    print(f"[Translate] Downloading language pack: {src} → {tgt}...")
    _init()
    available = argostranslate.package.get_available_packages()
    pkg = next(
        (p for p in available if p.from_code == src and p.to_code == tgt),
        None,
    )
    if pkg is None:
        # Try via English as pivot (src→en→tgt)
        raise ValueError(
            f"No direct Argos language pack for {src}→{tgt}. "
            f"Try installing manually: argospm install translate-{src}_{tgt}"
        )
    argostranslate.package.install_from_path(pkg.download())
    _installed_pairs.add(pair_key)
    print(f"[Translate] Pack installed: {src} → {tgt}")


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting before translation so Argos sees clean text."""
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\1', text)
    text = re.sub(r'_{3}(.+?)_{3}', r'\1', text)
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\1', text)
    text = re.sub(r'_{2}(.+?)_{2}', r'\1', text)
    text = re.sub(r'(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)', r'\1', text)
    return text


def _strip_html(text: str) -> str:
    """Convert HTML table structure to plain text with line breaks.

    dots.ocr outputs Table blocks as HTML.  We convert row/cell boundaries
    to newlines so the text preserves structure instead of becoming a blob.
    """
    # Replace row and cell boundaries with newlines
    text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</th>', '\t', text, flags=re.IGNORECASE)
    text = re.sub(r'</td>', '\t', text, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Collapse tabs to single space (column separator)
    text = re.sub(r'\t+', '  ', text)
    # Collapse multiple spaces (but keep newlines)
    text = re.sub(r'[^\S\n]+', ' ', text)
    # Remove blank lines
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()


def translate_text(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate a string, preserving line breaks. Returns original text on failure."""
    if not text.strip():
        return text
    if src_lang == tgt_lang:
        return text

    # Normalise common variants
    src = src_lang.lower().split("-")[0].split("_")[0]
    tgt = tgt_lang.lower().split("-")[0].split("_")[0]

    # Argos doesn't have an "unknown" pack — fall back to trying common langs
    if src == "unknown":
        return text

    try:
        _ensure_lang_pack(src, tgt)
        installed = argostranslate.translate.get_installed_languages()
        src_obj = next((l for l in installed if l.code == src), None)
        tgt_obj = next((l for l in installed if l.code == tgt), None)
        if src_obj is None or tgt_obj is None:
            return _strip_html(_strip_markdown(text))
        translation = src_obj.get_translation(tgt_obj)
        if translation is None:
            return _strip_html(_strip_markdown(text))

        # ── HTML table: translate cell contents in-place, keep structure ───
        if re.search(r'<tr[^>]*>', text, re.IGNORECASE):
            return _translate_html_table(text, translation)

        # ── Normal text: strip markdown/HTML ────────────────────────────
        clean = _strip_html(_strip_markdown(text))

        # ── Form-like block: alternating label/value lines ────────────
        #    Translate each line individually to preserve structure
        if _looks_like_form(clean):
            return _translate_form_lines(clean, translation)

        # ── Regular text: group short lines for better context ────────
        return _translate_lines(clean, translation)

    except Exception as e:
        print(f"[Translate] Warning: {e}")
        return _strip_html(_strip_markdown(text))


def _translate_html_table(html: str, translation) -> str:
    """Translate cell contents inside HTML table, preserving table structure."""
    def _translate_cell(match):
        tag = match.group(1)       # td or th
        attrs = match.group(2)     # any attributes
        content = match.group(3)   # cell text
        close = match.group(4)     # closing tag

        # Clean up content
        inner = re.sub(r'<[^>]+>', '', content).strip()
        if not inner or _is_numeric_line(inner):
            translated = inner
        else:
            translated = translation.translate(inner)
        return f"<{tag}{attrs}>{translated}</{close}>"

    result = re.sub(
        r'<(td|th)([^>]*)>(.*?)</(td|th)>',
        _translate_cell,
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return result


def _translate_lines(clean: str, translation) -> str:
    """Translate text line-by-line with short-line grouping for context."""
    lines = clean.split("\n")
    result_lines: list = []
    group: list = []

    def _flush_group():
        if not group:
            return
        joined = " ".join(group)
        translated = translation.translate(joined)
        # Try to re-split the translation to match the original count.
        parts = translated.split(". ")
        if len(parts) == len(group):
            for k, p in enumerate(parts):
                suffix = "." if k < len(parts) - 1 and not p.endswith(".") else ""
                result_lines.append(p + suffix)
        else:
            result_lines.append(translated)

    for line in lines:
        stripped = line.strip()
        if not stripped:
            _flush_group()
            group = []
            result_lines.append("")
        elif _is_numeric_line(stripped) or len(stripped) > 60:
            _flush_group()
            group = []
            if _is_numeric_line(stripped):
                result_lines.append(stripped)
            else:
                result_lines.append(translation.translate(stripped))
        else:
            group.append(stripped)
            if len(group) >= 3:
                _flush_group()
                group = []

    _flush_group()
    return "\n".join(result_lines)


def _looks_like_form(text: str) -> bool:
    """Detect alternating label/value structure (>=4 lines with short/data pattern).

    Form blocks alternate between short label lines and data-value lines
    (dates, numbers, addresses, names).  If we see at least 3 such pairs,
    treat the whole block as a form so we translate line-by-line.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        return False

    pairs = 0
    i = 0
    while i < len(lines) - 1:
        label = lines[i]
        value = lines[i + 1]
        # Label: short-ish text line (not a number/date)
        # Value: a data-like line (number, date, name, address, etc.)
        if (len(label) < 50 and not _is_numeric_line(label)
                and (_is_numeric_line(value) or _is_data_value(value))):
            pairs += 1
            i += 2
        else:
            i += 1
    return pairs >= 3


def _is_data_value(text: str) -> bool:
    """Heuristic: looks like a data value rather than a label.

    Matches dates, amounts, addresses (with numbers), proper nouns,
    and very short values (single word / code).
    """
    stripped = text.strip()
    # Dates: 10/04/2008, 27/06/2024
    if re.match(r'^\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4}', stripped):
        return True
    # Contains digits mixed with text (addresses, codes)
    digits = sum(1 for c in stripped if c.isdigit())
    if digits >= 2 and digits / max(len(stripped), 1) > 0.15:
        return True
    # Starts with uppercase word (proper noun / name)
    words = stripped.split()
    if words and words[0][0:1].isupper() and len(words) <= 8:
        return True
    # Very short (single code or value)
    if len(stripped) <= 20:
        return True
    return False


def _translate_form_lines(clean: str, translation) -> str:
    """Translate each line individually to preserve form label/value structure."""
    lines = clean.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            result.append("")
        elif _is_numeric_line(stripped):
            result.append(stripped)
        else:
            result.append(translation.translate(stripped))
    return "\n".join(result)


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
    # Infer majority language from blocks with a detected language,
    # then apply it to any "unknown" blocks (e.g. text without accented chars).
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
