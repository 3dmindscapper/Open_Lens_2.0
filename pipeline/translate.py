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

    # Strip markdown and HTML before translating — cleaner input for engine
    clean = _strip_html(_strip_markdown(text))

    try:
        _ensure_lang_pack(src, tgt)
        installed = argostranslate.translate.get_installed_languages()
        src_obj = next((l for l in installed if l.code == src), None)
        tgt_obj = next((l for l in installed if l.code == tgt), None)
        if src_obj is None or tgt_obj is None:
            return clean
        translation = src_obj.get_translation(tgt_obj)
        if translation is None:
            return clean
        # Translate line-by-line to preserve structural line breaks
        lines = clean.split("\n")
        translated_lines = []
        for line in lines:
            if line.strip():
                translated_lines.append(translation.translate(line))
            else:
                translated_lines.append("")
        return "\n".join(translated_lines)
    except Exception as e:
        print(f"[Translate] Warning: {e}")
        return clean


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
