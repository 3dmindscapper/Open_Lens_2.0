"""
translate_m2m.py — Translation backend using Facebook's M2M-100 (MIT license).

Supports 100 languages, any-to-any (9,900 directions) in a single model.
Two model sizes available:
  - 418M (~1.7 GB) — faster, lighter, good for CPU or limited VRAM
  - 1.2B (~4.9 GB) — higher translation quality, needs more memory
"""
import os
import re
from typing import Optional

import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# ── Model configuration ──────────────────────────────────────────────────────
MODELS = {
    "m2m100_418m": "facebook/m2m100_418M",
    "m2m100_1.2b": "facebook/m2m100_1.2B",
}
CACHE_DIR = os.path.join("models", "m2m100")

# M2M-100 language codes (ISO 639-1 where possible)
# Both 418M and 1.2B support the same 100 languages.
M2M_LANGS = {
    "af", "am", "ar", "ast", "az", "ba", "be", "bg", "bn", "br", "bs",
    "ca", "ceb", "cs", "cy", "da", "de", "el", "en", "es", "et", "fa",
    "ff", "fi", "fr", "fy", "ga", "gd", "gl", "gu", "ha", "he", "hi",
    "hr", "ht", "hu", "hy", "id", "ig", "ilo", "is", "it", "ja", "jv",
    "ka", "kk", "km", "kn", "ko", "lb", "lg", "ln", "lo", "lt", "lv",
    "mg", "mk", "ml", "mn", "mr", "ms", "my", "ne", "nl", "no", "ns",
    "oc", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "si", "sk",
    "sl", "so", "sq", "sr", "ss", "su", "sv", "sw", "ta", "th", "tl",
    "tn", "tr", "uk", "ur", "uz", "vi", "wo", "xh", "yi", "yo", "zh",
    "zu",
}

_model: Optional[M2M100ForConditionalGeneration] = None
_tokenizer: Optional[M2M100Tokenizer] = None
_active_model_key: Optional[str] = None


def set_model(key: str):
    """Switch to a different M2M-100 model size. Reloads on next translate call."""
    global _model, _tokenizer, _active_model_key
    key = key.lower()
    if key not in MODELS:
        raise ValueError(f"Unknown model: {key!r}. Choose from: {list(MODELS.keys())}")
    if key != _active_model_key:
        # Unload current model so _load_model picks up the new one
        _model = None
        _tokenizer = None
        _active_model_key = key
        print(f"[M2M-100] Model set to: {key} ({MODELS[key]})")


def get_model() -> str:
    return _active_model_key or "m2m100_418m"


def _load_model():
    """Load M2M-100 model and tokenizer (lazy, first call only)."""
    global _model, _tokenizer, _active_model_key
    if _model is not None:
        return

    if _active_model_key is None:
        _active_model_key = "m2m100_418m"

    model_id = MODELS[_active_model_key]
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"[M2M-100] Loading {model_id}...")

    _tokenizer = M2M100Tokenizer.from_pretrained(
        model_id, cache_dir=CACHE_DIR
    )
    _model = M2M100ForConditionalGeneration.from_pretrained(
        model_id, cache_dir=CACHE_DIR
    ).eval()

    # Move to GPU only if there's enough VRAM headroom after OCR model
    if torch.cuda.is_available():
        free_vram = torch.cuda.mem_get_info()[0] / (1024 ** 3)
        vram_threshold = 5.0 if _active_model_key == "m2m100_1.2b" else 2.5
        if free_vram > vram_threshold:
            _model = _model.to("cuda")
            print(f"[M2M-100] Loaded on CUDA ({free_vram:.1f} GB free)")
        else:
            print(f"[M2M-100] Loaded on CPU (only {free_vram:.1f} GB VRAM free)")
    else:
        print("[M2M-100] Loaded on CPU")


def supports_pair(src: str, tgt: str) -> bool:
    """Return True if M2M-100 supports the given language pair."""
    return src in M2M_LANGS and tgt in M2M_LANGS


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate a single string via M2M-100. Returns original on failure."""
    if not text.strip() or src_lang == tgt_lang:
        return text

    # Normalise codes
    src = src_lang.lower().split("-")[0].split("_")[0]
    tgt = tgt_lang.lower().split("-")[0].split("_")[0]

    if src == "unknown" or src not in M2M_LANGS or tgt not in M2M_LANGS:
        return text

    _load_model()

    try:
        _tokenizer.src_lang = src
        encoded = _tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        encoded = {k: v.to(_model.device) for k, v in encoded.items()}

        with torch.inference_mode():
            generated = _model.generate(
                **encoded,
                forced_bos_token_id=_tokenizer.get_lang_id(tgt),
                max_new_tokens=512,
            )
        result = _tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return result
    except Exception as e:
        print(f"[M2M-100] Translation error: {e}")
        return text


def translate_lines(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate multi-line text, preserving line structure.

    Short lines are grouped for better context. Numeric-only lines are preserved.
    """
    if not text.strip() or src_lang == tgt_lang:
        return text

    lines = text.split("\n")
    result_lines = []
    group = []

    def _flush():
        if not group:
            return
        joined = " ".join(group)
        translated = translate(joined, src_lang, tgt_lang)
        # Try to re-split to match original line count
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
            _flush()
            group = []
            result_lines.append("")
        elif _is_numeric(stripped) or len(stripped) > 120:
            _flush()
            group = []
            if _is_numeric(stripped):
                result_lines.append(stripped)
            else:
                result_lines.append(translate(stripped, src_lang, tgt_lang))
        else:
            group.append(stripped)
            if len(group) >= 3:
                _flush()
                group = []

    _flush()
    return "\n".join(result_lines)


def _is_numeric(text: str) -> bool:
    """True if the line is mostly numbers/dates/amounts."""
    alpha = sum(1 for c in text if c.isalpha())
    return alpha <= 2
