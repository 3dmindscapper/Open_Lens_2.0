"""
ocr.py — Run dots.ocr on a PIL Image and return structured text blocks with bounding boxes.

Each block is a dict:
    {
        "text":   str,            # raw OCR text for this block
        "bbox":   (x1, y1, x2, y2),  # pixel coordinates on the image
        "lang":   str,            # detected language code, e.g. "zh", "fr"
        "conf":   float,          # confidence 0.0–1.0 (if available)
    }
"""
import os
import re
import json
import math
import time
from typing import List, Dict, Any

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, AutoConfig

MODEL_ID = "rednote-hilab/dots.ocr"
CACHE_DIR = os.path.join("models", "dots_ocr")

# ── Performance tuning ────────────────────────────────────────────────────────
# Lower max_pixels = fewer image tokens = faster inference (at cost of detail).
# Default 1_003_520 matches Qwen2.5-VL-2B tier; original was 11_289_600.
OCR_MAX_PIXELS = 5_000_000 #directly controls impact of table and ocr acuracy on runtime; 5M is a good balance for our test document, but may need adjustment for other documents.

_model = None
_processor = None

# dots.ocr prompt for full layout extraction with bboxes + text (from official prompts)
LAYOUT_PROMPT = (
    "Please output the layout information from the PDF image, including each layout "
    "element's bbox, its category, and the corresponding text content within the bbox.\n\n"
    "1. Bbox format: [x1, y1, x2, y2]\n\n"
    "2. Layout Categories: The possible categories are ['Caption', 'Footnote', 'Formula', "
    "'List-item', 'Page-footer', 'Page-header', 'Picture', 'Section-header', 'Table', 'Text', 'Title'].\n\n"
    "3. Text Extraction & Formatting Rules:\n"
    "    - Picture: For the 'Picture' category, the text field should be omitted.\n"
    "    - Formula: Format its text as LaTeX.\n"
    "    - Table: Format its text as HTML.\n"
    "    - All Others (Text, Title, etc.): Format their text as Markdown.\n\n"
    "4. Constraints:\n"
    "    - The output text must be the original text from the image, with no translation.\n"
    "    - All layout elements must be sorted according to human reading order.\n\n"
    "5. Final Output: The entire output must be a single JSON object."
)


def _get_local_model_path():
    """Resolve the local HF cache snapshot directory for the dots.ocr model."""
    refs_file = os.path.join(
        CACHE_DIR, "models--rednote-hilab--dots.ocr", "refs", "main"
    )
    if os.path.isfile(refs_file):
        with open(refs_file) as f:
            commit_hash = f.read().strip()
        snapshot_dir = os.path.join(
            CACHE_DIR, "models--rednote-hilab--dots.ocr", "snapshots", commit_hash
        )
        if os.path.isdir(snapshot_dir):
            return snapshot_dir
    return None


def _load_model():
    global _model, _processor
    if _model is not None:
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[OCR] Loading dots.ocr on {device}...")

    local_path = _get_local_model_path()
    model_src = local_path or MODEL_ID
    model_kwargs = {"trust_remote_code": True}
    if not local_path:
        model_kwargs["cache_dir"] = CACHE_DIR

    # Load processor from LOCAL snapshot path so the custom DotsVLProcessor
    # (registered in configuration_dots.py) is used.  DotsVLProcessor sets
    # image_token="<|imgpad|>" (id 151665) which matches the model's
    # config.image_token_id.  Loading from the HF hub ID would give the base
    # Qwen2_5_VLProcessor with image_token="<|image_pad|>" (id 151655) causing
    # a fatal token-ID mismatch.
    _processor = AutoProcessor.from_pretrained(model_src, **model_kwargs)
    print(f"[OCR] Processor: {type(_processor).__name__}, "
          f"image_token={getattr(_processor, 'image_token', '?')}")

    # ── Override image resolution cap for speed ──────────────────────────
    if hasattr(_processor, 'image_processor'):
        old_max = getattr(_processor.image_processor, 'max_pixels', None)
        _processor.image_processor.max_pixels = OCR_MAX_PIXELS
        _processor.image_processor.min_pixels = 3136
        print(f"[OCR] max_pixels: {old_max} → {OCR_MAX_PIXELS}")

    # ── Load config and set SDPA attention for the vision encoder ────────
    config = AutoConfig.from_pretrained(model_src, **model_kwargs)
    if hasattr(config, 'vision_config'):
        config.vision_config.attn_implementation = "sdpa"

    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    _model = AutoModelForCausalLM.from_pretrained(
        model_src,
        config=config,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation="sdpa",   # SDPA for LLM layers (PyTorch native)
        **model_kwargs,
    ).eval()

    print(f"[OCR] Model loaded (attn=sdpa, dtype={dtype}).")


def _smart_resize(height: int, width: int, factor: int = 28,
                  min_pixels: int = 3136, max_pixels: int = OCR_MAX_PIXELS):
    """Compute resized dimensions matching the Qwen2.5-VL processor's smart_resize."""
    h_bar = max(factor, round(height / factor) * factor)
    w_bar = max(factor, round(width / factor) * factor)

    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(max_pixels / (h_bar * w_bar))
        h_bar = max(factor, math.floor(h_bar * beta / factor) * factor)
        w_bar = max(factor, math.floor(w_bar * beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h_bar * w_bar))
        h_bar = max(factor, math.ceil(h_bar * beta / factor) * factor)
        w_bar = max(factor, math.ceil(w_bar * beta / factor) * factor)

    return h_bar, w_bar


def _parse_ocr_output(raw_output: str, img_w: int, img_h: int) -> List[Dict[str, Any]]:
    """
    Parse the JSON output from dots.ocr into a list of block dicts.

    The model outputs JSON like:
        [{"bbox": [x1, y1, x2, y2], "category": "Text", "text": "..."}]

    Coordinates are in the resized input image's pixel space;
    we scale them back to the original image dimensions.
    """
    blocks: List[Dict[str, Any]] = []

    # Strip markdown code fences if present
    cleaned = raw_output.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```$', '', cleaned)
    cleaned = cleaned.strip()

    # Compute the resized dimensions the processor used
    input_h, input_w = _smart_resize(img_h, img_w)
    scale_x = img_w / input_w
    scale_y = img_h / input_h

    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            data = data.get("layout", data.get("cells", [data]))
        if not isinstance(data, list):
            data = [data]

        # First pass: collect all items with scaled bboxes, including Pictures
        all_items = []
        for item in data:
            bbox = item.get("bbox", [])
            category = item.get("category", "Text")
            text = item.get("text", "").strip()

            if len(bbox) < 4:
                continue

            # Scale from resized input space to original image space
            x1 = max(0, min(int(float(bbox[0]) * scale_x), img_w))
            y1 = max(0, min(int(float(bbox[1]) * scale_y), img_h))
            x2 = max(0, min(int(float(bbox[2]) * scale_x), img_w))
            y2 = max(0, min(int(float(bbox[3]) * scale_y), img_h))

            if x2 <= x1 or y2 <= y1:
                continue

            all_items.append({
                "text": text,
                "bbox": (x1, y1, x2, y2),
                "category": category,
            })

        # Collect Picture bboxes for overlap detection
        picture_bboxes = [
            it["bbox"] for it in all_items if it["category"] == "Picture"
        ]

        for it in all_items:
            category = it["category"]
            text = it["text"]
            scaled_bbox = it["bbox"]

            # Skip Picture blocks and empty text
            if category == "Picture" or not text:
                continue

            # Skip text blocks that overlap significantly with a Picture bbox
            # (these are typically logo text inside a logo graphic)
            if _overlaps_picture(scaled_bbox, picture_bboxes, threshold=0.4):
                continue

            blocks.append({
                "text": text,
                "bbox": scaled_bbox,
                "category": category,
                "lang": _detect_lang_hint(text),
                "conf": 1.0,
            })
    except (json.JSONDecodeError, TypeError) as e:
        print(f"[OCR] Warning: Failed to parse JSON output: {e}")
        print(f"[OCR] Raw output (first 500 chars): {raw_output[:500]}")

    return blocks


def _overlaps_picture(bbox, picture_bboxes, threshold=0.4):
    """Return True if bbox overlaps with any picture bbox by more than threshold."""
    x1, y1, x2, y2 = bbox
    area = max(1, (x2 - x1) * (y2 - y1))
    for px1, py1, px2, py2 in picture_bboxes:
        ix1 = max(x1, px1)
        iy1 = max(y1, py1)
        ix2 = min(x2, px2)
        iy2 = min(y2, py2)
        if ix1 < ix2 and iy1 < iy2:
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / area >= threshold:
                return True
    return False


def _detect_lang_hint(text: str) -> str:
    """Lightweight heuristic language hint based on character ranges."""
    for ch in text:
        cp = ord(ch)
        if 0x4E00 <= cp <= 0x9FFF:
            return "zh"
        if 0x3040 <= cp <= 0x30FF:
            return "ja"
        if 0xAC00 <= cp <= 0xD7AF:
            return "ko"
        if 0x0600 <= cp <= 0x06FF:
            return "ar"
        if 0x0400 <= cp <= 0x04FF:
            return "ru"

    # Latin-script language detection: two-pass approach.
    # Pass 1: check for uniquely distinguishing characters.
    # Pass 2: fall back to common stop-word counting for ambiguous cases.
    lower = text.lower()

    # Characters/markers unique (or near-unique) to one language
    if "·" in lower:                        # ela geminada — Catalan only
        return "ca"
    if "ñ" in lower or "¡" in lower or "¿" in lower:
        return "es"
    if "ß" in lower:
        return "de"
    if "ã" in lower or "õ" in lower:
        return "pt"
    if "œ" in lower or "æ" in lower:
        return "fr"

    # ò is common in Catalan and Italian but NOT French
    has_o_grave = "ò" in lower

    # Accented-character counting for languages that share accents
    catalan_chars = set("àèòíúçïü")
    french_chars = set("éèêëàâçùûüôîï")
    italian_chars = set("àèéìòù")
    german_chars = set("äöü")

    ca = sum(1 for c in lower if c in catalan_chars)
    fr = sum(1 for c in lower if c in french_chars)
    it = sum(1 for c in lower if c in italian_chars)
    de = sum(1 for c in lower if c in german_chars)

    # Stop-word boosting to break ties between Catalan, French, Italian
    words = set(re.findall(r'\b\w+\b', lower))
    ca_stops = {"del", "els", "les", "amb", "per", "què", "és", "dels",
                "ses", "les", "seva", "seu", "als", "pel", "uns"}
    fr_stops = {"les", "des", "une", "est", "sont", "dans", "avec",
                "pour", "sur", "aux", "qui", "que", "ont", "ces"}
    it_stops = {"gli", "delle", "degli", "della", "nella", "nelle",
                "sono", "alla", "alle", "anche", "questo", "questa"}

    ca += sum(3 for w in words if w in ca_stops)
    fr += sum(3 for w in words if w in fr_stops)
    it += sum(3 for w in words if w in it_stops)

    # ò presence strongly favours Catalan or Italian over French
    if has_o_grave:
        ca += 2
        it += 2

    best = max(ca, fr, it, de)
    if best > 0:
        if ca == best:
            return "ca"
        if fr == best:
            return "fr"
        if it == best:
            return "it"
        if de == best:
            return "de"

    # Latin script without distinctive accents — default to unknown
    return "unknown"


def run_ocr(image: Image.Image) -> List[Dict[str, Any]]:
    """Run dots.ocr on a PIL Image and return a list of text blocks with bounding boxes."""
    _load_model()
    from qwen_vl_utils import process_vision_info

    img_w, img_h = image.size

    # ── Build messages in the Qwen2-VL multimodal format ─────────────────
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": LAYOUT_PROMPT},
            ],
        }
    ]

    # ── Processor pipeline (matches official demo_hf.py) ────────────────
    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(_model.device)

    # Strip keys the model's forward() does not accept (e.g. mm_token_type_ids)
    valid_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    for key in list(inputs.keys()):
        if key not in valid_keys:
            del inputs[key]

    imgpad_count = (inputs["input_ids"] == _model.config.image_token_id).sum().item()
    print(f"[OCR] Input: {inputs['input_ids'].shape[1]} tokens, {imgpad_count} imgpad")

    # ── Generate ─────────────────────────────────────────────────────────
    t0 = time.perf_counter()
    with torch.inference_mode():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=False,
        )
    gen_time = time.perf_counter() - t0

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    raw = _processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    print(f"[OCR] Generated {len(generated_ids_trimmed[0])} tokens in {gen_time:.1f}s")
    print(f"[OCR] Raw output (first 500 chars): {raw[:500]}")

    blocks = _parse_ocr_output(raw, img_w, img_h)
    print(f"[OCR] Found {len(blocks)} text block(s).")
    return blocks
