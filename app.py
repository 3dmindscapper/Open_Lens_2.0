"""
app.py — Gradio web UI for OpenLens 2.0.

Launch with: python app.py
Then open:   http://localhost:7860
"""
import base64
import io
import os
import re
import tempfile
import traceback
from pathlib import Path
from typing import Optional, Tuple, List

import gradio as gr
from PIL import Image

from pipeline.pdf_utils import load_document, images_to_pdf
from pipeline.ocr import run_ocr
from pipeline.translate import translate_blocks, set_model, get_model
from pipeline.inpaint import erase_text_blocks
from pipeline.renderer import render_translations
from pipeline.export import export_all_pages_json, export_all_pages_markdown

# ── Supported target languages (M2M-100: 100 languages, any-to-any) ──────────
TARGET_LANGUAGES = {
    "English":               "en",
    "Spanish":               "es",
    "French":                "fr",
    "German":                "de",
    "Portuguese":            "pt",
    "Italian":               "it",
    "Dutch":                 "nl",
    "Russian":               "ru",
    "Arabic":                "ar",
    "Japanese":              "ja",
    "Korean":                "ko",
    "Chinese":               "zh",
    "Turkish":               "tr",
    "Polish":                "pl",
    "Swedish":               "sv",
    "Afrikaans":             "af",
    "Amharic":               "am",
    "Asturian":              "ast",
    "Azerbaijani":           "az",
    "Bashkir":               "ba",
    "Belarusian":            "be",
    "Bulgarian":             "bg",
    "Bengali":               "bn",
    "Breton":                "br",
    "Bosnian":               "bs",
    "Catalan":               "ca",
    "Cebuano":               "ceb",
    "Czech":                 "cs",
    "Welsh":                 "cy",
    "Danish":                "da",
    "Greek":                 "el",
    "Estonian":              "et",
    "Persian":               "fa",
    "Fulah":                 "ff",
    "Finnish":               "fi",
    "Western Frisian":       "fy",
    "Irish":                 "ga",
    "Scottish Gaelic":       "gd",
    "Galician":              "gl",
    "Gujarati":              "gu",
    "Hausa":                 "ha",
    "Hebrew":                "he",
    "Hindi":                 "hi",
    "Croatian":              "hr",
    "Haitian Creole":        "ht",
    "Hungarian":             "hu",
    "Armenian":              "hy",
    "Indonesian":            "id",
    "Igbo":                  "ig",
    "Iloko":                 "ilo",
    "Icelandic":             "is",
    "Javanese":              "jv",
    "Georgian":              "ka",
    "Kazakh":                "kk",
    "Central Khmer":         "km",
    "Kannada":               "kn",
    "Luxembourgish":         "lb",
    "Ganda":                 "lg",
    "Lingala":               "ln",
    "Lao":                   "lo",
    "Lithuanian":            "lt",
    "Latvian":               "lv",
    "Malagasy":              "mg",
    "Macedonian":            "mk",
    "Malayalam":             "ml",
    "Mongolian":             "mn",
    "Marathi":               "mr",
    "Malay":                 "ms",
    "Burmese":               "my",
    "Nepali":                "ne",
    "Northern Sotho":        "ns",
    "Occitan":               "oc",
    "Oriya":                 "or",
    "Punjabi":               "pa",
    "Pashto":                "ps",
    "Romanian":              "ro",
    "Sindhi":                "sd",
    "Sinhala":               "si",
    "Slovak":                "sk",
    "Slovenian":             "sl",
    "Somali":                "so",
    "Albanian":              "sq",
    "Serbian":               "sr",
    "Swati":                 "ss",
    "Sundanese":             "su",
    "Swahili":               "sw",
    "Tamil":                 "ta",
    "Thai":                  "th",
    "Tagalog":               "tl",
    "Tswana":                "tn",
    "Ukrainian":             "uk",
    "Urdu":                  "ur",
    "Uzbek":                 "uz",
    "Vietnamese":            "vi",
    "Wolof":                 "wo",
    "Xhosa":                 "xh",
    "Yiddish":               "yi",
    "Yoruba":                "yo",
    "Zulu":                  "zu",
}

OUTPUT_DIR = Path(tempfile.gettempdir()) / "openlens2_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _image_to_data_uri(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image as a base64 data URI for embedding in HTML."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


def _format_table_overlay(html: str) -> str:
    """Convert an HTML table block into a clean overlay HTML table for copy/paste."""
    # Parse header from <thead> (may have bare <th> without <tr>)
    header_cells = []
    thead_m = re.search(r'<thead[^>]*>(.*?)</thead>', html,
                        re.DOTALL | re.IGNORECASE)
    if thead_m:
        header_cells = [re.sub(r'<[^>]+>', '', c).strip()
                        for c in re.findall(r'<th[^>]*>(.*?)</th>',
                                            thead_m.group(1),
                                            re.DOTALL | re.IGNORECASE)]

    # Parse body rows
    body_rows = []
    thead_span = (thead_m.start(), thead_m.end()) if thead_m else None
    for m in re.finditer(r'<tr[^>]*>(.*?)</tr>', html,
                         re.DOTALL | re.IGNORECASE):
        if thead_span and m.start() >= thead_span[0] and m.end() <= thead_span[1]:
            continue
        cells = re.findall(r'<(?:td|th)[^>]*>(.*?)</(?:td|th)>', m.group(1),
                           re.DOTALL | re.IGNORECASE)
        row = [re.sub(r'<[^>]+>', '', c).strip() for c in cells]
        if row:
            body_rows.append(row)

    # Build a simple HTML table for the overlay
    parts = ['<table style="width:100%;border-collapse:collapse;'
             'font-size:inherit;line-height:inherit;">']
    if header_cells:
        parts.append('<tr>')
        for c in header_cells:
            safe = c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(f'<th style="text-align:left;padding:0 2px;">'
                         f'{safe}</th>')
        parts.append('</tr>')
    for row in body_rows:
        parts.append('<tr>')
        for c in row:
            safe = c.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            parts.append(f'<td style="padding:0 2px;">{safe}</td>')
        parts.append('</tr>')
    parts.append('</table>')
    return ''.join(parts)


def _format_text_overlay(text: str) -> str:
    """Format a text block for the overlay, detecting form layout.

    For form blocks (alternating label/value pairs), renders as a
    two-column table so labels and values are visually separated.
    For regular text, uses simple line breaks.
    """
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    if len(lines) < 4:
        safe = (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace("\n", "<br>"))
        return safe

    # Detect form pattern: alternating label / data-value lines
    # More lenient — skip non-matching lines instead of breaking
    from pipeline.translate import _is_numeric_line
    pairs = []
    standalone = []
    i = 0
    while i < len(lines):
        if i < len(lines) - 1:
            label = lines[i]
            value = lines[i + 1]
            is_data = (_is_numeric_line(value)
                       or any(c.isdigit() for c in value)
                       or len(value) <= 45
                       or (len(label) < 45 and len(value) > len(label)))
            if len(label) < 60 and not _is_numeric_line(label) and is_data:
                pairs.append((label, value))
                i += 2
                continue
        # Standalone line (header, long text, etc.)
        standalone.append((i, lines[i]))
        i += 1

    if len(pairs) >= 2:
        # Render as label: value table
        parts = ['<table style="width:100%;border-collapse:collapse;'
                 'font-size:inherit;line-height:inherit;">']

        # Rebuild in original order
        i = 0
        pair_idx = 0
        standalone_idx = 0
        pair_positions = {}
        for pi, (label, value) in enumerate(pairs):
            # Find original position
            idx = lines.index(label)
            pair_positions[idx] = pi

        for li, line in enumerate(lines):
            sl = (line.replace("&", "&amp;").replace("<", "&lt;")
                  .replace(">", "&gt;"))
            if li in pair_positions:
                pi = pair_positions[li]
                lbl, val = pairs[pi]
                sv = (val.replace("&", "&amp;").replace("<", "&lt;")
                      .replace(">", "&gt;"))
                slbl = (lbl.replace("&", "&amp;").replace("<", "&lt;")
                        .replace(">", "&gt;"))
                parts.append(
                    f'<tr><td style="padding:1px 2px;'
                    f'white-space:nowrap;vertical-align:top;">{slbl}</td>'
                    f'<td style="padding:1px 2px 1px 8px;">{sv}</td></tr>'
                )
            elif li > 0 and (li - 1) in pair_positions:
                continue  # value line already included with its label
            else:
                parts.append(
                    f'<tr><td colspan="2" style="padding:1px 2px;">'
                    f'{sl}</td></tr>'
                )
        parts.append('</table>')
        return ''.join(parts)

    # Regular text — simple line breaks
    safe = (text.replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace("\n", "<br>"))
    return safe


def _build_text_overlay_html(
    image: Image.Image,
    blocks: List[dict],
    text_key: str = "translated",
) -> str:
    """Build HTML with the image as background and selectable text boxes
    positioned via CSS absolute positioning over each block's bbox."""
    img_w, img_h = image.size
    data_uri = _image_to_data_uri(image)

    divs = []
    for block in blocks:
        text = block.get(text_key, "").strip()
        bbox = block.get("bbox")
        if not text or not bbox:
            continue
        x1, y1, x2, y2 = bbox
        # Convert pixel positions to percentages for responsive layout
        left_pct = (x1 / img_w) * 100
        top_pct = (y1 / img_h) * 100
        w_pct = ((x2 - x1) / img_w) * 100
        h_pct = ((y2 - y1) / img_h) * 100
        # Format table blocks as an HTML table for clean copy/paste
        is_table = bool(re.search(r'<tr[^>]*>', text, re.IGNORECASE))
        if is_table:
            safe_text = _format_table_overlay(text)
        else:
            # Strip any stray HTML tags from non-table blocks
            text = re.sub(r'<[^>]+>', '', text).strip()
            if not text:
                continue
            safe_text = _format_text_overlay(text)
        divs.append(
            f'<div class="text-box" style="'
            f"left:{left_pct:.2f}%;top:{top_pct:.2f}%;"
            f"width:{w_pct:.2f}%;height:{h_pct:.2f}%;"
            f'">{safe_text}</div>'
        )

    return f"""
    <div class="doc-container" style="
        position:relative;display:inline-block;width:100%;
        max-width:{img_w}px;
    ">
        <img src="{data_uri}" style="width:100%;display:block;" />
        {"".join(divs)}
    </div>
    """


def _build_preview_html(
    original_pages: List[Image.Image],
    translated_pages: List[Image.Image],
    all_blocks: List[List[dict]],
) -> str:
    """Build full HTML for side-by-side original + translated preview
    with selectable text overlays on the translated side."""

    style = """
    <style>
    .preview-wrap { font-family: Arial, sans-serif; }
    .page-row {
        display: flex; gap: 12px; margin-bottom: 24px;
        align-items: flex-start;
    }
    .page-col {
        flex: 1; min-width: 0;
    }
    .page-label {
        font-weight: bold; font-size: 14px; margin-bottom: 6px;
        color: #555; text-align: center;
    }
    .doc-container { position: relative; display: inline-block; width: 100%; }
    .doc-container img { width: 100%; display: block; }
    .text-box {
        position: absolute; overflow: hidden;
        font-size: 10px; line-height: 1.2;
        color: rgba(0,0,0,0) !important; cursor: text;
        user-select: text; -webkit-user-select: text;
        z-index: 2;
        background: transparent;
        transition: background 0.15s, color 0.15s;
        border-radius: 2px;
    }
    .text-box *, .text-box table, .text-box th, .text-box td,
    .text-box tr, .text-box thead, .text-box tbody {
        color: rgba(0,0,0,0) !important;
        background: transparent !important;
        border: none !important;
        user-select: text; -webkit-user-select: text;
    }
    .text-box::selection,
    .text-box *::selection { background: rgba(0,120,215,0.4) !important; color: #000 !important; }
    .text-box:hover {
        background: rgba(200, 200, 200, 0.75);
        color: #000 !important;
    }
    .text-box:hover *, .text-box:hover table, .text-box:hover tr,
    .text-box:hover th, .text-box:hover td,
    .text-box:hover thead, .text-box:hover tbody {
        color: #000 !important;
    }
    </style>
    """

    pages_html = []
    for i, (orig, trans, blocks) in enumerate(
        zip(original_pages, translated_pages, all_blocks)
    ):
        orig_uri = _image_to_data_uri(orig)
        trans_overlay = _build_text_overlay_html(trans, blocks, "translated")
        orig_overlay = _build_text_overlay_html(orig, blocks, "text")

        pages_html.append(f"""
        <div class="page-row">
            <div class="page-col">
                <div class="page-label">Original — Page {i+1}</div>
                {orig_overlay}
            </div>
            <div class="page-col">
                <div class="page-label">Translated — Page {i+1}</div>
                {trans_overlay}
            </div>
        </div>
        """)

    return f'<div class="preview-wrap">{style}{"".join(pages_html)}</div>'


def process_document(
    file_obj,
    target_lang_name: str,
    model_name: str,
    export_json: bool,
    export_md: bool,
    progress=gr.Progress(track_tqdm=True),
):
    """
    Main pipeline function called by Gradio.

    Returns:
        - preview_html: HTML string with side-by-side selectable-text preview
        - output_file: path to the downloadable translated PDF
        - json_file: path to JSON export (or None)
        - md_file: path to Markdown export (or None)
        - status: status message string
    """
    if file_obj is None:
        return None, None, None, None, "⚠️ Please upload a file first."

    # Set translation model size
    _model_map = {
        "M2M-100 418M (Faster)": "m2m100_418m",
        "M2M-100 1.2B (Better Quality)": "m2m100_1.2b",
    }
    set_model(_model_map.get(model_name, "m2m100_418m"))

    tgt_code = TARGET_LANGUAGES.get(target_lang_name, "en")
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        # ── Step 1: Load document pages ───────────────────────────────────────
        progress(0.05, desc="Loading document...")
        pages = load_document(file_path)
        n = len(pages)
        status_lines = [
            f"📄 Loaded {n} page(s).",
            f"🔤 Translation model: M2M-100 {get_model().upper()}",
        ]

        translated_pages = []
        original_pages = []
        all_blocks = []

        for i, page_img in enumerate(pages):
            page_label = f"Page {i+1}/{n}"
            progress((i / n) * 0.85 + 0.05, desc=f"Processing {page_label}...")

            # ── Step 2: OCR ───────────────────────────────────────────────────
            blocks = run_ocr(page_img)
            if not blocks:
                translated_pages.append(page_img)
                original_pages.append(page_img)
                all_blocks.append([])
                status_lines.append(f"  {page_label}: no text detected, kept original.")
                continue

            detected_langs = {b.get("lang", "?") for b in blocks}
            status_lines.append(
                f"  {page_label}: {len(blocks)} block(s), "
                f"lang(s): {', '.join(detected_langs)}"
            )

            # ── Step 3: Translate ─────────────────────────────────────────────
            blocks = translate_blocks(blocks, tgt_lang=tgt_code)

            # ── Step 4: Inpaint (erase original text) ─────────────────────────
            inpainted = erase_text_blocks(page_img, blocks)

            # ── Step 5: Render translated text ────────────────────────────────
            rendered = render_translations(inpainted, page_img, blocks)

            translated_pages.append(rendered)
            original_pages.append(page_img)
            all_blocks.append(blocks)

        # ── Step 6: Export PDF ────────────────────────────────────────────────
        progress(0.90, desc="Saving output PDF...")
        stem = Path(file_path).stem
        out_name = stem + f"_translated_{tgt_code}.pdf"
        out_path = str(OUTPUT_DIR / out_name)
        images_to_pdf(translated_pages, out_path)

        # ── Step 7: Optional JSON/Markdown exports ────────────────────────────
        json_path = None
        md_path = None

        if export_json:
            json_name = stem + "_ocr_output.json"
            json_path = str(OUTPUT_DIR / json_name)
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(export_all_pages_json(all_blocks))
            status_lines.append(f"📋 JSON export saved.")

        if export_md:
            md_name = stem + "_ocr_output.md"
            md_path = str(OUTPUT_DIR / md_name)
            with open(md_path, "w", encoding="utf-8") as f:
                f.write(export_all_pages_markdown(all_blocks))
            status_lines.append(f"📝 Markdown export saved.")

        progress(0.96, desc="Building preview...")
        preview = _build_preview_html(original_pages, translated_pages, all_blocks)

        progress(1.0, desc="Done.")
        status_lines.append(f"\n✅ Done! Output saved to: {out_path}")

        return preview, out_path, json_path, md_path, "\n".join(status_lines)

    except Exception:
        err = traceback.format_exc()
        print(err)
        return None, None, None, None, f"❌ Error:\n{err}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="OpenLens 2.0",
) as demo:
    gr.Markdown(
        """
        # 🌐 OpenLens 2.0
        **Local AI-powered document translation.** Upload a PDF or image —
        translated text is overlaid directly onto the original layout.
        All processing is done locally. No data leaves your machine.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload document",
                file_types=[".pdf", ".jpg", ".jpeg", ".png", ".webp"],
                type="filepath",
            )
            lang_dropdown = gr.Dropdown(
                choices=list(TARGET_LANGUAGES.keys()),
                value="English",
                label="Translate to",
            )
            backend_dropdown = gr.Dropdown(
                choices=["M2M-100 418M (Faster)", "M2M-100 1.2B (Better Quality)"],
                value="M2M-100 418M (Faster)",
                label="Translation model",
                info="418M: faster, less VRAM. 1.2B: higher quality, needs ~5 GB VRAM.",
            )

            with gr.Row():
                export_json_cb = gr.Checkbox(
                    label="Export JSON", value=False,
                    info="Raw OCR blocks with bboxes — for LLM pipelines",
                )
                export_md_cb = gr.Checkbox(
                    label="Export Markdown", value=False,
                    info="Structured document text — headings, tables, lists",
                )

            submit_btn = gr.Button("🚀 Translate", variant="primary")

        with gr.Column(scale=2):
            status_box = gr.Textbox(
                label="Status",
                lines=6,
                interactive=False,
                placeholder="Upload a file and click Translate...",
            )
            download_btn = gr.File(
                label="📥 Download translated PDF",
                interactive=False,
            )
            with gr.Row():
                download_json = gr.File(
                    label="📋 JSON export",
                    interactive=False,
                )
                download_md = gr.File(
                    label="📝 Markdown export",
                    interactive=False,
                )

    gr.Markdown(
        "### Preview (original ← | → translated)  \n"
        "*Hover over text on the translated side to highlight — "
        "select and copy translated text directly.*"
    )
    preview_html = gr.HTML(label="Preview")

    submit_btn.click(
        fn=process_document,
        inputs=[file_input, lang_dropdown, backend_dropdown,
                export_json_cb, export_md_cb],
        outputs=[preview_html, download_btn, download_json,
                 download_md, status_box],
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - **M2M-100 418M** (~1.7 GB): faster, runs well on CPU or limited VRAM.
        - **M2M-100 1.2B** (~4.9 GB): higher quality translations, needs ~5 GB VRAM.
        - Both models support 100 languages with any-to-any translation.
        - Models are downloaded automatically on first use (one-time).
        - CPU-only machines may take 20–60 seconds per page.
        - For best results, use clear high-resolution scans (150 DPI+).
        - Hover over translated text to highlight it; select to copy.
        - **JSON/Markdown exports** contain the raw OCR output
          with bounding boxes — useful for downstream LLM processing.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible on your local network
        server_port=7860,
        share=False,              # set True for a temporary public URL
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
