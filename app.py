"""
app.py — Gradio web UI for DocuTranslate.

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
from pipeline.translate import translate_blocks
from pipeline.inpaint import erase_text_blocks
from pipeline.renderer import render_translations

# ── Supported target languages (Argos Translate codes) ───────────────────────
TARGET_LANGUAGES = {
    "English":    "en",
    "Spanish":    "es",
    "French":     "fr",
    "German":     "de",
    "Portuguese": "pt",
    "Italian":    "it",
    "Dutch":      "nl",
    "Russian":    "ru",
    "Arabic":     "ar",
    "Japanese":   "ja",
    "Korean":     "ko",
    "Chinese (Simplified)": "zh",
    "Turkish":    "tr",
    "Polish":     "pl",
    "Swedish":    "sv",
}

OUTPUT_DIR = Path(tempfile.gettempdir()) / "docutranslate_output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _image_to_data_uri(img: Image.Image, fmt: str = "JPEG") -> str:
    """Encode a PIL image as a base64 data URI for embedding in HTML."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=85)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/jpeg" if fmt == "JPEG" else "image/png"
    return f"data:{mime};base64,{b64}"


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
        # Strip HTML table tags so raw <td>/<tr> don't leak into overlay
        text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
        text = re.sub(r'</t[dh]>', '  ', text, flags=re.IGNORECASE)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'  +', '  ', text).strip()
        if not text:
            continue
        # Escape HTML entities
        safe_text = (text.replace("&", "&amp;").replace("<", "&lt;")
                     .replace(">", "&gt;").replace("\n", "<br>"))
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
        color: transparent; cursor: text;
        user-select: text; -webkit-user-select: text;
        z-index: 2;
        background: transparent;
        transition: background 0.15s, color 0.15s;
        border-radius: 2px;
    }
    .text-box::selection { background: rgba(0,120,215,0.4); color: #000; }
    .text-box:hover {
        background: rgba(255, 255, 255, 0.70);
        color: #000;
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
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[str], Optional[str], str]:
    """
    Main pipeline function called by Gradio.

    Returns:
        - preview_html: HTML string with side-by-side selectable-text preview
        - output_file: path to the downloadable translated PDF
        - status: status message string
    """
    if file_obj is None:
        return None, None, "⚠️ Please upload a file first."

    tgt_code = TARGET_LANGUAGES.get(target_lang_name, "en")
    file_path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

    try:
        # ── Step 1: Load document pages ───────────────────────────────────────
        progress(0.05, desc="Loading document...")
        pages = load_document(file_path)
        n = len(pages)
        status_lines = [f"📄 Loaded {n} page(s)."]

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

        # ── Step 6: Export ────────────────────────────────────────────────────
        progress(0.92, desc="Saving output PDF...")
        out_name = Path(file_path).stem + f"_translated_{tgt_code}.pdf"
        out_path = str(OUTPUT_DIR / out_name)
        images_to_pdf(translated_pages, out_path)

        progress(0.96, desc="Building preview...")
        preview = _build_preview_html(original_pages, translated_pages, all_blocks)

        progress(1.0, desc="Done.")
        status_lines.append(f"\n✅ Done! Output saved to: {out_path}")

        return preview, out_path, "\n".join(status_lines)

    except Exception:
        err = traceback.format_exc()
        print(err)
        return None, None, f"❌ Error:\n{err}"


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="DocuTranslate",
) as demo:
    gr.Markdown(
        """
        # 🌐 DocuTranslate
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

    gr.Markdown(
        "### Preview (original ← | → translated)  \n"
        "*Hover over text on the translated side to highlight — "
        "select and copy translated text directly.*"
    )
    preview_html = gr.HTML(label="Preview")

    submit_btn.click(
        fn=process_document,
        inputs=[file_input, lang_dropdown],
        outputs=[preview_html, download_btn, status_box],
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - First run will download Argos language packs (~100 MB per language pair).
        - CPU-only machines may take 20–60 seconds per page.
        - For best results, use clear high-resolution scans (150 DPI+).
        - Hover over translated text to highlight it; select to copy.
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
