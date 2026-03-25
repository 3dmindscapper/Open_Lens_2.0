"""
app.py — Gradio web UI for DocuTranslate.

Launch with: python app.py
Then open:   http://localhost:7860
"""
import os
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


def process_document(
    file_obj,
    target_lang_name: str,
    progress=gr.Progress(track_tqdm=True),
) -> Tuple[Optional[List[Image.Image]], Optional[str], str]:
    """
    Main pipeline function called by Gradio.

    Returns:
        - gallery: list of translated page images for side-by-side preview
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

        for i, page_img in enumerate(pages):
            page_label = f"Page {i+1}/{n}"
            progress((i / n) * 0.85 + 0.05, desc=f"Processing {page_label}...")

            # ── Step 2: OCR ───────────────────────────────────────────────────
            blocks = run_ocr(page_img)
            if not blocks:
                # No text found — keep page as-is
                translated_pages.append(page_img)
                original_pages.append(page_img)
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

        # ── Step 6: Export ────────────────────────────────────────────────────
        progress(0.92, desc="Saving output PDF...")
        out_name = Path(file_path).stem + f"_translated_{tgt_code}.pdf"
        out_path = str(OUTPUT_DIR / out_name)
        images_to_pdf(translated_pages, out_path)

        progress(1.0, desc="Done.")
        status_lines.append(f"\n✅ Done! Output saved to: {out_path}")

        # Build side-by-side comparison images for the gallery
        comparison_images = []
        for orig, trans in zip(original_pages, translated_pages):
            w = orig.width + trans.width + 8
            h = max(orig.height, trans.height)
            combined = Image.new("RGB", (w, h), (240, 240, 240))
            combined.paste(orig, (0, 0))
            combined.paste(trans, (orig.width + 8, 0))
            comparison_images.append(combined)

        return comparison_images, out_path, "\n".join(status_lines)

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
                file_types=[".pdf", ".jpg", ".jpeg", ".png"],
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

    gr.Markdown("### Side-by-side preview (original ← | → translated)")
    gallery = gr.Gallery(
        label="Pages",
        columns=1,
        height="auto",
        object_fit="contain",
    )

    submit_btn.click(
        fn=process_document,
        inputs=[file_input, lang_dropdown],
        outputs=[gallery, download_btn, status_box],
    )

    gr.Markdown(
        """
        ---
        **Tips:**
        - First run will download Argos language packs (~100 MB per language pair).
        - CPU-only machines may take 20–60 seconds per page.
        - For best results, use clear high-resolution scans (150 DPI+).
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
