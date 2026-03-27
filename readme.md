# OpenLens 2.0 — Local AI Document Translation Tool

Translates PDFs and images by overlaying translated text directly onto the original document layout — similar to Google Lens, but fully local.  
Runs entirely on your machine — no internet required after setup, no external API calls, no data leaves your PC.

---

## How it works

1. **OCR** — [dots.ocr](https://huggingface.co/rednote-hilab/dots.ocr) (Qwen2.5-VL based) extracts text blocks with bounding boxes and layout categories (Title, Text, Table, Section-header, etc.).
2. **Translate** — [Argos Translate](https://github.com/argosopentech/argos-translate) translates each block line-by-line, preserving document structure. Fully offline after the first language-pack download.
3. **Inpaint** — The original text regions are erased by sampling the surrounding background colour and filling the bounding box.
4. **Render** — Translated text is drawn back onto the document at the correct position, recovering the original font size via binary search and rendering form-style label/value pairs side-by-side with tab-stop alignment.
5. **Export** — Output is served as a downloadable translated PDF (or image).

---

## Features

- **Side-by-side comparison** — Gradio UI shows original and translated documents next to each other.
- **Font size recovery** — Binary-searches the font size that makes the *original* text fit its bounding box, then uses that same size for the translation.
- **Form / table layout** — Detects alternating label–value line patterns and renders them side-by-side with tab-stop alignment instead of stacking vertically.
- **Markdown & HTML stripping** — Cleans bold/italic markers and HTML table tags (`<th>`, `<td>`, etc.) from OCR output before translation and rendering.
- **Logo protection** — Detects Picture bounding boxes and skips text blocks that overlap them, preventing logos from being erased.
- **Multi-language detection** — Heuristic language detection with accent-character counting and stop-word boosting for Latin-script languages (French, Catalan, Spanish, Italian, German, Portuguese) plus CJK, Arabic, and Cyrillic.

---

## Folder structure

```
Open_Lens_2.0/
├── readme.md
├── setup.bat          ← Run once to install everything (Windows)
├── run.bat            ← Run daily to launch the app
├── requirements.txt
├── app.py             ← Gradio web UI + pipeline orchestration
├── download_model.py  ← Downloads dots.ocr weights into models/
├── pipeline/
│   ├── init.py
│   ├── ocr.py         ← dots.ocr inference + layout parsing
│   ├── translate.py   ← Argos Translate (local, line-by-line)
│   ├── inpaint.py     ← Erase original text from image
│   ├── renderer.py    ← Overlay translated text (font recovery, form layout)
│   └── pdf_utils.py   ← PDF ↔ image conversion (PyMuPDF @ 150 DPI)
└── models/            ← Auto-created, stores downloaded model weights (~2–4 GB)
```

---

## One-time setup (Windows)

1. Install **Python 3.11** from https://www.python.org/downloads/  
   — During install, tick **"Add Python to PATH"**
2. Install **Git** from https://git-scm.com/download/win
3. Copy this entire folder to your PC.
4. Double-click **`setup.bat`**  
   — Creates a virtual environment, installs dependencies, downloads the dots.ocr model.  
   — Takes 5–15 minutes on first run depending on your connection.
5. Double-click **`run.bat`** to start the app.
6. Open your browser at **http://localhost:7860**

---

## Daily use

Double-click **`run.bat`** — the Gradio UI opens in your browser automatically.

---

## Supported inputs

| Format   | Notes                                    |
|----------|------------------------------------------|
| PDF      | All pages processed, multi-page output   |
| JPG/JPEG | Single image                             |
| PNG      | Single image                             |

---

## Supported languages

**Source** (auto-detected): French, Catalan, Spanish, Italian, German, Portuguese, Chinese, Japanese, Korean, Arabic, Russian — plus any other language dots.ocr can read (falls back to majority-language detection).

**Target** (selectable in UI):

| Language             | Code |
|----------------------|------|
| English              | en   |
| Spanish              | es   |
| French               | fr   |
| German               | de   |
| Portuguese           | pt   |
| Italian              | it   |
| Dutch                | nl   |
| Russian              | ru   |
| Arabic               | ar   |
| Japanese             | ja   |
| Korean               | ko   |
| Chinese (Simplified) | zh   |
| Turkish              | tr   |
| Polish               | pl   |
| Swedish              | sv   |

Argos Translate language packs are downloaded automatically on first use (~100 MB per pair).  
Full list: https://www.argosopentech.com/argospm/index/

---

## Hardware notes

| Setup                 | Expected speed      |
|-----------------------|---------------------|
| NVIDIA GPU (CUDA)     | ~3–8 s per page     |
| CPU only (no GPU)     | ~20–60 s per page   |
| Apple Silicon (MPS)   | ~5–15 s per page    |

dots.ocr auto-detects your GPU. No manual config needed.

---

## Known limitations

- **Latin-language disambiguation** — Catalan, French, Italian, and Spanish share many accented characters. The heuristic detector uses stop-word boosting but can still misidentify short texts. Longer documents are more reliable.
- **Table rendering** — Complex multi-column tables with merged cells may not render perfectly; the form-layout renderer handles simple label–value pairs well.
- **Translation quality** — Depends on Argos Translate's model quality for each language pair. Some pairs (e.g. Catalan→English) may be less polished than others (e.g. French→English).
- **Handwritten text** — dots.ocr is optimised for printed/typed text. Handwritten documents may produce poor OCR results.

---

## Troubleshooting

**"Python not found"** — Re-run the Python installer and tick "Add Python to PATH".

**"Model download failed"** — Check your internet connection. The model only needs to download once.

**"Translation failed"** — The Argos language pack for that pair may not be installed. Check the console for the detected language code and install manually:
```
argospm install translate-XX_en
```
Replace `XX` with the source language code (e.g. `ca`, `fr`, `de`).

**Blank/white overlay** — The inpainting step couldn't sample a background colour. Try a higher-resolution scan.

**Wrong source language detected** — The heuristic may misidentify short text blocks. A future update will add manual source-language override in the UI.

---

## Roadmap / Future plans

### REST API server mode
Expose the translation pipeline as a stateless HTTP API so any frontend (React, Vue, mobile app, etc.) can call it:
```
POST /api/translate
  - Body: multipart file upload + target language
  - Response: translated PDF/image bytes (or a job ID for async processing)

GET /api/status/{job_id}
  - Returns job progress and download URL when complete
```
Planned stack: **FastAPI** with background task queue, optional Redis/Celery for multi-worker scaling.

### Additional planned features
- **Manual source-language override** — dropdown in the UI to force a source language when auto-detection fails.
- **Batch processing** — upload a folder of documents and translate them all in one run.
- **Translation memory / glossary** — user-defined term mappings (e.g. company names, legal terms) that override Argos output.
- **Docker image** — single-command deployment with GPU passthrough for server environments.
- **Progress streaming** — real-time page-by-page progress updates via WebSocket or SSE.
- **Improved table rendering** — parse HTML table structure from OCR output and reconstruct grid layouts with cell borders.

---

## Privacy & security

- All processing happens on your machine.
- No files are sent to any external server.
- No telemetry, no analytics, no tracking.
- Uploaded documents are processed in memory and not stored to disk permanently
- Temporary files are cleaned up after each job
