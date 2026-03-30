# OpenLens 2.0 — Local AI Document Translation Tool

Translates PDFs and images by overlaying translated text directly onto the original document layout — similar to Google Lens, but fully local.  
Runs entirely on your machine — no internet required after setup, no external API calls, no data leaves your PC.

---

## How it works

1. **OCR** — [dots.mocr](https://huggingface.co/rednote-hilab/dots.mocr) (Qwen2.5-VL based, 3B params) extracts text blocks with bounding boxes and layout categories (Title, Text, Table, Section-header, etc.). This is the successor to dots.ocr with improved multilingual accuracy and structured-graphics parsing.
2. **Translate** — **M2M-100** (100 languages, any-to-any, MIT license). Two model sizes: 418M (faster) and 1.2B (higher quality). Fully offline after first model download.
3. **Inpaint** — The original text regions are erased by sampling the surrounding background colour and filling the bounding box.
4. **Render** — Translated text is drawn back onto the document at the correct position, recovering the original font size via binary search and rendering form-style label/value pairs side-by-side with tab-stop alignment.
5. **Export** — Output as downloadable translated PDF, plus optional **JSON** and **Markdown** exports for downstream LLM processing.

---

## Features

- **M2M-100 translation** — 100 languages, any-to-any. Two model sizes (418M / 1.2B) switchable from the UI.
- **JSON & Markdown export** — Toggle raw OCR output as downloadable `.json` (with bboxes, categories, text) or `.md` (structured headings, tables, lists) for downstream LLM pipelines.
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
├── setup_server.sh    ← Run once for Linux server deployment (+ Flash Attention)
├── run.bat            ← Run daily to launch the app
├── requirements.txt
├── app.py             ← Gradio web UI + pipeline orchestration
├── download_model.py  ← Downloads dots.mocr weights into models/
├── test_renderer.py   ← Visual test for renderer with mock OCR blocks
├── pipeline/
│   ├── init.py
│   ├── ocr.py         ← dots.mocr inference + layout parsing
│   ├── translate.py   ← Translation router (M2M-100)
│   ├── translate_m2m.py ← M2M-100 translation backend (100 languages)
│   ├── export.py      ← JSON + Markdown export from OCR blocks
│   ├── inpaint.py     ← Erase original text from image
│   ├── renderer.py    ← Overlay translated text (font recovery, form layout)
│   └── pdf_utils.py   ← PDF ↔ image conversion (PyMuPDF @ 150 DPI)
└── models/            ← Auto-created, stores downloaded model weights
```

---

## One-time setup (Windows)

1. Install **Python 3.11** from https://www.python.org/downloads/  
   — During install, tick **"Add Python to PATH"**
2. Install **Git** from https://git-scm.com/download/win
3. Copy this entire folder to your PC.
4. Double-click **`setup.bat`**  
   — Creates a virtual environment, installs dependencies, downloads the dots.mocr model.  
   — Takes 5–15 minutes on first run depending on your connection.
5. Double-click **`run.bat`** to start the app.
6. Open your browser at **http://localhost:7860**

## One-time setup (Linux server — with Flash Attention)

For production / mass-usage deployment on Linux with NVIDIA GPUs:

```bash
chmod +x setup_server.sh
./setup_server.sh
```

This installs everything including **Flash Attention 2** for ~30% faster OCR inference.  
Requires: CUDA toolkit, gcc/g++, ninja-build.

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

**Source** (auto-detected): French, Catalan, Spanish, Italian, German, Portuguese, Chinese, Japanese, Korean, Arabic, Russian — plus any other language dots.mocr can read (falls back to majority-language detection).

**Target** (selectable in UI): **100 languages** with M2M-100, including all of the above plus:

Afrikaans, Amharic, Asturian, Azerbaijani, Bashkir, Belarusian, Bengali, Breton, Bosnian, Bulgarian, Cebuano, Czech, Croatian, Danish, Estonian, Finnish, Fulah, Georgian, Greek, Gujarati, Haitian Creole, Hausa, Hebrew, Hindi, Hungarian, Armenian, Icelandic, Igbo, Iloko, Indonesian, Irish, Javanese, Kazakh, Khmer, Kannada, Lao, Latvian, Lithuanian, Luxembourgish, Macedonian, Malagasy, Malay, Malayalam, Marathi, Mongolian, Burmese, Nepali, Occitan, Oriya, Pashto, Persian, Punjabi, Romanian, Scottish Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Albanian, Sundanese, Swahili, Swati, Tagalog, Tamil, Thai, Tswana, Ukrainian, Urdu, Uzbek, Vietnamese, Welsh, Western Frisian, Wolof, Xhosa, Yiddish, Yoruba, Zulu.

Both model sizes support the same 100 language codes.

---

## Hardware notes

| Setup                 | Expected speed          |
|-----------------------|-------------------------|
| NVIDIA GPU + Flash Attn (Linux) | ~30–60 s per page |
| NVIDIA GPU + SDPA (Windows)     | ~50–80 s per page |
| CPU only (no GPU)     | ~2–5 min per page       |

dots.mocr auto-detects your GPU. On CUDA, the model runs in **bfloat16** with **SDPA** attention (or **Flash Attention 2** if installed on Linux).  
Speed depends heavily on two tunable constants in `pipeline/ocr.py`:

| Knob | What it does | Trade-off |
|------|-------------|-----------|
| `OCR_MAX_PIXELS` | Max image resolution fed to the model | Lower = faster, less detail |
| `MAX_NEW_TOKENS` | Max output tokens per page | Lower = faster, may truncate dense pages |

See the commented boxes in `pipeline/ocr.py` for detailed guidance.

---

## dots.mocr compatibility notes

dots.mocr ships custom model code (`trust_remote_code=True`) that may conflict with certain `transformers` versions. The OCR pipeline handles these automatically:

- **flash_attn not installed** — The upstream vision encoder hard-imports `flash_attn`. On Windows (or any system without it), a lightweight stub module is injected so the import succeeds. The model uses PyTorch SDPA instead — no functionality is lost.
- **Processor init TypeError** — The custom `DotsVLProcessor` may conflict with `Qwen2_5_VLProcessor.__init__` in some transformers versions. The pipeline falls back to building the processor manually with the correct chat template and token IDs.
- **Requires transformers 4.x** — Version 5.x breaks the model. Pinned in `requirements.txt`.

---

## Known limitations

- **Latin-language disambiguation** — Catalan, French, Italian, and Spanish share many accented characters. The heuristic detector uses stop-word boosting but can still misidentify short texts. Longer documents are more reliable.
- **Table rendering** — Complex multi-column tables with merged cells may not render perfectly; the form-layout renderer handles simple label–value pairs well.
- **Translation quality** — M2M-100 1.2B provides the best quality. The 418M model is faster but may produce lower-quality translations for complex sentences.
- **Handwritten text** — dots.mocr is optimised for printed/typed text. Handwritten documents may produce poor OCR results.

---

## Troubleshooting

**"Python not found"** — Re-run the Python installer and tick "Add Python to PATH".

**"Model download failed"** — Check your internet connection. The model only needs to download once.

**"Translation failed"** — Check the console for the detected language code. M2M-100 supports 100 languages — if the source language is not in the supported set, the text will be returned as-is.

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
- **Batch processing** — upload a folder of documents and translate them all in one run.
- **Translation memory / glossary** — user-defined term mappings (e.g. company names, legal terms) that override translation output.
- ~~Upgrading to dots.mocr for improved results~~ ✅ Done
- ~~M2M-100 translation backend (100 languages)~~ ✅ Done
- ~~Remove Argos dependency, dual M2M-100 model sizes~~ ✅ Done
- ~~JSON/Markdown export for LLM pipelines~~ ✅ Done

---

## Privacy & security

- All processing happens on your machine.
- No files are sent to any external server.
- No telemetry, no analytics, no tracking.
- Uploaded documents are processed in memory and not stored to disk permanently
- Temporary files are cleaned up after each job
