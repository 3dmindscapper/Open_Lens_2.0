# DocuTranslate — Local AI Document Translation Tool

Translates PDFs and images by overlaying translated text directly onto the original document.
Runs fully locally — no internet required after setup, no external API calls.

---

## Folder structure

```
docutranslate/
├── README.md
├── setup.bat          ← Run once to install everything (Windows)
├── run.bat            ← Run daily to launch the app
├── requirements.txt
├── app.py             ← Gradio web UI
├── pipeline/
│   ├── __init__.py
│   ├── ocr.py         ← dots.ocr inference
│   ├── translate.py   ← Argos Translate (local)
│   ├── inpaint.py     ← Erase original text from image
│   ├── renderer.py    ← Overlay translated text
│   └── pdf_utils.py   ← PDF ↔ image conversion
└── models/            ← Auto-created, stores downloaded model weights
```

---

## One-time setup (do this first, on your home PC)

1. Install **Python 3.11** from https://www.python.org/downloads/
   - During install, tick **"Add Python to PATH"**
2. Install **Git** from https://git-scm.com/download/win
3. Copy this entire `docutranslate/` folder to your PC
4. Double-click **`setup.bat`**
   - This installs all dependencies and downloads the dots.ocr model (~2–4 GB)
   - Takes 5–15 minutes on first run
5. Double-click **`run.bat`** to start the app
6. Open your browser at **http://localhost:7860**

---

## Daily use

Just double-click **`run.bat`** — the Gradio UI opens in your browser automatically.

---

## Supported inputs

| Format | Notes |
|--------|-------|
| PDF    | All pages processed, multi-page output preserved |
| JPG / JPEG | Single image |
| PNG    | Single image |

---

## Language support

Source language is auto-detected by dots.ocr.
Target language defaults to English. You can change it in the UI.

Argos Translate language packs are downloaded automatically on first use.
Supported pairs: see https://www.argosopentech.com/argospm/index/

---

## Hardware notes

| Setup | Expected speed |
|-------|---------------|
| NVIDIA GPU (CUDA) | ~3–8s per page |
| CPU only (no GPU) | ~20–60s per page |
| Apple Silicon (MPS) | ~5–15s per page |

dots.ocr will auto-detect your GPU. No manual config needed.

---

## Troubleshooting

**"Python not found"** — Re-run the Python installer and tick "Add Python to PATH"

**"Model download failed"** — Check your internet connection. The model only needs to download once.

**"Translation failed"** — The Argos language pack for that language pair may not be installed. Check the console output for the language code and install manually:
```
argospm install translate-XX_en
```
Replace `XX` with the detected source language code (e.g. `zh`, `fr`, `de`).

**Blank/white overlay on output** — The inpainting step couldn't sample a background color. Try uploading a higher-resolution scan.

---

## Privacy & security

- All processing happens on your machine
- No files are sent to any external server
- Uploaded documents are processed in memory and not stored to disk permanently
- Temporary files are cleaned up after each job
