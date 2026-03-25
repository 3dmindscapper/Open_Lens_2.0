"""
pdf_utils.py — Convert PDFs to page images and reassemble translated pages back into a PDF.
"""
import io
import os
from pathlib import Path
from typing import List

import fitz  # pymupdf
from PIL import Image


DPI = 150  # Balance between quality and speed. Increase to 200 for sharper output.


def pdf_to_images(pdf_path: str) -> List[Image.Image]:
    """Convert every page of a PDF to a PIL Image at DPI resolution."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        mat = fitz.Matrix(DPI / 72, DPI / 72)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    doc.close()
    return images


def image_file_to_images(image_path: str) -> List[Image.Image]:
    """Load a single JPG/PNG as a one-element list to unify the pipeline."""
    img = Image.open(image_path).convert("RGB")
    return [img]


def images_to_pdf(images: List[Image.Image], output_path: str) -> str:
    """Save a list of PIL Images as a multi-page PDF."""
    if not images:
        raise ValueError("No images to save.")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        resolution=DPI,
    )
    return output_path


def load_document(file_path: str) -> List[Image.Image]:
    """Auto-detect file type and return a list of page images."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        return pdf_to_images(file_path)
    elif ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"):
        return image_file_to_images(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
