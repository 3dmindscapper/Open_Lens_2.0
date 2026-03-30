"""
export.py — Export OCR blocks as JSON or Markdown files.

Produces structured output suitable for downstream LLM processing.
"""
import json
import re
from typing import List, Dict, Any


def blocks_to_json(
    blocks: List[Dict[str, Any]],
    page_index: int = 0,
    include_translated: bool = True,
) -> dict:
    """Convert OCR blocks into a JSON-serialisable dict.

    Args:
        include_translated: If True, include the 'translated' field when present.
    """
    out_blocks = []
    for b in blocks:
        entry = {
            "bbox": list(b["bbox"]) if isinstance(b.get("bbox"), tuple) else b.get("bbox", []),
            "category": b.get("category", "Text"),
            "text": b.get("text", ""),
            "lang": b.get("lang", "unknown"),
        }
        if include_translated and "translated" in b:
            entry["translated"] = b["translated"]
        out_blocks.append(entry)
    return {"page": page_index + 1, "blocks": out_blocks}


def blocks_to_markdown(
    blocks: List[Dict[str, Any]],
    page_index: int = 0,
    text_key: str = "text",
) -> str:
    """Convert OCR blocks into a Markdown string preserving document structure.

    Args:
        text_key: Which block field to use — 'text' for raw OCR output,
                  'translated' for translated text.

    Category mapping:
        Title          → # heading
        Section-header → ## heading
        Text           → paragraph
        List-item      → - bullet
        Table          → HTML table (passed through)
        Formula        → $$ LaTeX $$
        Caption        → *italic*
        Footnote       → > blockquote
        Page-header    → (skipped — usually noise)
        Page-footer    → (skipped — usually noise)
        Picture        → (skipped)
    """
    lines = [f"<!-- Page {page_index + 1} -->", ""]
    prev_category = None

    for b in blocks:
        category = b.get("category", "Text")
        text = b.get(text_key, b.get("text", "")).strip()
        if not text or category in ("Picture", "Page-header", "Page-footer"):
            continue

        # Clean HTML from non-table content
        if category != "Table":
            text = _strip_html_inline(text)
            text = text.strip()
            if not text:
                continue

        # Add blank line between different category types
        if prev_category and prev_category != category and lines[-1] != "":
            lines.append("")

        if category == "Title":
            lines.append(f"# {text}")
        elif category == "Section-header":
            lines.append(f"## {text}")
        elif category == "List-item":
            # Handle multi-line list items
            for li in text.split("\n"):
                li = li.strip()
                if li:
                    if not li.startswith("- ") and not li.startswith("* "):
                        li = f"- {li}"
                    lines.append(li)
        elif category == "Table":
            lines.append(text)  # Pass HTML table through
        elif category == "Formula":
            lines.append(f"$${text}$$")
        elif category == "Caption":
            lines.append(f"*{text}*")
        elif category == "Footnote":
            for fn_line in text.split("\n"):
                fn_line = fn_line.strip()
                if fn_line:
                    lines.append(f"> {fn_line}")
        else:
            # Text and anything else → paragraph
            lines.append(text)

        prev_category = category

    lines.append("")  # trailing newline
    return "\n".join(lines)


def export_all_pages_json(
    all_blocks: List[List[Dict[str, Any]]],
    include_translated: bool = True,
) -> str:
    """Combine all pages into a single JSON string."""
    pages = [
        blocks_to_json(blocks, i, include_translated=include_translated)
        for i, blocks in enumerate(all_blocks)
    ]
    return json.dumps({"pages": pages}, indent=2, ensure_ascii=False)


def export_all_pages_markdown(
    all_blocks: List[List[Dict[str, Any]]],
    text_key: str = "text",
) -> str:
    """Combine all pages into a single Markdown string."""
    parts = [
        blocks_to_markdown(blocks, i, text_key=text_key)
        for i, blocks in enumerate(all_blocks)
    ]
    return "\n---\n\n".join(parts)


def _strip_html_inline(text: str) -> str:
    """Remove HTML tags, converting table structures to line-separated text."""
    text = re.sub(r'</tr>', '\n', text, flags=re.IGNORECASE)
    text = re.sub(r'</t[dh]>', '  ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    return text.strip()
