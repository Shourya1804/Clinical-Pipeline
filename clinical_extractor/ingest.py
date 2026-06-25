"""
Multi-format, memory-safe document ingestion.

Reads clinical notes from many text formats and yields them ONE AT A TIME so a
folder of a million files never has to fit in memory.

Supported out of the box (no extra dependency):
  .txt .text .md .markdown .log .csv .tsv .json .ndjson .jsonl .rtf .htm .html

Supported if python-docx is installed:
  .docx   (Word).  PDFs are NOT text; use OCR/text-extraction first.

For structured formats (.csv/.json/...) the raw file text is passed through;
the NER model still finds medical terms inside the field values. HTML and RTF
are stripped to plain text first; .docx paragraphs are joined.
"""

from __future__ import annotations

import html as _html
import os
import re
from pathlib import Path
from typing import Iterator, Iterable, Optional, Tuple, List

# Formats handled by reading bytes directly (after light cleanup for html/rtf).
PLAIN_EXT = {".txt", ".text", ".md", ".markdown", ".log",
             ".csv", ".tsv", ".json", ".ndjson", ".jsonl"}
HTML_EXT = {".htm", ".html"}
RTF_EXT = {".rtf"}
DOCX_EXT = {".docx"}

SUPPORTED_EXT = PLAIN_EXT | HTML_EXT | RTF_EXT | DOCX_EXT


def supported_extensions() -> List[str]:
    return sorted(SUPPORTED_EXT)


def _read_plain(path: Path) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _strip_html(text: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = _html.unescape(text)
    return re.sub(r"[ \t]+\n", "\n", text)


def _strip_rtf(text: str) -> str:
    text = re.sub(r"\\par[d]?\b", "\n", text)
    text = re.sub(r"\\'[0-9a-fA-F]{2}", "", text)     # hex-escaped chars
    text = re.sub(r"\\[a-zA-Z]+-?\d* ?", "", text)    # control words
    text = text.replace("{", "").replace("}", "")
    return text


def _read_docx(path: Path) -> str:
    try:
        import docx  # python-docx
    except Exception:
        raise RuntimeError(
            "Reading .docx needs python-docx. Install it with: pip install python-docx"
        )
    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


def extract_text(path) -> str:
    """Read a single file of any supported type and return its plain text."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext in HTML_EXT:
        return _strip_html(_read_plain(path))
    if ext in RTF_EXT:
        return _strip_rtf(_read_plain(path))
    if ext in DOCX_EXT:
        return _read_docx(path)
    return _read_plain(path)


def _iter_paths(inputs: Iterable, recursive: bool, exts: set) -> Iterator[Path]:
    for raw in inputs:
        p = Path(raw)
        if p.is_dir():
            if recursive:
                for root, _dirs, files in os.walk(p):
                    for name in sorted(files):
                        fp = Path(root) / name
                        if fp.suffix.lower() in exts:
                            yield fp
            else:
                for fp in sorted(p.iterdir()):
                    if fp.is_file() and fp.suffix.lower() in exts:
                        yield fp
        elif p.is_file():
            yield p


def iter_documents(
    inputs: Iterable,
    recursive: bool = True,
    exts: Optional[Iterable[str]] = None,
    on_error=None,
) -> Iterator[Tuple[str, str]]:
    """Yield (source_name, text) for every readable document under `inputs`.

    `inputs` is one path or a list of paths (files and/or directories).
    Unreadable files are skipped (passed to `on_error(path, exc)` if given)
    instead of crashing the whole run - important for million-file batches.
    Memory stays flat: this is a generator, one document at a time.
    """
    use_exts = set(e.lower() if e.startswith(".") else "." + e.lower()
                   for e in exts) if exts else SUPPORTED_EXT
    if isinstance(inputs, (str, bytes, os.PathLike)):
        inputs = [inputs]

    for fp in _iter_paths(inputs, recursive, use_exts):
        try:
            text = extract_text(fp)
        except Exception as exc:           # unreadable / corrupt / missing dep
            if on_error:
                on_error(str(fp), exc)
            continue
        yield (fp.name, text)
