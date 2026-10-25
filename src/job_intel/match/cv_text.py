"""Get clean text out of an uploaded CV (PDF or text) or pasted text.

Everything stays on the local machine. Failures raise ``CVExtractionError`` with a message meant
for the end user; softer problems (scanned pages, very short text) become warnings.
OCR is deliberately not implemented: a scanned PDF is detected and reported instead of being
silently "read" as an empty or garbled document.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

MAX_BYTES = 10 * 1024 * 1024
MAX_PAGES = 12
MAX_CHARS = 60_000
MIN_USEFUL_CHARS = 200
SCANNED_CHARS_PER_PAGE = 120
TEXT_SUFFIXES = {".txt", ".md", ".text"}


class CVExtractionError(ValueError):
    """The CV could not be read; the message is safe to show to the user."""


@dataclass
class ExtractedCV:
    text: str
    source: str  # "pdf" | "text" | "pasted"
    pages: int | None = None
    warnings: list[str] = field(default_factory=list)


_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿­"))
_BULLETS = re.compile(r"^[\s]*[•●▪■◦○◆►▶➢✓✔*·–—-]\s+")


def clean_cv_text(raw: str) -> str:
    """Normalize extracted text: unicode, bullets, hyphenation, whitespace. Never adds content."""
    text = unicodedata.normalize("NFC", raw).translate(_INVISIBLE)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ").replace("\t", " ")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # A hyphen at a line end may be a real hyphen ("data-\nscientist") or a syllable break; keeping it loses nothing.
    text = re.sub(r"(?<=[a-zà-ÿ])-\n(?=[a-zà-ÿ])", "-", text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r" {2,}", " ", line).rstrip()
        line = _BULLETS.sub("- ", line) if _BULLETS.match(line) else line.strip()
        lines.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _finish(text: str, source: str, pages: int | None, warnings: list[str]) -> ExtractedCV:
    text = clean_cv_text(text)
    if not text:
        raise CVExtractionError("The CV is empty: no text could be extracted.")
    if len(text) > MAX_CHARS:
        warnings.append(f"The CV is very long; only the first {MAX_CHARS:,} characters were analysed.")
        text = text[:MAX_CHARS]
    if len(text) < MIN_USEFUL_CHARS:
        warnings.append(
            f"Only {len(text)} characters of text were found: the profile will probably be incomplete. "
            "Check the extracted text and add missing sections by hand."
        )
    return ExtractedCV(text=text, source=source, pages=pages, warnings=warnings)


def from_pasted_text(text: str | None) -> ExtractedCV:
    if text is None or not text.strip():
        raise CVExtractionError("The pasted CV text is empty.")
    return _finish(text, "pasted", None, [])


def from_pdf_bytes(data: bytes) -> ExtractedCV:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise CVExtractionError('PDF support needs pypdf: python -m pip install -e ".[app]"') from exc
    if not data:
        raise CVExtractionError("The file is empty.")
    if len(data) > MAX_BYTES:
        raise CVExtractionError(f"The file is larger than {MAX_BYTES // (1024 * 1024)} MB.")
    if not data.lstrip().startswith(b"%PDF"):
        raise CVExtractionError("This file is not a valid PDF (missing %PDF header).")
    import io

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise CVExtractionError("The PDF is password-protected. Remove the password and upload it again.")
        n_pages = len(reader.pages)
        warnings: list[str] = []
        if n_pages > MAX_PAGES:
            warnings.append(f"The PDF has {n_pages} pages; only the first {MAX_PAGES} were read.")
        parts = []
        for page in reader.pages[:MAX_PAGES]:
            try:
                parts.append(page.extract_text() or "")
            except Exception:  # one unreadable page must not lose the others
                parts.append("")
    except CVExtractionError:
        raise
    except (PdfReadError, ValueError, KeyError, OSError, TypeError, AssertionError, RecursionError) as exc:
        raise CVExtractionError(f"The PDF could not be read ({type(exc).__name__}). It may be corrupted.") from exc

    read_pages = min(n_pages, MAX_PAGES)
    joined = "\n\n".join(parts)
    chars = len(re.sub(r"\s", "", joined))
    if chars == 0:
        raise CVExtractionError(
            "No text could be extracted: this PDF looks like a scan or an image. OCR is not supported - "
            "paste the CV text instead or export a text-based PDF."
        )
    if chars / max(read_pages, 1) < SCANNED_CHARS_PER_PAGE:
        warnings.append(
            "Very little text was found for the number of pages: the PDF may be (partly) scanned. "
            "OCR is not supported, so the profile may be incomplete - please verify it and use pasted text if needed."
        )
    empty_pages = sum(1 for p in parts if not re.sub(r"\s", "", p))
    if 0 < empty_pages < read_pages:
        warnings.append(f"{empty_pages} of {read_pages} pages contained no extractable text (images or scans?).")
    return _finish(joined, "pdf", n_pages, warnings)


def from_file(path: str | Path) -> ExtractedCV:
    path = Path(path)
    if not path.exists() or not path.is_file():
        raise CVExtractionError("The uploaded file could not be found.")
    suffix = path.suffix.lower()
    data = path.read_bytes() if path.stat().st_size <= MAX_BYTES + 1 else b""
    if suffix == ".pdf":
        return from_pdf_bytes(data)
    if suffix in TEXT_SUFFIXES:
        if not data:
            raise CVExtractionError("The file is empty or too large.")
        for encoding in ("utf-8-sig", "cp1252"):
            try:
                return _finish(data.decode(encoding), "text", None, [])
            except UnicodeDecodeError:
                continue
        raise CVExtractionError("The text file could not be decoded (use UTF-8).")
    raise CVExtractionError(f"Unsupported file type {suffix or '(none)'!r}. Upload a PDF or a .txt file, or paste the text.")
