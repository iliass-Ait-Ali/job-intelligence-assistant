"""Pure helpers for cleaning text, dates, job types and URLs."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from html.parser import HTMLParser
from typing import Any
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

# --- HTML -> text -----------------------------------------------------------

_SKIP_TAGS = {"script", "style", "head", "template", "noscript"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "header", "footer", "ul", "ol", "table",
    "tr", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "br", "hr",
}  # fmt: skip
_INVISIBLE = dict.fromkeys(map(ord, "​‌‍⁠﻿"))


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS or tag == "li":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


_TAG_LEFTOVER = re.compile(r"</?[a-zA-Z][^>]*>")
_BULLET_GAP = re.compile(r"(\n- [^\n]*)\n\n(?=- )")  # blank line between two list items


def normalize_whitespace(text: str) -> str:
    """Collapse runs of spaces per line and limit blank lines to one."""
    text = text.translate(_INVISIBLE).replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    lines = [re.sub(r"[ \t\f\v]+", " ", line).strip() for line in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def _extract(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return "".join(parser.parts)


def html_to_text(value: Any) -> str | None:
    """Strip HTML, decode entities, keep paragraph/list structure. None if empty."""
    if value is None:
        return None
    raw = str(value)
    text = _extract(raw)
    # Some feeds double-escape their markup ("&lt;p&gt;"); decode once more if so.
    if "&lt;" in raw and _TAG_LEFTOVER.search(text):
        text = _extract(text)
    text = _BULLET_GAP.sub(r"\1\n", normalize_whitespace(text))
    return text or None


def clean_str(value: Any) -> str | None:
    """Trim and collapse whitespace in a short field; empty/None -> None."""
    if value is None:
        return None
    text = normalize_whitespace(str(value))
    return re.sub(r"\s*\n\s*", " ", text) or None


# --- dates ------------------------------------------------------------------


def parse_datetime(value: Any) -> str | None:
    """Return an ISO-8601 UTC string for epoch seconds/millis or ISO text, else None.

    Timestamps without a timezone are interpreted as UTC.
    """
    if value is None or isinstance(value, bool):
        return None
    try:
        if isinstance(value, (int, float)) or re.fullmatch(r"\d{9,13}", str(value).strip()):
            seconds = float(value)
            if seconds > 1e11:  # milliseconds
                seconds /= 1000
            dt = datetime.fromtimestamp(seconds, tz=UTC)
        else:
            text = str(value).strip()
            if not text:
                return None
            dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
            dt = dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)
    except (ValueError, OverflowError, OSError):
        return None
    return dt.isoformat()


# --- job type ---------------------------------------------------------------

# Checked in order; the first canonical type with a matching label wins.
_JOB_TYPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("internship", re.compile(r"intern(ship)?\b|praktik|stagiaire|convention de stage", re.I)),
    ("working_student", re.compile(r"werkstudent|working[\s-]*student", re.I)),
    ("freelance", re.compile(r"freelanc|freiberuf|selbst(st|ä)ndig|self-employed|auto-entrepreneur|ind[ée]pendant", re.I)),
    ("contract", re.compile(r"(?<!indefinite )(?<!employment )(?<!open-ended )\bcontract(or)?\b", re.I)),
    ("part_time", re.compile(r"part[\s_-]*time|teilzeit", re.I)),
    ("full_time", re.compile(r"full[\s_-]*time|vollzeit", re.I)),
    ("temporary", re.compile(r"temporary|fixed[\s-]*term|(?<!un)befristet|\bcdd\b", re.I)),
    ("permanent", re.compile(r"permanent|unbefristet|\bcdi\b|open-ended", re.I)),
)  # fmt: skip

_TITLE_INTERNSHIP = re.compile(
    r"\bintern(ship)?\b|(?:pflicht)?praktik(?:um|ant\w*)|stagiaire|\bwerkstudent\w*|working[\s-]*student",
    re.I,
)


def canonical_job_type(labels: list[str] | str | None, title: str | None = None) -> str | None:
    """Map provider labels (any language) to a small canonical vocabulary.

    A title that says internship / Praktikum / Werkstudent overrides the labels,
    because providers often tag such roles as plain "Full Time".
    Experience-level labels ("Mid", "Experienced", ...) are ignored.
    """
    if title and (m := _TITLE_INTERNSHIP.search(title)):
        return "working_student" if re.search(r"werkstudent|working", m.group(0), re.I) else "internship"
    if labels is None:
        return None
    if isinstance(labels, str):
        labels = [labels]
    joined = " | ".join(str(label) for label in labels if label)
    for canonical, pattern in _JOB_TYPE_RULES:
        if pattern.search(joined):
            return canonical
    return None


# --- URLs -------------------------------------------------------------------

_TRACKING_PARAMS = {"ref", "source", "gclid", "fbclid", "mc_cid", "mc_eid"}


def canonical_url(url: str) -> str:
    """Normalize a URL for duplicate detection (not for display)."""
    parts = urlsplit(url.strip())
    query = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS
    ]
    path = parts.path.rstrip("/") or "/"
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, urlencode(sorted(query)), ""))
