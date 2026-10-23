"""Turn a job record into embeddable chunks.

Every chunk is "header + body slice". The header (title | company | location | type) is repeated in
each chunk so a chunk from the middle of a long posting still says which job it belongs to; this
matters because several postings of one company share long identical introductions.

Chunking is token-aware: the counter passed in measures exactly what the model will receive
(prefix + special tokens included), and every finished chunk is re-measured. A chunk over the
limit is never emitted: the job is re-packed with a smaller budget, and if that still fails a
ChunkingError is raised, so text is never silently truncated by the model.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from job_intel.search.errors import ChunkingError

CHUNKER_VERSION = 1
MIN_BODY_TOKENS = 32
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?;:])\s+|\n")

TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class ChunkingConfig:
    max_tokens: int = 300  # full model input per chunk: prefix + header + body + special tokens
    overlap_tokens: int = 40  # trailing text repeated at the start of the next chunk


@dataclass(frozen=True)
class Chunk:
    index: int
    text: str  # body slice, shown as the search snippet
    embed_text: str  # header + body, what gets embedded (the embedder adds its own prefix)
    tokens: int


def job_header(job: Mapping[str, Any]) -> str:
    parts = [
        (job.get("title") or "")[:200],
        (job.get("company") or "")[:100],
        (job.get("location") or "")[:100],
        (job.get("job_type") or "").replace("_", " "),
        "remote" if job.get("remote") else "",
    ]
    return " | ".join(p for p in parts if p)


def _split_oversized(text: str, budget: int, cost: TokenCounter) -> list[str]:
    """Break text that is over budget into pieces that each fit (words, then characters)."""
    pieces, current = [], ""
    for word in text.split():
        candidate = f"{current} {word}".strip()
        if cost(candidate) <= budget:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = ""
        while cost(word) > budget:  # a single huge token (URL, base64...): cut by characters
            lo, hi = 1, len(word)
            while lo < hi:
                mid = (lo + hi + 1) // 2
                lo, hi = (mid, hi) if cost(word[:mid]) <= budget else (lo, mid - 1)
            pieces.append(word[:lo])
            word = word[lo:]
        current = word
    if current:
        pieces.append(current)
    return pieces


def _units(body: str, budget: int, cost: TokenCounter) -> list[tuple[str, int]]:
    """Paragraphs, falling back to sentences/lines, then words, each within budget."""
    units: list[tuple[str, int]] = []
    for paragraph in (p.strip() for p in body.split("\n\n")):
        if not paragraph:
            continue
        pieces = [paragraph]
        if cost(paragraph) > budget:
            pieces = [s.strip() for s in _SENTENCE_SPLIT.split(paragraph) if s.strip()]
        for piece in pieces:
            fitted = [piece] if cost(piece) <= budget else _split_oversized(piece, budget, cost)
            units.extend((f, cost(f)) for f in fitted)
    return units


def _overlap_tail(chunk: list[tuple[str, int]], overlap: int) -> list[tuple[str, int]]:
    """Trailing units of a finished chunk (at most `overlap` tokens) to repeat in the next one."""
    tail: list[tuple[str, int]] = []
    used = 0
    for unit in reversed(chunk[1:]):  # never the whole chunk: that would duplicate it
        if used + unit[1] + (1 if tail else 0) > overlap:
            break
        tail.insert(0, unit)
        used += unit[1] + (1 if len(tail) > 1 else 0)
    return tail


def _pack(units: list[tuple[str, int]], budget: int, overlap: int) -> list[str]:
    chunks: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    used = 0
    for unit in units:
        join = 1 if current else 0
        if current and used + join + unit[1] > budget:
            chunks.append(current)
            current, used = _overlap_tail(current, overlap), 0
            used = sum(u[1] for u in current) + max(0, len(current) - 1)
            if current and used + 1 + unit[1] > budget:
                current, used = [], 0
            join = 1 if current else 0
        current.append(unit)
        used += join + unit[1]
    if current:
        chunks.append(current)
    return ["\n".join(text for text, _ in chunk) for chunk in chunks]


def build_chunks(
    job: Mapping[str, Any], count_tokens: TokenCounter, cfg: ChunkingConfig, model_max_tokens: int
) -> list[Chunk]:
    """Chunk one job. `count_tokens` must count the exact model input for a passage text."""
    limit = min(cfg.max_tokens, model_max_tokens)
    header = job_header(job)
    body = (job.get("description") or "").strip()
    overhead = count_tokens("")
    header_cost = count_tokens(header + "\n") - overhead
    cost = lambda text: count_tokens(text) - overhead  # noqa: E731
    budget = limit - overhead - header_cost
    if budget < MIN_BODY_TOKENS:
        raise ChunkingError(
            f"job {job.get('id')!r}: header uses {header_cost} tokens of the {limit}-token limit; "
            f"raise chunk size or shorten the header"
        )

    for _attempt in range(8):
        bodies = _pack(_units(body, budget, cost), budget, cfg.overlap_tokens) if body else [""]
        chunks = []
        for i, text in enumerate(bodies):
            embed_text = f"{header}\n{text}" if text else header
            chunks.append(Chunk(i, text, embed_text, count_tokens(embed_text)))
        if all(c.tokens <= limit for c in chunks):
            return chunks
        budget = int(budget * 0.9)  # token counts are not exactly additive across joins; re-pack tighter
        if budget < MIN_BODY_TOKENS:
            break
    raise ChunkingError(f"job {job.get('id')!r}: could not build chunks within {limit} tokens")
