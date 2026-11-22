"""Keyword-matching baseline: the simplest thing a job board does.

score(q, d) = number of DISTINCT query terms that occur in the document (coordinate matching),
tie-broken by the number of query terms found in the title. No term weighting (unlike BM25's IDF),
no length normalisation, no synonyms. It is the floor every other system must beat; if BM25 does
not beat it, the IDF / saturation machinery is not earning its keep on this corpus.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

from job_intel.search.lexical import tokenize


class KeywordIndex:
    def __init__(self, documents: Mapping[str, str], titles: Mapping[str, str]) -> None:
        self._terms = {doc_id: frozenset(tokenize(text)) for doc_id, text in documents.items()}
        self._title_terms = {doc_id: frozenset(tokenize(titles.get(doc_id, ""))) for doc_id in documents}

    def search(self, query: str, k: int, eligible: Collection[str] | None = None) -> list[tuple[str, float]]:
        terms = set(tokenize(query))
        if not terms:
            return []
        allowed = None if eligible is None else set(eligible)
        scored = []
        for doc_id, doc_terms in self._terms.items():
            if allowed is not None and doc_id not in allowed:
                continue
            overlap = len(terms & doc_terms)
            if overlap:
                title_overlap = len(terms & self._title_terms[doc_id])
                scored.append((doc_id, overlap + title_overlap / 100.0))
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:k]
