"""BM25 lexical baseline, implemented here (no dependency) so it is easy to read and test.

Document = title, company, location, job type and description: the same facts the embedder sees.
Tokenization: accent-stripped, lower-cased alphanumeric words of 2+ chars, minus a small
EN/FR/DE/NL stopword list. IDF statistics are computed over the whole indexed corpus; eligibility
filters only restrict which documents may be returned. Documents sharing no query term score 0
and are not returned, so BM25 can return fewer than k results.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter, defaultdict
from collections.abc import Collection, Iterable, Mapping
from typing import Any

STOPWORDS = frozenset(
    """a an the and or of to in on for with at by from as is are be this that it its we you your our us
    le la les un une des du de et ou en pour avec sur dans au aux par est sont vous nous votre notre
    der die das und oder ein eine einer mit fur von zu im den dem ist sind wir sie ihr ihre auf bei
    het een en van voor met op bij ons je jij wij naar als zijn""".split()
)


def tokenize(text: str) -> list[str]:
    text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
    return [t for t in re.findall(r"[a-z0-9]+", text.casefold()) if len(t) >= 2 and t not in STOPWORDS]


def job_document(job: Mapping[str, Any]) -> str:
    fields = (job.get("title"), job.get("company"), job.get("location"), job.get("job_type"), job.get("description"))
    return "\n".join(str(f) for f in fields if f)


class Bm25Index:
    def __init__(self, documents: Mapping[str, str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._lengths: dict[str, int] = {}
        for doc_id, text in documents.items():
            tokens = tokenize(text)
            self._lengths[doc_id] = len(tokens)
            for term, tf in Counter(tokens).items():
                self._postings[term][doc_id] = tf
        self.n_docs = len(self._lengths)
        self._avg_len = (sum(self._lengths.values()) / self.n_docs) if self.n_docs else 0.0

    def _idf(self, term: str) -> float:
        df = len(self._postings.get(term, ()))
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score_all(self, query: str) -> dict[str, float]:
        scores: dict[str, float] = defaultdict(float)
        for term in set(tokenize(query)):
            idf = self._idf(term)
            for doc_id, tf in self._postings.get(term, {}).items():
                norm = 1 - self.b + self.b * self._lengths[doc_id] / (self._avg_len or 1.0)
                scores[doc_id] += idf * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return scores

    def search(self, query: str, k: int, eligible: Collection[str] | None = None) -> list[tuple[str, float]]:
        allowed = None if eligible is None else set(eligible)
        ranked: Iterable[tuple[str, float]] = (
            (d, s) for d, s in self.score_all(query).items() if allowed is None or d in allowed
        )
        return sorted(ranked, key=lambda pair: (-pair[1], pair[0]))[:k]


def best_snippet(description: str | None, query: str, max_chars: int = 300) -> str:
    """The line/paragraph sharing the most query terms (the first one if none do)."""
    paragraphs = [p.strip() for p in re.split(r"\n+", description or "") if len(re.findall(r"\w", p)) >= 3]
    if not paragraphs:
        return ""
    terms = set(tokenize(query))
    best = max(paragraphs, key=lambda p: len(terms & set(tokenize(p))))
    return best[:max_chars] + ("..." if len(best) > max_chars else "")
