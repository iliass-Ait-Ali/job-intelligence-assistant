"""Embedding back-ends: a real SentenceTransformers model, and a deterministic one for tests.

Both return L2-normalized float32 vectors, so cosine similarity is a dot product and
ChromaDB's cosine distance equals ``1 - similarity``.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from typing import Protocol

import numpy as np

from job_intel.search.errors import ChunkingError, DependencyError, ModelLoadError, SearchError

log = logging.getLogger(__name__)

DEFAULT_MODEL = "intfloat/multilingual-e5-base"


class Embedder(Protocol):
    name: str
    dim: int
    max_seq_length: int
    passage_prefix: str
    query_prefix: str

    def count_passage_tokens(self, text: str) -> int:
        """Tokens the model will see for this passage (prefix and special tokens included)."""

    def embed_passages(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def describe(self) -> str: ...


def prefixes_for(model_name: str) -> tuple[str, str]:
    """(query_prefix, passage_prefix). E5 models are trained with 'query: ' / 'passage: '."""
    return ("query: ", "passage: ") if "e5" in model_name.lower() else ("", "")


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str = DEFAULT_MODEL, *, device: str = "cpu", batch_size: int = 16) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise DependencyError(
                f'sentence-transformers could not be imported ({type(exc).__name__}: {exc}). '
                'If it is not installed, run: python -m pip install -e ".[search]"'
            ) from exc
        self.name = model_name
        self.batch_size = batch_size
        self.device = device
        self.query_prefix, self.passage_prefix = prefixes_for(model_name)
        try:
            self._model = SentenceTransformer(model_name, device=device)
        except Exception as exc:
            raise ModelLoadError(
                f"could not load embedding model {model_name!r}: {type(exc).__name__}: {exc}\n"
                "The first run downloads the model from huggingface.co (about 1.1 GB for the default). "
                "Check your connection, or pre-download it and set HF_HUB_OFFLINE=1."
            ) from exc
        self.max_seq_length = int(self._model.max_seq_length)
        dim_getter = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self.dim = int(dim_getter())
        self._tokenizer = self._model.tokenizer
        self._tokenizer.model_max_length = 10**9  # we measure lengths ourselves; silence the length warning

    def _token_count(self, text: str) -> int:
        return len(self._tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])

    def count_passage_tokens(self, text: str) -> int:
        return self._token_count(self.passage_prefix + text)

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        for text in texts:
            if self._token_count(text) > self.max_seq_length:
                raise ChunkingError(
                    f"input has more than {self.max_seq_length} tokens and would be silently truncated"
                )
        vectors = self._model.encode(
            texts, batch_size=self.batch_size, normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )  # fmt: skip
        return np.asarray(vectors, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode([self.passage_prefix + t for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        try:
            return self._encode([self.query_prefix + t for t in texts])
        except ChunkingError as exc:
            raise SearchError(f"query is too long for the model: {exc}") from exc

    def describe(self) -> str:
        import torch

        return (
            f"{self.name} (dim={self.dim}, max_seq_length={self.max_seq_length}, device={self.device}, "
            f"torch {torch.__version__}, {torch.get_num_threads()} CPU threads)"
        )


class HashingEmbedder:
    """Deterministic bag-of-words hashing embedder. FOR TESTS AND OFFLINE SMOKE RUNS ONLY.

    Texts that share words get a high cosine similarity, which is enough to test ranking,
    filtering and index logic without downloading a model. It has no semantic knowledge.
    """

    name = "hashing-test-v1"
    query_prefix = ""
    passage_prefix = ""

    def __init__(self, dim: int = 1024, max_seq_length: int = 128) -> None:
        self.dim = dim
        self.max_seq_length = max_seq_length

    @staticmethod
    def _words(text: str) -> list[str]:
        text = "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))
        return re.findall(r"\w+", text.casefold())

    def count_passage_tokens(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text)) + 2

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for word in self._words(text):
            digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
            value = int.from_bytes(digest, "big")
            vec[value % self.dim] += 1.0 if (value >> 40) & 1 else -1.0
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            vec[0], norm = 1.0, 1.0
        return vec / norm

    def _encode(self, texts: list[str]) -> np.ndarray:
        for text in texts:
            if self.count_passage_tokens(text) > self.max_seq_length:
                raise ChunkingError(f"input exceeds {self.max_seq_length} tokens")
        return np.stack([self._vector(t) for t in texts]) if texts else np.zeros((0, self.dim), dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def describe(self) -> str:
        return f"{self.name} (dim={self.dim}, TEST ONLY - no semantic knowledge)"


def load_embedder(model_name: str, *, device: str = "cpu", batch_size: int = 16) -> Embedder:
    if model_name == HashingEmbedder.name:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(model_name, device=device, batch_size=batch_size)
