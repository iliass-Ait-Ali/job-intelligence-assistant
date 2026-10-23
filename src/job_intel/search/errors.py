"""Errors the search layer raises on purpose; the CLI turns them into one-line messages."""

from __future__ import annotations


class SearchError(Exception):
    """Base class: an expected, user-actionable failure."""

    exit_code = 2


class DependencyError(SearchError):
    """An optional dependency (sentence-transformers, chromadb) is missing or broken."""


class ModelLoadError(SearchError):
    """The embedding model could not be loaded (offline, wrong name, corrupt cache...)."""


class IndexNotFoundError(SearchError):
    """No index has been built yet."""


class IndexConfigMismatch(SearchError):
    """The index was built with different model/chunking settings than requested."""


class JobsFileError(SearchError):
    """The processed jobs file is missing or malformed."""


class ChunkingError(SearchError):
    """A chunk would exceed the embedding model's input limit."""


class EvalError(SearchError):
    """Evaluation inputs (queries, judgments) are missing or invalid."""
