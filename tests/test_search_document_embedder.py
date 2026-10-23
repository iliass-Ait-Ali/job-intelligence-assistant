import sys
import types

import numpy as np
import pytest
from search_data import jobs

from job_intel.search.document import ChunkingConfig, build_chunks, job_header
from job_intel.search.embedder import HashingEmbedder, SentenceTransformerEmbedder, load_embedder, prefixes_for
from job_intel.search.errors import ChunkingError, DependencyError, ModelLoadError, SearchError

EMB = HashingEmbedder(max_seq_length=128)
CFG = ChunkingConfig(max_tokens=100, overlap_tokens=20)


def chunks_of(job, cfg=CFG, count=EMB.count_passage_tokens, limit=128):
    return build_chunks(job, count, cfg, limit)


def test_header_carries_title_company_location_type_and_remote():
    job = {"title": "AI Engineer", "company": "Delta", "location": "Remote job", "job_type": "full_time", "remote": True}
    assert job_header(job) == "AI Engineer | Delta | Remote job | full time | remote"
    assert job_header({"title": "T"}) == "T"


def test_every_chunk_is_within_the_limit_and_starts_with_the_header():
    job = jobs()[8]  # the long posting
    chunks = chunks_of(job)
    assert len(chunks) > 3
    header = job_header(job)
    for chunk in chunks:
        assert chunk.embed_text.startswith(header)
        assert chunk.tokens == EMB.count_passage_tokens(chunk.embed_text) <= 100
    assert [c.index for c in chunks] == list(range(len(chunks)))


def test_no_text_is_lost_the_requirements_at_the_end_survive():
    job = jobs()[8]
    joined = "\n".join(c.text for c in chunks_of(job))
    assert "kubernetes operators in Go" in joined
    for paragraph in job["description"].split("\n\n"):
        assert paragraph in joined


def test_consecutive_chunks_overlap():
    chunks = chunks_of(jobs()[8], cfg=ChunkingConfig(max_tokens=100, overlap_tokens=40))
    shared = [set(a.text.split("\n")) & set(b.text.split("\n")) for a, b in zip(chunks, chunks[1:])]
    assert any(shared)


def test_short_and_missing_descriptions_give_one_chunk():
    one = chunks_of(jobs()[0])
    assert len(one) == 1 and one[0].text == jobs()[0]["description"]
    empty = chunks_of(jobs()[12])  # description None
    assert len(empty) == 1 and empty[0].text == "" and empty[0].embed_text == job_header(jobs()[12])


def test_single_paragraph_longer_than_the_budget_is_split_without_loss():
    words = [f"w{i}" for i in range(400)]
    job = {"id": "x", "title": "T", "description": " ".join(words)}
    chunks = chunks_of(job)
    assert all(c.tokens <= 100 for c in chunks) and len(chunks) > 3
    assert {w for c in chunks for w in c.text.split()} == set(words)


def test_a_token_longer_than_the_budget_is_cut_by_characters():
    job = {"id": "x", "title": "T", "description": "before " + "A" * 5000 + " after"}
    chunks = chunks_of(job, count=lambda t: len(t) // 4 + 2)  # 4 chars per token
    assert all(c.tokens <= 100 for c in chunks)
    assert "".join(c.text for c in chunks).count("A") >= 5000  # nothing dropped (overlap may repeat)


def test_repacks_when_token_counts_are_not_additive():
    # Newlines cost 3 tokens each: unit-wise additive estimates are too optimistic, the final check forces a re-pack.
    count = lambda t: EMB.count_passage_tokens(t) + 3 * t.count("\n")  # noqa: E731
    chunks = chunks_of(jobs()[8], count=count)
    assert all(c.tokens <= 100 for c in chunks)


def test_header_too_long_is_an_error_not_a_truncation():
    with pytest.raises(ChunkingError, match="header"):
        build_chunks({"id": "x", "title": "word " * 60, "description": "d"}, EMB.count_passage_tokens, ChunkingConfig(60, 10), 128)


def test_limit_is_the_smaller_of_config_and_model_window():
    chunks = build_chunks(jobs()[8], EMB.count_passage_tokens, ChunkingConfig(500, 20), 128)
    assert max(c.tokens for c in chunks) <= 128


# ---- embedders ----------------------------------------------------------------------------


def test_hashing_embedder_is_deterministic_normalized_and_float32():
    a = EMB.embed_passages(["Machine learning engineer", "data analyst"])
    b = EMB.embed_passages(["Machine learning engineer", "data analyst"])
    assert a.dtype == np.float32 and np.array_equal(a, b)
    assert np.allclose(np.linalg.norm(a, axis=1), 1.0, atol=1e-5)


def test_hashing_embedder_similarity_reflects_shared_words_and_accents():
    q = EMB.embed_queries(["data engineer spark"])[0]
    close, far = EMB.embed_passages(["spark data engineer pipelines", "pastry chef bakery"])
    assert float(q @ close) > float(q @ far)
    assert float(EMB.embed_queries(["Zürich"])[0] @ EMB.embed_queries(["zurich"])[0]) == pytest.approx(1.0, abs=1e-5)


def test_embedder_refuses_inputs_over_the_window_instead_of_truncating():
    with pytest.raises(ChunkingError):
        EMB.embed_passages(["word " * 300])


def test_e5_prefixes():
    assert prefixes_for("intfloat/multilingual-e5-base") == ("query: ", "passage: ")
    assert prefixes_for("some/other-model") == ("", "")


def test_load_embedder_returns_the_test_embedder_by_name():
    assert isinstance(load_embedder("hashing-test-v1"), HashingEmbedder)


def test_missing_sentence_transformers_gives_an_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)  # makes `import` raise ImportError
    with pytest.raises(DependencyError, match=r"pip install -e"):
        SentenceTransformerEmbedder("any/model")


def test_model_download_failure_is_reported_cleanly(monkeypatch):
    def broken(*_a, **_k):
        raise OSError("offline: cannot reach huggingface.co")

    fake = types.SimpleNamespace(SentenceTransformer=broken)
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake)
    with pytest.raises(ModelLoadError, match="huggingface.co") as info:
        SentenceTransformerEmbedder("any/model")
    assert "HF_HUB_OFFLINE" in str(info.value)
    assert isinstance(info.value, SearchError)
