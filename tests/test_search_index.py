import copy

import numpy as np
import pytest

pytest.importorskip("chromadb")

from search_data import jobs  # noqa: E402

from job_intel.search.document import ChunkingConfig, build_chunks  # noqa: E402
from job_intel.search.embedder import HashingEmbedder  # noqa: E402
from job_intel.search.errors import IndexConfigMismatch, IndexNotFoundError, SearchError  # noqa: E402
from job_intel.search.index import IndexConfig, JobIndex, content_hash  # noqa: E402

EMB = HashingEmbedder(max_seq_length=128)
CHUNKING = ChunkingConfig(max_tokens=100, overlap_tokens=20)


class OtherModel(HashingEmbedder):
    name = "hashing-other-v2"


@pytest.fixture
def index_path(tmp_path):
    return tmp_path / "index"


def open_build(path, embedder=EMB, chunking=CHUNKING, rebuild=False):
    return JobIndex.open_for_build(path, IndexConfig.build(embedder, chunking), rebuild=rebuild)


def chunk_ids(index, job_id):
    page = index._col.get(include=["metadatas"])
    return sorted(i for i, m in zip(page["ids"], page["metadatas"]) if m["job_id"] == job_id)


def test_first_build_indexes_every_job_with_stable_chunk_ids(index_path):
    index = open_build(index_path)
    report = index.sync(jobs(), EMB)
    assert (report.added, report.updated, report.unchanged, report.removed) == (13, 0, 0, 0)
    assert report.total_jobs == 13 and report.total_chunks > 13 and report.embedded_chunks == report.total_chunks
    assert report.chunk_token_stats["max"] <= 100
    assert chunk_ids(index, "arbeitnow:syn-01") == ["arbeitnow:syn-01#0"]
    assert len(chunk_ids(index, "arbeitnow:syn-09")) > 3


def test_stored_metadata_has_no_nulls_and_expected_types(index_path):
    index = open_build(index_path)
    index.sync(jobs(), EMB)
    page = index._col.get(include=["metadatas"])
    for meta in page["metadatas"]:
        assert all(v is not None for v in meta.values())
        assert isinstance(meta["chunk_index"], int) and isinstance(meta["n_chunks"], int)
    undated = next(m for m in page["metadatas"] if m["job_id"] == "arbeitnow:syn-06")
    assert "published_ts" not in undated and "location" not in undated and "job_type" not in undated
    table = index.job_table()
    assert table["arbeitnow:syn-06"].published_ts is None and table["arbeitnow:syn-06"].location is None
    assert table["remotive:syn-05"].remote is True and table["arbeitnow:syn-02"].published_date.isoformat() == "2026-10-21"


def test_second_sync_is_a_no_op(index_path):
    index = open_build(index_path)
    first = index.sync(jobs(), EMB)
    second = index.sync(jobs(), EMB)
    assert (second.added, second.updated, second.removed, second.unchanged) == (0, 0, 0, 13)
    assert second.embedded_chunks == 0 and second.total_chunks == first.total_chunks == index.count_chunks()


def test_reopening_the_index_does_not_duplicate_records(index_path):
    open_build(index_path).sync(jobs(), EMB)
    again = open_build(index_path)
    report = again.sync(jobs(), EMB)
    assert report.unchanged == 13 and report.embedded_chunks == 0


def test_changed_job_is_replaced_and_old_chunks_are_removed(index_path):
    index = open_build(index_path)
    index.sync(jobs(), EMB)
    before = index.count_chunks()
    edited = copy.deepcopy(jobs())
    edited[8]["description"] = "Short new description."  # 25-chunk posting becomes one chunk
    report = index.sync(edited, EMB)
    assert (report.updated, report.unchanged) == (1, 12)
    assert chunk_ids(index, "arbeitnow:syn-09") == ["arbeitnow:syn-09#0"]
    assert report.embedded_chunks == 1 and index.count_chunks() < before


def test_metadata_only_change_triggers_an_update(index_path):
    index = open_build(index_path)
    index.sync(jobs(), EMB)
    edited = copy.deepcopy(jobs())
    edited[0]["job_type"] = "contract"
    assert index.sync(edited, EMB).updated == 1
    assert index.job_table()["arbeitnow:syn-01"].job_type == "contract"


def test_removed_jobs_are_deleted(index_path):
    index = open_build(index_path)
    index.sync(jobs(), EMB)
    report = index.sync(jobs()[:5], EMB)
    assert report.removed == 8 and report.unchanged == 5 and set(index.job_table()) == {j["id"] for j in jobs()[:5]}
    assert chunk_ids(index, "arbeitnow:syn-09") == []


def test_job_left_half_written_by_an_interrupted_run_is_repaired(index_path):
    index = open_build(index_path)
    index.sync(jobs(), EMB)
    ids = chunk_ids(index, "arbeitnow:syn-09")
    index._col.delete(ids=ids[2:])  # simulate a crash after two chunks
    report = index.sync(jobs(), EMB)
    assert report.updated == 1 and chunk_ids(index, "arbeitnow:syn-09") == ids


def test_duplicate_ids_in_the_input_are_counted_and_not_indexed_twice(index_path):
    index = open_build(index_path)
    report = index.sync([*jobs(), jobs()[0]], EMB)
    assert report.duplicate_inputs == 1 and report.total_jobs == 13
    assert len(chunk_ids(index, "arbeitnow:syn-01")) == 1


def test_records_without_id_or_title_are_rejected(index_path):
    with pytest.raises(SearchError, match="id/title"):
        open_build(index_path).sync([{"title": "no id"}], EMB)


def test_content_hash_ignores_collected_at_but_not_description():
    a, b = copy.deepcopy(jobs()[0]), copy.deepcopy(jobs()[0])
    b["collected_at"] = "2030-01-01T00:00:00+00:00"
    assert content_hash(a) == content_hash(b)
    b["description"] += "!"
    assert content_hash(a) != content_hash(b)


# ---- configuration validation -------------------------------------------------------------


def test_different_model_is_refused_with_an_explanation(index_path):
    open_build(index_path).sync(jobs(), EMB)
    with pytest.raises(IndexConfigMismatch, match="model_name") as info:
        open_build(index_path, embedder=OtherModel(max_seq_length=128))
    assert "--rebuild" in str(info.value)


def test_different_chunking_or_dimension_is_refused(index_path):
    open_build(index_path).sync(jobs(), EMB)
    with pytest.raises(IndexConfigMismatch, match="chunk_max_tokens"):
        open_build(index_path, chunking=ChunkingConfig(80, 20))
    with pytest.raises(IndexConfigMismatch, match="embedding_dim"):
        open_build(index_path, embedder=HashingEmbedder(dim=128, max_seq_length=128))


def test_rebuild_replaces_the_index_and_settings(index_path):
    open_build(index_path).sync(jobs(), EMB)
    rebuilt = open_build(index_path, chunking=ChunkingConfig(80, 20), rebuild=True)
    assert rebuilt.count_chunks() == 0
    report = rebuilt.sync(jobs(), EMB)
    assert report.added == 13 and JobIndex.open_existing(index_path).config.chunk_max_tokens == 80


def test_open_existing_returns_the_stored_config_and_complains_when_missing(index_path, tmp_path):
    with pytest.raises(IndexNotFoundError, match="jobintel index"):
        JobIndex.open_existing(index_path)
    open_build(index_path).sync(jobs(), EMB)
    config = JobIndex.open_existing(index_path).config
    assert config.model_name == "hashing-test-v1" and config.distance == "cosine" and config.chunk_overlap_tokens == 20
    with pytest.raises(IndexNotFoundError):
        JobIndex.open_existing(tmp_path / "elsewhere")


# ---- distances ----------------------------------------------------------------------------


def test_chroma_distance_is_one_minus_cosine_similarity(index_path):
    index = open_build(index_path)
    index.sync(jobs(), EMB)
    job = jobs()[0]
    chunk = build_chunks(job, EMB.count_passage_tokens, CHUNKING, 128)[0]
    stored = EMB.embed_passages([chunk.embed_text])[0]
    query = EMB.embed_queries(["machine learning gpu pytorch"])[0]

    exact_self = index.query(stored, 1)[0]
    assert exact_self.chunk_id == "arbeitnow:syn-01#0"
    assert exact_self.distance == pytest.approx(0.0, abs=1e-4) and exact_self.similarity == pytest.approx(1.0, abs=1e-4)

    hit = next(h for h in index.query(query, 50) if h.chunk_id == "arbeitnow:syn-01#0")
    assert hit.similarity == pytest.approx(float(query @ stored), abs=1e-4)
    assert hit.distance == pytest.approx(1.0 - float(query @ stored), abs=1e-4)
    assert hit.similarity == pytest.approx(1.0 - hit.distance)


def test_query_prefilter_restricts_candidates_before_the_vector_search(index_path):
    index = open_build(index_path)
    index.sync(jobs(), EMB)
    query = EMB.embed_queries(["kubernetes operators"])[0]
    assert {h.job_id for h in index.query(query, 50, job_ids=["arbeitnow:syn-02"])} == {"arbeitnow:syn-02"}
    assert index.query(query, 5, job_ids=[]) == []  # empty eligibility must not crash Chroma


def test_chroma_ranking_matches_brute_force_cosine(index_path):
    index = open_build(index_path)
    index.sync(jobs(), EMB)
    texts, ids = [], []
    for job in jobs():
        for c in build_chunks(job, EMB.count_passage_tokens, CHUNKING, 128):
            texts.append(c.embed_text)
            ids.append(f"{job['id']}#{c.index}")
    matrix = EMB.embed_passages(texts)
    for q in ("spark airflow pipelines", "stage reporting power bi", "llm gpu inference"):
        qv = EMB.embed_queries([q])[0]
        sims = matrix @ qv
        top = np.argsort(-sims, kind="stable")[:10]
        hits = index.query(qv, 10)
        # Scores must agree exactly (within float error); ids only where there is no tie (many chunks score 0).
        assert [h.similarity for h in hits] == pytest.approx([float(sims[i]) for i in top], abs=1e-4)
        positive = [ids[i] for i in top if sims[i] > 1e-6]
        assert [h.chunk_id for h in hits][: len(positive)] == positive
