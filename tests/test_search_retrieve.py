import pytest

pytest.importorskip("chromadb")

from search_data import jobs, make_job  # noqa: E402

from job_intel.search import retrieve  # noqa: E402
from job_intel.search.document import ChunkingConfig  # noqa: E402
from job_intel.search.embedder import HashingEmbedder  # noqa: E402
from job_intel.search.filters import JobFilter  # noqa: E402
from job_intel.search.index import IndexConfig, JobIndex  # noqa: E402
from job_intel.search.lexical import Bm25Index, best_snippet, tokenize  # noqa: E402
from job_intel.search.retrieve import BM25, SEMANTIC, Searcher, apply_company_cap  # noqa: E402

EMB = HashingEmbedder(max_seq_length=128)
CHUNKING = ChunkingConfig(max_tokens=100, overlap_tokens=20)


def dominant_job():
    """One posting whose ~60 chunks (one long paragraph each) outrank everything else for the test queries."""
    paragraph = "machine learning pipeline orchestration " * 12
    paragraphs = [f"{paragraph}step {i}" for i in range(60)]
    return make_job(50, "ML Pipeline Engineer", "Mega", "Berlin", "full_time", False, "\n\n".join(paragraphs))


def make_searcher(tmp_path, records=None):
    records = records or jobs()
    index = JobIndex.open_for_build(tmp_path / "idx", IndexConfig.build(EMB, CHUNKING))
    index.sync(records, EMB)
    return Searcher(index, EMB, {j["id"]: j for j in records})


@pytest.fixture
def searcher(tmp_path):
    return make_searcher(tmp_path)


def ids(resp):
    return [h.job.job_id for h in resp.hits]


# ---- semantic engine ----------------------------------------------------------------------


def test_semantic_search_ranks_the_matching_job_first_with_cosine_scores(searcher):
    resp = searcher.search("LLM inference latency and GPU performance", k=3)
    assert ids(resp)[0] == "arbeitnow:syn-01"
    top = resp.hits[0]
    assert top.score_kind == "cosine_similarity" and 0.0 < top.score <= 1.0
    assert top.distance == pytest.approx(1.0 - top.score)
    assert [h.score for h in resp.hits] == sorted((h.score for h in resp.hits), reverse=True)
    assert [h.rank for h in resp.hits] == [1, 2, 3]
    assert resp.eligible_jobs == resp.total_jobs == 13


def test_a_requirement_in_the_last_paragraph_of_a_long_posting_is_found(searcher):
    resp = searcher.search("kubernetes operators in Go", k=3)
    assert ids(resp)[0] == "arbeitnow:syn-09"
    assert "kubernetes" in resp.hits[0].snippet.lower()  # the matching chunk is shown, not the posting's start


def test_search_returns_distinct_jobs_even_when_one_job_dominates_the_top_chunks(tmp_path):
    # One posting with 80 near-identical chunks would fill any fixed-size chunk window on its own.
    dominant = dominant_job()
    s = make_searcher(tmp_path, [dominant, *jobs()])
    assert s.table[dominant["id"]].n_chunks > retrieve.INITIAL_CHUNKS
    resp = s.search("machine learning pipeline orchestration", k=6)
    assert len(ids(resp)) == len(set(ids(resp))) == 6
    assert resp.chunks_fetched > retrieve.INITIAL_CHUNKS  # it had to over-fetch
    assert ids(resp)[0] == dominant["id"]


def test_adaptive_overfetch_is_exact_against_exhaustive_ranking(tmp_path):
    dominant = dominant_job()
    s = make_searcher(tmp_path, [dominant, *jobs()])
    query = "machine learning pipeline data engineer spark"
    everything = s.search(query, k=len(s.table))  # asks for all jobs -> exhausts all chunks
    assert everything.exhausted and len(everything.hits) == len(s.table)
    assert ids(s.search(query, k=5)) == ids(everything)[:5]


def test_k_larger_than_corpus_returns_everything_and_says_so(searcher):
    resp = searcher.search("data", k=100)
    assert len(resp.hits) == 13 and resp.exhausted
    assert any("only 13 job(s)" in n for n in resp.notes)


def test_filters_are_applied_before_ranking_not_to_a_fixed_candidate_list(tmp_path):
    dominant = dominant_job()
    s = make_searcher(tmp_path, [dominant, *jobs()])
    # Globally the 80 'Mega' chunks (plus others) outrank the Lille job's single chunk by far.
    unfiltered = s.search("machine learning pipeline orchestration", k=13)
    lille = "arbeitnow:syn-12"
    assert ids(unfiltered).index(lille) > 5
    filtered = s.search("machine learning pipeline orchestration", k=3, flt=JobFilter(locations=("Lille",)))
    assert ids(filtered) == [lille] and filtered.eligible_jobs == 1


def test_job_type_unknown_and_include_unknown(searcher):
    interns = searcher.search("data", k=20, flt=JobFilter(job_types=("internship",)))
    assert set(ids(interns)) == {"arbeitnow:syn-02", "arbeitnow:syn-10", "arbeitnow:syn-13"}
    with_unknown = searcher.search("data", k=20, flt=JobFilter(job_types=("internship",), include_unknown=True))
    assert {"arbeitnow:syn-04", "arbeitnow:syn-06"} <= set(ids(with_unknown))
    assert set(ids(searcher.search("data", k=20, flt=JobFilter(job_types=("unknown",))))) == {"arbeitnow:syn-04", "arbeitnow:syn-06"}


def test_location_filter_is_spelling_insensitive_and_excludes_missing_locations(searcher):
    zurich = {"arbeitnow:syn-01", "arbeitnow:syn-11"}
    assert set(ids(searcher.search("engineer", k=20, flt=JobFilter(locations=("Zürich",))))) == zurich
    assert set(ids(searcher.search("engineer", k=20, flt=JobFilter(locations=("zurich",))))) == zurich
    nl = set(ids(searcher.search("engineer", k=20, flt=JobFilter(locations=("Nederland",)))))
    assert nl == {"arbeitnow:syn-03", "arbeitnow:syn-04"}
    assert "arbeitnow:syn-06" not in ids(searcher.search("research", k=20, flt=JobFilter(locations=("Berlin",))))
    assert "arbeitnow:syn-06" in ids(searcher.search("research", k=20, flt=JobFilter(locations=("Berlin",), include_unknown=True)))


def test_since_remote_and_source_filters(searcher):
    recent = searcher.search("data", k=20, flt=JobFilter.from_dict({"since": "2026-10-20"}))
    assert set(ids(recent)) == {"arbeitnow:syn-01", "arbeitnow:syn-02", "remotive:syn-05"}  # undated job excluded
    assert ids(searcher.search("data", k=20, flt=JobFilter(remote=True))) == ["remotive:syn-05"]
    assert "remotive:syn-05" not in ids(searcher.search("llm agents", k=20, flt=JobFilter(sources=("arbeitnow",))))


def test_filters_that_match_nothing_return_an_explained_empty_result(searcher):
    resp = searcher.search("data", flt=JobFilter(locations=("Atlantis",)))
    assert resp.hits == [] and resp.eligible_jobs == 0 and "no job satisfies" in resp.notes[0]
    assert searcher.search("data", engine=BM25, flt=JobFilter(locations=("Atlantis",))).hits == []


def test_company_cap_is_presentation_only_and_configurable(searcher):
    plain = searcher.search("machine learning engineer data pipelines", k=6)
    acme = [h for h in plain.hits if h.job.company == "Acme"]
    assert len(acme) >= 2 and plain.hidden_by_cap == 0
    capped = searcher.search("machine learning engineer data pipelines", k=6, max_per_company=1)
    assert sum(h.job.company == "Acme" for h in capped.hits) == 1 and capped.hidden_by_cap >= 1
    assert len(capped.hits) == 6  # over-fetched so the cap does not shrink the list
    two = searcher.search("machine learning engineer data pipelines", k=6, max_per_company=2)
    assert sum(h.job.company == "Acme" for h in two.hits) <= 2
    # the underlying ranking (no cap) is unchanged by earlier capped calls
    assert ids(searcher.search("machine learning engineer data pipelines", k=6)) == ids(plain)


def test_apply_company_cap_unit():
    from job_intel.search.filters import IndexedJob

    def j(i, company):
        return IndexedJob(f"s:{i}", "t", company, None, None, None, "s", "u", None)

    ranked = [(j(1, "A"), 0), (j(2, "a"), 0), (j(3, None), 0), (j(4, None), 0), (j(5, "B"), 0)]
    kept, hidden = apply_company_cap(ranked, 1)
    assert [x[0].job_id for x in kept] == ["s:1", "s:3", "s:4", "s:5"] and hidden == 1  # case-insensitive; no-company jobs never merged
    assert apply_company_cap(ranked, None) == (ranked, 0)


def test_unknown_engine_is_rejected(searcher):
    with pytest.raises(ValueError, match="engine"):
        searcher.search("x", engine="magic")


def test_missing_job_records_are_reported(tmp_path):
    records = jobs()
    index = JobIndex.open_for_build(tmp_path / "idx", IndexConfig.build(EMB, CHUNKING))
    index.sync(records, EMB)
    s = Searcher(index, EMB, {j["id"]: j for j in records[:-2]})
    assert any("missing from the jobs file" in n for n in s.search("data").notes)


# ---- BM25 ---------------------------------------------------------------------------------


def test_bm25_tokenizer_strips_accents_stopwords_and_short_tokens():
    assert tokenize("Les données à Zürich: ML & R, the Spark!") == ["donnees", "zurich", "ml", "spark"]


def test_bm25_ranks_rare_term_matches_first_and_skips_non_matches():
    docs = {"a": "data data data engineer", "b": "kafka streaming engineer", "c": "pastry chef"}
    bm = Bm25Index(docs)
    ranked = bm.search("kafka engineer", k=10)
    assert [d for d, _ in ranked] == ["b", "a"]  # 'c' shares no term -> not returned
    assert bm.search("kafka", k=10, eligible=["a", "c"]) == []
    assert bm.search("zzz", k=3) == []


def test_bm25_is_deterministic_on_ties():
    bm = Bm25Index({"b": "spark", "a": "spark"})
    assert [d for d, _ in bm.search("spark", k=2)] == ["a", "b"]


def test_bm25_engine_returns_bm25_scores_and_snippets(searcher):
    resp = searcher.search("kafka streaming", engine=BM25, k=5)
    assert ids(resp)[0] == "arbeitnow:syn-11" and resp.hits[0].score_kind == "bm25" and resp.hits[0].distance is None
    assert "Kafka" in resp.hits[0].snippet
    assert len(resp.hits) < 5 and any("share a term" in n for n in resp.notes)


def test_both_engines_see_the_same_eligible_candidates(searcher):
    flt = JobFilter(job_types=("internship",), locations=("Paris",))
    sem, lex = searcher.search("stage reporting", engine=SEMANTIC, k=10, flt=flt), searcher.search("stage reporting", engine=BM25, k=10, flt=flt)
    assert sem.eligible_jobs == lex.eligible_jobs == 2
    assert set(ids(sem)) <= {"arbeitnow:syn-02", "arbeitnow:syn-10"} and set(ids(lex)) <= {"arbeitnow:syn-02", "arbeitnow:syn-10"}


def test_best_snippet_prefers_the_paragraph_with_most_query_terms():
    text = "Intro about the company.\n\nWe need Spark and Airflow experience.\n\nBenefits."
    assert best_snippet(text, "spark airflow") == "We need Spark and Airflow experience."
    assert best_snippet(None, "x") == ""
    assert best_snippet("-\n\n- \nReal first line\nsecond", "nothing matches") == "- \nReal first line".split("\n")[1]  # bare bullets skipped
