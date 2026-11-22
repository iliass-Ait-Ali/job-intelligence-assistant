"""Search engines (keyword, FAISS, re-rank), scoring, skill gap, learning plan, market analytics."""

import pytest

from job_intel.match.gap import analyse_gap, implied_by, learning_plan
from job_intel.match.models import CandidateProfile, Evidence, Preferences, SkillItem
from job_intel.match.scoring import DEFAULT_WEIGHTS, load_weights, rank_by_score, score_match
from job_intel.search.keyword import KeywordIndex
from job_intel.search.rerank import rerank_jobs
from job_intel.search.retrieve import ALL_ENGINES

pytest.importorskip("chromadb")
pytest.importorskip("faiss")


def test_keyword_baseline_counts_distinct_terms_and_breaks_ties_by_title():
    idx = KeywordIndex({"a": "python sql python", "b": "python spark sql", "c": "java"}, {"a": "", "b": "python engineer", "c": ""})
    ranked = idx.search("python sql", k=10)
    assert [d for d, _ in ranked] == ["b", "a"]  # both have 2 terms; b also has one in its title
    assert idx.search("rust", k=10) == [] and idx.search("python", k=10, eligible={"c"}) == []


def test_faiss_matches_an_exact_scan_and_respects_prefilters(match_searcher):
    from job_intel.search.faiss_index import FaissChunkIndex

    fx = FaissChunkIndex(match_searcher.index)
    assert fx.count_chunks() == match_searcher.index.count_chunks()
    vec = match_searcher.embedder.embed_queries(["python machine learning"])[0]
    exact = match_searcher.index._exact_query(vec, 5, None)
    got = fx.query(vec, 5)
    assert [h.chunk_id for h in got] == [h.chunk_id for h in exact]
    assert all(abs(a.similarity - b.similarity) < 1e-5 for a, b in zip(got, exact))
    only = fx.query(vec, 50, job_ids={"arbeitnow:syn-03"})
    assert only and {h.job_id for h in only} == {"arbeitnow:syn-03"}
    assert fx.query(vec, 5, job_ids=set()) == []
    assert fx.chunks_of("arbeitnow:syn-03") == match_searcher.index.chunks_of("arbeitnow:syn-03")


def test_every_engine_returns_ranked_hits(platform_service):
    for engine in ALL_ENGINES:
        resp = platform_service.search("python machine learning engineer", engine=engine, k=3)
        assert 1 <= len(resp.hits) <= 3 and [h.rank for h in resp.hits] == list(range(1, len(resp.hits) + 1))


def test_rerank_scores_jobs_by_their_best_passage():
    class R:
        def score(self, q, passages):
            return [len(p) for p in passages]

    out = rerank_jobs(R(), "q", {"a": ["xx", "xxxxx"], "b": ["xxx"], "c": []})
    assert [(r.job_id, r.score) for r in out[:2]] == [("a", 5.0), ("b", 3.0)] and out[-1].job_id == "c"


@pytest.fixture(scope="module")
def ranked(platform_service, student_profile):
    prefs = Preferences(target_roles=["Data Scientist"], job_types=["internship", "full_time"], max_results=10, hard_constraints=[])
    return platform_service.recommend(student_profile, prefs)


def test_scores_are_bounded_explained_and_sorted(ranked, student_profile):
    assert ranked.ranked
    for _match, score in ranked.ranked:
        assert 0 <= score.score <= 100 and 0 < score.coverage <= 1
        assert {c.name for c in score.components} == {*DEFAULT_WEIGHTS} - {"contract"} | {"contract"}
        known = [c for c in score.components if c.known]
        assert sum(score.contribution(c.name) for c in known) == pytest.approx(score.score, abs=0.2)
        assert score.ranking_score == pytest.approx(score.coverage * score.score + (1 - score.coverage) * 50, abs=0.2)
    keys = [s.ranking_score for _, s in ranked.ranked]
    assert keys == sorted(keys, reverse=True)


def test_unknown_components_do_not_penalise(ranked, student_profile):
    match, base = ranked.ranked[0]
    w = dict(DEFAULT_WEIGHTS)
    unknown = [c.name for c in base.components if not c.known]
    if unknown:
        w[unknown[0]] = 10.0  # weight on an unobservable criterion changes coverage, not the observed score
        changed = score_match(match, student_profile, w)
        assert changed.score == pytest.approx(base.score, abs=0.1) and changed.coverage < base.coverage


def test_weights_file_is_validated(tmp_path):
    good = tmp_path / "w.yaml"
    good.write_text("weights:\n  semantic: 0.5\n", encoding="utf-8")
    assert load_weights(good)["semantic"] == 0.5 and load_weights(None) == DEFAULT_WEIGHTS
    bad = tmp_path / "bad.yaml"
    bad.write_text("weights:\n  salary: 1\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_weights(bad)


def _profile(*skills, context="listed"):
    return CandidateProfile(skills=[SkillItem(name=s, context=context, evidence=[Evidence(text=f"used {s}")]) for s in skills])


def test_implication_is_directional():
    assert implied_by(["PostgreSQL"])["SQL"] == "PostgreSQL"
    assert "PostgreSQL" not in implied_by(["SQL"])


def test_gap_statuses_and_declarations(ranked):
    match = ranked.ranked[0][0]
    names = [s.skill for s in match.skills]
    profile = _profile(names[0])
    report = analyse_gap(profile, match, declared={names[-1]: "have"} if len(names) > 1 else {})
    statuses = {i.skill: i.status for i in report.items}
    assert statuses[names[0]] == "listed"
    if len(names) > 1:
        assert statuses[names[-1]] == "self_declared"
    lacking = analyse_gap(_profile(), match, declared={n: "lack" for n in names})
    assert {i.status for i in lacking.items} <= {"absent", "implied", "transferable"}
    unknown = analyse_gap(_profile(), match)
    assert all(i.status == "not_mentioned" and i.hint for i in unknown.items)


def test_learning_plan_puts_prerequisites_first(ranked):
    from job_intel.match.gap import GapReport, SkillGap

    rep = GapReport("j", "Platform Engineer", "X", [SkillGap("Kubernetes", "not_mentioned", "required", "k8s"),
                                                     SkillGap("Docker", "not_mentioned", "preferred", "docker"),
                                                     SkillGap("Python", "listed", "required", "py")])  # fmt: skip
    plan = learning_plan([(rep, 80.0)], {"Kubernetes": 0.5, "Docker": 0.1})
    order = [s.skill for s in plan]
    assert order.index("Docker") < order.index("Kubernetes") and "Python" not in order
    assert plan[0].resources and plan[0].project


def test_market_analytics(platform_service):
    from job_intel.market import analytics as ma

    df = platform_service.market_df
    demand = ma.skill_demand(df, top=5)
    assert demand["share"].between(0, 1).all() and demand["postings"].is_monotonic_decreasing
    co = ma.cooccurrence(df, top_skills=10, min_count=1)
    assert (co["lift"] > 0).all()
    m = ma.cooccurrence_matrix(df, top_skills=5)
    assert all(m.loc[s, s] == 1.0 for s in m.index)
    seg = ma.filter_frame(df, contract_type=["internship"])
    assert set(seg["contract_type"]) == {"internship"}
    assert ma.counts_by(df, "role_family")["postings"].sum() >= len(df)
    trend = ma.weekly_trend(df, ["Python"])
    assert trend["postings"].sum() == len(df)
    assert ma.snapshot_comparison(df, df)["change_pts"].abs().max() == 0


def test_rank_by_score_is_deterministic(ranked, student_profile):
    matches = [m for m, _ in ranked.ranked]
    a = [m.job.job_id for m, _ in rank_by_score(matches, student_profile)]
    b = [m.job.job_id for m, _ in rank_by_score(list(reversed(matches)), student_profile)]
    assert a == b
