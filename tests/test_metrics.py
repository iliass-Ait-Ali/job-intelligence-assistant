"""IR / extraction metrics against values computed by hand."""

import math

import pytest

from job_intel.evaluation import metrics as M


def test_precision_recall_rr_ap_by_hand():
    ranked = [0, 2, None, 1, 0]  # None = unjudged, counts as non-relevant
    judged = [2, 1, 1, 0, 0]  # 3 relevant (>=1) in the pool
    assert M.precision_at_k(ranked, 5) == pytest.approx(2 / 5)
    assert M.precision_at_k(ranked, 5, t=2) == pytest.approx(1 / 5)
    assert M.recall_at_k(ranked, judged, 5) == pytest.approx(2 / 3)
    assert M.reciprocal_rank(ranked, 5) == pytest.approx(1 / 2)
    assert M.reciprocal_rank(ranked, 5, t=2) == pytest.approx(1 / 2)
    # AP: relevant at ranks 2 and 4 -> (1/2 + 2/4) / min(3, 5)
    assert M.average_precision(ranked, judged, 5) == pytest.approx((0.5 + 0.5) / 3)
    assert M.judged_at_k(ranked, 5) == pytest.approx(4 / 5)


def test_ndcg_by_hand_and_edge_cases():
    ranked, judged = [1, 2, 0], [2, 1, 0]
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert M.ndcg_at_k(ranked, judged, 3) == pytest.approx(dcg / idcg)
    assert M.ndcg_at_k([2, 1, 0], judged, 3) == pytest.approx(1.0)
    assert M.ndcg_at_k([0, 0], [0, 0], 2) is None  # nothing relevant: undefined, not 0
    assert M.recall_at_k([1], [0, 0], 5) is None


def test_bootstrap_is_deterministic_and_brackets_the_mean():
    vals = [0.2, 0.4, 0.6, 0.8, 1.0]
    mean, lo, hi = M.bootstrap_ci(vals, n=2000, seed=1)
    assert mean == pytest.approx(0.6) and lo <= mean <= hi
    assert M.bootstrap_ci(vals, n=2000, seed=1) == (mean, lo, hi)
    assert M.paired_bootstrap_p([1, 1, 1, 1], [0, 0, 0, 0], n=500) == 0.0


def test_prf_and_f1():
    tp, fp, fn = M.prf({"a", "b", "c"}, {"b", "c", "d", "e"})
    assert (tp, fp, fn) == (2, 1, 2)
    f = M.f1_from_counts(tp, fp, fn)
    assert f["precision"] == pytest.approx(2 / 3) and f["recall"] == pytest.approx(0.5)
    assert f["f1"] == pytest.approx(2 * (2 / 3) * 0.5 / (2 / 3 + 0.5))
    assert M.f1_from_counts(0, 0, 0) == {"precision": 0.0, "recall": 0.0, "f1": 0.0}
