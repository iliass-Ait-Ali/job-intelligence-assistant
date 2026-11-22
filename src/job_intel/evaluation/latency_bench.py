"""Latency and resource benchmark of every pipeline step on the live corpus (Module evaluation, section 5).

Cold steps are measured once (model loading, index construction). Warm steps are measured over repeated
calls and reported as median / p95. GPU memory is torch's peak allocation; process memory is the resident
set size when psutil is available. Also compares the vector backends: Chroma HNSW (approximate) vs FAISS
IndexFlatIP (exact), as top-10 agreement and latency over the evaluation queries.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path
from statistics import median

from job_intel.config import Settings
from job_intel.evaluation.data import load_queries, write_report

ROOT = Path(__file__).resolve().parents[3]


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def _timeit(fn, n: int) -> dict[str, float]:
    times = []
    for _ in range(n):
        started = time.perf_counter()
        fn()
        times.append(1000 * (time.perf_counter() - started))
    return {"median_ms": round(median(times), 1), "p95_ms": round(_p95(times), 1), "runs": n}


def run(settings: Settings) -> dict:
    import torch

    from job_intel.match.cv_parser import parse_cv
    from job_intel.match.cv_text import from_file
    from job_intel.match.models import Preferences
    from job_intel.search import service as search_service
    from job_intel.search.embedder import resolve_device
    from job_intel.search.faiss_index import FaissChunkIndex
    from job_intel.search.rerank import CrossEncoderReranker
    from job_intel.search.retrieve import Searcher
    from job_intel.services import JobIntelService

    device = resolve_device(settings.device)
    steps: dict[str, dict] = {}
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    base, _ = search_service.open_searcher(settings)
    steps["load embedding model + open Chroma (cold)"] = {"ms": round(1000 * (time.perf_counter() - started))}
    started = time.perf_counter()
    faiss_index = FaissChunkIndex(base.index)
    steps["build FAISS flat index from stored vectors (cold)"] = {"ms": round(1000 * (time.perf_counter() - started)),
                                                                  "chunks": faiss_index.count_chunks()}  # fmt: skip
    started = time.perf_counter()
    reranker = CrossEncoderReranker(device=device)
    steps["load cross-encoder (cold)"] = {"ms": round(1000 * (time.perf_counter() - started))}
    jobs = {jid: base.record(jid) for jid in base.table}
    fs = Searcher(faiss_index, base.embedder, jobs, reranker=reranker)
    queries = [q["text"] for q in load_queries(settings)]
    for engine in ("keyword", "bm25", "semantic", "hybrid", "hybrid_rerank"):
        fs.search(queries[0], engine=engine, k=10)  # warm-up
        i = iter(queries * 3)
        steps[f"search · {engine} (warm)"] = _timeit(lambda: fs.search(next(i), engine=engine, k=10), len(queries) * 2)
    i = iter(queries * 3)
    steps["embed one query"] = _timeit(lambda: base.embedder.embed_queries([next(i)]), 20)

    # vector backends: exact FAISS vs approximate HNSW (Chroma)
    agree, t_faiss, t_hnsw = [], [], []
    for q in queries:
        v = base.embedder.embed_queries([q])[0]
        s0 = time.perf_counter()
        a = [h.chunk_id for h in faiss_index.query(v, 10)]
        t_faiss.append(1000 * (time.perf_counter() - s0))
        s0 = time.perf_counter()
        b = [h.chunk_id for h in base.index.query(v, 10)]
        t_hnsw.append(1000 * (time.perf_counter() - s0))
        agree.append(len(set(a) & set(b)) / 10)
    backends = {"faiss_flat_ms_median": round(median(t_faiss), 2), "chroma_hnsw_ms_median": round(median(t_hnsw), 2),
                "hnsw_recall@10_vs_exact": round(sum(agree) / len(agree), 3)}  # fmt: skip

    cv_path = ROOT / "eval" / "synthetic_cvs" / "cv_fr_data_engineer.txt"
    steps["read + parse CV"] = _timeit(lambda: parse_cv(from_file(cv_path).text), 10)
    svc = JobIntelService(settings)
    svc._searcher = fs
    profile = parse_cv(cv_path.read_text(encoding="utf-8"))
    prefs = Preferences(**json.loads(cv_path.with_suffix(".prefs.json").read_text(encoding="utf-8")))
    started = time.perf_counter()
    recs = svc.recommend(profile, prefs)
    steps["recommendations (cold: requirement extraction for all jobs)"] = {"ms": round(1000 * (time.perf_counter() - started))}
    steps["recommendations (warm)"] = _timeit(lambda: svc.recommend(profile, prefs), 5)
    steps["skill gap + learning plan"] = _timeit(lambda: svc.plan(profile, recs.ranked), 5)
    session = svc.assistant_session(profile, recs.ranked)
    steps["assistant answer (deterministic)"] = _timeit(lambda: svc.ask(session, "Why does job 1 match my CV?"), 10)
    ok, _ = svc.llm_status()
    if ok:
        steps["assistant answer (local llama3.2:3b via Ollama)"] = _timeit(lambda: svc.ask(session, "Why does job 1 match my CV?", use_llm=True), 3)
    resources = {"device": device, "torch": torch.__version__,
                 "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                 "gpu_peak_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else None,
                 "cpu": platform.processor(), "python": platform.python_version()}  # fmt: skip
    try:
        import psutil

        resources["process_rss_mb"] = round(psutil.Process().memory_info().rss / 2**20)
    except ImportError:
        pass
    index_dir = settings.index_dir
    resources["chroma_index_mb"] = round(sum(p.stat().st_size for p in index_dir.rglob("*") if p.is_file()) / 2**20, 1)
    resources["faiss_vectors_mb"] = round(faiss_index.count_chunks() * faiss_index.config.embedding_dim * 4 / 2**20, 1)
    hardware = f"{resources['cpu']} · {resources['gpu'] or 'no GPU'} · torch {resources['torch']} on {device}"
    return {"steps": steps, "vector_backends": backends, "resources": resources, "hardware": hardware,
            "llm_cost": "local model: no per-call cost; an API model would cost tokens x price (see assistant benchmark for token volume)"}  # fmt: skip


def main(settings: Settings | None = None) -> dict:
    settings = settings or Settings.from_env()
    report = run(settings)
    write_report("latency.json", report)
    return report
