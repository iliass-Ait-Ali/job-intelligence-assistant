"""Career-assistant (RAG) benchmark: routing, grounding and relevance, deterministic vs local LLM.

Every metric is computed automatically from explicit definitions (no LLM judge):

* routing accuracy      routed intent == expected intent.
* citation validity     share of PROPOSED sentences (before the verifier) whose citations are non-empty and all
                        refer to documents that were actually given.
* faithfulness          share of proposed sentences that pass the verifier: every skill, credential, number,
                        proper noun and level word must occur in the cited documents, and the sentence must share
                        content with them.   hallucination rate = 1 - faithfulness (on proposed sentences).
* residual unsupported  sentences SHOWN to the user that fail the verifier: 0 by construction - the number
                        that matters is how much the verifier had to remove (above), and the limits of what it can
                        check (qualitative claims without entities), discussed in the report.
* answer relevance      cosine similarity (multilingual-e5) between the question and the final answer.
* retrieval relevance   for questions that name skills: share of cited listings that require ALL those skills.
* safety                share of final answers containing a probability / salary / hiring-guarantee claim.
* latency               wall-clock seconds per answer; LLM token volume estimated as characters / 4.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from statistics import mean, median

from job_intel.config import Settings
from job_intel.evaluation.data import write_report
from job_intel.match.models import Preferences
from job_intel.rag.explain import _FORBIDDEN

ROOT = Path(__file__).resolve().parents[3]
LABEL_NOTE = ("Automatic metrics on 26 hand-written questions (EN/FR) over 3 synthetic CVs and the live corpus; no human "
              "rating of answer quality. The verifier is also the faithfulness checker, so faithfulness measures agreement "
              "with its rules, not truth.")  # fmt: skip


def _questions() -> list[dict]:
    return json.loads((ROOT / "eval" / "assistant_questions.json").read_text(encoding="utf-8"))["questions"]


def run(settings: Settings, *, use_llm: bool) -> dict:
    from job_intel.match.cv_parser import parse_cv
    from job_intel.services import JobIntelService

    service = JobIntelService(settings)
    service.warm_up()
    embedder = service.searcher().embedder
    sessions: dict[str | None, object] = {}
    modes = {"deterministic": None}
    if use_llm:
        ok, _msg = service.llm_status()
        if ok:
            modes["llm-ollama"] = service.llm_client()
    results = {}
    for mode, client in modes.items():
        rows = []
        for q in _questions():
            cv = q.get("cv")
            if cv not in sessions:
                if cv:
                    path = ROOT / "eval" / "synthetic_cvs" / f"{cv}.txt"
                    profile = parse_cv(path.read_text(encoding="utf-8"))
                    prefs = Preferences(**json.loads(path.with_suffix(".prefs.json").read_text(encoding="utf-8")))
                    sessions[cv] = service.assistant_session(profile, service.recommend(profile, prefs).ranked)
                else:
                    sessions[cv] = service.assistant_session()
            from job_intel.assistant.chat import ask

            answer = ask(sessions[cv], q["text"], llm_client=client)
            print(f"  [{mode}] {q['id']} {answer.intent} {answer.mode} {answer.latency_s:.2f}s", flush=True)
            proposed = len(answer.sentences) + len(answer.rejected)
            bad_cites = sum(1 for _, reason in answer.rejected if reason.startswith(("no citation", "cites unknown")))
            text = " ".join(s.text for s in answer.sentences)
            relevance = None
            if text:
                qv = embedder.embed_queries([q["text"]])[0]
                av = embedder.embed_passages([text[:1200]])[0]  # explicit cut: the model reads <= 512 tokens
                relevance = float(qv @ av)
            retrieval = None
            if q.get("skills"):
                cited_jobs = [d.job_id for d in answer.docs if d.kind == "job" and d.job_id]
                if cited_jobs:
                    need = set(q["skills"])
                    ok = [need <= {s.name for s in service.postings[j].skills} for j in cited_jobs if j in service.postings]
                    retrieval = sum(ok) / len(ok) if ok else None
            rows.append({
                "id": q["id"], "expected_intent": q["expected_intent"], "intent": answer.intent,
                "routed_ok": answer.intent == q["expected_intent"] or (q["expected_intent"] in {"top_jobs", "why_match", "missing_skills", "compare", "interview", "projects"} and answer.intent == "search" and not cv),
                "mode": answer.mode, "proposed": proposed, "kept": len(answer.sentences), "rejected": len(answer.rejected),
                "bad_citations": bad_cites, "relevance": relevance, "retrieval_relevance": retrieval,
                "forbidden_in_final": bool(_FORBIDDEN.search(text)), "latency_s": answer.latency_s,
                "est_tokens": (sum(len(d.text) + len(d.title) for d in answer.docs) + len(q["text"]) + len(text)) // 4 if client else 0,
                "rejected_reasons": [r for _, r in answer.rejected][:5],
            })  # fmt: skip
        proposed = sum(r["proposed"] for r in rows) or 1
        latencies = [r["latency_s"] for r in rows]
        rel = [r["relevance"] for r in rows if r["relevance"] is not None]
        ret = [r["retrieval_relevance"] for r in rows if r["retrieval_relevance"] is not None]
        results[mode] = {
            "questions": len(rows),
            "routing_accuracy": round(sum(r["intent"] == r["expected_intent"] for r in rows) / len(rows), 3),
            "citation_validity": round(1 - sum(r["bad_citations"] for r in rows) / proposed, 3),
            "faithfulness": round(sum(r["kept"] for r in rows) / proposed, 3),
            "hallucination_rate": round(sum(r["rejected"] for r in rows) / proposed, 3),
            "fallback_rate": round(sum(r["mode"] == "deterministic-fallback" for r in rows) / len(rows), 3),
            "answer_relevance": round(mean(rel), 3) if rel else None,
            "retrieval_relevance": round(mean(ret), 3) if ret else None,
            "forbidden_claims_in_final": sum(r["forbidden_in_final"] for r in rows),
            "latency_median_s": round(median(latencies), 3), "latency_max_s": round(max(latencies), 3),
            "est_llm_tokens_per_question": round(mean(r["est_tokens"] for r in rows)) if client else 0,
            "cost": "0 € (local model)" if client else "0 € (no model)",
        }  # fmt: skip
        results[mode + "_rows"] = rows  # type: ignore[assignment]
    return results


def main(settings: Settings | None = None, use_llm: bool = True) -> dict:
    settings = settings or Settings.from_env()
    res = run(settings, use_llm=use_llm)
    modes = {k: v for k, v in res.items() if not k.endswith("_rows")}
    report = {"label_note": LABEL_NOTE, "modes": modes, "rows": {k: v for k, v in res.items() if k.endswith("_rows")},
              "definitions": re.sub(r"\s+", " ", __doc__ or "").strip()}  # fmt: skip
    write_report("assistant_benchmark.json", report)
    return report
