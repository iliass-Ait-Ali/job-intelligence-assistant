"""`jobintel bench <what>`: run benchmarks and write reports/*.json (then reports/evaluation_report.md)."""

from __future__ import annotations

from job_intel.config import Settings


def run_bench(what: str, settings: Settings, *, use_llm: bool = True) -> int:
    from job_intel.evaluation import assistant_bench, latency_bench, retrieval_bench, scoring_bench, skills_bench

    jobs = {
        "retrieval": lambda: retrieval_bench.main(settings),
        "skills": lambda: skills_bench.main(settings),
        "scoring": lambda: scoring_bench.main(settings),
        "assistant": lambda: assistant_bench.main(settings, use_llm=use_llm),
        "latency": lambda: latency_bench.main(settings),
    }
    for name in jobs if what == "all" else [what]:
        print(f"== {name} ==", flush=True)
        jobs[name]()
    from job_intel.evaluation.report import write_markdown

    path = write_markdown()
    print(f"wrote {path}")
    return 0
