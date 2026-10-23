"""SYNTHETIC jobs for the Phase 2 tests. None of these are real listings; URLs use the .invalid TLD."""

from __future__ import annotations

import json
from pathlib import Path


def make_job(n: int, title: str, company: str | None, location: str | None, job_type: str | None, remote: bool,
             description: str | None, *, source: str = "arbeitnow", published_at: str | None = "2026-10-20T10:00:00+00:00") -> dict:  # fmt: skip
    return {
        "id": f"{source}:syn-{n:02d}", "title": title, "company": company, "location": location, "job_type": job_type,
        "remote": remote, "description": description, "source": source,
        "source_url": f"https://fixtures.invalid/{source}/syn-{n:02d}", "published_at": published_at,
        "collected_at": "2026-10-23T00:00:00+00:00",
    }  # fmt: skip


def long_description() -> str:
    """Many paragraphs; the only mention of 'kubernetes operators' is in the LAST paragraph."""
    filler = [
        f"Paragraph {i}: we build recommendation models, feature stores and evaluation dashboards for retail customers "
        f"across several teams and quarters number {i}."
        for i in range(1, 25)
    ]
    return "\n\n".join([*filler, "Requirements: experience writing kubernetes operators in Go."])


def jobs() -> list[dict]:
    return [
        make_job(1, "Machine Learning Engineer", "Acme", "Zürich", "full_time", False,
                 "Optimize LLM inference latency and GPU performance. PyTorch, CUDA kernels, quantization."),  # fmt: skip
        make_job(2, "Data Analyst Internship", "Beta", "Paris, Île-de-France, France", "internship", False,
                 "Stage: analyse de données avec SQL, reporting Power BI et tableaux de bord.",
                 published_at="2026-10-21T10:00:00+00:00"),  # fmt: skip
        make_job(3, "Data Engineer", "Gamma", "Amsterdam, Noord-Holland, Nederland", "full_time", False,
                 "Build Spark and Airflow pipelines on Databricks for batch data processing.",
                 published_at="2026-10-19T10:00:00+00:00"),  # fmt: skip
        make_job(4, "Analytics Engineer", "Gamma", "Rotterdam, Zuid-Holland, Netherlands", None, False,
                 "Model data with dbt and SQL in the warehouse for analysts.", published_at="2026-10-18T10:00:00+00:00"),  # fmt: skip
        make_job(5, "AI Engineer", "Delta", "Remote job", "full_time", True,
                 "Build LLM agents and retrieval augmented generation applications.", source="remotive",
                 published_at="2026-10-22T10:00:00+00:00"),  # fmt: skip
        make_job(6, "NLP Researcher", "Epsilon", None, None, False,
                 "Research on natural language processing, text mining and computational linguistics.", published_at=None),  # fmt: skip
        make_job(7, "Data Scientist", "Zeta", "London", "full_time", False,
                 "Credit risk modelling in Python with gradient boosting.", published_at="2026-10-17T10:00:00+00:00"),  # fmt: skip
        make_job(8, "Werkstudent KI", "Eta", "Munich HQ", "working_student", False,
                 "Werkstudent im Bereich Künstliche Intelligenz mit Python.", published_at="2026-10-16T10:00:00+00:00"),  # fmt: skip
        make_job(9, "Machine Learning Engineer", "Acme", "Lausanne, Switzerland", "full_time", False, long_description(),
                 published_at="2026-10-15T10:00:00+00:00"),  # fmt: skip
        make_job(10, "Data Analyst", "Beta", "Paris", "internship", False,
                 "Stage analyste de données: Tableau, reporting, SQL.", published_at="2026-10-14T10:00:00+00:00"),  # fmt: skip
        make_job(11, "Senior Data Engineer", "Acme", "Zurich", "full_time", False,
                 "Kafka streaming pipelines and event processing.", published_at="2026-10-13T10:00:00+00:00"),  # fmt: skip
        make_job(12, "Data Scientist", "Theta", "Lille - Jardin des Plantes, Nord; Paris, Paris", "full_time", False,
                 "Forecasting and time series analysis for logistics.", published_at="2026-10-12T10:00:00+00:00"),  # fmt: skip
        make_job(13, "ML Intern", "Iota", "Berlin", "internship", False, None,
                 published_at="2026-10-11T10:00:00+00:00"),  # fmt: skip
    ]


def write_jobs(path: Path, records: list[dict] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(j, ensure_ascii=False) for j in (records or jobs())) + "\n", encoding="utf-8")
    return path
