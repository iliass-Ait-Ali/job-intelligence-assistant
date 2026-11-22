"""SQLite store for structured postings (the MVP database).

Why SQLite: a single file, zero administration, full SQL for the market analytics and the API's
filters, and the schema below ports to PostgreSQL unchanged (``TEXT``/``INTEGER``/``REAL`` only,
no SQLite-specific features). Normalised tables let analytics count skills with plain SQL:

    jobs(id PK, title, company, location, country, region, remote, contract_type, sector,
         experience_level, min_years, degree_level, description_language, published_at, source, apply_url)
    job_skills(job_id FK, skill, category, level, evidence)       -- one row per technical skill
    job_soft_skills(job_id FK, skill, evidence)
    job_roles(job_id FK, role_family)
    job_languages(job_id FK, language)
    job_countries(job_id FK, country)

The database is rebuilt from ``jobs_enriched.jsonl`` in one transaction (idempotent).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable
from pathlib import Path

from job_intel.enrich.postings import JobPosting

SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY, title TEXT NOT NULL, company TEXT, location TEXT, country TEXT, region TEXT,
    remote INTEGER, contract_type TEXT, sector TEXT, experience_level TEXT, min_years INTEGER,
    degree_level TEXT, description_language TEXT, published_at TEXT, source TEXT NOT NULL, apply_url TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS job_skills (job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE, skill TEXT, category TEXT,
                                       level TEXT, evidence TEXT);
CREATE TABLE IF NOT EXISTS job_soft_skills (job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE, skill TEXT, evidence TEXT);
CREATE TABLE IF NOT EXISTS job_roles (job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE, role_family TEXT);
CREATE TABLE IF NOT EXISTS job_languages (job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE, language TEXT);
CREATE TABLE IF NOT EXISTS job_countries (job_id TEXT REFERENCES jobs(id) ON DELETE CASCADE, country TEXT);
CREATE INDEX IF NOT EXISTS ix_skills_skill ON job_skills(skill);
CREATE INDEX IF NOT EXISTS ix_skills_job ON job_skills(job_id);
CREATE INDEX IF NOT EXISTS ix_jobs_country ON jobs(country);
"""
TABLES = ("job_countries", "job_languages", "job_roles", "job_soft_skills", "job_skills", "jobs")


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def rebuild(path: Path, postings: Iterable[JobPosting]) -> int:
    con = connect(path)
    n = 0
    try:
        with con:  # one transaction: readers never see a half-written database
            for table in TABLES:
                con.execute(f"DELETE FROM {table}")  # noqa: S608 - constant table names
            for p in postings:
                n += 1
                con.execute(
                    "INSERT INTO jobs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (p.id, p.title, p.company, p.location, p.countries[0] if p.countries else p.region, p.region,
                     None if p.remote is None else int(p.remote), p.contract_type, p.sector, p.experience_level,
                     p.min_years, p.degree_level, p.description_language, p.published_at, p.source, str(p.apply_url)),
                )  # fmt: skip
                con.executemany("INSERT INTO job_skills VALUES (?,?,?,?,?)",
                                [(p.id, s.name, s.category, s.level, s.evidence) for s in p.skills])  # fmt: skip
                con.executemany("INSERT INTO job_soft_skills VALUES (?,?,?)", [(p.id, k, v) for k, v in p.soft_skills.items()])
                con.executemany("INSERT INTO job_roles VALUES (?,?)", [(p.id, r) for r in p.role_family])
                con.executemany("INSERT INTO job_languages VALUES (?,?)", [(p.id, lang) for lang in p.languages])
                con.executemany("INSERT INTO job_countries VALUES (?,?)", [(p.id, c) for c in p.countries])
    finally:
        con.close()
    return n
