"""Job-market intelligence (Module 6): descriptive statistics over the structured postings.

Every function takes a pandas DataFrame from ``postings_frame`` (one row per posting) or
``skills_frame`` (one row per posting x skill) so it can be filtered by any segment first
(role family, country, sector, experience level, contract type) and stays unit-testable.

Statistics used
* **demand share**     P(skill) = #postings asking for the skill / #postings in the segment.
* **co-occurrence lift**  lift(a, b) = P(a, b) / (P(a) P(b)). 1 = independent; > 1 = asked together
  more often than chance. Pairs need a minimum support (``min_count``) because lift is unstable
  on rare pairs.
* **distinctiveness**  for role-family profiles: P(skill | family) / P(skill) - which skills
  characterise a family rather than being asked everywhere (Python is everywhere; dbt is not).

Caveats (repeated in the UI): the corpus is a *sample* of three job boards, filtered to data / AI
roles, collected at a few dates; it is not the labour market. Counts describe this dataset.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import combinations

import pandas as pd

ROLE_LABELS = {
    "ai_engineer": "AI engineer", "ml_engineer": "ML engineer", "data_engineer": "Data engineer",
    "data_scientist": "Data scientist", "data_analyst": "Data analyst", "nlp": "NLP / LLM",
}  # fmt: skip
SEGMENTS = ("role_family", "country", "sector", "experience_level", "contract_type", "source")


def postings_frame(postings: Iterable) -> pd.DataFrame:
    rows = []
    for p in postings:
        rows.append({
            "id": p.id, "title": p.title, "company": p.company or "Unknown",
            "country": (p.countries[0].title() if p.countries else (p.region or "unknown").title()),
            "region": p.region, "remote": p.remote,
            "contract_type": p.contract_type or "unknown",
            "role_family": ROLE_LABELS.get(p.role_family[0], p.role_family[0]) if p.role_family else "Other data/AI",
            "role_families": [ROLE_LABELS.get(r, r) for r in p.role_family] or ["Other data/AI"],
            "sector": p.sector or "Unspecified", "experience_level": p.experience_level,
            "min_years": p.min_years, "degree_level": p.degree_level or "not stated",
            "language": p.description_language, "source": p.source,
            "published_at": pd.to_datetime(p.published_at, utc=True, errors="coerce"),
            "skills": sorted({s.name for s in p.skills}),
            "required_skills": sorted({s.name for s in p.skills if s.level == "required"}),
            "soft_skills": sorted(p.soft_skills),
            "url": str(p.apply_url),
        })  # fmt: skip
    return pd.DataFrame(rows)


def filter_frame(df: pd.DataFrame, **segments: list[str] | None) -> pd.DataFrame:
    out = df
    for column, values in segments.items():
        if values:
            if column == "role_family":
                out = out[out["role_families"].apply(lambda rs: bool(set(rs) & set(values)))]
            else:
                out = out[out[column].isin(values)]
    return out


def skill_demand(df: pd.DataFrame, column: str = "skills", top: int = 25) -> pd.DataFrame:
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["skill", "postings", "share"])
    counts = df[column].explode().dropna().value_counts()
    out = counts.rename_axis("skill").reset_index(name="postings")
    out["share"] = out["postings"] / n
    return out.head(top)


def counts_by(df: pd.DataFrame, column: str, top: int = 20) -> pd.DataFrame:
    if column == "role_family":
        series = df["role_families"].explode()
    else:
        series = df[column]
    out = series.value_counts().rename_axis(column).reset_index(name="postings").head(top)
    out["share"] = out["postings"] / max(len(df), 1)
    return out


def cooccurrence(df: pd.DataFrame, top_skills: int = 20, min_count: int = 5) -> pd.DataFrame:
    """Pairs among the most frequent skills, with support, conditional probability and lift."""
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["a", "b", "count", "p_b_given_a", "lift"])
    demand = skill_demand(df, top=top_skills)
    keep = set(demand["skill"])
    single = dict(zip(demand["skill"], demand["postings"]))
    pairs: dict[tuple[str, str], int] = {}
    for skills in df["skills"]:
        present = sorted(set(skills) & keep)
        for a, b in combinations(present, 2):
            pairs[(a, b)] = pairs.get((a, b), 0) + 1
    rows = []
    for (a, b), c in pairs.items():
        if c < min_count:
            continue
        lift = (c / n) / ((single[a] / n) * (single[b] / n))
        rows.append({"a": a, "b": b, "count": c, "p_b_given_a": c / single[a], "lift": lift})
    return pd.DataFrame(rows).sort_values(["lift", "count"], ascending=False).reset_index(drop=True) if rows else \
        pd.DataFrame(columns=["a", "b", "count", "p_b_given_a", "lift"])


def cooccurrence_matrix(df: pd.DataFrame, top_skills: int = 15) -> pd.DataFrame:
    """Square matrix of P(column skill | row skill) among the top skills (diagonal = 1)."""
    skills = list(skill_demand(df, top=top_skills)["skill"])
    sets = [set(s) for s in df["skills"]]
    data = {}
    for a in skills:
        with_a = [s for s in sets if a in s]
        data[a] = {b: (sum(b in s for s in with_a) / len(with_a) if with_a else 0.0) for b in skills}
    return pd.DataFrame(data).T.loc[skills, skills]


def role_profiles(df: pd.DataFrame, top: int = 8, min_postings: int = 10) -> pd.DataFrame:
    """For each role family: its most distinctive skills (share in family / share overall)."""
    overall = skill_demand(df, top=10_000).set_index("skill")["share"]
    rows = []
    exploded = df.explode("role_families")
    for family, group in exploded.groupby("role_families"):
        if len(group) < min_postings:
            continue
        fam = skill_demand(group, top=10_000)
        fam = fam[fam["postings"] >= max(5, 0.08 * len(group))].copy()  # stable shares only
        fam["distinctiveness"] = fam.apply(lambda r: r["share"] / overall.get(r["skill"], r["share"]), axis=1)
        fam = fam.sort_values(["distinctiveness", "share"], ascending=False).head(top)
        for _, r in fam.iterrows():
            rows.append({"role_family": family, "skill": r["skill"], "share_in_family": r["share"],
                         "distinctiveness": r["distinctiveness"], "postings": len(group)})  # fmt: skip
    return pd.DataFrame(rows)


def weekly_trend(df: pd.DataFrame, skills: list[str]) -> pd.DataFrame:
    """Postings per publication week and the share asking for each skill (published_at is the provider date)."""
    d = df.dropna(subset=["published_at"]).copy()
    if d.empty:
        return pd.DataFrame(columns=["week", "postings", *skills])
    d["week"] = d["published_at"].dt.tz_convert(None).dt.to_period("W").dt.start_time
    rows = []
    for week, g in d.groupby("week"):
        row = {"week": week, "postings": len(g)}
        for s in skills:
            row[s] = sum(s in sk for sk in g["skills"]) / len(g)
        rows.append(row)
    return pd.DataFrame(rows)


def snapshot_comparison(old: pd.DataFrame, new: pd.DataFrame, top: int = 15) -> pd.DataFrame:
    """Demand share of the top skills in two collection snapshots (different dates, overlapping sources)."""
    a = skill_demand(old, top=10_000).set_index("skill")["share"]
    b = skill_demand(new, top=10_000).set_index("skill")["share"]
    skills = list(skill_demand(new, top=top)["skill"])
    out = pd.DataFrame({"skill": skills, "share_old": [a.get(s, 0.0) for s in skills],
                        "share_new": [b.get(s, 0.0) for s in skills]})  # fmt: skip
    out["change_pts"] = 100 * (out["share_new"] - out["share_old"])
    return out


def top_companies_for_skill(df: pd.DataFrame, skill: str, top: int = 10) -> pd.DataFrame:
    hit = df[df["skills"].apply(lambda s: skill in s)]
    return counts_by(hit, "company", top=top)
