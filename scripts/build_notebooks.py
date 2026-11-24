"""Build and execute the two notebooks (exploration, evaluation) so their outputs are visible on GitHub.

    python scripts/build_notebooks.py
Needs the collected data (data/) and the reports (reports/*.json).
"""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"


def md(text):
    return nbformat.v4.new_markdown_cell(text)


def code(text):
    return nbformat.v4.new_code_cell(text)


SETUP = """import os, sys, json, sqlite3
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
os.chdir(ROOT); sys.path.insert(0, str(ROOT / "src"))
import pandas as pd, matplotlib.pyplot as plt
pd.set_option("display.max_colwidth", 80); plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})
BLUE = "#2a78d6"
"""

EXPLORATION = [
    md("# 01 · Exploration of the job corpus\n\nWhat is in the data the system searches? Sources, completeness of the "
       "extracted fields, skill demand, skill associations and role profiles. Data: `data/processed/jobs_enriched.jsonl` "
       "and `data/jobintel.db` (created by `jobintel collect && jobintel enrich`)."),
    code(SETUP),
    code("""from job_intel.enrich.postings import load_postings
from job_intel.market import analytics as ma
postings = load_postings(Path("data/processed/jobs_enriched.jsonl"))
df = ma.postings_frame(postings)
print(len(df), "listings,", df.company.nunique(), "companies")
df.source.value_counts()"""),
    md("## Field completeness\nUnknown stays unknown: the share of listings where each structured field could be read."),
    code("""q = json.loads(Path("data/processed/quality_report.json").read_text())["completeness"]
s = pd.Series(q).sort_values()
ax = s.plot.barh(color=BLUE, figsize=(7, 4)); ax.set_xlim(0, 1); ax.set_title("Share of listings where the field is known")
for i, v in enumerate(s): ax.text(v + 0.01, i, f"{v:.0%}", va="center", fontsize=8)
plt.show()"""),
    md("## Skill demand (SQL on the SQLite store)"),
    code("""con = sqlite3.connect("data/jobintel.db")
top = pd.read_sql("SELECT skill, COUNT(DISTINCT job_id) AS listings FROM job_skills GROUP BY skill ORDER BY listings DESC LIMIT 20", con)
top["share"] = top.listings / len(df)
ax = top[::-1].plot.barh(x="skill", y="share", color=BLUE, legend=False, figsize=(7, 6)); ax.set_title("Share of listings asking for each skill")
plt.show(); top.head(10)"""),
    md("## Required vs merely mentioned\nThe extractor separates *required* (requirements section or cue words) from *preferred* and *mentioned*."),
    code("""pd.read_sql("SELECT level, COUNT(*) AS n FROM job_skills GROUP BY level", con)"""),
    md("## Skill associations: lift\n`lift(A,B) = P(A,B) / (P(A)P(B))` - above 1 means asked together more often than chance."),
    code("""ma.cooccurrence(df, top_skills=25, min_count=10).head(12)"""),
    md("## What characterises each role family? (distinctiveness = P(skill | family) / P(skill))"),
    code("""prof = ma.role_profiles(df)
prof.groupby("role_family").head(4)[["role_family", "skill", "share_in_family", "distinctiveness"]]"""),
    md("## Countries and experience levels\nHimalayas is remote-only and US-heavy: this is a sample of three boards, not the labour market."),
    code("""fig, axes = plt.subplots(1, 2, figsize=(11, 4))
ma.counts_by(df, "country", 10).set_index("country").postings[::-1].plot.barh(ax=axes[0], color=BLUE, title="Listings by country")
df.experience_level.value_counts()[::-1].plot.barh(ax=axes[1], color=BLUE, title="Experience level")
plt.tight_layout(); plt.show()"""),
]

EVALUATION = [
    md("# 02 · Evaluation of the system\n\nReads the benchmark outputs (`reports/*.json`, produced by `jobintel bench all`). "
       "**Labels are assistant-proposed (not human-verified)** - see `reports/evaluation_report.md` for the caveats."),
    code(SETUP + "R = lambda name: json.loads(Path('reports', name).read_text(encoding='utf-8'))"),
    md("## 1. Retrieval: five systems, 10 queries, 95% bootstrap intervals"),
    code("""r = R("retrieval_benchmark.json")
rows = [{"system": k, **{m: v[m]["mean"] for m in ("ndcg@10", "p@5", "map@10", "recall@10", "mrr@10_strict")},
         "ci_low": v["ndcg@10"]["ci_low"], "ci_high": v["ndcg@10"]["ci_high"], "latency_ms": v["latency_ms"]["p50"]}
        for k, v in r["systems"].items()]
t = pd.DataFrame(rows).set_index("system"); t.round(3)"""),
    code("""fig, ax = plt.subplots(figsize=(7, 3.5))
ax.bar(t.index, t["ndcg@10"], color=BLUE, yerr=[t["ndcg@10"] - t.ci_low, t.ci_high - t["ndcg@10"]], capsize=4)
ax.set_ylim(0, 1); ax.set_title("nDCG@10 with 95% bootstrap CI (over 10 queries)"); plt.show()"""),
    md("Per-query nDCG@10 - where do the systems disagree?"),
    code("""pd.DataFrame(r["per_query"]).T.round(2)"""),
    md("## 2. Skill extraction: dev (before / after error analysis) and held-out test"),
    code("""s = R("skills_benchmark.json")
pd.DataFrame({"dev before fixes": s["dev_before_fixes"]["micro"], "dev after fixes": s["dev_after_fixes"]["micro"],
              "held-out test": s["micro"]}).T.round(3)"""),
    code("""pd.DataFrame(s["errors"]["false_negatives"], columns=["missed skill", "count"]).head(12)"""),
    md("## 3. Matching: ordering a candidate's eligible jobs (leave-one-candidate-out)"),
    code("""c = R("scoring_calibration.json")
pd.DataFrame(c["methods"]).T.round(3)"""),
    code("""print(c["conclusion"]); c["eligibility_agreement"]"""),
    md("## 4. Assistant (RAG)"),
    code("""a = R("assistant_benchmark.json"); pd.DataFrame(a["modes"]).T"""),
    md("## 5. Latency"),
    code("""l = R("latency.json"); pd.DataFrame(l["steps"]).T"""),
]


def build(name: str, cells: list) -> None:
    nb = nbformat.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "jobintel", "display_name": "job-intel (.venv)", "language": "python"}})
    NotebookClient(nb, timeout=600, kernel_name="jobintel", resources={"metadata": {"path": str(NB)}}).execute()
    NB.mkdir(exist_ok=True)
    nbformat.write(nb, NB / name)
    print("wrote", NB / name)


if __name__ == "__main__":
    import sys

    wanted = sys.argv[1:] or ["exploration", "evaluation"]
    if "exploration" in wanted:
        build("01_exploration.ipynb", EXPLORATION)
    if "evaluation" in wanted:
        build("02_evaluation.ipynb", EVALUATION)
