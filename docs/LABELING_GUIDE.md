# Human labeling guide

Every quality number in this project comes from **your** judgments. The tools never fill in a grade, and they refuse
to print metrics until the relevant items are graded. This guide says what to do and how long it takes.

## What you will label (all built and ready; nothing is graded yet)

| Task | Command | Items | What a judgment is used for |
|---|---|---|---|
| 1. Retrieval relevance | `jobintel label retrieval` | 167 query/job pairs over 10 queries | nDCG@10, P@5, MRR@10, recall for **BM25 vs semantic vs hybrid (RRF)** |
| 2. CV-to-job fit | `jobintel label matching` | 84 candidate/job pairs over 3 synthetic CVs | ranking quality, role relevance, **eligibility correctness, incorrect exclusions, jobs the ranking never surfaces** |
| 3. Skill claims | `jobintel label skills` | 32 claims | precision of "exact" vs "transferable" skill matching |
| 4. Explanation audit | edit `data/eval/grounding_audit.csv` | about 95 claims (deterministic) | share of explanation claims actually supported by the evidence |

Total: roughly 3 to 4 hours if you do everything. You do not have to: **a query / candidate is scored as soon as all of
its pairs are graded**, so partial work still yields (smaller) measurements. A good first session (about 45 minutes):

1. `jobintel label retrieval --only q01`, `--only q02`, `--only q05` (about 54 cards)
2. `jobintel label matching --only cv_en_student_data_science` (26 cards)
3. `jobintel label skills` (32 quick y/n)

Then run `jobintel eval run`, `jobintel match-eval run`, `jobintel match-eval skills-score` and share the output.

## How the tool works

* One card at a time. It shows the query (or the candidate's CV summary and stated preferences) and the job:
  title, company, location, type, URL, the opening of the listing and the lines that share words with the query.
* **Blinded**: nothing tells you which system returned a job, at what rank or score, or whether the system excluded it.
* **Saved after every answer**; quit with `q` and resume later; `b` goes back; `s` skips; `n` adds a note; `?` shows help.
* Open the listing URL if the excerpt is not enough (listings expire, so judge soon after the data snapshot).

## Retrieval grades (task 1)

Judge the job against the **query only**: does this listing answer what the query asks for?

* **2 - highly relevant**: role/field, level and every explicit constraint in the query are met.
* **1 - partially relevant**: clearly related, but misses an important part (wrong level, wrong stack, adjacent role).
* **0 - not relevant**.

Queries with filters (for example `q02` = internships in Paris) are evaluated among the filtered jobs only. `q02`, `q06`
and `q09` are *exhaustive*: **every** eligible job is in the pool, which is what allows true recall to be reported.

## CV-to-job fit (task 2)

Each card shows a synthetic candidate (their CV summary and **stated preferences**) and a job. Three questions:

1. **Grade** 0 / 1 / 2: how good a fit is this job for this candidate, considering skills, level, role and the stated preferences.
2. **Role**: does the job's *role* match the candidate's target role or field? `y` / `n` / `?`.
3. **Eligible**: given the stated preferences (job type, location, remote), is the job *acceptable*? `y` = acceptable,
   `n` = violates a stated preference, `?` = the listing does not say.

Some pairs are jobs the system **excluded** or never showed: you do not know which, and should not try to guess. Answer
honestly; an excluded job that you mark acceptable is exactly the "incorrect exclusion" the audit is designed to find.

To test **your real CV** locally: put it (and an optional `<name>.prefs.json`) in a private folder such as `private/mycv/`
(git-ignored), then `jobintel match-eval pool --cvs private/mycv` and `jobintel label matching --cvs private/mycv`.
Judgments, pool metadata and the review sheet for it stay under `data/eval/` (git-ignored); the CV text is never copied
into a tracked file.

## Skill claims (task 3: `label skills`)

Each card shows a skill the listing asks for and the CV evidence the system offers.

* If the card says the CV **exactly** shows the skill: `y` only if the evidence really demonstrates *that* skill.
* If it says a **related** skill transfers: `y` only if the CV skill is genuinely close enough to count as a transferable
  strength for this requirement (for example PyTorch for TensorFlow), `n` otherwise.

## Explanation audit

Open `data/eval/grounding_audit.csv` and fill the `supported` column with `yes`, `partial` or `no` for each claim, judging it
against the two evidence columns next to it. Then `jobintel match-eval grounding-score`. Results are reported **per source**
(`deterministic`, `llm-stub`, `llm-ollama`) so a template result is never mistaken for a real Llama result.

## What the numbers will and will not mean

* One annotator, 10 queries, 3 synthetic candidates: results are descriptive. Per-query tables, win/tie/loss counts and
  bootstrap intervals (n >= 5 queries) are provided instead of significance claims.
* Unjudged results count as not relevant, and `judged@10` shows how many of the top 10 were judged.
* **Recall@10** is printed only for queries whose eligible jobs were *all* judged; otherwise only *pooled* recall is shown,
  and it is labelled as such.
