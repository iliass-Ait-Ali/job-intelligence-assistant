# Detailed usage and design notes (Phases 1-3)

Phase 1 collects Data / ML / AI job and internship listings from two public APIs, cleans and
normalizes them, filters for relevant roles, removes duplicates and exports
`data/processed/jobs.csv` and `data/processed/jobs.jsonl`.

**Phase 2** (after the Phase 1 sections) adds local semantic job search: SentenceTransformers + ChromaDB, a BM25 baseline and a human-judged evaluation framework.

**Phase 3** adds CV matching and grounded explanations with a local Ollama model; **Phase 5** (last section) adds the platform: enrichment + SQLite, five search engines, scoring, skill gaps, market analytics, the RAG assistant, the FastAPI service, the Streamlit UI and the benchmarks.

**Not built yet:** Docker, public deployment.

# Phase 1 - Data collection

| Source | Endpoint | Notes |
|---|---|---|
| Arbeitnow | `https://www.arbeitnow.com/api/job-board-api` | Free, paginated (up to ~300 jobs/page), mostly Europe |
| Remotive | `https://remotive.com/api/remote-jobs` | Remote jobs only; the public endpoint currently returns a small sample (9 jobs at the time of writing) |

## Install and run (Windows PowerShell, Python 3.11+)

```powershell
cd job-intelligence-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install -e ".[dev]"

Copy-Item .env.example .env           # optional: put a contact address in JOBINTEL_CONTACT

jobintel collect                      # or: python -m job_intel collect
jobintel collect --max-pages 3        # fewer Arbeitnow pages
jobintel collect --sources arbeitnow  # one source only
jobintel collect --force-refresh      # ignore fresh raw snapshots and call the APIs again

pytest                                # offline tests, no network needed
```

`collect` prints a per-source table and the totals, and writes:

| File | Content |
|---|---|
| `data/processed/jobs.csv` | final records (UTF-8 with BOM so Excel opens it correctly) |
| `data/processed/jobs.jsonl` | same records, one JSON object per line |
| `data/processed/collection_summary.json` | all counts, per-source stats, errors, attribution |
| `data/raw/<source>_<UTC time>.json.gz` | untouched API responses (git-ignored, last 3 kept) |

Record schema: `id, title, company, location, job_type, remote, description, source,
source_url, published_at, collected_at`. Missing values are `null` in JSONL and empty cells in CSV.
`published_at`/`collected_at` are ISO-8601 UTC. `job_type` is one of `internship`,
`working_student`, `freelance`, `contract`, `part_time`, `full_time`, `temporary`, `permanent`
or null (providers use mixed-language labels such as "Unbefristeter arbeitsvertrag"; a title with
"Praktikum", "Internship" or "Werkstudent" overrides the label).

## Architecture

```
src/job_intel/
  cli.py            argparse entry point (`jobintel collect`), prints the summary
  config.py         Settings from environment / .env (timeouts, delay, TTLs, User-Agent)
  http.py           HttpClient: User-Agent, connect+read timeouts, spacing between requests, retry with backoff
  connectors/
    base.py         Connector interface: fetch() -> raw records, normalize(raw) -> Job
    arbeitnow.py    pagination + record normalization
    remotive.py     single request + record normalization
  textutils.py      HTML -> text, whitespace, dates, job_type mapping, URL canonicalization (pure functions)
  models.py         Job dataclass and the exported column list
  relevance.py      rule-based role filter and ranking score
  dedup.py          duplicate removal
  storage.py        raw snapshot cache + atomic CSV/JSONL/JSON writers
  pipeline.py       orchestrates the ETL run and collects the counts
  search/           Phase 2, see below
tests/              offline pytest suite + SYNTHETIC fixtures (tests/fixtures)
```

Adding a source means adding one `Connector` subclass; nothing downstream changes.

## The ETL pipeline

1. **Extract** - each connector fetches raw JSON. If a raw snapshot for that source is younger than its
   TTL (Arbeitnow 1 h, Remotive 6 h) it is reused instead of calling the API again; this keeps
   reruns polite and respects Remotive's "at most ~4 requests/day" guidance. Every live response is
   saved to `data/raw/` before anything is parsed.
2. **Transform - normalize** - HTML is stripped (scripts/styles dropped, entities decoded, list items become
   `- ` bullets, paragraph breaks kept), whitespace collapsed, dates converted to UTC, `null`/empty values
   become `None`. A record without a title or URL is counted as *invalid* and skipped; one bad record never
   aborts the run.
3. **Transform - filter** (`relevance.py`) - a listing is kept if its **title** names a target role
   (Data Scientist, Data Analyst, ML Engineer, AI Engineer, Data Engineer, NLP, plus BI / analytics
   engineer / LLM variants in English, German and French), or if it is an internship / working-student role
   with a data/AI signal (data term in title or tags, an AI term in a title that is not about marketing,
   design, sales etc., or two strong data-science skills in the description). Descriptions never qualify a
   regular job on their own, so keyword-stuffed text cannot smuggle in unrelated roles. Survivors are
   ranked: title match 10 (+2 per extra family), +5 internship, up to +5 for skill mentions, newest first.
4. **Transform - deduplicate** (`dedup.py`) - the best-ranked copy wins. Duplicates are matched by stable
   `id` (`<source>:<provider id>`), by canonical `source_url` (host case, trailing slash, `utm_*`, fragment
   ignored) and, because live data showed reposts under new ids and URLs, by identical
   title + company + location. Removals are reported per rule.
5. **Load** - atomic write of CSV, JSONL and the summary JSON (a failed run never leaves half-written files,
   and if *every* source fails the previous exports are left untouched and the exit code is 1).

Counts reported: `total fetched -> invalid skipped -> normalized -> filtered out (not relevant) -> relevant
-> duplicates removed -> final`.

## Results of the live run (2026-10-23)

| Step | Count |
|---|---|
| Total fetched | 2,915 (Arbeitnow 2,906 over 10 pages, Remotive 9) |
| Invalid, skipped | 0 |
| Filtered out as not relevant | 2,722 |
| Relevant | 193 |
| Duplicates removed | 2 (title + company + location reposts) |
| **Final records** | **191** (all Arbeitnow; 28 internships, 2 working-student roles) |

The 50+ target is met, but only through Arbeitnow: the Remotive endpoint returned 9 listings, none of
them a target role. Numbers change on every run because both boards update continuously.

## Failure handling

Timeouts, connection errors, HTTP 429/5xx are retried with backoff (honouring `Retry-After`); other 4xx and
invalid JSON fail fast. If a later Arbeitnow page fails, the earlier pages are kept and a warning is
reported. A schema change (missing `data` / `jobs` list) raises a clear error. One source failing does not
stop the other. All of this is covered by the offline tests.

## Terms of use and attribution

* Every record keeps the original listing URL (`source_url`) and `source` name. Please keep showing both
  wherever the data is displayed.
* **Arbeitnow**: free public API, "please do not abuse", link back to the site. The client sends a
  descriptive User-Agent, waits 2 s between pages and caps pages per run.
* **Remotive**: jobs may be shared further only with a link to the Remotive URL and Remotive named as the
  source; they must **not** be submitted to third-party job aggregators (e.g. Jooble, Google Jobs, LinkedIn
  Jobs); the data is delayed by 24 h. Do not use the data to collect sign-ups. Check
  <https://remotive.com/api-documentation> before publishing any Remotive data. This repo's
  `data/processed/` files contain Remotive records whenever Remotive returns relevant ones, so review
  the terms before pushing them to a public repository.

## Limitations

* Relevance is rule-based on titles; roles with unusual titles are missed and a few early-career
  postings with a weak data signal get through (e.g. "Electromagnetic Simulation Intern").
  Titles in languages other than English/German/French/Dutch-ish are only partly covered.
* The title + company + location rule can merge two genuinely separate openings with identical wording.
* Arbeitnow pagination is capped at 10 pages (about 3,000 listings, roughly the newest week); raise
  `--max-pages` for more.
* Remotive's public endpoint is a small sample, so it contributes little.
* Remotive timestamps carry no timezone and are interpreted as UTC.
* `remote` is `false` for Arbeitnow jobs that do not say they are remote (not necessarily on-site).
* Test fixtures are synthetic (URLs use the reserved `.invalid` domain); they are not real listings.

---

# Phase 2 - Semantic job search

Natural-language search over the processed jobs (French, English, German, Dutch...) with a local
vector index, plus a lexical BM25 baseline and a framework to compare them on **human relevance
judgments**. No performance claim is made until those judgments exist.

## Install and run (Windows PowerShell)

```powershell
cd job-intelligence-assistant
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,search]"     # adds sentence-transformers (CPU torch) and chromadb

jobintel index                               # embed data/processed/jobs.jsonl -> data/index/chroma
jobintel index-status                        # what is indexed and with which settings

jobintel search "stage data analyst a Paris"
jobintel search "LLM agents and RAG" --engine both -k 5          # semantic and BM25, same filters
jobintel search "data engineer" --location Zurich --job-type full_time --since 2026-10-15
jobintel search "machine learning" --max-per-company 2           # optional diversity (display only)

pytest                                       # offline tests (search/app tests need the [search]/[app] extras)
```

The first `jobintel index` downloads `intfloat/multilingual-e5-base` (about 1.1 GB, cached under
`~/.cache/huggingface`). Everything after that is offline. CPU only; no GPU is used or needed.
`jobintel collect` still works with the plain Phase 1 install (the search libraries are imported lazily).
If the model cannot be downloaded or a dependency is missing, the CLI prints a one-line error with the fix
(`pip install -e ".[search]"`, or set `HF_HUB_OFFLINE=1` once the model is cached) instead of a traceback.

## How it works

* **Documents.** Each job is split into chunks of at most 300 *model* tokens (about 40 tokens of overlap).
  Every chunk starts with a header `title | company | location | job type | remote`, so a chunk from the middle of a
  1,300-token posting still says which job it is. Tokens are counted with the model's own tokenizer on the exact
  text the model receives (including the `passage: ` prefix and special tokens); each chunk is re-measured, and a job
  that cannot be chunked within the limit raises an error instead of being truncated.
* **Index.** ChromaDB `PersistentClient` at `data/index/chroma`, cosine space. Chunk id = `<job id>#<n>`. Embeddings are
  computed by this project (normalized), never by Chroma. A job's score is the cosine similarity of its best chunk.
  Chroma reports cosine *distance* = `1 - similarity`; the code keeps both explicitly and the CLI prints similarity.
* **Idempotent updates.** Each job stores a content hash. `jobintel index` adds new jobs, re-embeds changed ones (old chunks
  are deleted first), deletes jobs that left the dataset, repairs jobs left half-written by an interrupted run and skips the
  rest. Running it twice embeds nothing the second time.
* **Configuration check.** Model name, embedding dimension, prefixes, chunk size/overlap, chunker version and distance are stored
  with the index. Indexing with different settings, or querying with another `--model`, is refused with an explanation;
  `jobintel index --rebuild` drops and rebuilds. A stale index (jobs file changed since indexing) triggers a warning.
* **Retrieval completeness.** Search fetches chunks, groups them by job and **doubles the number of chunks until k distinct jobs
  are covered** or every eligible chunk has been seen, so a posting with dozens of similar chunks cannot crowd out other jobs.
  Once k jobs are covered the top-k is exact: any unseen job has all its chunks below the lowest score fetched.
* **Filters run before the vector search.** The eligible job ids are computed first and passed to Chroma as a `where job_id in (...)`
  clause, so a filter never operates on a fixed-size candidate list. BM25 uses the very same eligible set.
* **Company cap.** `--max-per-company N` is a display option (off by default). Evaluation always uses the uncapped ranking.
* **Snippets** show the line of the best-matching chunk with the most words in common with the query (or the start of the chunk);
  this is a readability aid, not an explanation of the model's reasoning.
* **Scores.** E5 cosine similarities live in a narrow band (about 0.8-0.9), so only the *ranking* and the matched text are meaningful;
  scores are not probabilities and BM25 scores are not comparable with cosine scores.

### Filter semantics (missing / unknown values)

| Option | Behaviour |
|---|---|
| `--job-type T` | exact canonical type. Jobs without a type (about 25% of the data) are **excluded**, unless you pass `--job-type unknown` or `--include-unknown`. |
| `--remote` | only listings that **explicitly** say remote. Phase 1 stores `remote=false` for "not stated", so those listings are excluded and `--include-unknown` does not bring them back. There is no "not remote" filter. |
| `--location L` | accent- and spelling-insensitive (Zurich = Zürich, Nederland = Netherlands, UK = United Kingdom), city-to-country aware for the cities in the data (Amsterdam matches Netherlands), repeatable (OR). Jobs without a location are excluded unless `--include-unknown`. Cities outside the small gazetteer match only by name. |
| `--since DATE` | published on/after the date. Jobs without a publication date are excluded unless `--include-unknown`. |
| `--source S` | `arbeitnow` or `remotive`. |

## Human evaluation (semantic vs BM25)

All relevance labels come from a person. The tool never creates or guesses a grade.

```powershell
jobintel eval pool          # unions the top-10 of both systems per query into eval/judgments.csv (blinded, shuffled, ungraded)
# read the postings in data\eval\review.md, then fill the `grade` column of eval/judgments.csv: 0, 1 or 2
jobintel eval run           # scores both systems; refuses to print metrics while no query is fully graded
jobintel eval run --per-query --report-json data\eval\report.json
```

* `eval/queries.json` has 10 queries (EN/FR/DE, some with filters); append more, up to 20. Each query is evaluated with its own eligibility
  filters, identically for both engines, with no company cap.
* Grades: 2 = highly relevant, 1 = partially relevant, 0 = not relevant. A query is scored only when its whole pool is graded;
  partially graded queries are listed as pending. Re-running `eval pool` only appends new items and keeps existing grades.
* Metrics: nDCG@10 (graded; gain 2^grade-1; the ideal ranking comes from the judged pool), Precision@5 and MRR@10 at **two explicit binary
  thresholds** (grade >= 1 and grade >= 2), and *pooled* recall@10. Pooled recall is relevant results found divided by relevant items in the
  judged pool; it is **not** full-corpus recall and it favours the systems that built the pool.
* `judged@10` reports how many of each system's top 10 have a grade; unjudged results count as not relevant.
* With 10-20 queries and one annotator the averages are descriptive: use the per-query table and win/tie/loss counts, not significance claims.

## Phase 2 limitations

* Embedding on a laptop CPU is slow (about 2 chunks/s with `multilingual-e5-base` on this machine); a re-index after changing settings takes minutes.
* Several postings of one company share long identical text (for example one role posted per city). They are not duplicates, so they are all
  kept and can fill the top of a ranking; use `--max-per-company` for display.
* The location gazetteer only covers the cities seen so far; spelling aliases are manual.
* `job_type` is missing for about a quarter of the jobs and `remote` cannot express "not remote" (see the filter table).
* The index and `data/` are git-ignored on purpose: they contain third-party listings. `eval/judgments.csv` stores only ids and titles.
  Remotive listings must not be passed to third-party sites; use `jobintel index --exclude-source remotive` to keep them out of the index.
* Retrieval quality has not been measured: it needs your judgments in `eval/judgments.csv`.

---

# Phase 3 - CV matching and grounded explanations

Upload a CV (PDF, text or pasted; French or English) and get job recommendations from the local job database, each with the
evidence behind it, what is missing or unverified, and a grounded explanation. Everything runs on this computer.

## Install and run (Windows PowerShell)

```powershell
cd job-intelligence-assistant
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev,app]"        # pypdf, python-docx, streamlit, fastapi (plus the search libraries)

jobintel ui                                  # http://127.0.0.1:8501 (loopback only)

jobintel profile --cv my_cv.pdf              # structured profile as JSON (printed, not saved)
jobintel match --cv my_cv.pdf --job-type internship --location Paris -n 10 --explain 3
jobintel llm-status                          # is the local Ollama model available?
```

The recommendations need the Phase 2 index (`jobintel index`). Optional local language model:

```powershell
# install Ollama from https://ollama.com, then:
ollama pull llama3.2:3b                      # default; set JOBINTEL_LLM_MODEL for another, e.g. llama3.1:8b
jobintel match --cv my_cv.pdf --explain 3 --llm
```

Without Ollama everything still works: explanations are produced by a **deterministic, clearly labelled** mode.

## Pipeline

1. **CV text** (`match/cv_text.py`): pypdf extraction with limits (10 MB, 12 pages), password/corruption/empty handling and a warning for
   (partly) scanned PDFs. OCR is deliberately not implemented. Uploaded temp files are deleted after reading; nothing is written to disk.
2. **Profile** (`match/cv_parser.py`, `match/models.py`): rule-based sections (EN/FR/DE headings), skills from a curated taxonomy with the
   CV lines that mention them, education level, experience and project entries, spoken languages, target roles (only from "looking for..."
   statements). Dates give month **bounds**, never guesses. No contact details, proficiency levels or years of experience are inferred.
   The profile is a Pydantic model you can inspect and correct in the interface.
3. **Candidate representation** (`match/matcher.py: build_queries`): one query per target role, one for the skills, one per experience, one per
   project, one for the summary, so nothing is collapsed into a single keyword bag.
4. **Retrieval**: every query runs on both Phase 2 engines (semantic and BM25); the lists are fused with Reciprocal Rank Fusion into a
   broad pool (up to 80 jobs).
5. **Eligibility**: job type, location, remote, experience, education, language, each `satisfied | violated | unknown`. Unknown (missing
   information) never excludes a job. Only constraints you mark hard (default: job type and location) exclude, and only when clearly violated;
   excluded jobs are listed with the reason. "Remote only" keeps listings that explicitly state remote work (Phase 1 cannot tell "on-site"
   from "not stated").
6. **Requirements** (`match/requirements.py`): skills with their level (required / preferred / mentioned, from section headings and cues),
   years of experience, degree level, language requirements and seniority, each keeping the listing line it came from. Listing lines that look
   like instructions to an AI are ignored.
7. **Ranking**: each listing skill is `exact` (the CV states it), `transferable` (the CV states a related skill from the closest related group,
   e.g. PyTorch for TensorFlow, never counted as exact) or `not evidenced`. The order is: evidence level (strong / moderate / weak, from
   documented rules and thresholds in `matcher.py`), role match, number of clearly unmet stated requirements, share of skills with exact
   evidence, retrieval rank. **There is no percentage or probability score**: it could not be calibrated.

## Grounded explanations (RAG)

* **Retrieve**: for the chosen job, the ChromaDB retriever (restricted to that job) finds the passages closest to the candidate's queries; the
  lines where each requirement was found are always added. CV evidence lines are chosen the same way. All snippets are numbered (CV1.., JOB1..).
* **Generate**: the local model gets the snippets as a JSON message, a strict system prompt (snippets are untrusted data, cite ids, name nothing
  that is not cited, no probabilities/salaries) and a JSON schema (`strengths`, `gaps`, `preparation`), temperature 0.
* **Verify** (`rag/explain.py`): every item must cite existing snippets of the right kind; a strength needs a CV and a listing citation; every
  technology, credential and number named must appear in the cited text; each citation must be about the claim; probabilities, salaries, URLs and
  markup are rejected; a "missing skill" that the CV does state is rejected as a contradiction. Rejected items are shown separately.
* **Fallback**: if Ollama is down, the model is missing, times out, returns invalid JSON or nothing verifiable, the deterministic explanation
  (same structure, same verifier) is shown with a note saying why.
* **Prompt injection**: CV and listing lines that look like instructions are withheld from the evidence and reported; snippets are JSON-escaped
  data; the model has no tools; and the verifier means an obedient model still cannot add unsupported facts.
* **Privacy**: the Ollama client refuses any non-loopback URL; no other module opens network connections; the app binds 127.0.0.1 and
  analytics are off. Uploaded CVs and extracted profiles are never committed (`.gitignore` covers `data/`, PDFs/Word files, `*.profile.json`,
  `cvs/`, `uploads/`; only `eval/synthetic_cvs/` is allowed).

## Evaluation (nothing is pre-filled)

```powershell
jobintel match-eval pool              # blinded (synthetic candidate, job) pairs -> eval/match_judgments.csv (grade 0/1/2 by YOU)
jobintel match-eval run               # pipeline vs retrieval-only ordering of the SAME eligible jobs; refuses until a candidate is fully graded
jobintel match-eval grounding --top 3 # explanations for the top jobs; counts verifier outcomes; writes data/eval/grounding_audit.csv
jobintel match-eval grounding-score   # summarises YOUR yes / partial / no marks in the audit file
```

Matching metrics reuse the Phase 2 definitions (nDCG@10, Precision@5 and MRR@10 at grade >= 1 and >= 2, pooled recall labelled as such, judged@10
coverage). The grounding run reports how many proposed items pass the automatic verifier; that is a consistency check of citations and named
facts, not proof that statements are true, which is why verified items are exported for a human audit. The three synthetic CVs in
`eval/synthetic_cvs/` (with `.prefs.json` preferences) are fictitious.

## Phase 3 limitations

* The CV parser is rule-based: unusual layouts (two-column PDFs, graphics-heavy CVs), unusual headings or skills outside the taxonomy are missed;
  the interface lets you correct the profile, and warnings say what could not be found.
* Skill matching is vocabulary-based; the taxonomy and the "related" groups are hand-made and incomplete.
* Language requirements are matched by name only (levels are not compared); education and experience checks use explicit statements only.
* Evidence levels and their thresholds are heuristics chosen for transparency, not validated; matching quality is unmeasured until you grade
  the pool.
* The local-LLM path is covered by tests against a mock Ollama server and scripted answers; response quality of a particular model must be
  checked by the human audit.


---

# Phase 4 - evaluation, reliability and portfolio readiness

New commands (see `README.md` for the measured results and `docs/LABELING_GUIDE.md` for the human workflow):

```powershell
# retrieval: BM25 vs semantic vs hybrid (Reciprocal Rank Fusion, k=60)
jobintel search "LLM agents and RAG" --engine all -k 5
jobintel eval pool                              # blinded pool incl. exhaustive queries (q02, q06, q09)
jobintel label retrieval [--only q01] [--stats] # interactive, blinded, resumable
jobintel eval run --per-query --save-baseline first   # after labelling; later: --compare-baseline first

# CV matching
jobintel match-eval diagnose                    # label-free: why jobs are excluded, pool coverage of eligible jobs
jobintel match-eval pool                        # candidate/job pairs + exclusion audit + unseen-job sample (strata hidden)
jobintel label matching                         # grade + role + eligibility, per pair
jobintel match-eval skills-pool ; jobintel label skills ; jobintel match-eval skills-score
jobintel match-eval run                         # ranking metrics + audit of the system's decisions (Wilson intervals)

# parser, explanations, regression
jobintel match-eval extraction --compare-phase3 # synthetic CVs with known ground truth, Phase 3 vs now
jobintel match-eval grounding --top 3           # provenance + verifier stress test + audit export (deterministic)
jobintel match-eval grounding --llm --model llama3.2:3b   # REAL Ollama run (only after you approve installing Ollama)
jobintel regress [--update]                     # label-free baselines; exit code 1 on regression
```

* `--engine all` prints semantic, BM25 and hybrid. `hybrid` fuses the two full rankings of the eligible jobs with
  `1/(60+rank)`; the constant is the published default and was **not tuned**.
* `match_candidate(..., prefilter=True)` applies the hard constraints to the whole index **before** retrieval.
  `prefilter=False` reproduces the Phase 3 behaviour for comparison.
* Explanations are tagged `deterministic`, `llm-stub` (a mock/scripted client) or `llm-ollama` (a real local model);
  reports never merge them and state "Real language-model output evaluated: NO" unless an Ollama run happened.
* `scripts/capture_screenshots.py` regenerates `docs/screenshots/` with a synthetic CV (needs `pip install playwright`
  and an installed Edge or Chrome).

---

# Phase 5 - the platform (enrichment, five engines, scoring, gaps, market, assistant, API, UI, benchmarks)

```powershell
jobintel collect                 # Arbeitnow + Himalayas + Remotive (polite, cached snapshots)
jobintel enrich                  # structured postings -> data/processed/jobs_enriched.jsonl + data/jobintel.db + quality_report.json
jobintel index                   # embeddings (GPU if available: JOBINTEL_DEVICE=auto|cuda|cpu)
jobintel search "stage data analyst Paris" --engine all
jobintel ui                      # Streamlit app, 9 pages
jobintel api                     # REST API, http://127.0.0.1:8000/docs
jobintel bench all               # reports/*.json + reports/evaluation_report.md (add --no-llm without Ollama)
python scripts/capture_screenshots.py   # with the UI running: drives it with Playwright and saves docs/screenshots/
python scripts/build_notebooks.py       # executes notebooks/01_exploration.ipynb and 02_evaluation.ipynb
python scripts/check_links.py           # checks the learning-resource URLs of the knowledge base
```

Configuration (environment variables, see `.env.example`): `JOBINTEL_DEVICE`, `JOBINTEL_HIMALAYAS_PAGES`,
`JOBINTEL_HIMALAYAS_TTL_H`, `JOBINTEL_LLM_MODEL`, `JOBINTEL_OLLAMA_URL`, `JOBINTEL_LLM_TRUSTED_HOSTS` (private hosts you
control, e.g. the `ollama` container), and the scoring weights in `config/scoring.yaml`.
