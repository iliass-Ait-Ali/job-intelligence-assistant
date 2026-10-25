# Interview guide

How to present this project honestly and defend it. Read the **"claim / do not claim"** section first.

## 1. The 60-second pitch

> I built a local job-intelligence assistant on 191 real job listings. A pipeline collects and cleans them; a multilingual search combines
> embeddings in ChromaDB with BM25; and a CV matcher turns a CV into a structured profile, ranks jobs with explicit rules and explains each
> recommendation with evidence quoted from the CV and the listing. The part I am proudest of is that I treated it as an *evaluation* problem:
> blinded human labelling, metrics with confidence intervals, label-free regression baselines, and an explanation verifier that I stress-tested.
> I can show you what I measured, what I found wrong and fixed, and what I have **not** measured yet.

*Version française (30 s) :* « J'ai construit un assistant local d'analyse d'offres : collecte et nettoyage de 191 offres réelles, recherche
multilingue (embeddings + BM25), et matching CV-offres avec explications sourcées et vérifiées. Le plus important pour moi : l'évaluation -
annotation humaine en aveugle, métriques avec intervalles, tests de régression, et transparence sur ce qui n'est pas encore mesuré. »

## 2. What you can and cannot claim

| You CAN say | You CANNOT say (yet) |
|---|---|
| "476 offline tests, including real ChromaDB and a mock Ollama server" | "semantic / hybrid search beats BM25" - **no human judgments exist yet** |
| "hybrid = RRF of semantic and BM25; implemented and evaluable" | "the matcher is X% accurate" |
| "a diagnostic showed my candidate pool missed 5 of 8 eligible jobs; filtering before retrieval fixed the coverage (56% -> 100% on one CV)" | "recommendations are good" (coverage is not quality) |
| "the explanation verifier rejected 351/351 constructed corruptions" (say *constructed by me*) | "the system does not hallucinate" - no real LLM was ever run |
| "the parser is 1.00 on my synthetic layouts" (say *synthetic, written by me*) | "the parser works on real CVs" |
| "everything runs locally; CVs never go to an external API" | "it is secure against prompt injection" - say *reduces the risk* |

If you later enter the judgments, replace the left column with measured numbers *and their intervals*, and keep the caveats.

## 3. Design decisions and trade-offs

| Decision | Why | Trade-off / alternative |
|---|---|---|
| `multilingual-e5-base` on CPU | multilingual (FR/EN/DE/NL listings), MIT license, 512-token window, runs locally | slower to index (about 8 min for 1,074 chunks) than `-small`; BGE-M3 is heavier; I never compared models (that needs labels) |
| Chunking <=300 tokens, header on every chunk, tokenizer-counted | 182 of 191 listings exceed 512 tokens and the requirements section sits late in the text; a repeated header tells which job a chunk belongs to | more vectors; job score = best chunk (max), which can favour long, repetitive postings |
| Over-fetch until k distinct jobs are covered; filter *before* ANN search | one posting with many similar chunks must not crowd out others; a post-filter on a fixed top-N silently loses results | extra queries when a filter is selective |
| Own BM25 baseline | a baseline should be readable and dependency-free | no stemming / language-specific analysis |
| Hybrid = RRF with k=60 | parameter-free, scale-free (cosine and BM25 are incomparable), the published default | untuned on purpose (tuning on 10 queries would overfit); a weighted blend might do better |
| Rule-based CV parser + Pydantic | deterministic, auditable, testable, "never invent" is enforceable, no CV text goes to a model | brittle on exotic layouts; an LLM extractor would be more flexible but harder to guarantee and test |
| Tri-state constraints (`unknown` != mismatch) | 25% of listings lack a job type; excluding unknowns would hide valid jobs | more candidates shown with a "check this" flag |
| No percentage score | it would not be calibrated; people read it as a probability | less "demo-friendly" - I show counts + evidence instead |
| Verifier over claims, deterministic fallback | a small local LLM will not always follow citation rules; the user must still get a correct answer | the verifier checks *consistency* (ids, entities, numbers), not truth |
| Local Ollama only, loopback enforced | CVs are personal data | a bigger hosted model would write better prose |

## 4. The evaluation story (what to emphasise)

1. **Blinded human labelling** I can defend: no system names/ranks/scores on the card, saved per answer, resumable.
2. **Pooling bias is handled explicitly**: the pool unions BM25, semantic and hybrid; three small queries are judged *exhaustively* so true
   recall is computable; otherwise recall is reported only as "pooled" with that label.
3. **Honest statistics**: 10 queries, one annotator -> per-query tables, win/tie/loss, bootstrap interval over queries (n>=5), Wilson intervals
   for proportions. No p-values.
4. **Label-free regression baselines** catch regressions without annotators: parser accuracy on synthetic CVs, eligible-job pool coverage,
   evidence provenance, verifier stress test (`jobintel regress`, exit code 1 on regression).
5. **Mock vs real is enforced in code**: explanation sources are `deterministic`, `llm-stub`, `llm-ollama`; reports never merge them.

## 5. A good "tell me about a problem you found" story (STAR)

* **Situation.** The matcher printed "pool 80, 77 excluded" for a student CV, which looked broken.
* **Task.** Decide whether the logic, the data, or the retrieval was at fault - *without loosening constraints to make it look nicer*.
* **Action.** I wrote a label-free diagnostic that applies the constraints to the *whole* index independently of retrieval: of 191 jobs only 18
  satisfied "internship in France" (8 for "Paris"); the exclusions were correct (full-time jobs, other countries). The real bug was
  upstream: retrieval ran first, so an 80-job pool was mostly ineligible jobs and **5 of 8 eligible jobs never entered it**.
* **Result.** Constraints are now applied before retrieval; same decisions, same reasons; coverage of eligible jobs 56% -> 100% (student) and
  53% -> 92% (French CV). It is a coverage metric, so I do not claim better recommendations - only that eligible jobs are no longer lost.
  A human audit (incorrect exclusions, jobs never surfaced) is built to measure the rest.

## 6. Questions you will probably get

**Why not just use an LLM to extract the CV and score the jobs?**
Auditability and honesty. A rule-based extractor guarantees "no inference"; a verifier can then check claims against known evidence. I would add
an LLM extractor as an *option* behind the same Pydantic schema, evaluated on the same synthetic corpus.

**Why RRF and why k=60?** RRF fuses rankings without calibrating different score scales; 60 is the standard constant from the original paper.
I deliberately did not tune it on 10 queries.

**How do you know hybrid is better?** I do not. It is implemented and evaluable; the judgments are pending. (Say this plainly.)

**How do you evaluate without ground truth?** Pooled human judgments with explicit rubric, blinded cards, graded 0/1/2, nDCG with 2^g-1 gain,
P@5/MRR at two thresholds, recall only where judgments are complete, and uncertainty reported.

**Why nDCG and not accuracy?** Retrieval is a ranking problem with graded relevance and few relevant items per query; nDCG rewards
putting the best jobs first. P@5 and MRR are given for readability.

**What is the risk of pooling?** Unjudged items are treated as non-relevant, which favours the systems that built the pool; so `judged@10` is
reported and recall is gated on completeness.

**How do you prevent hallucination in explanations?** Four layers: (1) the model only sees numbered evidence snippets, (2) every item must cite
them, (3) a verifier checks that named skills/degrees/employers/numbers appear in the cited text and rejects probabilities/salaries,
(4) failures fall back to a deterministic template. I measured the verifier against constructed corruptions, but I have **not** run a real
Llama model, so I do not know its real hallucination rate.

**What about prompt injection in a CV or a listing?** Instruction-like lines are removed before they reach the model and are never cited;
snippets are JSON string values; the model has no tools; and the verifier blocks unsupported facts, so an obeyed injection cannot add content.
It reduces the risk; I would not call it solved.

**Why is a job "unknown" instead of "not matching"?** Because missing data is not evidence. 25% of listings lack a job type; excluding them would
silently hide valid jobs. The UI says what is unknown.

**How do you handle French/English/German?** The embedding model is multilingual; the skill taxonomy has FR/DE aliases; headings and dates are
parsed in three languages; language requirements are extracted by name (levels are not compared automatically).

**What does "transferable skill" mean?** A CV skill from the closest related group (PyTorch for TensorFlow). It is shown separately and is never
counted as an exact match.

**What would break on a real CV?** Two-column PDFs interleaving text, scanned PDFs (detected and reported, no OCR), unusual headings
(fallbacks help but are measured only on synthetic layouts), skills outside the ~150-entry taxonomy.

**Why Chroma and not FAISS/pgvector?** Persistent, metadata filtering and a simple local API were enough for 1,074 chunks; at this size an exact
scan would also be instant. I tested that Chroma's ranking matches brute-force cosine (ties aside).

**What is your biggest weakness?** No measured retrieval or matching quality yet, and no real LLM run. The tooling is ready; the labels are the gap.

**What would you do next with more time?** Grade the pools; run the Llama audit; compare e5-small/base and a weighted fusion on held-out queries;
an LLM-based extractor evaluated against the synthetic corpus; near-duplicate collapsing for multi-city postings; real-CV pilot with consent.

**How would you scale it?** The job set is tiny; the design changes are an ANN index with server-side filters, incremental indexing (already
idempotent by content hash), a scheduled collector with provider quotas, and per-user sessions.

## 7. A 3-minute demo script

1. `jobintel app --open`, paste `eval/synthetic_cvs/cv_en_student_data_science.txt`, **Analyse CV**: show the extracted profile and the warning behaviour.
2. Preferences: internship, France, hard constraints on. **Find matching jobs**: point at "173 excluded, 18 eligible", the exact / related / not-evidenced
   chips, and the `?` marks (unknown location).
3. AI analysis: **Explain match**: show the citations `[CV2, JOB1]`, the banner "deterministic - no language model was used", the removed-items section.
4. Terminal: `jobintel match-eval diagnose` (why jobs are excluded) and `jobintel regress` (label-free baselines).
5. Say what is *not* measured and show `docs/LABELING_GUIDE.md`.

## 8. CV bullet points (French) - only what is implemented and verified

* Conçu un pipeline de données reproductible (ingestion d'API publiques avec quotas et cache, nettoyage, déduplication) alimentant un
  corpus de 191 offres réelles, soutenu par 476 tests automatisés exécutables hors ligne.
* Développé une recherche d'offres multilingue (FR/EN/DE) combinant embeddings *multilingual-e5* + ChromaDB, BM25 et fusion RRF, avec un cadre
  d'évaluation en aveugle (nDCG, MRR, intervalles de Wilson/bootstrap) et des baselines de non-régression.
* Réalisé un assistant local de matching CV-offres (Gradio, Ollama optionnel) produisant des explications sourcées et vérifiées
  automatiquement (citations CV/offre, contrôle des entités, défenses anti-injection) ; un diagnostic a fait passer la couverture des offres
  éligibles dans le pool candidat de 56 % à 100 % sur un CV de test.

**After you enter the judgments**, a measured bullet may replace the evaluation clause - for example "... hybrid vs BM25 : nDCG@10 = X
[IC 95 % a-b] sur 10 requêtes annotées" - but *only* with the real numbers and their interval. Do not use this template before then.
