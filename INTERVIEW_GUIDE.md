# Interview guide

A one-page brief for presenting this project. The full French version (pitch, demo script, ~10 likely questions
with answers) is [docs/learn/10_entretien.md](docs/learn/10_entretien.md); the illustrated walkthrough is
[docs/RAPPORT_PEDAGOGIQUE.md](docs/RAPPORT_PEDAGOGIQUE.md).

## Two-minute pitch

A search, matching and recommendation engine for data/AI jobs — not a chatbot. 720 real listings from three
authorised APIs are cleaned, deduplicated and structured (required/preferred skills with the sentence that states
them, level, country, languages) into SQLite. Retrieval fuses BM25 and multilingual embeddings searched exactly
with FAISS through Reciprocal Rank Fusion, with optional cross-encoder re-ranking. Matching applies the user's
eliminatory constraints first (unknown never excludes), then an interpretable compatibility index whose points are
attributable to criteria; a weight-calibration experiment was run with leave-one-candidate-out cross-validation and
**rejected** because it over-fitted. Skill gaps separate "absent" from "not mentioned" by asking the user, and the
learning plan is ordered by a topological sort of prerequisites. A RAG assistant cites numbered documents for every
sentence and a verifier removes unsupported sentences. Everything runs locally: FastAPI, Streamlit, 530+ tests, CI,
Docker.

## Numbers to know (labels proposed by an AI, not yet human-verified — say it)

| What | Value |
|---|---|
| Hybrid vs BM25, nDCG@10 (10 queries) | 0.842 [0.75, 0.93] vs 0.754 [0.60, 0.89]; 6 wins / 1 tie / 3 losses |
| Cross-encoder re-rank | 0.851, several times slower: kept optional |
| Skill extraction, held-out test | F1 0.80 (P 0.91, R 0.70); 19% of gold skills outside the taxonomy |
| Score weights | fitted 0.79 in-sample → 0.64 out-of-sample vs 0.66 default: defaults kept |
| Eliminatory constraints | precision 1.00 (no eligible job wrongly excluded), recall 0.72 |

## Limits to state before being asked

AI-proposed labels; 10 queries and 3 synthetic CVs; rule-based parser and finite taxonomy; weak sector heuristic;
the corpus is three boards, not the labour market; a small local LLM whose proposed sentences are often rejected.
