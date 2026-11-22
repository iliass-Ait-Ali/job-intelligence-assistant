"""Manual smoke test of the assistant on the real corpus (not part of the test suite)."""
import json
import sys

from job_intel.assistant.chat import AssistantSession, ask
from job_intel.config import Settings
from job_intel.enrich.postings import load_postings
from job_intel.match.cv_parser import parse_cv
from job_intel.match.matcher import match_candidate
from job_intel.match.models import Preferences
from job_intel.match.scoring import rank_by_score
from job_intel.rag.ollama import OllamaClient
from job_intel.search import service
from job_intel.search.faiss_index import FaissChunkIndex
from job_intel.search.retrieve import Searcher

s = Settings.from_env()
searcher, _ = service.open_searcher(s)
jobs = {j["id"]: j for j in service.load_jobs(s.jobs_path)}
fs = Searcher(FaissChunkIndex(searcher.index), searcher.embedder, jobs)
postings = {p.id: p for p in load_postings(s.enriched_path)}
p = sys.argv[1] if len(sys.argv) > 1 else "eval/synthetic_cvs/cv_fr_data_engineer.txt"
prof = parse_cv(open(p, encoding="utf-8").read())
prefs = Preferences(**json.load(open(p.replace(".txt", ".prefs.json"))))
res = match_candidate(prof, prefs, fs)
sess = AssistantSession(postings, fs, prof, rank_by_score(res.matches, prof))
client = OllamaClient(model="llama3.2:3b")
QS = ["Quelles sont les cinq offres les plus pertinentes pour mon profil ?", "Why does job 2 match my CV?",
      "Which companies are looking for Machine Learning and PyTorch?", "Compare these three opportunities",
      "Quels projets dois-je réaliser pour améliorer mon profil ?"]
for q in QS:
    a = ask(sess, q)
    print("Q:", q, "|", a.intent, a.mode, a.latency_s, "kept", len(a.sentences), "rejected", len(a.rejected))
    print(a.markdown()[:500])
    print(a.rejected[:2])
for q in QS[1:3]:
    a = ask(sess, q, llm_client=client)
    print("LLM Q:", q, a.mode, a.latency_s, "kept", len(a.sentences), "rejected", a.rejected[:3])
    print(a.markdown())
