"""Run one benchmark with progress logging: python scripts/run_bench.py <retrieval|skills|scoring|assistant|latency|all> [--no-llm]"""

import logging
import sys

from job_intel.config import Settings
from job_intel.evaluation.runner import run_bench

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpcore", "urllib3", "huggingface_hub", "sentence_transformers", "chromadb", "faiss"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    sys.exit(run_bench(what, Settings.from_env(), use_llm="--no-llm" not in sys.argv))
