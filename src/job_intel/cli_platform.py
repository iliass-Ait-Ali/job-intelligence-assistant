"""CLI commands for the platform layer: `jobintel api`, `jobintel ui`, `jobintel bench`."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from job_intel.config import Settings

PLATFORM_COMMANDS = {"api", "ui", "bench"}
UI_ENTRY = Path(__file__).resolve().parent / "ui" / "app.py"


def add_platform_commands(sub: argparse._SubParsersAction) -> None:
    api = sub.add_parser("api", help="serve the REST API (FastAPI + uvicorn); docs at /docs")
    api.add_argument("--host", default="127.0.0.1")
    api.add_argument("--port", type=int, default=8000)
    api.add_argument("--reload", action="store_true")

    ui = sub.add_parser("ui", help="launch the Streamlit web application")
    ui.add_argument("--port", type=int, default=8501)
    ui.add_argument("--host", default="127.0.0.1")

    bench = sub.add_parser("bench", help="run the evaluation benchmarks and write reports/")
    bench.add_argument("what", choices=["retrieval", "skills", "scoring", "assistant", "latency", "all"])
    bench.add_argument("--no-llm", action="store_true", help="assistant benchmark without the local LLM")


def run(args: argparse.Namespace, settings: Settings) -> int:
    if args.command == "api":
        import uvicorn

        uvicorn.run("job_intel.api.main:app", host=args.host, port=args.port, reload=args.reload)
        return 0
    if args.command == "ui":
        cmd = [sys.executable, "-m", "streamlit", "run", str(UI_ENTRY), "--server.port", str(args.port),
               "--server.address", args.host, "--browser.gatherUsageStats", "false"]  # fmt: skip
        return subprocess.call(cmd)
    from job_intel.evaluation.runner import run_bench

    return run_bench(args.what, settings, use_llm=not args.no_llm)
