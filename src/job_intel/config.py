"""Runtime settings, read from environment variables (optionally via a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

from job_intel import __version__

DEFAULT_TTL_HOURS = {"arbeitnow": 1.0, "remotive": 6.0}


@dataclass(frozen=True)
class Settings:
    user_agent: str
    data_dir: Path = Path("data")
    connect_timeout: float = 5.0
    read_timeout: float = 30.0
    request_delay: float = 2.0
    max_retries: int = 2
    arbeitnow_max_pages: int = 10
    ttl_hours: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TTL_HOURS))
    # Phase 2 (semantic search)
    embedding_model: str = "intfloat/multilingual-e5-base"
    device: str = "cpu"
    chunk_max_tokens: int = 300
    chunk_overlap_tokens: int = 40
    eval_dir: Path = Path("eval")

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def jobs_path(self) -> Path:
        return self.processed_dir / "jobs.jsonl"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index" / "chroma"

    @property
    def eval_work_dir(self) -> Path:
        """Git-ignored scratch space for evaluation artefacts that contain job text."""
        return self.data_dir / "eval"

    @classmethod
    def from_env(cls) -> Settings:
        load_dotenv()
        contact = os.getenv("JOBINTEL_CONTACT", "").strip()
        agent = f"job-intel-assistant/{__version__} (portfolio project"
        agent += f"; contact: {contact})" if contact else ")"
        return cls(
            user_agent=agent,
            data_dir=Path(os.getenv("JOBINTEL_DATA_DIR", "data")),
            connect_timeout=float(os.getenv("JOBINTEL_CONNECT_TIMEOUT_S", 5)),
            read_timeout=float(os.getenv("JOBINTEL_READ_TIMEOUT_S", 30)),
            request_delay=float(os.getenv("JOBINTEL_REQUEST_DELAY_S", 2.0)),
            max_retries=int(os.getenv("JOBINTEL_MAX_RETRIES", 2)),
            arbeitnow_max_pages=int(os.getenv("JOBINTEL_ARBEITNOW_MAX_PAGES", 10)),
            embedding_model=os.getenv("JOBINTEL_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
            device=os.getenv("JOBINTEL_DEVICE", "cpu"),
            chunk_max_tokens=int(os.getenv("JOBINTEL_CHUNK_MAX_TOKENS", 300)),
            chunk_overlap_tokens=int(os.getenv("JOBINTEL_CHUNK_OVERLAP_TOKENS", 40)),
            eval_dir=Path(os.getenv("JOBINTEL_EVAL_DIR", "eval")),
            ttl_hours={
                "arbeitnow": float(os.getenv("JOBINTEL_ARBEITNOW_TTL_H", 1)),
                "remotive": float(os.getenv("JOBINTEL_REMOTIVE_TTL_H", 6)),
            },
        )
