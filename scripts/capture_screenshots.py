"""Capture the README screenshots of the Gradio app.

Uses a SYNTHETIC CV (eval/synthetic_cvs), the real local index, the deterministic explanation mode, and the
Edge/Chrome already installed on the machine (Playwright's `channel`, so no browser download).

    python -m pip install playwright
    python scripts/capture_screenshots.py
"""

from __future__ import annotations

import re
import socket
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "docs" / "screenshots"
CV = ROOT / "eval" / "synthetic_cvs" / "cv_en_student_data_science.txt"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main() -> int:
    from playwright.sync_api import sync_playwright

    from job_intel.app.controller import AppContext
    from job_intel.app.gradio_app import build_app
    from job_intel.config import Settings

    OUT.mkdir(parents=True, exist_ok=True)
    port = free_port()
    ctx = AppContext(Settings.from_env())
    demo = build_app(ctx)
    demo.launch(server_name="127.0.0.1", server_port=port, prevent_thread_lock=True, quiet=True)
    try:
        with sync_playwright() as p:
            browser = None
            for channel in ("msedge", "chrome"):
                try:
                    browser = p.chromium.launch(channel=channel, headless=True)
                    break
                except Exception as exc:  # try the next installed browser
                    print(f"{channel}: {exc}", file=sys.stderr)
            if browser is None:
                print("No Edge/Chrome found for Playwright.", file=sys.stderr)
                return 1
            page = browser.new_page(viewport={"width": 1320, "height": 900})
            page.goto(f"http://127.0.0.1:{port}/")
            page.get_by_role("tab", name="1 · Candidate profile").wait_for()

            page.get_by_label("…or paste the CV text (French or English)").fill(CV.read_text(encoding="utf-8"))
            page.get_by_role("button", name="Analyse CV").click()
            page.get_by_text("CV parsed").wait_for(timeout=60000)
            page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / "01_profile.png"), full_page=True)

            page.get_by_role("tab", name="2 · Preferences").click()
            page.get_by_label("internship", exact=True).check()
            page.get_by_label("Preferred locations (city, region or country; comma-separated)").fill("France")
            page.wait_for_timeout(400)
            page.screenshot(path=str(OUT / "02_preferences.png"), full_page=True)

            page.get_by_role("tab", name="3 · Recommendations").click()
            page.get_by_role("button", name="Find matching jobs").click()
            page.get_by_text("recommendations.", exact=False).first.wait_for(timeout=180000)
            page.wait_for_timeout(800)
            page.screenshot(path=str(OUT / "03_recommendations.png"), full_page=False)

            page.get_by_role("tab", name="4 · AI analysis").click()
            page.get_by_role("button", name="Explain match").click()
            page.get_by_text("Why it matches", exact=False).first.wait_for(timeout=120000)  # only present in a rendered result
            page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / "04_explanation.png"), full_page=True)

            for number in (1, 2, 3):
                page.get_by_role("checkbox", name=re.compile(rf"^{number}\. ")).check()
            page.get_by_role("button", name="Compare selected").click()
            page.get_by_text("not evidenced in the CV:", exact=False).first.wait_for(timeout=120000)
            page.wait_for_timeout(600)
            page.screenshot(path=str(OUT / "05_comparison.png"), full_page=True)
            browser.close()
    finally:
        demo.close()
    for png in sorted(OUT.glob("*.png")):
        print(png.relative_to(ROOT), png.stat().st_size // 1024, "KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
