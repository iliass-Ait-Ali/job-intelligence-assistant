"""Drive the running Streamlit app like a user and save screenshots to docs/screenshots/.

    jobintel ui &            # or: streamlit run src/job_intel/ui/app.py
    python scripts/capture_screenshots.py [--url http://127.0.0.1:8501]

Uses a SYNTHETIC CV (eval/synthetic_cvs); listings shown are real and link to their sources.
Also works as an end-to-end smoke test: it fails if any page shows a Streamlit exception.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT = Path(__file__).resolve().parents[1] / "docs" / "screenshots"


def wait_idle(page: Page, timeout: float = 240.0) -> None:
    """Wait until Streamlit stops running (no 'Running...' status widget, no spinner)."""
    page.wait_for_timeout(1200)
    deadline = time.time() + timeout
    while time.time() < deadline:
        running = page.locator('[data-testid="stStatusWidget"]').count() > 0
        spinning = page.locator('[data-testid="stSpinner"]').count() > 0
        if not running and not spinning:
            page.wait_for_timeout(800)
            return
        page.wait_for_timeout(500)
    raise TimeoutError("the app did not become idle")


def check_no_exception(page: Page, name: str) -> None:
    if page.locator('[data-testid="stException"]').count():
        text = page.locator('[data-testid="stException"]').first.inner_text()[:800]
        raise RuntimeError(f"{name}: Streamlit exception on the page:\n{text}")


def shot(page: Page, name: str, full: bool = True) -> None:
    wait_idle(page)
    check_no_exception(page, name)
    OUT.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(OUT / f"{name}.png"), full_page=full)
    print("saved", name)


def goto(page: Page, url: str, path: str) -> None:
    page.goto(f"{url}/{path}")
    wait_idle(page)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8501")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900}, device_scale_factor=1)
        page.set_default_timeout(240_000)
        goto(page, url, "")
        shot(page, "01_overview")

        # load a synthetic CV through the UI (client-side navigation keeps the session)
        page.get_by_role("link", name="CV & Profile").click()
        wait_idle(page)
        page.get_by_role("tab", name="Use a synthetic example").click()
        page.wait_for_timeout(500)
        page.locator('[data-testid="stSelectbox"]').first.click()
        page.get_by_role("option", name="en student data science").click()
        page.get_by_role("button", name="Load example").click()
        shot(page, "02_profile")

        page.get_by_role("link", name="Recommendations").click()
        shot(page, "04_recommendations")

        page.get_by_role("button", name="Details").first.click()
        wait_idle(page)
        shot(page, "05_job_details")

        page.get_by_role("link", name="Skill Gap & Plan").click()
        shot(page, "06_skill_gap")

        page.get_by_role("link", name="Job Search").click()
        wait_idle(page)
        shot(page, "03_job_search")
        page.get_by_text("Compare all five engines side by side").click()
        shot(page, "03b_engine_comparison", full=False)

        page.get_by_role("link", name="Market Intelligence").click()
        shot(page, "07_market")
        page.get_by_role("tab", name="Skill associations").click()
        shot(page, "07b_market_associations")

        page.get_by_role("link", name="Career Assistant").click()
        wait_idle(page)
        page.get_by_role("button", name="Quelles compétences me manquent pour l'offre 1 ?").click()
        wait_idle(page)
        page.get_by_text("Sources (").first.click()  # show the cited documents
        shot(page, "08_assistant")

        page.get_by_role("link", name="Evaluation & Method").click()
        shot(page, "09_evaluation")
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
