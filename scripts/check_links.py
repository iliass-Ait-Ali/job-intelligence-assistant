"""Check that every learning-resource URL of the knowledge base answers (run occasionally; network needed)."""

import sys

import requests
import yaml

from job_intel.match.gap import KB_PATH


def main() -> int:
    kb = yaml.safe_load(KB_PATH.read_text(encoding="utf-8"))
    urls = sorted({r["url"] for e in kb["skills"].values() if e for r in (e.get("resources") or [])})
    print(f"{len(kb['skills'])} skills in the knowledge base, {len(urls)} distinct resource URLs")
    bad = []
    for url in urls:
        try:
            r = requests.get(url, timeout=20, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0 (link check)"})
            status = r.status_code
        except requests.RequestException as exc:
            status = type(exc).__name__
        ok = isinstance(status, int) and status < 400
        print(("OK  " if ok else "FAIL"), status, url)
        if not ok:
            bad.append(url)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
