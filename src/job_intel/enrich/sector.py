"""Industry-sector heuristic for listings and CV experiences.

Providers give no reliable industry field, so the sector is *estimated* from keyword evidence in the
text (company description, product words). Rules:

* each sector has a short multilingual keyword list (word-boundary, accent-insensitive);
* a sector needs at least ``MIN_HITS`` distinct keywords to be assigned (one stray "bank" is not
  enough), the best-scoring sector wins, ties are broken alphabetically for determinism;
* otherwise the sector is ``None`` ("unspecified"), never a guess.

This is a transparent baseline, NOT a validated classifier: its errors are discussed in
``docs/learn/01_data_pipeline.md`` and every assignment keeps the keywords that triggered it.
"""

from __future__ import annotations

import re

from job_intel.match.cv_parser import fold

MIN_HITS = 2

SECTOR_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Finance & Banking": ("bank", "banking", "banque", "fintech", "payments", "payment", "trading", "asset management",
                          "credit", "lending", "investment", "financial services", "insurance", "insurtech",
                          "assurance", "versicherung", "wealth", "brokerage", "crypto", "blockchain"),
    "Healthcare & Life sciences": ("health", "healthcare", "medical", "clinical", "patients", "patient", "hospital",
                                   "pharma", "pharmaceutical", "biotech", "life sciences", "sante", "medtech",
                                   "gesundheit", "drug discovery", "genomics"),
    "E-commerce & Retail": ("e-commerce", "ecommerce", "retail", "marketplace", "shopping", "customers order",
                            "online shop", "fashion", "consumer goods", "grocery", "commerce"),
    "Consulting & Services": ("consulting", "consultancy", "conseil", "beratung", "advisory", "our clients",
                              "client projects", "consultants", "esn", "it services"),
    "Software & SaaS": ("saas", "software company", "platform for", "b2b software", "developer tools", "cloud platform",
                        "api platform", "open source", "devtools"),
    "Energy & Climate": ("energy", "energie", "renewable", "solar", "wind", "climate", "carbon", "electricity",
                         "utilities", "battery", "grid", "sustainability"),
    "Mobility & Automotive": ("automotive", "mobility", "vehicle", "vehicles", "cars", "autonomous driving",
                              "transport", "logistics", "fleet", "railway", "aviation", "aerospace"),
    "Media, Gaming & Entertainment": ("media", "gaming", "games", "video", "music", "streaming", "entertainment",
                                      "publishing", "news", "advertising", "adtech"),
    "Telecom": ("telecom", "telecommunications", "network operator", "5g", "mobile operator"),
    "Industry & Manufacturing": ("manufacturing", "industrial", "industry 4.0", "factory", "production plant",
                                 "semiconductor", "chemicals", "robotics", "hardware", "machinery"),
    "Public sector & Education": ("public sector", "government", "ministry", "university", "research institute",
                                  "education", "edtech", "students", "non-profit", "ngo", "school"),
    "HR & Recruitment": ("recruitment", "staffing", "talent acquisition", "hr tech", "hr software", "recruiting agency",
                         "personalvermittlung", "interim"),
    "Cybersecurity": ("cybersecurity", "security operations", "threat", "siem", "soc analyst", "zero trust"),
    "Real estate & Construction": ("real estate", "proptech", "construction", "immobilier", "buildings"),
}

_PATTERNS = {
    sector: [(kw, re.compile(rf"(?<![a-z0-9]){re.escape(fold(kw))}(?![a-z0-9])")) for kw in kws]
    for sector, kws in SECTOR_KEYWORDS.items()
}


def classify_sector(*texts: str | None) -> tuple[str | None, list[str]]:
    """(sector or None, keywords that triggered it)."""
    folded = fold(" \n ".join(t for t in texts if t))
    best: tuple[int, str] | None = None
    hits_by_sector: dict[str, list[str]] = {}
    for sector, patterns in _PATTERNS.items():
        hits = [kw for kw, rx in patterns if rx.search(folded)]
        if len(hits) >= MIN_HITS:
            hits_by_sector[sector] = hits
            key = (-len(hits), sector)
            if best is None or key < best:
                best = key  # type: ignore[assignment]
    if best is None:
        return None, []
    return best[1], hits_by_sector[best[1]]
