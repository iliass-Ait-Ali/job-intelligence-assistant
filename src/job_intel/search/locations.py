"""Location normalization for filtering.

Locations are free text from the providers ("Zürich", "Zurich", "Amsterdam, Noord-Holland,
Nederland", "UK - London", "Lille - Jardin des Plantes, Nord; Paris, Paris"). A job is described
by a set of *keys*: its accent-stripped, lower-cased segments and words, with spelling variants
mapped to one canonical name, plus the country inferred from a small city gazetteer. A wanted
location is normalized the same way and must be one of the job's keys.
"""

from __future__ import annotations

import re
import unicodedata

# spelling variant (already normalized) -> canonical name
ALIASES = {
    "nederland": "netherlands", "pays bas": "netherlands", "niederlande": "netherlands", "holland": "netherlands",
    "deutschland": "germany", "allemagne": "germany",
    "frankreich": "france",
    "schweiz": "switzerland", "suisse": "switzerland", "svizzera": "switzerland",
    "uk": "united kingdom", "great britain": "united kingdom", "royaume uni": "united kingdom",
    "koln": "cologne", "munchen": "munich", "geneve": "geneva", "the hague": "den haag", "la haye": "den haag",
    "ile de france": "ile de france",
}  # fmt: skip

CITY_COUNTRY = {
    **dict.fromkeys(
        ["amsterdam", "den haag", "s hertogenbosch", "dongen", "ede", "heerlen", "leiden", "rijssen",
         "rotterdam", "tricht", "utrecht", "venlo", "zoetermeer", "zwolle", "eindhoven"],
        "netherlands",
    ),
    **dict.fromkeys(
        ["berlin", "cologne", "munich", "hamburg", "heidelberg", "karlsruhe", "frankfurt", "stuttgart", "dusseldorf"],
        "germany",
    ),
    **dict.fromkeys(
        ["bern", "birr", "dubendorf", "lausanne", "lyss", "rapperswil jona", "solothurn", "zurich", "geneva"],
        "switzerland",
    ),
    **dict.fromkeys(
        ["bordeaux", "boulogne billancourt", "chaponost", "issy les moulineaux", "lille", "lyon", "nantes",
         "paris", "pouzauges", "toulouse", "marseille", "nice", "strasbourg"],
        "france",
    ),
    **dict.fromkeys(["london", "manchester", "edinburgh"], "united kingdom"),
    "sydney": "australia",
}  # fmt: skip

_SEGMENT_SPLIT = re.compile(r"[,;/|()]|\s[-–—]\s")


def normalize(text: str) -> str:
    """Accent-strip, casefold, turn punctuation into single spaces."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).casefold()
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def canonical(term: str) -> str:
    norm = normalize(term)
    return ALIASES.get(norm, norm)


def location_keys(location: str | None) -> frozenset[str]:
    """All match keys for a job location; empty when the location is missing."""
    if not location:
        return frozenset()
    keys: set[str] = set()
    for segment in _SEGMENT_SPLIT.split(location):
        norm = canonical(segment)
        if not norm:
            continue
        keys.add(norm)
        keys.update(canonical(word) for word in norm.split())
    keys.update(CITY_COUNTRY[k] for k in list(keys) if k in CITY_COUNTRY)
    keys.discard("")
    return frozenset(keys)


def location_matches(location: str | None, wanted: str) -> bool:
    return canonical(wanted) in location_keys(location)
