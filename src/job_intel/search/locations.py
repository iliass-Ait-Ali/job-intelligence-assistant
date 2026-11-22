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
    "usa": "united states", "us": "united states", "united states of america": "united states", "etats unis": "united states",
    "maroc": "morocco", "marokko": "morocco", "espagne": "spain", "spanien": "spain", "espana": "spain",
    "italie": "italy", "italien": "italy", "belgique": "belgium", "belgien": "belgium", "osterreich": "austria",
    "autriche": "austria", "polen": "poland", "pologne": "poland", "inde": "india", "irlande": "ireland",
    "bruxelles": "brussels", "wien": "vienna", "lisboa": "lisbon", "anywhere": "worldwide", "global": "worldwide",
    "remote": "remote", "emea": "europe", "eu": "europe", "european union": "europe",
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
    **dict.fromkeys(["london", "manchester", "edinburgh", "cambridge", "oxford", "bristol"], "united kingdom"),
    **dict.fromkeys(["casablanca", "rabat", "marrakech", "tanger", "tangier", "fes", "agadir", "meknes", "oujda",
                     "kenitra", "tetouan"], "morocco"),
    **dict.fromkeys(["madrid", "barcelona", "valencia", "seville"], "spain"),
    **dict.fromkeys(["brussels", "antwerp", "ghent", "leuven"], "belgium"),
    **dict.fromkeys(["vienna", "graz", "linz"], "austria"),
    **dict.fromkeys(["warsaw", "krakow", "wroclaw", "gdansk"], "poland"),
    **dict.fromkeys(["lisbon", "porto"], "portugal"),
    **dict.fromkeys(["dublin", "cork"], "ireland"),
    **dict.fromkeys(["milan", "rome", "turin"], "italy"),
    **dict.fromkeys(["new york", "san francisco", "seattle", "boston", "austin", "chicago"], "united states"),
    **dict.fromkeys(["toronto", "montreal", "vancouver"], "canada"),
    "sydney": "australia",
}  # fmt: skip

EUROPE = frozenset({
    "france", "germany", "netherlands", "switzerland", "united kingdom", "spain", "italy", "belgium", "austria",
    "poland", "portugal", "ireland", "sweden", "denmark", "norway", "finland", "czech republic", "czechia", "romania",
    "greece", "hungary", "luxembourg", "estonia", "latvia", "lithuania", "croatia", "bulgaria", "slovakia", "slovenia",
    "serbia", "ukraine", "cyprus", "malta", "iceland",
})  # fmt: skip
COUNTRIES = EUROPE | frozenset({
    "morocco", "united states", "canada", "india", "australia", "brazil", "mexico", "argentina", "colombia", "chile",
    "south africa", "nigeria", "kenya", "egypt", "tunisia", "algeria", "turkey", "israel", "united arab emirates",
    "saudi arabia", "singapore", "philippines", "japan", "china", "south korea", "new zealand", "pakistan",
    "indonesia", "vietnam", "malaysia", "thailand",
})  # fmt: skip
REGIONS = {"europe": EUROPE, "worldwide": COUNTRIES}

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


def countries_of(location: str | None) -> list[str]:
    """Countries named or implied (through a known city) by a location, sorted; may be empty."""
    keys = location_keys(location)
    return sorted(k for k in keys if k in COUNTRIES)


def region_of(location: str | None) -> str | None:
    keys = location_keys(location)
    if "worldwide" in keys:
        return "worldwide"
    if "europe" in keys:
        return "europe"
    return None


def remote_allows(restriction: str | None, wanted: str) -> bool | None:
    """Does a remote job restricted to `restriction` (e.g. "United States, Canada", "Worldwide", "Europe")
    accept candidates in `wanted`? None when the restriction is missing or `wanted` is not a known country/city."""
    keys = location_keys(restriction)
    if not keys:
        return None
    want = canonical(wanted)
    want_country = CITY_COUNTRY.get(want, want)
    if "worldwide" in keys:
        return True
    if want in keys or want_country in keys:
        return True
    if "europe" in keys and want_country in EUROPE:
        return True
    if want_country not in COUNTRIES:
        return None
    return False
