"""Show-name resolution via the unofficial IMDb suggestion API (no key needed)."""

import json
import re
import urllib.parse
import urllib.request


def resolve_show_candidates(name: str, year: int | None = None) -> list[tuple[int, str]]:
    """All plausible (numeric imdb id, canonical title) matches for a show name.

    Collects every tvSeries/tvMiniSeries hit from the suggestion API, ranked
    so a hit whose start year matches `year` sorts first (ties/no-year-given
    keep the API's own relevance order, which is already rank-sorted).
    """
    q = urllib.parse.quote(name.lower())
    url = f"https://v2.sg.media-imdb.com/suggestion/{name[0].lower()}/{q}.json"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.load(resp)
    hits = [hit for hit in data.get("d", []) if hit.get("qid") in ("tvSeries", "tvMiniSeries")]
    if year is not None:
        hits.sort(key=lambda hit: hit.get("y") != year)
    return [(int(hit["id"].removeprefix("tt")), hit["l"]) for hit in hits]


def resolve_show(name: str) -> tuple[int, str]:
    """Resolve a show name to (numeric imdb id, canonical title).

    Raises LookupError if IMDb has no TV series for the query.
    """
    candidates = resolve_show_candidates(name)
    if not candidates:
        raise LookupError(f"no TV series found on IMDb for {name!r}")
    return candidates[0]


def parse_title_year(dirname: str) -> tuple[str, int | None]:
    """Parse ARM-style 'Title (Year)' folder names; year is optional."""
    m = re.fullmatch(r"(.+?)\s*\((\d{4})\)", dirname.strip())
    if m:
        return m.group(1), int(m.group(2))
    return dirname.strip(), None
