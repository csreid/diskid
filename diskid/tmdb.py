"""Map episodes to TMDB numbering via IMDb episode ids.

The subtitle archive keys episodes by IMDb's aired-order numbering, but media
servers (Jellyfin uses TMDB by default) can order episodes differently —
Futurama's held-over broadcast episodes are the classic case. Each archive
directory embeds the episode's own IMDb id, so the matched episode can be
remapped exactly, without trusting season/episode numbers at all.
"""

import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.themoviedb.org/3"


def api_key() -> str | None:
    """TMDB_APIKEY from the environment, or from ./.env."""
    key = os.environ.get("TMDB_APIKEY")
    if key:
        return key
    env = Path(".env")
    if env.is_file():
        for line in env.read_text().splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "TMDB_APIKEY":
                return value.strip().strip("'\"") or None
    return None


def _get(path: str, key: str, **params) -> dict:
    if "." in key and len(key) > 60:  # v4 read access token
        url = f"{API}{path}?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"})
    else:  # v3 api key
        url = f"{API}{path}?{urllib.parse.urlencode({**params, 'api_key': key})}"
        req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.load(resp)


class EpisodeMapper:
    """Resolves IMDb episode ids to TMDB (season, episode) numbering."""

    def __init__(self, key: str, show_imdb_id: int):
        self.key = key
        self.show_imdb_id = show_imdb_id
        self._show_tmdb_id: int | None = None
        self._cache: dict[int, tuple[int, int] | None] = {}

    def show_tmdb_id(self) -> int | None:
        if self._show_tmdb_id is None:
            found = _get(f"/find/tt{self.show_imdb_id:07d}", self.key,
                         external_source="imdb_id")
            shows = found.get("tv_results", [])
            self._show_tmdb_id = shows[0]["id"] if shows else -1
        return None if self._show_tmdb_id == -1 else self._show_tmdb_id

    def map_episode(self, episode_imdb_id: int) -> tuple[int, int] | None:
        """(season, episode) in TMDB numbering, or None if unknown to TMDB."""
        if episode_imdb_id in self._cache:
            return self._cache[episode_imdb_id]
        result = None
        try:
            found = _get(f"/find/tt{episode_imdb_id:07d}", self.key,
                         external_source="imdb_id")
            for hit in found.get("tv_episode_results", []):
                if hit.get("show_id") == self.show_tmdb_id():
                    result = (hit["season_number"], hit["episode_number"])
                    break
        except OSError as e:
            print(f"warning: TMDB lookup failed for tt{episode_imdb_id:07d}: {e}",
                  file=sys.stderr)
        self._cache[episode_imdb_id] = result
        return result
