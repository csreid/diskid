"""Access to the OPUS OpenSubtitles raw archive (subtitle_data/en.zip).

The archive stays zipped forever; entries are read lazily via random access.
Layout: OpenSubtitles/raw/en/<year>/<key>/<file_id>.xml where <key> is a bare
IMDb id (movies) or <episode_imdb>_<show_imdb>_<season>_<episode> (TV).
"""

import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from collections import defaultdict
from pathlib import Path

DEFAULT_ZIP = Path(__file__).resolve().parent.parent / "subtitle_data" / "en.zip"

EPISODE_RE = re.compile(
    r"OpenSubtitles/raw/en/[^/]+/(\d+)_(\d+)_(\d+)_(\d+)/[^/]+\.xml$"
)

MOVIE_RE = re.compile(r"OpenSubtitles/raw/en/(\d+)/(\d+)/[^/]+\.xml$")

EpisodeKey = tuple[int, int]  # (season, episode)


def episode_index(zip_path: Path) -> Path:
    """Build (once) a TSV of show_id, season, episode, zip entry for all episodes."""
    index = zip_path.with_suffix(".episodes.tsv")
    if index.exists():
        return index
    print(f"building episode index for {zip_path} (one-time)...", file=sys.stderr)
    with zipfile.ZipFile(zip_path) as z, open(index, "w") as out:
        for entry in z.namelist():
            m = EPISODE_RE.fullmatch(entry)
            if m:
                _, show, season, ep = m.groups()
                out.write(f"{show}\t{season}\t{ep}\t{entry}\n")
    return index


def show_subtitles(zip_path: Path, show_id: int) -> dict[EpisodeKey, list[str]]:
    """Map (season, episode) -> list of zip entries for the given show."""
    episodes: dict[EpisodeKey, list[str]] = defaultdict(list)
    with open(episode_index(zip_path)) as f:
        for line in f:
            show, season, ep, entry = line.rstrip("\n").split("\t")
            if int(show) == show_id:
                episodes[(int(season), int(ep))].append(entry)
    return dict(episodes)


def parse_timestamp(value: str) -> float:
    """Parse a subtitle "<time>" value ("HH:MM:SS,mmm") into seconds."""
    hms, ms = value.split(",")
    h, m, s = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + s + int(ms) / 1000


def subtitle_words(z: zipfile.ZipFile, entry: str) -> list[tuple[str, float]]:
    """(word, timestamp) pairs, where timestamp is the enclosing sentence's start time.

    Needed for clip-position matching: each word inherits the time its
    sentence started, so a matched window's word span can be read back as a
    point in the episode's timeline.
    """
    root = ET.fromstring(z.read(entry))
    words = []
    last_time = 0.0
    for s in root.iter("s"):
        times = s.findall("time")
        if times:
            last_time = parse_timestamp(times[0].get("value"))
        text = " ".join(t.strip() for t in s.itertext() if t.strip())
        # drop hearing-impaired annotations like [laughing]; whisper won't produce them
        text = re.sub(r"\[[^\]]*\]", " ", text)
        words.extend((w, last_time) for w in text.split())
    return words


def episode_imdb_ids(zip_path: Path, show_id: int) -> dict[EpisodeKey, int]:
    """Map (season, episode) -> the episode's own IMDb id.

    Parsed from the archive directory names (<episode_imdb>_<show>_<s>_<e>);
    where duplicate uploads disagree, the most common id wins.
    """
    counts: dict[EpisodeKey, dict[int, int]] = {}
    for key, entries in show_subtitles(zip_path, show_id).items():
        per_key = counts.setdefault(key, {})
        for entry in entries:
            ep_id = int(entry.split("/")[-2].split("_")[0])
            per_key[ep_id] = per_key.get(ep_id, 0) + 1
    return {key: max(ids, key=ids.get) for key, ids in counts.items()}


def show_word_docs(zip_path: Path, show_id: int) -> dict[EpisodeKey, list[list[tuple[str, float]]]]:
    """Map (season, episode) -> list of subtitle files, each a (word, timestamp) sequence."""
    episodes = show_subtitles(zip_path, show_id)
    docs: dict[EpisodeKey, list[list[tuple[str, float]]]] = {}
    with zipfile.ZipFile(zip_path) as z:
        for key, entries in episodes.items():
            docs[key] = [subtitle_words(z, e) for e in entries]
    return docs


def movie_index(zip_path: Path) -> Path:
    """Build (once) a TSV of imdb_id, year, zip entry for all movie subtitle files.

    Movies live under a bare IMDb id directory (no `_`), inside the movie's
    release-year directory. Kept beside the zip as `<zip>.movies.tsv`.
    """
    index = zip_path.with_suffix(".movies.tsv")
    if index.exists():
        return index
    print(f"building movie index for {zip_path} (one-time)...", file=sys.stderr)
    tmp = index.with_name(index.name + ".part")
    with zipfile.ZipFile(zip_path) as z, open(tmp, "w") as out:
        for entry in z.namelist():
            m = MOVIE_RE.fullmatch(entry)
            if m:
                year, imdb_id = m.groups()
                out.write(f"{imdb_id}\t{year}\t{entry}\n")
    tmp.replace(index)  # never leave a half-written index that looks complete
    return index


def movie_subtitles(zip_path: Path, imdb_id: int) -> list[str]:
    """Zip entries of every subtitle file for the movie with this numeric IMDb id."""
    entries = []
    with open(movie_index(zip_path)) as f:
        for line in f:
            movie, _, entry = line.rstrip("\n").split("\t")
            if int(movie) == imdb_id:
                entries.append(entry)
    return entries


def movie_catalog(zip_path: Path) -> dict[int, tuple[int, int]]:
    """Map imdb id -> (release year, number of subtitle files), for every movie."""
    catalog: dict[int, tuple[int, int]] = {}
    with open(movie_index(zip_path)) as f:
        for line in f:
            movie, year, _ = line.rstrip("\n").split("\t")
            _, n = catalog.get(int(movie), (int(year), 0))
            catalog[int(movie)] = (int(year), n + 1)
    return catalog


def movie_word_docs(zip_path: Path, imdb_id: int) -> list[list[tuple[str, float]]]:
    """Every subtitle file of one movie as a (word, timestamp) sequence."""
    entries = movie_subtitles(zip_path, imdb_id)
    with zipfile.ZipFile(zip_path) as z:
        return [subtitle_words(z, e) for e in entries]
