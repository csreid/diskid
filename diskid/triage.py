"""Decide whether a disc ARM filed as a *movie* is really a TV disc.

ARM searches a DVD's volume label as a movie first, so a TV disc labelled
`30_ROCK` can come out as "Barbie in Rock 'N Royals" in `movies/`. This looks
at the files themselves, plus (when known) the claimed film's real runtime:

- **movie**: the longest file matches the claimed runtime, or the disc has a
  long feature and no episode-shaped run of files. Leave it alone.
- **tv**: three or more files of similar length between 15 and 70 minutes, and
  nothing over 80, unless the claimed runtime matches the longest file.
  Identify it as a series.
- **review**: the longest file doesn't fit the claimed runtime but the disc
  isn't episode-shaped either. Don't touch it; flag it for a person.

Pure function of durations so it is easy to test; the watcher supplies them.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Literal

#: Episode-length window (minutes) and how alike episodes on one disc must be.
EPISODE_MIN, EPISODE_MAX = 15.0, 70.0
EPISODE_SPREAD = 0.25
MIN_EPISODES = 3
#: Anything longer than this is a feature, not an episode.
FEATURE_MIN = 80.0
#: How far the longest file may differ from the claimed runtime (fraction).
RUNTIME_TOLERANCE = 0.20


@dataclass(frozen=True)
class Verdict:
    kind: Literal["movie", "tv", "review"]
    reason: str


def classify(durations_s: list[float], claimed_runtime_min: float | None = None) -> Verdict:
    """Classify a disc from its file durations (seconds) and the claimed film's runtime."""
    if not durations_s:
        return Verdict("movie", "no readable files")
    mins = sorted(d / 60 for d in durations_s)
    longest = mins[-1]

    runtime_fits = None
    if claimed_runtime_min:
        runtime_fits = abs(longest - claimed_runtime_min) <= RUNTIME_TOLERANCE * claimed_runtime_min
        if runtime_fits:
            return Verdict("movie", f"longest file {longest:.0f} min matches claimed "
                                    f"runtime {claimed_runtime_min:.0f} min")

    in_range = [m for m in mins if EPISODE_MIN <= m <= EPISODE_MAX]
    if in_range:
        med = statistics.median(in_range)
        similar = [m for m in in_range if abs(m - med) <= EPISODE_SPREAD * med]
    else:
        similar = []
    if len(similar) >= MIN_EPISODES and longest <= FEATURE_MIN:
        why = (f"{len(similar)} files of ~{statistics.median(similar):.0f} min and nothing "
               f"over {FEATURE_MIN:.0f}")
        if runtime_fits is False:
            why += f" (claimed runtime {claimed_runtime_min:.0f} min doesn't fit)"
        return Verdict("tv", why)

    if runtime_fits is False:
        return Verdict("review", f"longest file {longest:.0f} min doesn't fit claimed "
                                 f"runtime {claimed_runtime_min:.0f} min, but not episode-shaped")
    return Verdict("movie", f"longest file {longest:.0f} min, no run of episode-length files")
