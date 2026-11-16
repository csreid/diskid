"""Calibration data: what the matcher saw and decided, plus (later) what was true.

The model constants in `match.py` (`STT_ERROR_RATE`, `TEXT_LLR_CAP`, ...) are
hand-picked. To fit them, and to check that "posterior 0.99" means ~99% right,
we need real observations with ground truth. This module records both, in one
append-only JSON-lines file (`<state>/calibration.jsonl`), as two event kinds:

`observation` -- written whenever a file is identified. Holds the raw inputs, so
    any constants can be refit offline by replaying `BayesianIdentifier.observe`
    over `clips[*].text` at `clips[*].t`: `id`, `show`, `show_imdb`, `file`,
    `size`, `duration`, `config` (eps, clip length, thresholds, STT model),
    `constants` (the match.py constants in force), `clips` (per clip: `t` capture
    time, `text` transcript -- empty when the clip was silent -- and the posterior
    right after: `top` [season, episode, score], `offset`, `null`), `ranked` (top 5
    [season, episode, score], IMDb numbering), `null`, `n_clips`.

`label` -- ground truth for an observation: `id` (of the observation),
    `episode` [season, episode] in IMDb numbering, and `source`:
      - "plan_applied": a plan was applied, i.e. the model's own confident answer
        was accepted. Weak: it can only confirm what the model already thought, and
        errors nobody noticed count as correct. Selection-biased toward easy files.
      - "manual": a person checked it (strong).
    Records without a label are unresolved, not wrong.

Join on `id`; the latest label for an id wins. Transcripts are of the user's own
media, so the file stays in the private state directory and is never uploaded.
"""

from __future__ import annotations

import uuid
from pathlib import Path

from . import match
from .eventlog import EventLog

#: The `match.py` constants that shape the posterior, recorded with every observation.
CONSTANT_NAMES = (
    "STT_ERROR_RATE", "RETRIEVAL_HIT_RATE", "TIME_JITTER_SIGMA", "TEXT_LLR_CAP",
    "GRID_RANGE", "GRID_STEP", "LENGTH_SIGMA", "LENGTH_PAD", "LENGTH_NULL_RANGE",
)


def new_id() -> str:
    return uuid.uuid4().hex[:10]


def log(path: Path | None) -> EventLog:
    """The calibration log at `path` (a no-op log when None)."""
    return EventLog(path)


def default_path(state_dir: Path) -> Path:
    return state_dir / "calibration.jsonl"


def model_constants() -> dict[str, float]:
    return {name: getattr(match, name) for name in CONSTANT_NAMES}


def write_label(path: Path | None, obs_id: str | None, episode: tuple[int, int],
                source: str, **extra) -> None:
    """Record that observation `obs_id` is really `episode` (IMDb numbering)."""
    if path is None or not obs_id:
        return
    log(path).write("label", id=obs_id, episode=list(episode), source=source, **extra)
