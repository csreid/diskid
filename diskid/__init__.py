"""diskid -- identify ripped TV-episode files by subtitle fingerprint.

Library entry point (see `diskid.identify` for details)::

    from diskid import Identifier, IdentifierConfig, apply_rename

    ident = Identifier(IdentifierConfig(model="small"))
    disc = ident.identify_disc("/completed/Firefly (2002)")
    for f in disc.files:
        if f.identified:
            apply_rename(f)          # 'Firefly - S01E02.mkv'

Lower-level building blocks (Whisper wrapper, subtitle archive access, the
Bayesian matcher) are importable from their submodules: `diskid.stt`,
`diskid.subdb`, `diskid.match`, `diskid.imdb`, `diskid.tmdb`, `diskid.audio`.
"""

from importlib.metadata import PackageNotFoundError, version

from .identify import (
    DiscResult,
    EpisodeRef,
    FileResult,
    Identifier,
    IdentifierConfig,
    ProgressEvent,
    RenameOutcome,
    Show,
    ShowNotFoundError,
    apply_rename,
    apply_renames,
    episode_filename,
)
from .imdb import parse_title_year
from .match import ConsensusScore

try:
    __version__ = version("diskid")
except PackageNotFoundError:  # running from a source tree without install metadata
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    "Identifier",
    "IdentifierConfig",
    "Show",
    "EpisodeRef",
    "FileResult",
    "DiscResult",
    "ConsensusScore",
    "RenameOutcome",
    "ProgressEvent",
    "ShowNotFoundError",
    "apply_rename",
    "apply_renames",
    "episode_filename",
    "parse_title_year",
]
