"""High-level library API: identify ripped episode files and plan renames.

This is the entry point for programs embedding diskid (e.g. a sidecar that
watches ARM's completed-rips folder). The CLI is a thin wrapper over it.

    from diskid import Identifier, IdentifierConfig, apply_rename

    ident = Identifier(IdentifierConfig(model="small"))
    disc = ident.identify_disc("/completed/Firefly (2002)")
    for f in disc.files:
        if f.identified:
            print(f.path, f.episode.season, f.episode.episode, f.target_name)
            apply_rename(f)
        else:
            print(f.path, f.status, f.error or f.best)

An `Identifier` is reusable and expensive to build up: it lazily loads the
Whisper model once and caches each show's subtitle data, so keep one around
for the life of the process rather than creating one per file.

Everything returned is a plain dataclass; call `.to_dict()` on a result for a
JSON-serializable form. Nothing here prints or renames behind your back --
renaming only happens through `apply_rename`/`apply_renames`.

Numbering: matching happens in the subtitle archive's IMDb aired-order
numbering. `FileResult.episode` is in the order requested by
`IdentifierConfig.order` (TMDB by default, falling back to IMDb order with a
warning if no TMDB key is available or the episode is unmapped), and always
carries the IMDb numbering too.
"""

from __future__ import annotations

import dataclasses
import errno
import logging
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from . import audio, calibration, imdb, layout, stt, subdb, tmdb
from .match import (
    STT_ERROR_RATE,
    BayesianIdentifier,
    ConsensusScore,
    assign_episodes,
    is_confident,
    sig,
)

log = logging.getLogger("diskid")

Status = Literal["identified", "uncertain", "failed"]
Order = Literal["tmdb", "imdb"]

#: How many ranked candidate episodes to keep on each `FileResult`.
TOP_CANDIDATES = 10
#: Max distinct shows tried when guessing the show from disc metadata.
MAX_SHOW_GUESSES = 3


class ShowNotFoundError(LookupError):
    """No show with subtitles in the archive could be determined."""


# --------------------------------------------------------------------------
# configuration and result types
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IdentifierConfig:
    """Tunables for an `Identifier`. Defaults match the CLI's defaults."""

    model: str = "small"
    """faster-whisper model size. Uses CUDA when available, else CPU."""
    language: str | None = None
    """Language code for STT (e.g. "en"); None autodetects."""
    clip_duration: float = 8.0
    """Seconds of audio per transcribed clip."""
    max_clips: int = 10
    """Cap on clips sampled per file; sampling stops early once confident."""
    min_clips: int = 3
    """Clips that must be observed before sampling may stop early. One
    coincidental line match can clear `min_score` on its own; requiring several
    clips lets a wrong episode be contradicted. Capped by the clips available."""
    start: float | None = None
    """If set, sample a single clip at this offset (seconds) instead."""
    min_score: float = 0.99
    """Posterior probability the top episode must reach to count as identified."""
    eps: float = STT_ERROR_RATE
    """Assumed STT word-error rate (a tuning knob; see match.py)."""
    output_dir: Path | None = None
    """Root that `template` is relative to. None: relative to each file's own folder
    (i.e. rename in place)."""
    template: str = layout.DEFAULT_TEMPLATE
    """Jinja template for the destination path; see `diskid.layout`."""
    order: Order = "tmdb"
    """Numbering for `FileResult.episode`/filenames: "tmdb" or "imdb"."""
    zip_path: Path = subdb.DEFAULT_ZIP
    """The OPUS OpenSubtitles raw English archive."""
    tmdb_api_key: str | None = None
    """TMDB key; None falls back to $TMDB_APIKEY / ./.env."""
    stt_url: str | None = None
    """Base URL of an OpenAI-compatible transcription server (e.g.
    "http://host:8000/v1"). If set, audio is sent there and local Whisper (and
    its `faster-whisper` dependency) is never loaded; `model` is then ignored."""
    stt_model: str | None = None
    """Model name to request from the server; None means
    "Systran/faster-whisper-<model>" (Speaches' naming)."""
    stt_api_key: str | None = None
    stt_timeout: float = 120.0
    """Seconds to wait for one clip's transcription."""
    calibration_log: Path | None = None
    """If set, append one JSON line per identified file (clip transcripts, per-clip
    posterior, final ranking, the model constants) so the constants can be refit
    offline; see `diskid.calibration`. None: record nothing."""

    def __post_init__(self) -> None:
        if self.order not in ("tmdb", "imdb"):
            raise ValueError(f"order must be 'tmdb' or 'imdb', not {self.order!r}")
        if not 0 < self.min_score < 1:
            raise ValueError("min_score must be in (0, 1)")
        if self.max_clips < 1:
            raise ValueError("max_clips must be >= 1")
        try:
            layout.validate(self.template)
        except layout.TemplateError as e:
            raise ValueError(f"bad template: {e}") from e
        if self.min_clips < 1:
            raise ValueError("min_clips must be >= 1")


@dataclass(frozen=True)
class Show:
    imdb_id: int
    title: str
    year: int | None = None
    """Year the series began, when known (IMDb's suggestion API supplies it)."""

    @property
    def imdb_tt(self) -> str:
        """The id in IMDb's own form, e.g. 'tt0303461'."""
        return f"tt{self.imdb_id:07d}"


@dataclass(frozen=True)
class EpisodeRef:
    """An identified episode, in both the requested and IMDb numbering."""

    season: int
    episode: int
    imdb_season: int
    imdb_episode: int
    order: Order
    """Which numbering `season`/`episode` are in (what was actually applied,
    which is "imdb" if a requested TMDB mapping was unavailable)."""
    imdb_id: int | None = None
    """The episode's own IMDb numeric id, when the archive records it."""
    title: str | None = None
    """The episode's name, when TMDB supplied it (needs a TMDB key)."""

    @property
    def code(self) -> str:
        return f"S{self.season:02d}E{self.episode:02d}"


@dataclass(frozen=True)
class FileResult:
    """The outcome of identifying one video file."""

    path: Path
    status: Status
    """"identified" (confident, unique episode), "uncertain" (best guess below
    `min_score`, or its episode was claimed by a more confident file on the
    same disc), or "failed" (couldn't be read; see `error`)."""
    show: Show | None = None
    episode: EpisodeRef | None = None
    """Set iff status == "identified"."""
    candidates: list[ConsensusScore] = field(default_factory=list)
    """Top episodes by posterior (IMDb numbering), best first, incl. losers."""
    null_probability: float | None = None
    """Posterior that the file is not any episode in the database
    (extra, trailer, wrong show...)."""
    clips_used: int = 0
    duration: float | None = None
    """File runtime in seconds."""
    target_path: Path | None = None
    """Where the file should go, from the template (under `output_dir`, else
    beside the source); set iff identified."""
    target_rel: str | None = None
    """`target_path` relative to its base, e.g. 'Show (2020)/Season 01/Show - S01E02.mkv'."""
    warnings: list[str] = field(default_factory=list)
    error: str | None = None
    calibration_id: str | None = None
    """Id of this file's record in the calibration log (None when not recording)."""

    @property
    def identified(self) -> bool:
        return self.status == "identified"

    @property
    def best(self) -> ConsensusScore | None:
        """Top-ranked candidate, whether or not it was confident enough."""
        return self.candidates[0] if self.candidates else None

    @property
    def target_name(self) -> str | None:
        """Just the filename of `target_path`."""
        return self.target_path.name if self.target_path else None

    def template_vars(self) -> dict[str, Any] | None:
        """Structured identification facts (what the path template is rendered
        with), or None when the file isn't identified."""
        if not (self.identified and self.show and self.episode):
            return None
        ep = self.episode
        return layout.template_vars(
            show=self.show.title, year=self.show.year, season=ep.season,
            episode=ep.episode, episode_title=ep.title, ext=self.path.suffix,
            stem=self.path.stem, order=ep.order, show_imdb=self.show.imdb_tt,
            episode_imdb=f"tt{ep.imdb_id:07d}" if ep.imdb_id else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class DiscResult:
    """Results for every file under a ripped-disc folder."""

    directory: Path
    show: Show | None
    files: list[FileResult]

    @property
    def identified(self) -> list[FileResult]:
        return [f for f in self.files if f.identified]

    @property
    def unidentified(self) -> list[FileResult]:
        return [f for f in self.files if not f.identified]

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass(frozen=True)
class RenameOutcome:
    source: Path
    target: Path | None
    renamed: bool
    reason: str = ""
    """Why nothing was renamed ("" when it was)."""


@dataclass(frozen=True)
class ProgressEvent:
    """Passed to the optional `progress` callback."""

    stage: Literal["loading_show", "trying_show", "transcribing", "file_done"]
    path: Path | None = None
    clip: int = 0
    max_clips: int = 0
    text: str | None = None
    """Transcript of the clip just processed (stage == "transcribing")."""
    message: str = ""


ProgressCallback = Callable[[ProgressEvent], None]


def _jsonable(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        d = {f.name: _jsonable(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
        if isinstance(obj, ConsensusScore):
            d["key"] = [obj.season, obj.episode]
        if isinstance(obj, EpisodeRef):
            d["code"] = obj.code
        if isinstance(obj, FileResult):
            d["target_name"] = obj.target_name
        return d
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    return obj


# --------------------------------------------------------------------------
# small pure helpers (also used by the CLI)
# --------------------------------------------------------------------------

def sample_clips(total: float, start: float | None, n_clips: int, duration: float) -> list[float]:
    """Pick up to `n_clips` capture start times for short clips.

    Spread across the middle 70% of the runtime (15%-85%): cold opens and
    recaps quote *other* episodes, and endings run into credits/previews.
    `total` is the file's full duration in seconds. An explicit `start`
    yields just that one clip.
    """
    if start is not None:
        return [start]
    if total <= duration / 0.7:
        return [max(0.0, (total - duration) / 2)]
    lo, hi = 0.15 * total, 0.85 * total
    if n_clips == 1:
        return [(lo + hi) / 2]
    step = (hi - lo) / (n_clips - 1)
    return [lo + i * step for i in range(n_clips)]


sanitize_filename = layout.sanitize_filename


def episode_filename(show_title: str, season: int, episode: int, suffix: str) -> str:
    """'Show Title - S01E02.mkv' (Jellyfin/Plex-friendly)."""
    return f"{sanitize_filename(show_title)} - S{season:02d}E{episode:02d}{suffix}"


def rename_to_target(video: Path, target: Path) -> tuple[Path | None, str]:
    """Move `video` to `target`, creating folders; never overwrites.

    Returns (new path, "") or (None, reason). Falls back to copy-and-delete
    when `target` is on another filesystem.
    """
    if target == video:
        return video, ""
    if not video.is_file():
        return None, "source file missing"
    if target.exists():
        return None, "target exists"
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            video.rename(target)
        except OSError as e:
            if e.errno != errno.EXDEV:
                raise
            shutil.move(video, target)
    except OSError as e:
        return None, f"rename failed: {e.strerror}"
    return target, ""


def apply_rename(result: FileResult) -> RenameOutcome:
    """Rename an identified file to its `target_name`. Never overwrites."""
    target = result.target_path
    if not result.identified or target is None:
        return RenameOutcome(result.path, None, False, "not identified")
    new, reason = rename_to_target(result.path, target)
    return RenameOutcome(result.path, target, new is not None, reason)


def apply_renames(results: list[FileResult]) -> list[RenameOutcome]:
    """`apply_rename` over many results (e.g. `DiscResult.files`)."""
    return [apply_rename(r) for r in results]


# --------------------------------------------------------------------------
# the Identifier
# --------------------------------------------------------------------------

@dataclass
class _ShowData:
    show: Show
    docs: dict
    imdb_ids: dict
    mapper: tmdb.EpisodeMapper | None = None


@dataclass
class MovieCheck:
    """Does a file's audio match a claimed movie's subtitles?"""

    imdb_tt: str
    year: int | None
    score: float            # posterior of the claimed movie among itself + decoys + null
    null: float             # posterior that the file is none of them
    clips_used: int
    duration: float
    subtitle_runtime: float | None   # canonical runtime from the subtitles, seconds
    n_subtitle_files: int
    confident: bool
    top_other: float        # best decoy's posterior (should be ~0)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(self)


@dataclass
class _Raw:
    """A file's matching outcome before disc-level assignment."""

    path: Path
    ranked: list[ConsensusScore] = field(default_factory=list)
    null_probability: float | None = None
    clips_used: int = 0
    duration: float | None = None
    error: str | None = None
    data: _ShowData | None = None
    calibration_id: str | None = None


class Identifier:
    """Identifies video files against the subtitle archive.

    Not thread-safe (it shares one Whisper model). Reuse one instance across
    files so the model and per-show subtitle data are loaded once.
    """

    def __init__(
        self,
        config: IdentifierConfig | None = None,
        *,
        progress: ProgressCallback | None = None,
    ):
        self.config = config or IdentifierConfig()
        self.progress = progress
        self._model = None
        self._device: str | None = None
        self._remote: stt.RemoteTranscriber | None = None
        if self.config.stt_url:
            self._remote = stt.RemoteTranscriber(
                self.config.stt_url,
                self.config.stt_model or f"Systran/faster-whisper-{self.config.model}",
                self.config.stt_api_key,
                self.config.stt_timeout,
            )
        self._shows: dict[int, _ShowData] = {}

    # -- public ----------------------------------------------------------

    @property
    def device(self) -> str:
        """"remote", "cuda" or "cpu" -- loads the local Whisper model if needed."""
        if self._remote:
            return "remote"
        self._load_model()
        assert self._device is not None
        return self._device

    def resolve_show(self, name: str, year: int | None = None) -> Show:
        """Look up a show by name on IMDb. Raises `ShowNotFoundError`."""
        hits = imdb.resolve_show_hits(name, year)
        if not hits:
            raise ShowNotFoundError(f"no TV series found on IMDb for {name!r}")
        return Show(*hits[0])

    def episode_ref(self, show: Show | str, candidate: ConsensusScore) -> EpisodeRef:
        """Number any candidate (e.g. a non-winning `FileResult.candidates` entry)
        in the configured order, like `FileResult.episode` is for the winner."""
        data = self._load_show(show)
        return self._numberer(data)(candidate)[0]

    def identify_file(
        self,
        path: str | Path,
        show: Show | str | None = None,
        *,
        hint: str | None = None,
    ) -> FileResult:
        """Identify one video file.

        `show`: a `Show` or a show name. If omitted, the show is guessed from
        `hint` (an ARM-style 'Title (Year)' string; default: the file's parent
        folder name), then the container's title tag.

        Raises `FileNotFoundError` for a missing file/archive and
        `ShowNotFoundError` if the show can't be determined. A file ffmpeg
        can't read yields a result with status "failed" instead of raising.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if show is not None:
            data = self._load_show(show)
        else:
            data = None
        raw = self._process(path, data, hint or path.resolve().parent.name)
        if raw.data is None and raw.error is None:
            raise ShowNotFoundError(f"could not determine show for {path}")
        numberer = self._numberer(raw.data) if raw.data else None
        best = raw.ranked[0] if raw.ranked and is_confident(raw.ranked, self.config.min_score) else None
        result = self._finalize(raw, best, numberer)
        self._emit("file_done", path=path)
        return result

    def identify_disc(
        self,
        directory: str | Path,
        show: Show | str | None = None,
        *,
        hint: str | None = None,
    ) -> DiscResult:
        """Identify every file under `directory` (recursively).

        Files are assigned *distinct* episodes: a disc never holds the same
        episode twice, so when two files claim one, only the more confident
        keeps it. If `show` is omitted it is guessed from `hint` (default: the folder name)
        ('Title (Year)') and container title tags; raises `ShowNotFoundError`
        if no file allows that. Every file is attempted -- non-video files
        come back as status "failed".
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(directory)
        files = sorted(p for p in directory.rglob("*") if p.is_file())
        data = self._load_show(show) if show is not None else None
        hint = hint or directory.resolve().name

        raws: list[_Raw] = []
        for video in files:
            raw = self._process(video, data, hint)
            data = raw.data or data
            raws.append(raw)
            self._emit("file_done", path=video)
        if data is None:
            raise ShowNotFoundError(f"could not determine show for {directory}")
        # files seen before the show could be guessed were never matched
        for i, raw in enumerate(raws):
            if raw.error is None and raw.data is None:
                raws[i] = self._process(raw.path, data, hint)

        ranked_by_file = {str(r.path): r.ranked for r in raws if r.error is None}
        assigned = assign_episodes(ranked_by_file, self.config.min_score)
        numberer = self._numberer(data)
        results = [self._finalize(r, assigned.get(str(r.path)), numberer, data) for r in raws]
        return DiscResult(directory, data.show, results)

    # -- internals -------------------------------------------------------

    def _emit(self, stage: str, **kw: Any) -> None:
        if self.progress:
            self.progress(ProgressEvent(stage=stage, **kw))  # type: ignore[arg-type]

    def _load_model(self):
        if self._model is None:
            self._model, self._device = stt.load_model(self.config.model)
            log.info("whisper model=%s device=%s", self.config.model, self._device)
        return self._model

    def _load_show(self, show: Show | str) -> _ShowData:
        if isinstance(show, str):
            show = self.resolve_show(show)
        if show.imdb_id in self._shows:
            return self._shows[show.imdb_id]
        zip_path = self.config.zip_path
        if not Path(zip_path).is_file():
            raise FileNotFoundError(zip_path)
        self._emit("loading_show", message=f"{show.title} ({show.imdb_tt})")
        docs = subdb.show_word_docs(zip_path, show.imdb_id)
        if not docs:
            raise ShowNotFoundError(f"no subtitles for {show.imdb_tt} in {zip_path}")
        data = _ShowData(show, docs, subdb.episode_imdb_ids(zip_path, show.imdb_id))
        self._shows[show.imdb_id] = data
        return data

    def check_movie(self, video: Path, imdb_id: int, decoys: int = 8) -> MovieCheck:
        """Verify `video` is the movie with this numeric IMDb id.

        Reuses the episode matcher: the claimed movie competes against a few
        random other movies (which give the background word model something
        to compare to) and the null hypothesis. A file of some other movie, or
        an extra, ends with its probability mass on the null/decoys.
        """
        import random
        from .match import _episode_length

        zip_path = self.config.zip_path
        if not Path(zip_path).is_file():
            raise FileNotFoundError(zip_path)
        claimed = subdb.movie_word_docs(zip_path, imdb_id)
        claimed = [d for d in claimed if d]
        if not claimed:
            raise ShowNotFoundError(f"no subtitles for tt{imdb_id:07d} in {zip_path}")
        catalog = subdb.movie_catalog(zip_path)
        rng = random.Random(imdb_id)
        others = rng.sample([m for m in catalog if m != imdb_id], decoys)
        docs: dict = {(1, 0): claimed}  # (0, 0) is the matcher's null key
        for i, other in enumerate(others, 1):
            other_docs = [d for d in subdb.movie_word_docs(zip_path, other)[:3] if d]
            if other_docs:
                docs[(1, i)] = other_docs
        total = audio.probe_duration(video)
        data = _ShowData(Show(imdb_id, f"tt{imdb_id:07d}", None), docs, {})
        ident = self._run_clips(video, data, total)
        ranked = {c.key: c.score for c in ident.ranked()}
        return MovieCheck(
            imdb_tt=f"tt{imdb_id:07d}", year=catalog.get(imdb_id, (None, 0))[0],
            score=ranked.get((1, 0), 0.0), null=ident.null_probability(),
            clips_used=ident.n_clips, duration=total,
            subtitle_runtime=_episode_length(claimed), n_subtitle_files=len(claimed),
            confident=ranked.get((1, 0), 0.0) >= self.config.min_score,
            top_other=max((v for k, v in ranked.items() if k != (1, 0)), default=0.0),
        )

    def _transcribe_clip(self, video: Path, start: float) -> str:
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "clip.wav"
            audio.extract_audio(video, start, self.config.clip_duration, wav)
            if self._remote:
                return self._remote.transcribe(wav, self.config.language)
            model = self._load_model()
            segments, _ = stt.transcribe_wav(model, wav, self.config.language)
            return " ".join(seg.text for seg in segments)

    def _run_clips(self, video: Path, data: _ShowData, total: float) -> BayesianIdentifier:
        cfg = self.config
        starts = sample_clips(total, cfg.start, cfg.max_clips, cfg.clip_duration)
        ident = BayesianIdentifier(data.docs, eps=cfg.eps, duration=total)
        for i, start in enumerate(starts, 1):
            text = self._transcribe_clip(video, start)
            ident.observe(start + cfg.clip_duration / 2, text)
            self._emit("transcribing", path=video, clip=i, max_clips=len(starts), text=text)
            if i >= cfg.min_clips and ident.confident(cfg.min_score):
                break
        return ident

    def _process(self, video: Path, data: _ShowData | None, hint: str) -> _Raw:
        """Match one file; guesses the show (returned in `.data`) when `data` is None."""
        raw = _Raw(path=video, data=data)
        try:
            total = audio.probe_duration(video)
            raw.duration = total
            if data is None:
                ident, data = self._guess_show(video, total, hint)
                if data is None:
                    return raw  # no show resolvable from this file
                raw.data = data
            else:
                ident = self._run_clips(video, data, total)
        except subprocess.CalledProcessError as e:
            raw.error = f"ffmpeg/ffprobe failed ({e.returncode})"
            log.warning("%s: %s", video, raw.error)
            return raw
        except ValueError as e:  # e.g. ffprobe printed no duration
            raw.error = f"unreadable media: {e}"
            return raw
        raw.ranked = ident.ranked()
        raw.null_probability = ident.null_probability()
        raw.clips_used = ident.n_clips
        raw.calibration_id = self._record_calibration(raw, ident)
        return raw

    def _record_calibration(self, raw: _Raw, ident: BayesianIdentifier) -> str | None:
        """Append this file's observations to the calibration log; returns the record id."""
        if self.config.calibration_log is None or raw.data is None:
            return None
        cfg = self.config
        try:
            size = raw.path.stat().st_size
        except OSError:
            size = None
        rec_id = calibration.new_id()
        calibration.log(cfg.calibration_log).write(
            "observation", id=rec_id, show=raw.data.show.title, show_imdb=raw.data.show.imdb_tt,
            file=str(raw.path), size=size, duration=raw.duration,
            config=dict(eps=cfg.eps, clip_duration=cfg.clip_duration, min_score=cfg.min_score,
                        min_clips=cfg.min_clips, max_clips=cfg.max_clips,
                        stt_model=cfg.stt_model or cfg.model, remote_stt=bool(cfg.stt_url)),
            constants=calibration.model_constants(),
            clips=ident.trace,
            ranked=[[c.season, c.episode, sig(c.score)] for c in raw.ranked[:5]],
            null=raw.null_probability, n_clips=raw.clips_used,
        )
        return rec_id

    def _guess_show(
        self, video: Path, total: float, hint: str,
    ) -> tuple[BayesianIdentifier, _ShowData] | tuple[None, None]:
        """Try disc-metadata show guesses in turn, stopping at the first confident one.

        Falls back to the best-scoring attempt if none is confident; returns
        (None, None) only if no guess resolves to a show in the archive.
        """
        guesses = [imdb.parse_title_year(hint)]
        try:
            tag = audio.probe_title(video)
        except (subprocess.CalledProcessError, OSError):
            tag = None
        if tag and tag.strip() and tag.strip() != guesses[0][0]:
            guesses.append((tag.strip(), None))

        seen: set[int] = set()
        best: tuple[float, BayesianIdentifier, _ShowData] | None = None
        for name, year in guesses:
            try:
                hits = imdb.resolve_show_hits(name, year)
            except OSError as e:
                log.warning("IMDb lookup failed for %r: %s", name, e)
                continue
            for show_id, title, show_year in hits[:2]:
                if show_id in seen or len(seen) >= MAX_SHOW_GUESSES:
                    continue
                seen.add(show_id)
                try:
                    data = self._load_show(Show(show_id, title, show_year))
                except ShowNotFoundError:
                    continue
                self._emit("trying_show", path=video,
                           message=f"{title} ({data.show.imdb_tt}) [guessed from {name!r}]")
                ident = self._run_clips(video, data, total)
                if ident.confident(self.config.min_score):
                    return ident, data
                ranked = ident.ranked()
                score = ranked[0].score if ranked else 0.0
                if best is None or score > best[0]:
                    best = (score, ident, data)
        return (best[1], best[2]) if best else (None, None)

    def _numberer(self, data: _ShowData) -> Callable[[ConsensusScore], tuple[EpisodeRef, list[str]]]:
        """Build ConsensusScore -> (EpisodeRef, warnings) in the configured order."""
        cfg = self.config
        warnings: list[str] = []
        mapper = None
        if cfg.order == "tmdb":
            key = cfg.tmdb_api_key or tmdb.api_key()
            if key:
                mapper = data.mapper or tmdb.EpisodeMapper(key, data.show.imdb_id)
                data.mapper = mapper
            else:
                warnings.append("no TMDB_APIKEY; using IMDb episode order")
                log.warning(warnings[-1])

        def number(ep: ConsensusScore) -> tuple[EpisodeRef, list[str]]:
            ep_id = data.imdb_ids.get(ep.key)
            mapped = mapper.map_episode(ep_id) if (mapper and ep_id) else None
            if mapped is not None:
                return EpisodeRef(mapped.season, mapped.episode, ep.season, ep.episode,
                                  "tmdb", ep_id, mapped.title), []
            ws = list(warnings)
            if mapper is not None:
                ws.append(f"no TMDB mapping for IMDb S{ep.season:02d}E{ep.episode:02d}; "
                          "using IMDb order")
                log.warning(ws[-1])
            return EpisodeRef(ep.season, ep.episode, ep.season, ep.episode, "imdb", ep_id), ws

        return number

    def _finalize(
        self,
        raw: _Raw,
        assigned: ConsensusScore | None,
        numberer: Callable[[ConsensusScore], tuple[EpisodeRef, list[str]]] | None,
        disc_data: _ShowData | None = None,
    ) -> FileResult:
        data = raw.data or disc_data
        common = dict(
            path=raw.path,
            show=data.show if data else None,
            candidates=raw.ranked[:TOP_CANDIDATES],
            null_probability=raw.null_probability,
            clips_used=raw.clips_used,
            duration=raw.duration,
            calibration_id=raw.calibration_id,
        )
        if raw.error is not None:
            return FileResult(status="failed", error=raw.error, **common)
        if assigned is None or numberer is None:
            warnings = []
            best = raw.ranked[0] if raw.ranked else None
            if best is not None and best.score >= self.config.min_score:
                warnings.append(
                    f"best match {best.season:02d}x{best.episode:02d} was claimed by a more "
                    "confident file on this disc"
                )
            return FileResult(status="uncertain", warnings=warnings, **common)
        ref, warnings = numberer(assigned)
        result = FileResult(status="identified", episode=ref, warnings=warnings, **common)
        try:
            rel = layout.render(self.config.template, result.template_vars())
        except layout.TemplateError as e:  # e.g. a bad value for a filter at render time
            return dataclasses.replace(
                result, status="failed", episode=None, error=f"template: {e}")
        base = self.config.output_dir or raw.path.parent
        return dataclasses.replace(result, target_path=base / rel, target_rel=str(rel))
