# diskid

Identify ripped TV-episode video files by subtitle fingerprint. Subtitles are
derived from a ripped file by speech-to-text (faster-whisper; GPU optional) and
matched against the OPUS OpenSubtitles v2024 raw English archive with a
sequential Bayesian model. Matched files can be renamed to `Show - S01E02.mkv`
using TMDB episode numbering (Jellyfin/Plex-friendly).

Built for a DVD-ripping pipeline (ARM): point it at a ripped-disc folder named
`Title (Year)` and it works out which episode each file is.

## Install

```sh
uv sync                  # or: pip install .
uv run diskid --help
```

Requires Python >= 3.12 and `ffmpeg`/`ffprobe` on `PATH`. For local transcription
install the `local` extra (`pip install '.[local]'`); Whisper then uses CUDA when
available and falls back to CPU. Alternatively use a remote server (below).

The subtitle archive must be at `subtitle_data/en.zip` (35 GB, gitignored, from
`https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/raw/en.zip`) or passed
explicitly. Keep it zipped; entries are read lazily. Set `TMDB_APIKEY` (env or
`./.env`) for TMDB numbering; without it, IMDb aired order is used with a warning.

## Library usage

```python
from diskid import Identifier, IdentifierConfig, apply_rename

ident = Identifier(IdentifierConfig(model="small"))   # reuse across files!

disc = ident.identify_disc("/completed/Firefly (2002)")   # show guessed from folder name
print(disc.show)                                          # Show(imdb_id=303461, title='Firefly')
for f in disc.files:
    if f.identified:
        print(f.path.name, f.episode.code, f.target_name)  # title_t03.mkv S01E02 Firefly - S01E02.mkv
        apply_rename(f)                                     # never overwrites
    else:
        print(f.path.name, f.status, f.error or f.best)
```

An `Identifier` lazily loads the Whisper model once and caches each show's
subtitle data, so keep one for the life of your process. It is not thread-safe.
Nothing is renamed unless you call `apply_rename`/`apply_renames`.

### Entry points

| Call | Returns |
|---|---|
| `Identifier.identify_file(path, show=None, hint=None)` | `FileResult` |
| `Identifier.identify_disc(directory, show=None)` | `DiscResult` (episodes assigned uniquely across files) |
| `Identifier.resolve_show(name, year=None)` | `Show` |
| `Identifier.episode_ref(show, candidate)` | `EpisodeRef` for any candidate, in the configured numbering |
| `apply_rename(result)` / `apply_renames(results)` | `RenameOutcome` / list |

`show` is a `Show` or a name; if omitted it is guessed from the folder name
(`hint` for `identify_file`, default: parent folder) and the container title tag.
Pass a `progress` callback to `Identifier(...)` to receive `ProgressEvent`s.

### Result types

- **`FileResult`**: `path`, `status` (`"identified"` / `"uncertain"` / `"failed"`),
  `show`, `episode` (`EpisodeRef`, only when identified), `candidates` (top 10
  `ConsensusScore`s, best first), `null_probability` (posterior the file is *not* any
  episode, e.g. an extra), `clips_used`, `duration`, `target_name`, `target_path`,
  `warnings`, `error`. Plus `.identified`, `.best`, `.to_dict()`.
- **`EpisodeRef`**: `season`, `episode` (in the requested numbering),
  `imdb_season`, `imdb_episode`, `order` (`"tmdb"`/`"imdb"`: what was *actually*
  applied), `imdb_id`, `.code` (`"S01E02"`).
- **`DiscResult`**: `directory`, `show`, `files`, `.identified`, `.unidentified`, `.to_dict()`.
- **`ConsensusScore`**: `season`, `episode` (IMDb numbering), `score` (posterior),
  `offset`, `n_clips`, `n_files`, `.key`.

`"uncertain"` means the best posterior is below `min_score` (extras, wrong
show, bad audio) *or* a more confident file on the same disc claimed that
episode (see `warnings`). `"failed"` means ffmpeg/ffprobe couldn't read the file.

### Errors

`ShowNotFoundError` (a `LookupError`): the show can't be determined, or the
archive holds no subtitles for it. `FileNotFoundError` / `NotADirectoryError` for
bad paths or a missing archive. Network failures from IMDb/TMDB surface as `OSError`
when resolving a show by name.

### Configuration

`IdentifierConfig` fields (all optional): `model="small"`, `language=None`,
`clip_duration=8.0`, `max_clips=10` (a cap; sampling stops early once confident),
`start=None` (sample one clip at this offset), `min_score=0.99`, `eps` (STT
word-error rate; uncalibrated tuning knob), `order="tmdb"`, `zip_path`,
`tmdb_api_key`.

`min_score` is a posterior probability, but the model's constants are not yet
calibrated against labeled data (see `plan.md`), so treat it as a ranking
signal rather than a literal probability.

### Lower-level modules

Used by `Identifier`, importable directly: `diskid.match` (`BayesianIdentifier`
and helpers; feed it `(capture_time, transcript)` pairs), `diskid.subdb`
(archive access), `diskid.stt` (Whisper), `diskid.audio` (ffprobe/ffmpeg),
`diskid.imdb`, `diskid.tmdb`.

## Command line

```sh
uv run diskid identify FILE --show "Firefly" [--apply] [--json]
uv run diskid identify-disc "Firefly (2002)" [--show NAME] [--apply] [--json]
uv run diskid apply MAPPING_ID
uv run diskid watch DIR [--dry-run]    # service mode, see below
```

Dry runs (no `--apply`) cache the proposed renames and print a mapping id;
`diskid apply MAPPING_ID` replays them without re-running matching. Exit code 2
means at least one file was not confidently identified. Useful flags: `--clips`,
`--clip-duration`, `--min-score`, `--model`, `--language`, `--order {tmdb,imdb}`,
`--zip`, `--start`, `--eps`.

## Running as a service (Raspberry Pi / ARM)

`diskid watch DIR` polls ARM's `COMPLETED_PATH`. Each subfolder is one disc
(`Title (Year)/title_tNN.mkv`); once a disc has been unmodified for `--settle`
seconds (default 120) it is identified and its confidently identified files
are renamed in place (never overwriting). Files that aren't confidently
identified, and discs whose show can't be determined (movies, unknown shows),
are left untouched. A disc is processed once; it is reprocessed only if its file
list or sizes change. Use `--dry-run` first to see what it would do.

State lives in `$STATE_DIRECTORY` (or `~/.local/state/diskid`): `state.json`
plus a JSON report per disc under `reports/` (the full `DiscResult`, including
candidate rankings, for reviewing misses).

Install (as root on the Pi):

```sh
git clone https://github.com/csreid/diskid.git /opt/diskid && cd /opt/diskid
uv sync --no-dev        # skips faster-whisper: transcription is done by the --stt-url server
mkdir -p /etc/diskid /var/lib/diskid-subtitles
cp deploy/diskid.env.example /etc/diskid/diskid.env && chmod 640 /etc/diskid/diskid.env
$EDITOR /etc/diskid/diskid.env        # watch dir, archive location, TMDB key, model
cp deploy/diskid.service /etc/systemd/system/    # edit User=/Group= to the ARM user
# put en.zip in /var/lib/diskid-subtitles (35 GB) and make it writable by that user
systemctl daemon-reload && systemctl enable --now diskid
journalctl -u diskid -f
```

Practical notes for a Pi:
- Don't run Whisper on the Pi. Point it at a transcription server with
  `--stt-url` (see below); the Pi then only runs ffmpeg and the matcher.
- It needs `ffmpeg`/`ffprobe`, ~35 GB of disk for the archive, and network access
  the first time (Whisper model download to `/var/cache/diskid`; IMDb/TMDB lookups).
- The unit runs at idle CPU/IO priority so it shouldn't slow ripping.
- First use of a show builds an index of the whole archive
  (`en.episodes.tsv`, next to the zip) which takes a while; later runs reuse it.
- `systemctl stop` aborts a disc in progress at the next clip; nothing is
  renamed until a disc's identification finishes.

## Remote transcription server

Set `--stt-url http://host:8000/v1` (or `IdentifierConfig(stt_url=...)`) to
send each 8-second clip (16 kHz mono wav, ~250 KB) to any OpenAI-compatible
`POST /v1/audio/transcriptions` server, e.g.
[Speaches](https://github.com/speaches-ai/speaches), instead of running
faster-whisper locally. Only the audio clip is sent; video never leaves the host.

- `--stt-model` picks the server's model name (default `Systran/faster-whisper-<--model>`;
  that naming is Speaches'. Other servers may use different names).
- `$DISKID_STT_API_KEY` sets a bearer token; `--stt-timeout` (default 120 s) bounds one clip.
- Connection failures and HTTP errors fail the disc; the watcher retries it after 15 minutes.
- With a server configured, `faster-whisper` is never imported, so install without it:
  `uv sync --no-dev` / `pip install .`. For local transcription use
  `pip install '.[local]'` (the dev group includes it).

## How it works

Clips (default 8 s, at most 10, spread across the middle 70% of the runtime) are
transcribed one at a time and fed to a Bayesian identifier, which stops as soon as
the top episode's posterior clears `--min-score`. Key design points (full
derivation in `plan.md`, details in `diskid/match.py`):

- **TF-IDF is retrieval only**: it finds one candidate line per subtitle file;
  the evidence is a word-level noise-model likelihood.
- **Timing offset is a marginalized nuisance parameter**, so the true episode
  accumulates evidence at a consistent offset across clips.
- **Median across an episode's subtitle files, never max**, since the archive has
  misfiled uploads.
- **File duration** is folded into the prior, keeping extras from matching.
- **A null hypothesis ("not in this database")** competes with every episode.
- **IMDb vs TMDB numbering**: matching happens in IMDb aired order; results are
  remapped to TMDB via the episode's own IMDb id.
