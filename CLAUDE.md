# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

diskid is part of a DVD ripping pipeline that identifies TV episodes so ripped files can be tagged with correct metadata. It uses subtitles as a fingerprint: subtitles are obtained from a ripped file — either extracted directly from the disc's subtitle streams or derived via speech-to-text from the audio — and matched against a database of known subtitles to determine which episode the file contains.

## Key Constraints

- **GPU optional**: Speech-to-text and any other heavy compute should use a GPU when one is available, but everything must degrade gracefully and run CPU-only.
- **Docker delivery**: The project is delivered as a Docker image. Keep runtime dependencies containerizable; the image should work with or without GPU passthrough (e.g. `--gpus all`).

## Development

The project uses [uv](https://docs.astral.sh/uv/) for Python management (Python 3.12, pinned in `.python-version`).

- Identify one file: `uv run diskid identify FILE --show NAME [--apply] [--json]`
- Identify a ripped-disc folder: `uv run diskid identify-disc DIR [--show NAME] [--apply]` (show name inferred from ARM-style `Title (Year)` folder names)
- Tests: `uv run pytest`
- Add a dependency: `uv add <package>`

There is no linter or Dockerfile yet.

## Subtitle Database

The subtitle reference database is the OPUS OpenSubtitles v2024 raw English archive at `subtitle_data/en.zip` (35 GB, gitignored; from `https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2024/raw/en.zip`). Keep it as a zip — entries are read lazily via random access; never extract it wholesale.

- Layout: `OpenSubtitles/raw/en/<year>/<key>/<file_id>.xml`, where `<key>` is either a bare IMDb id (movies) or `<episode_imdb>_<show_imdb>_<season>_<episode>` for TV episodes.
- `diskid/subdb.py` builds a derived index (`subtitle_data/en.episodes.tsv`) on first use: one row per episode subtitle file, `show_id<TAB>season<TAB>episode<TAB>zip_entry`.
- **Caveat**: some archive files are misfiled under the wrong episode (community upload errors). Aggregate per-episode scores with a median across subtitle files, never a max, so one bad file can't poison a ranking.
- Show names resolve to IMDb ids via the unofficial IMDb suggestion API (`https://v2.sg.media-imdb.com/suggestion/<initial>/<query>.json`), no key needed.
- **Episode numbering caveat**: archive keys use IMDb *aired-order* numbering, which can disagree with the production/DVD order used by TMDB/TVDB and expected by media servers (canonical example: Futurama 1ACV10–13 = IMDb S2E1–4). Matching happens in IMDb space; filenames are remapped to TMDB numbering by default (`--order tmdb`, `diskid/tmdb.py`, needs `TMDB_APIKEY` in env or `.env`) via the matched episode's own IMDb id — never assume the two numberings agree.

## Architecture

The `diskid/` package: `identify.py` (the library API: `Identifier`, result dataclasses, rename helpers; what the ARM sidecar imports) ← `cli.py` (thin argparse wrapper over it) → `stt.py` (faster-whisper, CUDA→CPU fallback, optional `local` extra; or `RemoteTranscriber` for an OpenAI-compatible server via `--stt-url`) + `subdb.py` (OPUS zip access, preserves per-word timestamps) → `match.py` (`BayesianIdentifier`: sequential Bayesian episode identification). `imdb.py` resolves show names and parses `Title (Year)` folder names.

Design points that aren't obvious from the code (full derivation in `plan.md`):

- Clips are sampled one at a time (`--clips` is a *cap*, default 10, `--clip-duration` 8s) at capture offsets spread across the middle 70% of the runtime, never the first/last 15%: cold opens and recaps quote *other* episodes, and endings run into credits/previews. Sampling stops as soon as the posterior clears `--min-score` (default 0.99), rather than always taking the full `--clips` count.
- **TF-IDF is retrieval only, not evidence.** `_retrieve_per_file` uses windowed TF-IDF cosine similarity purely to cheaply find one candidate line per subtitle file — short transcripts drown against whole-episode documents, so snippet-sized sliding windows are load-bearing for retrieval, but the cosine score itself is discarded once a candidate is found. The actual evidence is `_text_llr`: a word-level noise model (`STT_ERROR_RATE`) where each transcript word contributes `log((1-eps)/P_bg(w))` if aligned to the candidate line (`-log P_bg(w)` is IDF — rare words are worth far more than "yeah okay") or `log(eps)` if not. Alignment is sequence-based (`difflib`), not set overlap, so a common word only counts as matched when it's in the same relative order as the candidate — otherwise every episode's best-effort window would rack up bogus credit for "the"/"you"/"know".
- **The implied timing offset is a nuisance parameter, marginalized over a grid, not point-estimated.** Each episode keeps a running belief over "how many seconds ahead of this episode's subtitles does this rip run" (`BayesianIdentifier.beliefs[key].log_belief`, a grid over ±`GRID_RANGE` seconds). `_time_llr` scores each clip's implied offset as a mixture of "retrieval found the true line" (Gaussian, `TIME_JITTER_SIGMA` wide, centered on the grid value) and "retrieval found some other coincidentally-similar line" (uniform — absorbs catchphrases/recaps). The true episode accumulates evidence at a consistent offset across clips; wrong episodes have no reason to.
- **Median, never max, across an episode's subtitle files** (misfiled-files caveat above) — but applied *per delta-grid bin*, before marginalizing over the grid, so a misfiled file's spurious peak at some offset gets outvoted bin-by-bin rather than surviving as an outlier average.
- **File duration is folded into the prior as a length check, not accumulated per clip.** Each episode's canonical runtime (median across its subtitle files of the last line's timestamp, plus `LENGTH_PAD`) is compared against the ripped file's actual `ffprobe` duration via a tight Gaussian (`LENGTH_SIGMA`, provisional ~30s) against a broad reference density (`LENGTH_NULL_RANGE`), applied once at `BayesianIdentifier` construction since duration is known before any clip is transcribed. This is what keeps behind-the-scenes featurettes and other extras — which can still text-match an episode's dialogue — from outscoring the null hypothesis on text evidence alone.
- **A null hypothesis (`NULL_KEY`, "not in this database") competes on equal footing** with every real episode — without it, a file that matches nothing would still be forced to look "confident" about its least-bad option, since probability has to sum to 1 across *something*. This is what makes `--min-score` a meaningful stopping threshold rather than a heuristic cutoff.
- `STT_ERROR_RATE`, `RETRIEVAL_HIT_RATE`, and `TIME_JITTER_SIGMA` in `match.py` are provisional constants, not yet calibrated against real data — see `plan.md`'s Calibration section for how to fit them (and sanity-check the resulting posterior with a reliability curve) before trusting `--min-score` as a literal probability rather than a ranking signal.
- `identify-disc` assigns episodes uniquely across a disc's files (greedy by top score) — a duplicate episode claim means a bad match, not two copies of an episode.
- Renames never overwrite an existing file.

## ARM Integration

diskid runs as a systemd service on the ARM host (`diskid watch DIR`, `diskid/watch.py`; unit and env example in `deploy/`; install steps in README). ARM has no post-job hook, so it polls `COMPLETED_PATH` for disc folders (`<Title> (<Year>)/title_tNN.mkv`) that have been quiet for `--settle` seconds, runs `Identifier.identify_disc`, renames confident files, and records state/reports. Still to build: a Dockerfile/compose (the Docker constraint above still stands) and a VobSub/PGS subtitle-stream + OCR input path (preferred over Whisper when streams exist). An `OPENSUBTITLES_APIKEY` lives in `.env` (gitignored) for future API gap-filling; SubDL (50 downloads/day free) is the preferred quota fallback over OpenSubtitles (5/day).
