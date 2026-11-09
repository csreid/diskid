# Plan: Bayesian TV Episode Identifier via Subtitle Matching

## Context

Automated media-naming heuristics (filename guessing) fail on DVD rips of TV shows. This tool identifies *which episode* is in a video file by transcribing short random audio windows and Bayesian-updating over episodes using a subtitle database. The show name is given; the episode is unknown.

The design was derived from first principles in a prior conversation. The key ideas, which must be preserved:

1. **TF-IDF is retrieval only, not evidence.** It finds candidate subtitle lines cheaply. The actual evidence is a likelihood ratio from an explicit noise model.
2. **Per-episode timestamp offset is a nuisance parameter, marginalized, not fixed.** The correct episode accumulates matches at a *consistent* offset; wrong episodes match shared dialogue at inconsistent offsets. This is the Shazam offset-histogram trick and it's what makes the system robust to catchphrases, recaps, and theme songs.
3. **A null hypothesis "not in database" is mandatory** or the confidence score is a lie.

## Mathematical specification

### State

For each episode `e` in the show (plus null hypothesis `e0`):
- Log-posterior `log P(e | o_1..n)`, initialized uniform.
- A discrete grid posterior over alignment offset `delta_e`: bins of 1s over ±180s (configurable), initialized uniform. Optionally a small set of scale hypotheses `alpha ∈ {1.0, 25/23.976}` (PAL speedup) — implement `alpha=1` first, add scale later.

### Observation

An observation is `o = (t, tau)`: transcript `t` from an audio window starting at file-time `tau` (window length ~10–15s).

### Likelihood for episode e

```
P(o | e) = sum over delta grid of P(delta) * P_match(o | e, delta)
```

For the dominant retrieved line `l*` in episode e (found via TF-IDF over that episode's subtitle lines/spans):

```
P_match(o | e, delta) ≈ P_noise(t | l*) * P_time(tau | tau_l*, delta)
```

**Multi-file aggregation.** Episodes have multiple subtitle files (community uploads), and some archive files are misfiled under the wrong episode — a known corpus defect (see CLAUDE.md). Never take the max across an episode's files; one misfiled file would inject a spurious high-confidence `l*` straight into the episode's evidence. Instead:

- Retrieve `l*_f` and compute `P_match_f(o | e, delta)` independently per file `f` of episode `e`, over the full delta grid.
- Take the **elementwise median across files, per delta bin**, to get the episode's `P_match(o | e, delta)` vector, *then* marginalize (sum) over delta.
- Median-then-marginalize (not marginalize-then-median) so a misfiled file's spurious peak at some delta gets outvoted bin-by-bin rather than surviving as an outlier average.

**Text noise model** (token level). With STT word-error rate `eps` (calibrated, see below) and background unigram distribution `P_bg(w)` estimated over the entire subtitle corpus for the show (or better, a larger corpus):

- Align transcript tokens to subtitle tokens (simple: bag-of-words overlap after normalization; better: token-level alignment via difflib/Levenshtein).
- Log-likelihood ratio vs background:

```
LLR_text = sum over matched words w of log((1 - eps) / P_bg(w))
         + sum over unmatched transcript words of log(eps)   # optional penalty term
```

Note `-log P_bg(w)` **is** IDF — rare words carry the evidence. "yeah okay sure" ≈ 0 bits; a character name ≈ many bits.

**Time model.** Implied offset `delta_hat = tau_l* - tau`. Mixture:

```
P_time = pi * Normal(delta_hat | delta, sigma^2) + (1 - pi) * Uniform(over grid range)
```

- `pi` ≈ 0.7–0.9: probability TF-IDF retrieved the right line given the episode is correct.
- `sigma` ≈ 2–5s: subtitle timing jitter + window-start ambiguity.
- The uniform component absorbs wrong-line retrievals (repeated dialogue).

**Null hypothesis e0**: `P(o | e0) = P_bg(t)` (pure background text likelihood), uniform time. This is the denominator that keeps confidence honest.

**Length prior.** Real episodes of a show are almost always the same runtime; extras (behind-the-scenes featurettes, deleted scenes, trailers) are not, but their dialogue can still cosine-match an episode's subtitle text well enough to fool the text/time evidence alone. Each episode's canonical runtime is estimated once (median, across its subtitle files, of the last subtitle line's timestamp, plus a small pad for the credits tail subtitles don't cover) and compared against the ripped file's actual duration (`ffprobe`) via a tight Gaussian (`LENGTH_SIGMA`, provisional ~30s) against a broad "could be anything" reference density (`LENGTH_NULL_RANGE`). Unlike text/time evidence this isn't per-clip: duration is known before a single clip is transcribed, so it's folded once into each episode's log-prior at `BayesianIdentifier` construction, not accumulated via `observe`. This is what keeps a behind-the-scenes clip that quotes real dialogue from still winning against the null hypothesis on text alone.

### Update

All in log space:

```
log P(e | o_1..n) = log P(e) + sum_i log P(o_i | e)   (then normalize across e ∪ {e0})
```

Per episode, also update the delta grid posterior:

```
P(delta | o_1..n, e) ∝ P(delta) * P_match(o_n | e, delta)
```

The episode marginal likelihood is the sum over the delta grid (marginalization), NOT the max.

### Calibration

The stopping rule only means what it claims to mean ("0.99 confident" ≈ "wrong 1% of the time") if `eps`, `pi`, and `sigma` reflect reality rather than intuition. Calibrate against the user's existing correctly-named library (e.g. `~/Videos/Media/shows/...`), where the filename is ground truth:

- **`eps` (STT word-error rate)**: transcribe known-correct episodes, align each transcript window to the *true* subtitle line at the *true* offset, and measure the fraction of transcript words that don't match. Do this across enough shows/episodes to get a stable estimate — likely also model-size-dependent (whisper `small` vs `large`), so recalibrate if `--model` changes.
- **`pi` (retrieval hit rate)**: for the same known-correct runs, check how often the TF-IDF top-retrieved `l*` is actually the true line (i.e. `tau_l*` close to the true position) versus a wrong line elsewhere in the episode (recap/catchphrase/generic dialogue).
- **`sigma` (timing jitter)**: for the observations where retrieval *did* hit the true line, look at the empirical spread of `tau_l* - tau_true` and fit its standard deviation.
- **`LENGTH_SIGMA`/`LENGTH_PAD` (runtime prior)**: for known-correct files, compare `ffprobe` duration against the episode's median-last-subtitle-timestamp estimate; the mean of that gap is `LENGTH_PAD`, its spread is `LENGTH_SIGMA`. Different shows may need different pads (e.g. shows with post-credits scenes, or long end-credit rolls with no dialogue).
- **Sanity check the whole posterior, not just its inputs**: run the full pipeline against a held-out slice of the known-correct library and plot a reliability curve (claimed posterior vs. empirical accuracy across episodes/shows). If "0.99" isn't actually ~99% right in practice, the independence/model assumptions are leaking somewhere upstream and the threshold needs either a fix to the model or a post-hoc calibration mapping (e.g. isotonic regression) before it's trustworthy as a stopping criterion.

### Stopping rule

Stop when `max_e P(e | o_1..n) > threshold` (default 0.99) and argmax ≠ e0. Hard cap on window count (default 10); if cap reached, report best candidate with its actual posterior and flag as low-confidence. If e0 wins, report "not identified — possibly wrong show or missing subtitles."
