"""Bayesian episode identification from short transcribed clips.

Full design rationale lives in plan.md; the short version:

- **TF-IDF is retrieval only, not evidence.** Cosine similarity finds a cheap
  candidate line per subtitle file (`_retrieve_per_file`); it is never used
  directly as a confidence score. The actual evidence is an explicit
  word-level noise model against that candidate (`_text_llr`).
- **Timing offset is a nuisance parameter, marginalized, not fixed.** Each
  episode carries a belief (`_DeltaBelief`) over "how many seconds ahead of
  this episode's subtitles does this rip run" (recap/intro length
  difference). The correct episode accumulates evidence at a *consistent*
  offset across clips; wrong episodes don't, so their marginal likelihood
  stays low even if one lucky line matches.
- **A null hypothesis ("not in this database") is mandatory.** Without it,
  the posterior is forced to sum to 1 across only the known episodes, so
  even a file that matches nothing looks "confident" about its least-bad
  option. `NULL_KEY` competes on equal footing and anchors what "0.99
  confident" actually means.
- **Median, never max, across an episode's subtitle files.** Some archive
  files are misfiled under the wrong episode (community upload errors); a
  max would let one such file's spurious match dominate the episode's
  evidence.
- **File duration is evidence too, folded into the prior.** Each episode's
  canonical runtime (median last-subtitle timestamp across its files, see
  `_episode_length`) is compared against the ripped file's actual duration
  via a tight Gaussian (`LENGTH_SIGMA`). This is what keeps behind-the-scenes
  featurettes, deleted scenes, and other extras from getting mislabeled as
  episodes on text similarity alone -- their runtime doesn't match *any*
  real episode, so the null hypothesis wins on length even before a single
  clip is transcribed.

Constants below (`STT_ERROR_RATE`, `RETRIEVAL_HIT_RATE`, `TIME_JITTER_SIGMA`)
are provisional -- see plan.md's Calibration section. Until fit against real
labeled data, treat the posterior as a reasonable ranking signal more than a
literal, calibrated probability.
"""

import difflib
import math
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from .subdb import EpisodeKey

TimedDoc = list[tuple[str, float]]  # (word, enclosing-sentence start time)

NULL_KEY: EpisodeKey = (0, 0)  # "not in this database" -- not a real (season, episode)

STT_ERROR_RATE = 0.15       # eps: STT word-error rate
RETRIEVAL_HIT_RATE = 0.8    # pi: P(TF-IDF's top line is the true line | episode is correct)
TIME_JITTER_SIGMA = 3.0     # seconds: subtitle timing jitter + window-start ambiguity
GRID_RANGE = 180.0          # seconds, +/-, of alignment offset considered
GRID_STEP = 1.0

LENGTH_SIGMA = 30.0         # seconds: stddev of true runtime around the canonical estimate
LENGTH_PAD = 45.0           # seconds: subtitles typically stop a bit before credits roll
LENGTH_NULL_RANGE = 3600.0  # seconds, +/-: reference "could be anything" duration density

WORD_RE = re.compile(r"[a-z0-9']+")


def _normalize(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _windows(doc: TimedDoc, win: int, stride: int) -> list[tuple[str, float, float]]:
    """(window text, start time, end time) slices of a timed document."""
    if not doc:
        return []
    out = []
    for i in range(0, max(1, len(doc) - win + 1), stride):
        chunk = doc[i:i + win]
        out.append((" ".join(w for w, _ in chunk), chunk[0][1], chunk[-1][1]))
    return out


@dataclass(frozen=True)
class BackgroundModel:
    """Unigram P_bg(w) over the show's subtitle corpus.

    `-log P_bg(w)` is IDF: rare words carry evidence, "yeah okay sure"
    carries almost none. Laplace-smoothed so unseen words get a small floor
    instead of a zero probability.
    """
    probs: dict[str, float]
    floor: float

    def logp(self, word: str) -> float:
        return math.log(self.probs.get(word, self.floor))

    def text_logp(self, words: list[str]) -> float:
        return sum(self.logp(w) for w in words)

    @classmethod
    def fit(cls, docs: dict[EpisodeKey, list[TimedDoc]]) -> "BackgroundModel":
        counts: Counter[str] = Counter()
        for files in docs.values():
            for doc in files:
                counts.update(_normalize(" ".join(w for w, _ in doc)))
        total = sum(counts.values())
        vocab = len(counts)
        floor = 1 / (total + vocab)
        probs = {w: (c + 1) / (total + vocab) for w, c in counts.items()}
        return cls(probs, floor)


def _retrieve_per_file(
    docs: dict[EpisodeKey, list[TimedDoc]], clip_text: str
) -> dict[EpisodeKey, list[tuple[str, float]]]:
    """Cheap TF-IDF retrieval: the best-scoring window per subtitle file for this clip.

    One global vectorizer is fit across every file's windows so IDF weighting
    is shared and comparable; the best window is then tracked per file (not
    pooled per episode) so the caller can median-aggregate across an
    episode's files downstream. The cosine score itself is discarded here --
    it only picked the candidate, it isn't the evidence.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    qlen = max(len(clip_text.split()), 20)
    win, stride = 2 * qlen, max(qlen // 2, 10)

    file_keys: list[EpisodeKey] = []
    windows: list[str] = []
    window_times: list[tuple[float, float]] = []
    file_of_window: list[int] = []
    for key, files in docs.items():
        for doc in files:
            file_keys.append(key)
            fidx = len(file_keys) - 1
            for text, start, end in _windows(doc, win, stride):
                windows.append(text)
                window_times.append((start, end))
                file_of_window.append(fidx)

    if not windows:
        return {}

    vectorizer = TfidfVectorizer(strip_accents="unicode")
    win_vecs = vectorizer.fit_transform(windows)
    query_vec = vectorizer.transform([clip_text])
    scores = cosine_similarity(query_vec, win_vecs)[0]

    best_score = [-1.0] * len(file_keys)
    best_window = [""] * len(file_keys)
    best_time = [0.0] * len(file_keys)
    for i, (fidx, score, (start, end)) in enumerate(zip(file_of_window, scores, window_times)):
        if score > best_score[fidx]:
            best_score[fidx] = float(score)
            best_window[fidx] = windows[i]
            best_time[fidx] = (start + end) / 2

    per_episode: dict[EpisodeKey, list[tuple[str, float]]] = {}
    for key, text, ts in zip(file_keys, best_window, best_time):
        per_episode.setdefault(key, []).append((text, ts))
    return per_episode


def _aligned_indices(clip_words: list[str], window_words: list[str]) -> set[int]:
    """Indices into `clip_words` covered by a real in-order alignment to `window_words`.

    Deliberately sequence alignment (`difflib`), not set overlap: a set-based
    "does this word appear anywhere in the window" check would treat common
    words (the, you, know) as strong matches just because they're common
    enough to appear in nearly every window -- including wrong episodes'
    windows -- which defeats the point of weighting matches by rarity below.
    Alignment only counts a word as matched where it sits in the same
    relative order as the candidate line, which common words satisfy far
    less often by coincidence.
    """
    matcher = difflib.SequenceMatcher(None, clip_words, window_words, autojunk=False)
    matched: set[int] = set()
    for block in matcher.get_matching_blocks():
        matched.update(range(block.a, block.a + block.size))
    return matched


def _text_llr(clip_words: list[str], matched: set[int], bg: BackgroundModel, eps: float) -> float:
    """log P(transcript | candidate line) / P(transcript | background), a text-only LLR.

    Aligned words (see `_aligned_indices`) carry evidence weighted by
    rarity; unaligned words (STT insertion/hallucination, or just the wrong
    line) are charged the flat error rate `eps`. Derivation: absolute
    log P(transcript | line) is `sum_matched log(1-eps) + sum_unmatched
    [log(eps) + log P_bg(w)]` (an unmatched word is modeled as an
    eps-probability hallucination drawn from the background distribution);
    the null's log P(transcript | e0) is `sum_all log P_bg(w)`. Subtracting
    cancels the P_bg(w) terms on matched words only -- exactly the formula
    below.

    `eps` is not yet fit against labeled data (see plan.md's Calibration
    section) -- it's exposed as a knob (`--eps`) so it can be hand-tuned
    toward a more skeptical (higher eps) or more trusting (lower eps)
    classifier in the meantime.
    """
    total = 0.0
    for i, w in enumerate(clip_words):
        if i in matched:
            total += math.log(1 - eps) - bg.logp(w)
        else:
            total += math.log(eps)
    return total


def _time_llr(delta_hat: float, deltas: np.ndarray) -> np.ndarray:
    """log P(delta_hat | delta) / P(delta_hat | uniform), over the delta grid.

    Mixture of "retrieval found the true line" (Gaussian around the
    hypothesized true offset, `TIME_JITTER_SIGMA` wide) and "retrieval found
    some other, coincidentally-similar line" (uniform over the grid, which
    absorbs repeated dialogue/catchphrases). Relative to a pure-uniform
    reference so it composes additively with `_text_llr` in log-odds terms.
    """
    gaussian = np.exp(-0.5 * ((delta_hat - deltas) / TIME_JITTER_SIGMA) ** 2) / (
        TIME_JITTER_SIGMA * math.sqrt(2 * math.pi)
    )
    uniform = 1.0 / (2 * GRID_RANGE)
    p_time = RETRIEVAL_HIT_RATE * gaussian + (1 - RETRIEVAL_HIT_RATE) * uniform
    return np.log(p_time) - math.log(uniform)


def _episode_length(files: list[TimedDoc]) -> float | None:
    """Canonical runtime estimate for an episode: median last-subtitle-line
    timestamp across its files, plus `LENGTH_PAD` for the credits tail.

    Same median-not-max discipline as text/time evidence: a misfiled file
    (wrong episode, likely a different length entirely) shouldn't get to set
    the estimate. None if the episode has no timestamped files at all.
    """
    lasts = [max(t for _, t in doc) for doc in files if doc]
    if not lasts:
        return None
    return float(np.median(lasts)) + LENGTH_PAD


def _length_llr(duration: float, canonical: float | None) -> float:
    """log P(duration | episode) / P(duration | "could be anything"), a length-only LLR.

    Real episodes of a show are almost always the same runtime, give or take
    editing; extras (behind-the-scenes, deleted scenes, trailers) are not.
    A tight Gaussian around the episode's own canonical length (`LENGTH_SIGMA`)
    against a broad reference density does the discriminating -- it doesn't
    need to know what an "extra" looks like, only that its duration doesn't
    match this episode's.
    """
    if canonical is None:
        return 0.0
    log_hit = (
        -0.5 * ((duration - canonical) / LENGTH_SIGMA) ** 2
        - math.log(LENGTH_SIGMA * math.sqrt(2 * math.pi))
    )
    log_null = -math.log(2 * LENGTH_NULL_RANGE)
    return log_hit - log_null


def _logsumexp(log_weights: np.ndarray) -> float:
    m = np.max(log_weights)
    if math.isinf(m):
        return m
    return float(m + math.log(np.sum(np.exp(log_weights - m))))


@dataclass
class _EpisodeBelief:
    """One episode's running state: accumulated evidence and delta-offset belief."""
    log_belief: np.ndarray  # log P(delta | observations so far), unnormalized is fine
    llr: float = 0.0        # accumulated log P(o_1..n | e) / P(o_1..n | e0)
    n_files: int = 0


@dataclass(frozen=True)
class ConsensusScore:
    season: int
    episode: int
    score: float        # posterior probability, P(episode | clips so far)
    offset: float        # this episode's current best-guess alignment offset (seconds)
    n_clips: int
    n_files: int

    @property
    def key(self) -> EpisodeKey:
        return (self.season, self.episode)


class BayesianIdentifier:
    """Sequentially updates a posterior over a show's episodes (plus "not in database").

    Feed clips one at a time via `observe`; call `confident` after each to
    decide whether to stop sampling more, or `ranked` for the current
    snapshot. See module docstring / plan.md for the model.
    """

    def __init__(
        self,
        docs: dict[EpisodeKey, list[TimedDoc]],
        eps: float = STT_ERROR_RATE,
        duration: float | None = None,
    ):
        """`duration`: the ripped file's actual runtime in seconds, if known.

        Folded into the prior (not accumulated per-clip, unlike text/time
        evidence) since it's a property of the whole file, known before any
        clip is even transcribed. Omit it (default) to fall back to a flat
        prior across episodes, e.g. when the caller doesn't have a duration.
        """
        self.docs = docs
        self.eps = eps
        self.bg = BackgroundModel.fit(docs)
        self.deltas = np.arange(-GRID_RANGE, GRID_RANGE + GRID_STEP, GRID_STEP)
        self.episode_length = {key: _episode_length(files) for key, files in docs.items()}
        n_hypotheses = len(docs) + 1  # + null
        base_log_prior = -math.log(n_hypotheses)
        self.log_prior: dict[EpisodeKey, float] = {key: base_log_prior for key in docs}
        self.log_prior[NULL_KEY] = base_log_prior
        if duration is not None:
            for key in docs:
                self.log_prior[key] += _length_llr(duration, self.episode_length[key])
        self.beliefs: dict[EpisodeKey, _EpisodeBelief] = {
            key: _EpisodeBelief(log_belief=np.zeros_like(self.deltas), n_files=len(files))
            for key, files in docs.items()
        }
        self.n_clips = 0

    def observe(self, capture_time: float, clip_text: str) -> None:
        """Update the posterior with one (capture offset, transcript) observation."""
        if not clip_text.strip():
            return
        self.n_clips += 1
        clip_words = _normalize(clip_text)
        per_file = _retrieve_per_file(self.docs, clip_text)

        for key, windows in per_file.items():
            belief = self.beliefs[key]
            file_llrs = []
            for window_text, window_time in windows:
                window_words = _normalize(window_text)
                matched = _aligned_indices(clip_words, window_words)
                text_llr = _text_llr(clip_words, matched, self.bg, self.eps)
                delta_hat = window_time - capture_time
                file_llrs.append(text_llr + _time_llr(delta_hat, self.deltas))

            # median across an episode's files, per delta bin -- a misfiled
            # file's spurious peak gets outvoted bin-by-bin, not averaged in.
            obs_llr = np.median(np.stack(file_llrs), axis=0)

            log_prior_belief = belief.log_belief - _logsumexp(belief.log_belief)
            step_llr = _logsumexp(log_prior_belief + obs_llr)
            belief.llr += step_llr
            belief.log_belief = log_prior_belief + obs_llr

    def _posterior(self) -> dict[EpisodeKey, float]:
        log_unnorm = {key: self.log_prior[key] + b.llr for key, b in self.beliefs.items()}
        log_unnorm[NULL_KEY] = self.log_prior[NULL_KEY]
        m = max(log_unnorm.values())
        exps = {k: math.exp(v - m) for k, v in log_unnorm.items()}
        total = sum(exps.values())
        return {k: v / total for k, v in exps.items()}

    def ranked(self) -> list[ConsensusScore]:
        """Real episodes (never the null hypothesis) ranked by posterior, best first."""
        posterior = self._posterior()
        results = []
        for key, belief in self.beliefs.items():
            season, episode = key
            offset = float(self.deltas[np.argmax(belief.log_belief)]) if self.n_clips else 0.0
            results.append(ConsensusScore(
                season, episode, posterior[key], offset, self.n_clips, belief.n_files,
            ))
        return sorted(results, key=lambda e: -e.score)

    def null_probability(self) -> float:
        return self._posterior()[NULL_KEY]

    def confident(self, threshold: float = 0.99) -> ConsensusScore | None:
        """The top episode, if its posterior clears `threshold`; None otherwise.

        The null hypothesis can itself "win" (nothing clears the bar and
        null holds the most mass) -- that's reported as None, same as any
        other case with no confident episode.
        """
        top = self.ranked()[0] if self.beliefs else None
        if top is None or top.score < threshold:
            return None
        return top


def rank_clips(
    docs: dict[EpisodeKey, list[TimedDoc]],
    clips: list[tuple[float, str]],
    desc: str | None = None,
    eps: float = STT_ERROR_RATE,
    duration: float | None = None,
) -> list[ConsensusScore]:
    """Run a fixed batch of clips through a fresh `BayesianIdentifier`, ranked best first.

    Convenience wrapper for callers that don't need the incremental
    stop-when-confident loop (e.g. `identify-disc`'s per-file batch scoring).
    """
    identifier = BayesianIdentifier(docs, eps=eps, duration=duration)
    bar = tqdm(clips, desc=desc, unit="clip", leave=False, disable=desc is None)
    for capture_time, text in bar:
        identifier.observe(capture_time, text)
    return identifier.ranked()


def is_confident(ranked: list[ConsensusScore], min_score: float = 0.99) -> bool:
    """True when the top episode's posterior clears `min_score`."""
    return bool(ranked) and ranked[0].score >= min_score


def assign_episodes(
    per_file: dict[str, list[ConsensusScore]], min_score: float = 0.99
) -> dict[str, ConsensusScore | None]:
    """Assign each file a distinct episode (disc sets never repeat an episode).

    Files are visited most-confident first; each takes its best
    still-unclaimed episode, re-checked against the confidence rule with the
    claimed episodes removed. Unconfident files get None.
    """
    def top_score(ranked: list[ConsensusScore]) -> float:
        return ranked[0].score if ranked else 0.0

    assigned: dict[str, ConsensusScore | None] = {}
    taken: set[EpisodeKey] = set()
    for name in sorted(per_file, key=lambda n: -top_score(per_file[n])):
        remaining = [e for e in per_file[name] if e.key not in taken]
        if is_confident(remaining, min_score):
            assigned[name] = remaining[0]
            taken.add(remaining[0].key)
        else:
            assigned[name] = None
    return assigned
