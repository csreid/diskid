import random

from diskid import match
from diskid.match import (
    BayesianIdentifier,
    ConsensusScore,
    assign_episodes,
    is_confident,
)

WPS = 3.0  # words per second, for fabricating sentence timestamps


def fake_dialogue(rng: random.Random, n_words: int = 400) -> list[tuple[str, float]]:
    # vocab needs to be large enough that unrelated docs share few words,
    # as in real dialogue; a tiny vocab makes every doc look like every other
    vocab = [f"word{i}" for i in range(5000)]
    return [(rng.choice(vocab), i / WPS) for i in range(n_words)]


def noisy_copy(rng: random.Random, doc: list[tuple[str, float]], error_rate: float = 0.1):
    # different community uploads: wording differs, timing doesn't
    return [
        (f"variant{rng.randrange(5000)}" if rng.random() < error_rate else w, t)
        for w, t in doc
    ]


def make_show(n_episodes: int = 6, files_per_ep: int = 3, seed: int = 0):
    """Each episode is one dialogue; its files are noisy transcriptions of it,
    like the multiple community uploads of a real episode."""
    rng = random.Random(seed)
    show = {}
    for ep in range(1, n_episodes + 1):
        base = fake_dialogue(rng)
        show[(1, ep)] = [noisy_copy(rng, base) for _ in range(files_per_ep)]
    return show


def snippet_of(docs, key, start: int = 100, n_words: int = 40) -> str:
    words = docs[key][0][start:start + n_words]
    return " ".join(w for w, _ in words)


def snippet_midpoint(docs, key, start: int = 100, n_words: int = 40) -> float:
    words = docs[key][0][start:start + n_words]
    return (words[0][1] + words[-1][1]) / 2


def run(docs, clips: list[tuple[float, str]]) -> BayesianIdentifier:
    identifier = BayesianIdentifier(docs)
    for capture_time, text in clips:
        identifier.observe(capture_time, text)
    return identifier


def test_finds_source_episode():
    docs = make_show()
    transcript = snippet_of(docs, (1, 4))
    ident = run(docs, [(1000.0, transcript)])
    ranked = ident.ranked()
    assert ranked[0].key == (1, 4)
    assert ranked[0].score > 0.99
    assert ranked[1].score < 0.01


def test_resists_misfiled_file():
    docs = make_show()
    transcript = snippet_of(docs, (1, 4))
    # a copy of episode 4's subtitle misfiled under episode 2
    docs[(1, 2)].append(docs[(1, 4)][0])
    ident = run(docs, [(1000.0, transcript)])
    ranked = ident.ranked()
    assert ranked[0].key == (1, 4)
    # episode 2's median must not be dragged up by its one perfect file
    ep2 = next(e for e in ranked if e.key == (1, 2))
    assert ep2.score < 0.01


def test_rewards_consistent_offsets():
    docs = make_show()
    key = (1, 4)
    # three clips from the same episode; capture offsets chosen so that
    # (matched timestamp - capture offset) is the same constant for all three
    shift = 50.0
    starts = [50, 150, 300]
    clips = [
        (snippet_midpoint(docs, key, start=s) - shift, snippet_of(docs, key, start=s))
        for s in starts
    ]
    consistent = run(docs, clips)

    # last clip's implied offset no longer agrees with the other two, but
    # stays within the delta grid (otherwise timing evidence is uniformly
    # uninformative regardless of *how* inconsistent it is)
    inconsistent_clips = list(clips)
    inconsistent_clips[-1] = (
        snippet_midpoint(docs, key, start=300) - shift + 100.0, clips[-1][1],
    )
    inconsistent = run(docs, inconsistent_clips)

    # compare raw accumulated evidence, not the final softmaxed score: with
    # this much text evidence the posterior saturates near 1.0 either way
    assert inconsistent.beliefs[key].llr < consistent.beliefs[key].llr


def test_wrong_length_favors_null_despite_text_match():
    # a short behind-the-scenes-style clip that happens to quote episode 4's
    # dialogue verbatim (e.g. a clip package) shouldn't be labeled episode 4
    # if its overall runtime is nowhere near a real episode's length.
    docs = make_show()
    transcript = snippet_of(docs, (1, 4))
    ep4_canonical = match._episode_length(docs[(1, 4)])
    bts_duration = ep4_canonical / 10  # wildly shorter than any real episode

    identifier = match.BayesianIdentifier(docs, duration=ep4_canonical)
    identifier.observe(1000.0, transcript)
    assert identifier.ranked()[0].key == (1, 4)

    bts_identifier = match.BayesianIdentifier(docs, duration=bts_duration)
    bts_identifier.observe(1000.0, transcript)
    assert bts_identifier.null_probability() > identifier.null_probability()


def test_null_hypothesis_wins_on_unrelated_text():
    docs = make_show()
    rng = random.Random(999)
    vocab = [f"randomjunk{i}" for i in range(5000)]
    unrelated_text = " ".join(rng.choice(vocab) for _ in range(40))
    ident = run(docs, [(500.0, unrelated_text)])
    assert ident.null_probability() > 0.99


def test_is_confident():
    def ep(key, score):
        return ConsensusScore(*key, score, 0.0, 1, 3)

    assert is_confident([ep((1, 1), 0.995), ep((1, 2), 0.001)])
    assert not is_confident([ep((1, 1), 0.5), ep((1, 2), 0.3)])  # below min_score
    assert not is_confident([])
    assert is_confident([ep((1, 1), 0.995)])


def test_assign_episodes_no_duplicates():
    def ep(key, score):
        return ConsensusScore(*key, score, 0.0, 1, 3)

    # both files' best is episode 1, but file b wins it on score;
    # file a falls through to its (still confident) runner-up
    per_file = {
        "a.mkv": [ep((1, 1), 0.996), ep((1, 2), 0.995), ep((1, 3), 0.001)],
        "b.mkv": [ep((1, 1), 0.999), ep((1, 3), 0.001)],
    }
    assigned = assign_episodes(per_file, min_score=0.99)
    assert assigned["b.mkv"].key == (1, 1)
    assert assigned["a.mkv"].key == (1, 2)


def test_assign_episodes_unconfident_is_none():
    def ep(key, score):
        return ConsensusScore(*key, score, 0.0, 1, 3)

    per_file = {"a.mkv": [ep((1, 1), 0.5), ep((1, 2), 0.3)]}
    assigned = assign_episodes(per_file, min_score=0.99)
    assert assigned["a.mkv"] is None
