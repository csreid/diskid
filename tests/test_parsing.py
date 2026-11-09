from diskid.cli import episode_filename, sanitize_filename
from diskid.imdb import parse_title_year
from diskid.subdb import EPISODE_RE, episode_imdb_ids


def test_episode_imdb_ids(monkeypatch):
    import diskid.subdb as subdb
    monkeypatch.setattr(subdb, "show_subtitles", lambda zp, sid: {
        (2, 4): [
            "OpenSubtitles/raw/en/1999/584439_149460_2_4/1.xml",
            "OpenSubtitles/raw/en/1999/584439_149460_2_4/2.xml",
            "OpenSubtitles/raw/en/1999/99999_149460_2_4/3.xml",  # misfiled dup dir
        ],
        (2, 1): ["OpenSubtitles/raw/en/1999/584426_149460_2_1/4.xml"],
    })
    ids = episode_imdb_ids(None, 149460)
    assert ids == {(2, 4): 584439, (2, 1): 584426}


def test_parse_title_year():
    assert parse_title_year("Breaking Bad (2008)") == ("Breaking Bad", 2008)
    assert parse_title_year("  M*A*S*H (1972) ") == ("M*A*S*H", 1972)
    assert parse_title_year("Some Show") == ("Some Show", None)
    assert parse_title_year("Blade Runner 2049") == ("Blade Runner 2049", None)


def test_episode_re():
    m = EPISODE_RE.fullmatch(
        "OpenSubtitles/raw/en/2009/1232250_903747_2_13/1956248252.xml"
    )
    assert m and m.groups() == ("1232250", "903747", "2", "13")
    # movies (bare imdb id) don't match
    assert not EPISODE_RE.fullmatch("OpenSubtitles/raw/en/2011/1478804/1953352016.xml")
    assert not EPISODE_RE.fullmatch("OpenSubtitles/raw/en/2011/1478804/")


def test_episode_filename():
    assert episode_filename("Breaking Bad", 2, 13, ".mkv") == "Breaking Bad - S02E13.mkv"
    # '*' is stripped for Windows-safe filenames
    assert episode_filename("M*A*S*H", 2, 13, ".mkv") == "MASH - S02E13.mkv"
    assert sanitize_filename('What/If: A "Question"') == "WhatIf A Question"
