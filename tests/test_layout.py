from pathlib import PurePosixPath

import pytest

from diskid import layout
from diskid.identify import EpisodeRef, FileResult, IdentifierConfig, Show, rename_to_target
from diskid.watch import prune_empty_dirs

VARS = layout.template_vars(
    show="Mr/Robot: Test", year=2015, season=2, episode=3, episode_title="eps2.1/a",
    ext=".mkv", stem="title_t01", order="tmdb", show_imdb="tt1", episode_imdb=None)


def test_default_template():
    assert layout.render(layout.DEFAULT_TEMPLATE, VARS) == PurePosixPath("MrRobot Test - S02E03.mkv")


def test_library_template_and_year_conditional():
    assert str(layout.render(layout.LIBRARY_TEMPLATE, VARS)) == (
        "MrRobot Test (2015)/Season 02/MrRobot Test - S02E03.mkv")
    no_year = dict(VARS, year="")
    assert str(layout.render(layout.LIBRARY_TEMPLATE, no_year)).startswith("MrRobot Test/Season 02/")


def test_values_cannot_inject_path_separators():
    out = layout.render("{{ show }}/{{ episode_title }}{{ ext }}", VARS)
    assert out.parts == ("MrRobot Test", "eps2.1a.mkv")


@pytest.mark.parametrize("bad", [
    "{{ nope }}", "{{ show", "/abs/{{ show }}", "../{{ show }}", "{{ show }}/../x", "  ", "{{ ''.__class__ }}",
])
def test_bad_templates_rejected(bad):
    with pytest.raises(layout.TemplateError):
        layout.render(bad, VARS)


def test_config_validates_template():
    with pytest.raises(ValueError):
        IdentifierConfig(template="{{ unknown_var }}")
    IdentifierConfig(template=layout.LIBRARY_TEMPLATE)


def test_file_result_template_vars_and_json(tmp_path):
    ref = EpisodeRef(1, 2, 1, 2, "tmdb", 42, "Pilot")
    r = FileResult(path=tmp_path / "t.mkv", status="identified", show=Show(7, "S", 2011), episode=ref)
    v = r.template_vars()
    assert v["year"] == 2011 and v["episode_title"] == "Pilot" and v["episode_imdb"] == "tt0000042"
    assert r.to_dict()["episode"]["code"] == "S01E02"
    assert FileResult(path=tmp_path / "x", status="uncertain").template_vars() is None


def test_rename_creates_parents_and_never_overwrites(tmp_path):
    src = tmp_path / "a.mkv"
    src.write_bytes(b"x")
    dst = tmp_path / "lib" / "Show" / "Season 01" / "e.mkv"
    assert rename_to_target(src, dst) == (dst, "") and dst.read_bytes() == b"x"
    src.write_bytes(b"y")
    assert rename_to_target(src, dst) == (None, "target exists") and dst.read_bytes() == b"x"


def test_prune_empty_dirs_keeps_anything_with_files(tmp_path):
    disc = tmp_path / "disc"
    (disc / "extras" / "deep").mkdir(parents=True)
    prune_empty_dirs(disc)
    assert not disc.exists()
    (disc / "extras").mkdir(parents=True)
    (disc / "left.mkv").write_bytes(b"x")
    prune_empty_dirs(disc)
    assert (disc / "left.mkv").exists() and not (disc / "extras").exists()
