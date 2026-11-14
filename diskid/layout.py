"""Render a file's destination path from a Jinja template.

The template produces a path *relative to the output directory* (or, when no
output directory is configured, relative to the file's current folder). It is
rendered in a sandbox with strict variables, so a typo fails loudly instead of
silently producing a folder called "None".

Variables (strings are made filename-safe first, so a title containing "/"
can't create extra folders):

    show, series    show title, e.g. "Orange Is the New Black"
    year            show start year, or "" when unknown
    season, episode integers, in the numbering `order` says (see below)
    code            "S01E02"
    episode_title   episode name when TMDB supplied it, else ""
    ext             original file extension with the dot, e.g. ".mkv"
    stem            original filename without extension
    order           "tmdb" or "imdb": which numbering season/episode are in
    show_imdb       "tt2372162"
    episode_imdb    "tt3456789", or "" when unknown

Filter: `{{ season|pad }}` zero-pads to 2 digits (`|pad(3)` for 3).

Examples:

    {{ show }} - {{ code }}{{ ext }}                                 (the default)
    {{ show }}{% if year %} ({{ year }}){% endif %}/Season {{ season|pad }}/{{ show }} - {{ code }}{{ ext }}
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

from jinja2 import StrictUndefined, TemplateError as _JinjaError
from jinja2.sandbox import SandboxedEnvironment

DEFAULT_TEMPLATE = "{{ show }} - {{ code }}{{ ext }}"

#: A fuller layout most media servers (Jellyfin, Plex) understand.
LIBRARY_TEMPLATE = (
    "{{ show }}{% if year %} ({{ year }}){% endif %}/Season {{ season|pad }}/"
    "{{ show }} - {{ code }}{{ ext }}"
)


class TemplateError(ValueError):
    """The template is malformed, uses an unknown variable, or renders a bad path."""


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', "", name).strip()


def _pad(value: Any, width: int = 2) -> str:
    return f"{int(value):0{width}d}"


_env = SandboxedEnvironment(undefined=StrictUndefined, autoescape=False)
_env.filters["pad"] = _pad

#: Sample values used to validate a template before any file is processed.
_SAMPLE = dict(
    show="Show", series="Show", year=2000, season=1, episode=2, code="S01E02",
    episode_title="Title", ext=".mkv", stem="title_t00", order="tmdb",
    show_imdb="tt0000001", episode_imdb="tt0000002",
)


def template_vars(
    *, show: str, year: int | None, season: int, episode: int, episode_title: str | None,
    ext: str, stem: str, order: str, show_imdb: str, episode_imdb: str | None,
) -> dict[str, Any]:
    """The variable set a template is rendered with (see the module docstring)."""
    clean = sanitize_filename
    return dict(
        show=clean(show), series=clean(show), year=year if year else "",
        season=season, episode=episode, code=f"S{season:02d}E{episode:02d}",
        episode_title=clean(episode_title or ""), ext=ext, stem=clean(stem), order=order,
        show_imdb=show_imdb, episode_imdb=episode_imdb or "",
    )


def render(template: str, variables: dict[str, Any]) -> PurePosixPath:
    """Render `template` to a safe relative path.

    Raises `TemplateError` for syntax errors, unknown variables, an empty
    result, an absolute path, or any `.`/`..` component.
    """
    try:
        text = _env.from_string(template).render(**variables)
    except _JinjaError as e:
        raise TemplateError(f"{type(e).__name__}: {e}") from e
    except (TypeError, ValueError) as e:  # e.g. a filter given a non-number
        raise TemplateError(str(e)) from e
    if text.startswith(("/", "\\")):
        raise TemplateError(f"template must be relative, rendered {text!r}")
    parts = [p.strip() for p in text.replace("\\", "/").split("/")]
    parts = [p for p in parts if p]
    if not parts:
        raise TemplateError("template rendered an empty path")
    if any(p in (".", "..") for p in parts):
        raise TemplateError(f"template rendered a '.' or '..' component: {text!r}")
    return PurePosixPath(*parts)


def validate(template: str) -> None:
    """Fail fast on a bad template. Raises `TemplateError`."""
    render(template, _SAMPLE)
