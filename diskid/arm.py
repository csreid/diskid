"""Show-name hints from ARM's own database.

With ARM's disc identification off, ripped folders are called `None (None)`,
which says nothing. ARM's `job` table does record the disc's volume `label`
(e.g. `Orange_is_the_New_Black_Season_1_Disc_1`) and the output `path`, so a
folder can be mapped back to its label and the label cleaned into a show name.

The database is opened read-only. Paths in it are the *container's* view
(`/home/arm/media/...`), so folders are matched by their last two components
(`unidentified/None (None)`), newest job first.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

_TRAILER = re.compile(
    r"[\s._-]*\b(?:season|series|disc|disk|volume|vol|s|d)[\s._-]*\d+\b.*$", re.IGNORECASE)
_TRAILER_WORDS = re.compile(
    r"[\s._-]*\b(?:complete|the complete|collection|box ?set)\b.*$", re.IGNORECASE)


def clean_label(label: str) -> str | None:
    """'Orange_is_the_New_Black_Season_1_Disc_1' -> 'Orange is the New Black'.

    Returns None when nothing usable is left (e.g. a bare serial like 'HGG0NNM1'
    can't be told from a title here; IMDb simply won't find it).
    """
    text = re.sub(r"[_]+", " ", label).strip()
    text = _TRAILER.sub("", text)
    text = _TRAILER_WORDS.sub("", text).strip(" -._")
    return text or None


def label_for(db_path: Path, disc_dir: Path) -> str | None:
    """The ARM disc label for the job that produced `disc_dir`, if any."""
    tail = f"%/{disc_dir.parent.name}/{disc_dir.name}"
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        try:
            row = con.execute(
                "SELECT label FROM job WHERE path LIKE ? AND label IS NOT NULL "
                "ORDER BY job_id DESC LIMIT 1", (tail,)).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return row[0] if row else None


def show_hint(db_path: Path, disc_dir: Path) -> str | None:
    """A show-name hint for `disc_dir` from its ARM label, or None."""
    label = label_for(db_path, disc_dir)
    return clean_label(label) if label else None
