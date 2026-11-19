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


def job_for(db_path: Path, disc_dir: Path) -> dict | None:
    """The newest ARM job whose output folder is `disc_dir`: label, title, imdb_id, video_type, status."""
    tail = f"%/{disc_dir.parent.name}/{disc_dir.name}"
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        con.row_factory = sqlite3.Row
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(job)")}
            want = [c for c in ("label", "title", "imdb_id", "video_type", "status") if c in cols]
            if not want:
                return None
            row = con.execute(
                f"SELECT {', '.join(want)} FROM job WHERE path LIKE ? "
                "ORDER BY job_id DESC LIMIT 1", (tail,)).fetchone()
        finally:
            con.close()
    except sqlite3.Error:
        return None
    return dict(row) if row else None


#: ARM job statuses that mean the job is finished (anything else may still write to its folder).
FINISHED = ("success", "fail")


def job_active(db_path: Path, disc_dir: Path) -> bool:
    """Is the newest ARM job for `disc_dir` still running (ripping, transcoding, ...)?"""
    job = job_for(db_path, disc_dir)
    return bool(job and job.get("status") and job["status"] not in FINISHED)


def label_for(db_path: Path, disc_dir: Path) -> str | None:
    """The ARM disc label for the job that produced `disc_dir`, if any."""
    job = job_for(db_path, disc_dir)
    return job.get("label") if job else None


def show_hint(db_path: Path, disc_dir: Path) -> str | None:
    """A show-name hint for `disc_dir` from its ARM label, or None."""
    label = label_for(db_path, disc_dir)
    return clean_label(label) if label else None


def recent_jobs(db_path: Path, n: int = 3) -> list[dict]:
    """The newest `n` ARM jobs (id, title, label, status, titles, times, ejected, errors)."""
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=5)
        con.row_factory = sqlite3.Row
        try:
            cols = {r[1] for r in con.execute("PRAGMA table_info(job)")}
            want = [c for c in ("job_id", "title", "label", "status", "no_of_titles",
                                "start_time", "stop_time", "ejected", "errors") if c in cols]
            rows = con.execute(
                f"SELECT {', '.join(want)} FROM job ORDER BY job_id DESC LIMIT ?", (n,)).fetchall()
        finally:
            con.close()
    except sqlite3.Error:
        return []
    return [dict(r) for r in rows]
