"""Small filesystem helpers shared by the watcher and `apply`."""

from __future__ import annotations

from pathlib import Path


def prune_empty_dirs(root: Path) -> list[Path]:
    """Remove `root`'s empty subfolders, and `root` itself if that leaves it empty.

    Only ever calls `rmdir`, so a folder holding any file is left alone. Used
    after files are moved out of a disc folder. Returns the folders removed.
    """
    removed: list[Path] = []
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            d.rmdir()
            removed.append(d)
        except OSError:
            pass
    try:
        root.rmdir()
        removed.append(root)
    except OSError:
        pass
    return removed
