"""On-disk cache for dry-run identification mappings.

`identify`/`identify-disc` dry runs write their proposed renames here and
print a mapping id; `diskid apply <mapping-id>` replays them without
re-running whisper/matching.
"""

import json
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path


def cache_dir() -> Path:
    d = Path(tempfile.gettempdir()) / "diskid" / "mappings"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_mapping(command: str, show: str, entries: list[dict]) -> str:
    mapping_id = uuid.uuid4().hex[:8]
    path = cache_dir() / f"{mapping_id}.json"
    path.write_text(json.dumps({
        "id": mapping_id,
        "created": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "show": show,
        "entries": entries,
    }, indent=2))
    return mapping_id


def load_mapping(mapping_id: str) -> dict | None:
    path = cache_dir() / f"{mapping_id}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())
