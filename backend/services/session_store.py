"""
session_store.py — Minimal disk-backed persistence for session-scoped state
(forecast cache, chat history) so it survives a backend restart instead of
vanishing with the in-memory dict that holds it day-to-day.

Not a database: one JSON file per session, under backend/session_data/<kind>/.
That's the right amount of durability for a single-process capstone deploy —
callers should still keep an in-memory copy as the fast path and treat this
purely as a restart-safe backstop.
"""
import json
import re
from pathlib import Path
from typing import Optional

SESSION_DATA_DIR = Path(__file__).parent.parent / "session_data"

_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")


def _path_for(kind: str, session_id: str) -> Path:
    if not _SAFE_ID.match(session_id):
        # session_id is client-generated (crypto.randomUUID() / "default") — this
        # guards against it ever being used as a path-traversal payload.
        raise ValueError(f"Invalid session_id: {session_id!r}")
    kind_dir = SESSION_DATA_DIR / kind
    kind_dir.mkdir(parents=True, exist_ok=True)
    return kind_dir / f"{session_id}.json"


def save(kind: str, session_id: str, data: dict) -> None:
    """Write session state to disk atomically (write to a temp file, then rename)."""
    path = _path_for(kind, session_id)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(data))
    tmp_path.replace(path)


def load(kind: str, session_id: str) -> Optional[dict]:
    """Read session state from disk. Returns None if it doesn't exist or is corrupt."""
    path = _path_for(kind, session_id)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def delete(kind: str, session_id: str) -> None:
    path = _path_for(kind, session_id)
    path.unlink(missing_ok=True)
