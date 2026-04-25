"""
Session counts per driver from SQLite (for sparse-data disclaimer).
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

log = logging.getLogger(__name__)

# driver code -> number of distinct session_ids with at least one lap
_DRIVER_VALID_SESSIONS: dict[str, int] = {}


def load_driver_session_counts() -> dict[str, int]:
    """Count distinct sessions per driver across the DB. Call at startup; call again
    after rebuilding ``pitwall.db`` from the pipeline to refresh in-memory stats."""
    global _DRIVER_VALID_SESSIONS
    _DRIVER_VALID_SESSIONS = {}
    db = Path(DB_PATH)
    if not db.is_file():
        log.warning("DB not found for driver session stats: %s", db)
        return {}
    try:
        conn = sqlite3.connect(str(db))
        try:
            cur = conn.execute(
                """
                SELECT driver, COUNT(DISTINCT session_id) AS n
                FROM laps
                GROUP BY driver
                """
            )
            for row in cur:
                d, n = str(row[0]), int(row[1])
                _DRIVER_VALID_SESSIONS[d] = n
        finally:
            conn.close()
    except Exception as exc:
        log.warning("Could not load driver session counts: %s", exc)
        return {}
    log.info("Driver session stats loaded: %d drivers", len(_DRIVER_VALID_SESSIONS))
    return _DRIVER_VALID_SESSIONS


def get_session_count(driver_code: str) -> int | None:
    return _DRIVER_VALID_SESSIONS.get(driver_code.upper())


# Alias after pipeline re-imports data
refresh_driver_session_counts = load_driver_session_counts
