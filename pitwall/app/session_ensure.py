"""
On-demand: ensure a (circuit, year, Q/Race) session row exists in SQLite
by loading from FastF1 if missing. FP sessions are not ingested (pipeline
only extracts Q and Race in SESSION_TYPES).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

log = logging.getLogger(__name__)

UNAVAILABLE = (
    "Session data is not available for the requested calendar slot: "
    "the event could not be loaded from the FastF1 data source. "
    "Try a different season, check the circuit name, and use Q or Race. "
    "If you need practice (FP) data, the local database may not include it—run a full "
    "pipeline extract for that season, or rephrase in terms of quali or race."
)


def try_ingest_session(
    circuit: str,
    year: int | None,
    session_type: str,
) -> bool:
    """
    If missing in DB, run FastF1 extraction for (circuit, season, Q|Race).

    Returns True if a sessions row is present after the call (or was already
    there). Returns False if ingest failed or the session type is not extractable
    (e.g. FP) or no matching event exists in the season schedule.
    """
    st = session_type
    if st in ("Q1", "Q2", "Q3"):
        st = "Q"
    if st in ("R", "Race"):
        st = "Race"
    if st not in ("Q", "Race"):
        log.info("session_ensure: skip FastF1 ingest for session_type=%r (not Q/Race)", session_type)
        return False

    from config import SEASONS
    from pipeline.db import init_db
    from pipeline.extract import ingest_one_session_by_names

    init_db()
    years: list[int] = [int(year)] if year is not None else list(reversed(SEASONS))
    for y in years:
        log.info("session_ensure: trying ingest season=%d circuit=%r session=%s", y, circuit, st)
        if ingest_one_session_by_names(y, circuit, st):
            log.info("session_ensure: success season=%d circuit=%r session=%s", y, circuit, st)
            return True
    return False
