"""
SQL retrieval layer — parameterized queries only, no string formatting.
Used in telemetry mode to fetch lap/sector data from SQLite
and inject into the model prompt as structured context.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

log = logging.getLogger(__name__)

# Circuit name mapping: what intent.py produces → what the DB stores
_CIRCUIT_MAP: dict[str, str] = {
    "Great Britain": "British", "UK": "British", "Silverstone": "British",
    "Belgium": "Belgian", "Spa": "Belgian",
    "Italy": "Italian", "Monza": "Italian",
    "Australia": "Australian", "Albert Park": "Australian",
    "Austria": "Austrian", "Red Bull Ring": "Austrian",
    "Japan": "Japanese", "Suzuka": "Japanese",
    "Hungary": "Hungarian", "Hungaroring": "Hungarian",
    "Netherlands": "Dutch", "Zandvoort": "Dutch",
    "Spain": "Spanish", "Barcelona": "Spanish",
    "France": "French", "Paul Ricard": "French",
    "Canada": "Canadian", "Montreal": "Canadian",
    "Saudi Arabia": "Saudi Arabian", "Jeddah": "Saudi Arabian",
    "Sao Paulo": "São Paulo", "Brazil": "São Paulo", "Interlagos": "São Paulo",
    "United States": "United States", "Austin": "United States", "COTA": "United States",
    # These already match DB names
    "Abu Dhabi": "Abu Dhabi", "Azerbaijan": "Azerbaijan", "Bahrain": "Bahrain",
    "Chinese": "Chinese", "Monaco": "Monaco", "Singapore": "Singapore",
    "Qatar": "Qatar", "Miami": "Miami", "Las Vegas": "Las Vegas",
    "Mexico City": "Mexico City", "Emilia Romagna": "Emilia Romagna",
}


def _resolve_circuit(name: str | None) -> str | None:
    """Map intent circuit name → DB circuit name."""
    if not name:
        return None
    # Direct match first
    if name in _CIRCUIT_MAP:
        return _CIRCUIT_MAP[name]
    # Case-insensitive search
    for k, v in _CIRCUIT_MAP.items():
        if k.lower() == name.lower():
            return v
    return name  # Try raw name as fallback


def _get_conn() -> sqlite3.Connection:
    """Open a read-only connection to the telemetry database."""
    db = Path(DB_PATH)
    if not db.exists():
        raise FileNotFoundError(f"Database not found: {db}")
    return sqlite3.connect(str(db))


def _fmt_time(seconds: float | None) -> str:
    """Format seconds as M:SS.sss or SS.sss."""
    if seconds is None:
        return "N/A"
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m}:{s:06.3f}"
    return f"{seconds:.3f}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_telemetry(intent: dict) -> str:
    """Query SQLite for lap data matching the parsed intent.

    Returns a formatted text block suitable for injection into the model prompt.
    Raises RuntimeError if no data is found.
    """
    drivers = intent.get("drivers") or []
    circuit_raw = intent.get("circuit")
    year = intent.get("year")
    session = intent.get("session") or "Race"
    lap_num = intent.get("lap")

    circuit = _resolve_circuit(circuit_raw)

    conn = _get_conn()
    cur = conn.cursor()

    try:
        # ── Build session filter ─────────────────────────────────────────
        session_sql = """
            SELECT id, season, circuit, session_type, date
            FROM sessions
            WHERE circuit = ?
        """
        params: list = [circuit]

        if year:
            session_sql += " AND season = ?"
            params.append(year)

        # Map session type
        sess_type = session
        if session in ("Q", "Q1", "Q2", "Q3"):
            sess_type = "Q"
        elif session in ("Race", "R"):
            sess_type = "Race"
        session_sql += " AND session_type = ?"
        params.append(sess_type)

        session_sql += " ORDER BY season DESC LIMIT 1"
        cur.execute(session_sql, params)
        sess_row = cur.fetchone()

        if not sess_row:
            raise RuntimeError(
                f"No session found for circuit={circuit}, year={year}, session={sess_type}"
            )

        session_id, season, db_circuit, db_sess_type, date = sess_row

        # ── Fetch laps ───────────────────────────────────────────────────
        lines = [
            f"=== TELEMETRY DATA ===",
            f"Circuit: {db_circuit} | Season: {season} | Session: {db_sess_type} | Date: {date}",
            "",
        ]

        for drv in (drivers if drivers else ["all"]):
            lap_sql = """
                SELECT driver, lap_number, lap_time, sector1, sector2, sector3,
                       compound, tyre_age
                FROM laps
                WHERE session_id = ?
            """
            lap_params: list = [session_id]

            if drv != "all":
                lap_sql += " AND driver = ?"
                lap_params.append(drv)

            if lap_num and lap_num != "fastest":
                lap_sql += " AND lap_number = ?"
                lap_params.append(int(lap_num))
            elif lap_num == "fastest":
                lap_sql += " AND is_valid = 1 ORDER BY lap_time ASC LIMIT 1"
            else:
                # No specific lap — get a summary (fastest + last stint)
                lap_sql += " AND is_valid = 1 ORDER BY lap_time ASC LIMIT 5"

            cur.execute(lap_sql, lap_params)
            rows = cur.fetchall()

            if not rows:
                lines.append(f"Driver {drv}: No lap data found.")
                continue

            lines.append(f"--- Driver: {drv if drv != 'all' else 'ALL'} ---")
            lines.append(f"{'Lap':>4}  {'LapTime':>9}  {'S1':>7}  {'S2':>7}  {'S3':>7}  {'Compound':>10}  {'TyreAge':>7}")

            for row in rows:
                d, ln, lt, s1, s2, s3, comp, ta = row
                driver_label = d if drv == "all" else ""
                prefix = f"{d} " if drv == "all" else ""
                lines.append(
                    f"{prefix}{ln:>4}  {_fmt_time(lt):>9}  "
                    f"{_fmt_time(s1):>7}  {_fmt_time(s2):>7}  {_fmt_time(s3):>7}  "
                    f"{comp or 'N/A':>10}  {ta if ta is not None else 'N/A':>7}"
                )
            lines.append("")

        result = "\n".join(lines)
        log.info("Telemetry fetched: %d chars for %s @ %s %s", len(result), drivers, circuit, season)
        return result

    finally:
        conn.close()
