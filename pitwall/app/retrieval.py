"""
SQL retrieval layer — parameterized queries only, no string formatting.
Fetches lap/sector data from SQLite. Laps are pre-filtered at ingest (FastF1);
queries add is_valid, non-null sectors, race outlap exclusion, fastest-per-driver.
"""

from __future__ import annotations

import logging
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH
from delta_interpretation import (
    append_sector_time_delta_interpretation,
    filter_delta_lines_for_drivers,
)

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
    "Abu Dhabi": "Abu Dhabi", "Azerbaijan": "Azerbaijan", "Bahrain": "Bahrain",
    "Chinese": "Chinese", "Monaco": "Monaco", "Singapore": "Singapore",
    "Qatar": "Qatar", "Miami": "Miami", "Las Vegas": "Las Vegas",
    "Mexico City": "Mexico City", "Emilia Romagna": "Emilia Romagna",
}

IDENTICAL_LAP_ERROR = (
    "Query returned identical lap times for both drivers which is not physically possible. "
    "This likely indicates a data retrieval error — the session may not be cached or the "
    "driver filter is not working correctly. Try a different season or session."
)

SUSPECT_CLOSE_QUALI = (
    "These qualifying lap times differ by less than 0.050s between two drivers, which is "
    "implausible to three decimals for an independent per-driver query. Treated as a data "
    "retrieval error; do not use these figures as a reliable comparison. Try a different "
    "season or session, or re-run a full session extract."
)

MISSING_DRIVER_ERROR = (
    "The query did not return a distinct valid lap for each requested driver. "
    "The session data may be incomplete or a driver may not have set a representative lap. "
    "Try a different season or session."
)


def _resolve_circuit(name: str | None) -> str | None:
    if not name:
        return None
    if name in _CIRCUIT_MAP:
        return _CIRCUIT_MAP[name]
    for k, v in _CIRCUIT_MAP.items():
        if k.lower() == name.lower():
            return v
    return name


def _get_conn() -> sqlite3.Connection:
    db = Path(DB_PATH)
    if not db.exists():
        raise FileNotFoundError(f"Database not found: {db}")
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _fmt_time(seconds: float | None) -> str:
    if seconds is None:
        return "N/A"
    if seconds >= 60:
        m = int(seconds // 60)
        s = seconds - m * 60
        return f"{m}:{s:06.3f}"
    return f"{seconds:.3f}"


def _dedupe_preserve(codes: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in codes:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


@dataclass
class TelemetryFetchResult:
    """Result of fetch_telemetry_structured."""

    text: str
    meta: dict[str, Any] = field(default_factory=dict)
    has_lap_rows: bool = False
    query_found_session: bool = False
    rejection_reason: str | None = None


def fetch_telemetry(intent: dict) -> str:
    """Backward-compatible: return text only."""
    return fetch_telemetry_structured(intent).text


def _try_fetch_session_row(
    cur: sqlite3.Cursor, circuit: str, year: int | None, sess_type: str
) -> Any:
    session_sql = """
        SELECT id, season, circuit, session_type, date
        FROM sessions
        WHERE circuit = ?
    """
    params: list = [circuit]
    if year:
        session_sql += " AND season = ?"
        params.append(year)
    session_sql += " AND session_type = ?"
    params.append(sess_type)
    session_sql += " ORDER BY season DESC LIMIT 1"
    cur.execute(session_sql, params)
    return cur.fetchone()


def _validate_two_driver_times(
    requested: list[str],
    times: list[tuple[str, float]],
    *,
    is_quali: bool,
    wants_avg: bool,
) -> str | None:
    if len(requested) != 2 or len(times) < 2:
        return None
    by_driver = {c: t for c, t in times}
    if not all(d in by_driver for d in requested):
        return MISSING_DRIVER_ERROR
    t1, t2 = by_driver[requested[0]], by_driver[requested[1]]
    if round(t1, 3) == round(t2, 3):
        return IDENTICAL_LAP_ERROR
    # Quali head-to-head: independent per-driver pulls should not match within 50 ms
    if is_quali and not wants_avg and abs(t1 - t2) < 0.05:
        return SUSPECT_CLOSE_QUALI
    return None


def fetch_telemetry_structured(intent: dict) -> TelemetryFetchResult:
    """Run SQL and return structured result for validation and source labeling."""
    drivers = intent.get("drivers") or []
    circuit_raw = intent.get("circuit")
    year = intent.get("year")
    session = intent.get("session") or "Race"
    lap_num = intent.get("lap")

    circuit = _resolve_circuit(circuit_raw)
    meta: dict[str, Any] = {
        "circuit": circuit,
        "year": year,
        "session_type": session,
    }

    if not circuit:
        return TelemetryFetchResult(
            text="",
            meta=meta,
            has_lap_rows=False,
            query_found_session=False,
        )

    sess_type = session
    if session in ("Q", "Q1", "Q2", "Q3"):
        sess_type = "Q"
    elif session in ("Race", "R"):
        sess_type = "Race"
    elif session and session.startswith("FP"):
        sess_type = session

    conn = _get_conn()
    cur = conn.cursor()

    try:
        sess_row = _try_fetch_session_row(cur, circuit, year, sess_type)
        if not sess_row:
            from session_ensure import UNAVAILABLE, try_ingest_session

            if try_ingest_session(circuit, year, sess_type):
                sess_row = _try_fetch_session_row(cur, circuit, year, sess_type)
            if not sess_row:
                log.info(
                    "No session in DB and FastF1 ingest failed or skipped: "
                    "circuit=%s year=%s session=%s",
                    circuit,
                    year,
                    sess_type,
                )
                return TelemetryFetchResult(
                    text="",
                    meta={**meta, "session_type": sess_type, "ingest_tried": True},
                    has_lap_rows=False,
                    query_found_session=False,
                    rejection_reason=UNAVAILABLE,
                )

        session_id = int(sess_row["id"])
        season = int(sess_row["season"])
        db_circuit = str(sess_row["circuit"])
        db_sess_type = str(sess_row["session_type"])
        date = sess_row["date"]
        meta.update(
            {
                "circuit": db_circuit,
                "year": season,
                "session_type": db_sess_type,
                "session_id": session_id,
            }
        )

        is_race = db_sess_type == "Race"
        race_extra = " AND l.lap_number > 1 " if is_race else ""
        wants_avg = bool(intent.get("wants_average"))
        is_quali = db_sess_type == "Q"

        lines: list[str] = [
            "=== TELEMETRY DATA ===",
            f"Circuit: {db_circuit} | Season: {season} | Session: {db_sess_type} | Date: {date or 'N/A'}",
            f"(SQL filters: is_valid; non-null full sectors;{race_extra and ' outlap (lap 1) excluded for race session;' or ''} "
            f"{' average lap time' if wants_avg else ' fastest valid lap per driver'})",
            "",
        ]
        has_any = False
        driver_list = _dedupe_preserve(list(drivers) if drivers else ["all"])
        all_specific = "all" not in driver_list
        spec_codes: list[str] = [d for d in driver_list if d != "all"]

        def build_driver_sql() -> tuple[str, list]:
            if not all_specific or not spec_codes:
                return "", [session_id]
            placeholders = ", ".join("?" * len(spec_codes))
            return f" AND l.driver IN ({placeholders})", [session_id, *spec_codes]

        dsql, pfx = build_driver_sql()
        log.info(
            "telemetry_lap_query start session_id=%s season=%s circuit=%r session_type=%s "
            "wants_avg=%s lap=%r drivers=%r (single IN-query=%s) params=%r",
            session_id,
            season,
            db_circuit,
            db_sess_type,
            wants_avg,
            lap_num,
            driver_list,
            bool(spec_codes) and "all" not in driver_list,
            pfx,
        )

        sql_branch = "unknown"
        if wants_avg and lap_num in (None, "fastest"):
            sql_branch = "avg_lap"
            sql = f"""
                SELECT l.driver, AVG(l.lap_time), AVG(l.sector1), AVG(l.sector2), AVG(l.sector3)
                FROM laps l
                WHERE l.session_id = ?
                  {dsql}
                  AND l.is_valid = 1
                  AND l.lap_time IS NOT NULL AND l.lap_time > 0
                  AND l.sector1 IS NOT NULL AND l.sector2 IS NOT NULL AND l.sector3 IS NOT NULL
                  {race_extra}
                GROUP BY l.driver
            """
            cur.execute(sql, pfx)
            rows = cur.fetchall()
        elif lap_num not in (None, "fastest") and isinstance(lap_num, int):
            sql_branch = f"lap_{lap_num}"
            sql = f"""
                SELECT l.driver, l.lap_number, l.lap_time, l.sector1, l.sector2, l.sector3, l.compound, l.tyre_age
                FROM laps l
                WHERE l.session_id = ?
                  {dsql}
                  AND l.lap_number = ?
                  AND l.is_valid = 1
                  AND l.lap_time IS NOT NULL
                  AND l.sector1 IS NOT NULL AND l.sector2 IS NOT NULL AND l.sector3 IS NOT NULL
                  {race_extra}
            """
            p = list(pfx) + [lap_num]
            cur.execute(sql, p)
            rows = cur.fetchall()
        else:
            sql_branch = "fastest_per_driver"
            sql = f"""
                SELECT driver, lap_number, lap_time, sector1, sector2, sector3, compound, tyre_age
                FROM (
                    SELECT
                        l.driver, l.lap_number, l.lap_time, l.sector1, l.sector2, l.sector3,
                        l.compound, l.tyre_age,
                        ROW_NUMBER() OVER (
                            PARTITION BY l.driver
                            ORDER BY l.lap_time ASC, l.lap_number ASC
                        ) AS rn
                    FROM laps l
                    WHERE l.session_id = ?
                      {dsql}
                      AND l.is_valid = 1
                      AND l.lap_time IS NOT NULL AND l.lap_time > 0
                      AND l.sector1 IS NOT NULL
                      AND l.sector2 IS NOT NULL
                      AND l.sector3 IS NOT NULL
                      {race_extra}
                ) x
                WHERE x.rn = 1
            """
            cur.execute(sql, pfx)
            rows = cur.fetchall()

        nrows = len(rows) if rows else 0
        if rows:
            sample = []
            for r in rows[:3]:
                if wants_avg and lap_num in (None, "fastest"):
                    sample.append(f"{r[0]}:avgLap={float(r[1]):.3f}")
                else:
                    d = r["driver"] if isinstance(r, sqlite3.Row) else r[0]
                    lt = r["lap_time"] if isinstance(r, sqlite3.Row) else r[2]
                    sample.append(f"{d}:lapTime={float(lt):.3f}")
        else:
            sample = []
        log.info(
            "telemetry_lap_query result session_id=%s branch=%s row_count=%s sample=%r",
            session_id,
            sql_branch,
            nrows,
            sample,
        )

        if rows and len(spec_codes) == 2:
            pair_times: list[tuple[str, float]] = []
            if nrows < 2:
                log.warning("telemetry two-driver: expected 2 rows, got %d", nrows)
                return TelemetryFetchResult(
                    text=MISSING_DRIVER_ERROR,
                    meta={**meta, "two_driver_reject": True, "reason": "missing_row"},
                    has_lap_rows=False,
                    query_found_session=True,
                    rejection_reason=MISSING_DRIVER_ERROR,
                )
            if wants_avg and lap_num in (None, "fastest"):
                for r in rows:
                    pair_times.append((str(r[0]), float(r[1])))
            else:
                for row in rows:
                    pair_times.append((str(row["driver"]), float(row["lap_time"])))
            two_reject = _validate_two_driver_times(
                spec_codes, pair_times, is_quali=is_quali, wants_avg=bool(wants_avg)
            )
            if two_reject:
                log.warning("telemetry two-driver check failed: %s", two_reject[:160])
                return TelemetryFetchResult(
                    text=two_reject,
                    meta={**meta, "two_driver_reject": True},
                    has_lap_rows=False,
                    query_found_session=True,
                    rejection_reason=two_reject,
                )

        if not rows:
            who = (
                ", ".join(spec_codes)
                if spec_codes
                else ("ALL" if "all" in driver_list else ",".join(driver_list))
            )
            lines.append("--- " + f"Requested drivers: {who}" + " ---")
            lines.append("No valid lap data after filters (outlap/incomplete sectors excluded).")
            lines.append("")
        else:
            has_any = True
            if spec_codes:
                hdr = ", ".join(spec_codes) if all_specific and spec_codes else "ALL"
            else:
                hdr = "ALL"
            lines.append(f"--- Drivers: {hdr} (one combined query) ---")
            if len(spec_codes) >= 2 and "all" not in driver_list:
                lines.append(
                    "INSTRUCTION: Answer must compare *only* the drivers named above (3-letter codes). "
                    "Do not name, substitute, or compare to any other driver (including not the session’s "
                    "fastest or pole-sitter) unless that code appears in the line above. "
                    "Lap/sector numbers below are for those drivers only."
                )
            if wants_avg and lap_num in (None, "fastest"):
                lines.append(f"  {'Driver':4}  {'AvgLap':>9}  {'S1':>7}  {'S2':>7}  {'S3':>7}")
                for r in rows:
                    d, al, a1, a2, a3 = r[0], r[1], r[2], r[3], r[4]
                    log.info("telemetry per-row driver=%r avgLap=%.3f (avg branch)", d, float(al))
                    lines.append(
                        f"  {d:4}  {_fmt_time(float(al)):>9}  "
                        f"{_fmt_time(float(a1)):>7}  {_fmt_time(float(a2)):>7}  {_fmt_time(float(a3)):>7}"
                    )
            else:
                lines.append(
                    f"{'Dr':4}  {'Lap':>4}  {'LapTime':>9}  "
                    f"{'S1':>7}  {'S2':>7}  {'S3':>7}  {'Compound':>10}  {'TyreAge':>7}"
                )
                for row in rows:
                    d = row["driver"]
                    ln = row["lap_number"]
                    lt = row["lap_time"]
                    s1, s2, s3 = row["sector1"], row["sector2"], row["sector3"]
                    try:
                        comp = row["compound"]
                    except (KeyError, IndexError):
                        comp = None
                    try:
                        ta = row["tyre_age"]
                    except (KeyError, IndexError):
                        ta = None
                    log.info(
                        "telemetry per-row driver=%r lap=%s lapTime=%.3f",
                        d,
                        ln,
                        float(lt),
                    )
                    lines.append(
                        f"{d:4}  {int(ln):>4}  {_fmt_time(float(lt)):>9}  "
                        f"{_fmt_time(float(s1)):>7}  {_fmt_time(float(s2)):>7}  {_fmt_time(float(s3)):>7}  "
                        f"{(comp or 'N/A'):>10}  {ta if ta is not None else 'N/A':>7}"
                    )
            lines.append("")

        if has_any:
            try:
                append_sector_time_delta_interpretation(
                    lines,
                    cur,
                    session_id,
                    race_extra,
                    wants_avg=wants_avg,
                    lap_num=lap_num,
                )
            except Exception as exc:
                log.warning("Sector delta interpretation failed: %s", exc, exc_info=True)
        result_text = "\n".join(lines)
        if has_any:
            result_text = filter_delta_lines_for_drivers(
                result_text, intent.get("drivers") or None
            )
        if not has_any and driver_list:
            log.info("Telemetry: no valid lap rows after strict filters (session %s)", session_id)
        log.info("Telemetry fetch %d chars session=%s", len(result_text), session_id)
        return TelemetryFetchResult(
            text=result_text,
            meta=meta,
            has_lap_rows=has_any,
            query_found_session=True,
        )
    finally:
        conn.close()
