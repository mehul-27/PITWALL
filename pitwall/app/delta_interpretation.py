"""
Sector time deltas (seconds): single convention used by SQL telemetry + prompts.

Convention (time only):
  – Positive delta  => driver SLOWER than the stated reference in that sector
  – Negative delta  => driver FASTER  than the stated reference in that sector

Reference (retrieval + driver_sector_stats table):
  – Session’s fastest full lap: overall minimum lap time among valid laps, then
    that lap’s sector splits as baseline. (Same as pipeline driver_sector_stats.)

Corner min-speed "delta_vs_field" (km/h) is not a lap-time delta; see
config.CORNER_SPEED_DELTA_CONVENTION.
"""

from __future__ import annotations

import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    SECTOR_TIME_DELTA_CONVENTION_BANNER,
    TIME_DELTA_SIGN_RULE,
)


def _targets_from_intent_drivers(drivers: Sequence[str] | None) -> set[str] | None:
    if not drivers:
        return None
    if "all" in drivers:
        return None
    return {d for d in drivers if d != "all"}


def _fmt_interpretation_line(driver: str, sn: int, delta: float, ref_phrase: str) -> str:
    d_abs = abs(delta)
    if delta < 0:
        return (
            f"  {driver}  S{sn}: {delta:+.3f}s  →  {d_abs:.3f}s FASTER than {ref_phrase}"
        )
    if delta > 0:
        return (
            f"  {driver}  S{sn}: {delta:+.3f}s  →  {d_abs:.3f}s SLOWER than {ref_phrase}"
        )
    return f"  {driver}  S{sn}: {delta:+.3f}s  →  same as {ref_phrase}"


def append_sector_time_delta_interpretation(
    lines: list[str],
    cur: sqlite3.Cursor,
    session_id: int,
    race_extra: str,
    *,
    wants_avg: bool,
    lap_num: Any,
) -> None:
    """
    Appends a block with the sign rule, reference description, and one line
    per driver × sector. Uses full session to define reference, then filters
    display to intent drivers (handled at call site via separate query is ok:
    we always compute from full session for a fair reference).
    """
    rows, ref_label, ref_mode = _fetch_reference_rows(
        cur, session_id, race_extra, wants_avg, lap_num
    )
    if not rows or not ref_label:
        return

    b1, b2, b3, ref_phrase = _baseline_sectors_and_phrase(rows, ref_mode, ref_label)
    if b1 is None or b2 is None or b3 is None:
        return

    lines.append("")
    lines.append("--- SECTOR TIME DELTAS (seconds) ---")
    lines.append(SECTOR_TIME_DELTA_CONVENTION_BANNER)
    lines.append(TIME_DELTA_SIGN_RULE)
    lines.append(
        f"Reference ({ref_mode}): {ref_phrase}. "
        f"+ means slower in that sector vs reference, − means faster."
    )
    for r in sorted(rows, key=lambda x: (x[0], x[1] or 0)):
        drv, _ln, _lt, s1, s2, s3 = r
        for sn, (sv, bv) in enumerate(
            ((float(s1), b1), (float(s2), b2), (float(s3), b3)), start=1
        ):
            delta = sv - bv
            lines.append(_fmt_interpretation_line(str(drv), sn, delta, ref_phrase))


def _fetch_reference_rows(
    cur: sqlite3.Cursor,
    session_id: int,
    race_extra: str,
    wants_avg: bool,
    lap_num: Any,
) -> tuple[list[tuple], str, str]:
    """Return (row tuples, ref_label, ref_mode) where ref_mode is 'field_average' or 'session_fastest_lap'."""
    if wants_avg and lap_num in (None, "fastest"):
        cur.execute(
            f"""
            SELECT l.driver, NULL, AVG(l.lap_time), AVG(l.sector1), AVG(l.sector2), AVG(l.sector3)
            FROM laps l
            WHERE l.session_id = ?
              AND l.is_valid = 1
              AND l.lap_time IS NOT NULL AND l.lap_time > 0
              AND l.sector1 IS NOT NULL AND l.sector2 IS NOT NULL AND l.sector3 IS NOT NULL
              {race_extra}
            GROUP BY l.driver
            """,
            (session_id,),
        )
        qrows = cur.fetchall()
        if not qrows:
            return [], "", ""
        return (
            [
                (r[0], r[1], float(r[2]), r[3], r[4], r[5])
                for r in qrows
            ],
            f"field average over {len(qrows)} drivers (each value is that driver's mean sector time in this session)",
            "field_average",
        )

    if lap_num not in (None, "fastest") and isinstance(lap_num, int):
        cur.execute(
            f"""
            SELECT l.driver, l.lap_number, l.lap_time, l.sector1, l.sector2, l.sector3
            FROM laps l
            WHERE l.session_id = ?
              AND l.lap_number = ?
              AND l.is_valid = 1
              AND l.lap_time IS NOT NULL
              AND l.sector1 IS NOT NULL AND l.sector2 IS NOT NULL AND l.sector3 IS NOT NULL
              {race_extra}
            """,
            (session_id, lap_num),
        )
    else:
        cur.execute(
            f"""
            SELECT driver, lap_number, lap_time, sector1, sector2, sector3
            FROM (
                SELECT
                    l.driver, l.lap_number, l.lap_time, l.sector1, l.sector2, l.sector3,
                    ROW_NUMBER() OVER (
                        PARTITION BY l.driver
                        ORDER BY l.lap_time ASC, l.lap_number ASC
                    ) AS rn
                FROM laps l
                WHERE l.session_id = ?
                  AND l.is_valid = 1
                  AND l.lap_time IS NOT NULL AND l.lap_time > 0
                  AND l.sector1 IS NOT NULL
                  AND l.sector2 IS NOT NULL
                  AND l.sector3 IS NOT NULL
                  {race_extra}
            ) x
            WHERE x.rn = 1
            """,
            (session_id,),
        )

    raw = cur.fetchall()
    if not raw:
        return [], "", ""
    rows = [
        (
            r[0],
            r[1],
            float(r[2]),
            r[3],
            r[4],
            r[5],
        )
        for r in raw
    ]
    if lap_num not in (None, "fastest") and isinstance(lap_num, int):
        b = min(rows, key=lambda t: t[2])
        label = f"the fastest lap {lap_num} in this session ({b[0]})"
        return rows, label, "session_fastest_lap"
    b = min(rows, key=lambda t: t[2])
    label = f"the session's fastest full lap ({b[0]} lap {int(b[1])})"
    return rows, label, "session_fastest_lap"


def _baseline_sectors_and_phrase(
    rows: list[tuple],
    ref_mode: str,
    ref_label: str,
) -> tuple[float | None, float | None, float | None, str]:
    if ref_mode == "field_average":
        n = len(rows)
        b1 = sum(float(r[3]) for r in rows) / n
        b2 = sum(float(r[4]) for r in rows) / n
        b3 = sum(float(r[5]) for r in rows) / n
        phrase = f"{ref_label}"
        return b1, b2, b3, phrase
    b = min(rows, key=lambda t: t[2])
    phrase = ref_label
    return float(b[3]), float(b[4]), float(b[5]), phrase


def filter_delta_lines_for_drivers(
    text: str, drivers: list[str] | None
) -> str:
    """If intent lists specific drivers, drop interpretation lines for others."""
    targets = _targets_from_intent_drivers(drivers)
    if not targets or not text:
        return text
    if "--- SECTOR TIME DELTAS" not in text:
        return text
    lines = text.splitlines()
    out: list[str] = []
    in_block = False
    for line in lines:
        if line.strip() == "--- SECTOR TIME DELTAS (seconds) ---":
            in_block = True
            out.append(line)
            continue
        if in_block:
            stripped = line.strip()
            if stripped.startswith("---") and "SECTOR TIME DELTAS" not in line:
                in_block = False
                out.append(line)
                continue
            m = re.match(r"^(\s*)([A-Z]{2,3})\s+S[123]:", line)
            if m and m.group(2) not in targets:
                continue
        out.append(line)
    return "\n".join(out)


def telemetry_has_negative_sector_time_delta(telemetry: str) -> bool:
    """True if a sector time delta line shows a negative value (faster than reference)."""
    if not telemetry or "--- SECTOR TIME DELTAS" not in telemetry:
        return False
    return bool(re.search(r"(?m)^\s+\w+\s+S[123]:\s*-\d", telemetry))


SLOWER_SENTIMENT_WORDS = re.compile(
    r"\b(struggling|deficit|slower|behind|losing time|losing out)\b",
    re.IGNORECASE,
)


def check_response_sector_sign_misinterpretation(
    telemetry: str, model_response: str
) -> str | None:
    """
    If session telemetry says the driver was faster in a sector (negative delta)
    but the model uses language implying slowness, flag as possible sign error.
    """
    if not telemetry or not model_response:
        return None
    if not telemetry_has_negative_sector_time_delta(telemetry):
        return None
    if SLOWER_SENTIMENT_WORDS.search(model_response):
        return (
            "Possible sign mismatch: telemetry shows a faster sector (negative time delta vs reference) "
            "but the reply uses language usually associated with being slower. Re-check the data."
        )
    return None
