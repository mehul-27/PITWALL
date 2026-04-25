"""
Phase 1 -- FastF1 extraction.
Loops over seasons / events / session types, filters invalid laps,
resamples Race telemetry to a uniform distance axis, and batch-inserts
everything into SQLite (sessions, laps, telemetry tables).

Resumable: already-loaded sessions are skipped on re-run.
Atomic:    each session is committed in one transaction; partial failures
           leave the DB unchanged so the session retries cleanly.

Usage:
    python pipeline/extract.py                  # all configured seasons
    python pipeline/extract.py --season 2022    # one year only
    python pipeline/extract.py --clear          # wipe DB + cache, then run all
    python pipeline/extract.py --clear --season 2022
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path
from typing import Optional

import fastf1
import numpy as np
import pandas as pd
from tqdm import tqdm

# -- project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CACHE_DIR, DB_PATH, SEASONS, SESSION_TYPES, TELEMETRY_SESSIONS
from pipeline.db import init_db, managed_connection

log = logging.getLogger(__name__)

# -- tuning constants ----------------------------------------------------------
# Metres between interpolated telemetry distance points.
# 50 m gives ~65-115 points/lap -- half the DB rows vs the original 25 m setting.
TELEMETRY_STEP_M: float = 50.0

# FastF1 session identifier map: config keys -> FastF1 identifiers
_SESSION_ID_MAP = {"Q": "Q", "FP2": "FP2", "Race": "Race"}


# -- helpers -------------------------------------------------------------------

def _td_to_seconds(td) -> Optional[float]:
    """pd.Timedelta -> float seconds, or None for NaT."""
    try:
        if pd.isna(td):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(td, "total_seconds"):
        return td.total_seconds()
    return None


def _circuit_name_from_event(event_name: str) -> str:
    """Strip 'Grand Prix' suffix to get the short ALL_CIRCUITS key."""
    return event_name.replace(" Grand Prix", "").strip()


# -- filters -------------------------------------------------------------------

def filter_laps(laps) -> "fastf1.core.Laps":
    """
    Return a Laps slice keeping only analysis-worthy laps.

    Removes:
    - Any null in LapTime / Sector1-3Time
    - Outlaps  (PitOutTime is not NaT)
    - Inlaps   (PitInTime  is not NaT)
    - SC / VSC laps (TrackStatus contains '4' or '6')
    - Steward-deleted laps (Deleted == True)

    Returns a FastF1 Laps instance so rows retain .get_telemetry().
    """
    mask = pd.Series(True, index=laps.index)

    for col in ("LapTime", "Sector1Time", "Sector2Time", "Sector3Time"):
        if col in laps.columns:
            mask &= laps[col].notna()

    if "PitOutTime" in laps.columns:
        mask &= laps["PitOutTime"].isna()

    if "PitInTime" in laps.columns:
        mask &= laps["PitInTime"].isna()

    if "TrackStatus" in laps.columns:
        sc_mask = laps["TrackStatus"].astype(str).str.contains(r"[46]", regex=True, na=True)
        mask &= ~sc_mask

    if "Deleted" in laps.columns:
        deleted = laps["Deleted"].infer_objects(copy=False).fillna(False).astype(bool)
        mask &= ~deleted

    valid = laps.loc[mask]
    log.debug("    filter_laps: %d / %d laps kept", len(valid), len(laps))
    return valid


# -- telemetry -----------------------------------------------------------------

def resample_telemetry(tel: pd.DataFrame) -> pd.DataFrame:
    """
    Interpolate Speed, Throttle, Brake, nGear to a fixed distance grid
    spaced TELEMETRY_STEP_M metres apart.

    Returns DataFrame: distance, speed, throttle, brake, gear.
    Returns empty DataFrame on missing or malformed telemetry.
    """
    empty = pd.DataFrame(columns=["distance", "speed", "throttle", "brake", "gear"])
    if tel is None or tel.empty or "Distance" not in tel.columns or len(tel) < 2:
        return empty

    dist = tel["Distance"].values.astype(float)
    valid_mask = ~np.isnan(dist)
    dist = dist[valid_mask]
    if len(dist) < 2:
        return empty

    grid = np.arange(0.0, dist[-1], TELEMETRY_STEP_M)
    if len(grid) == 0:
        return empty

    def _interp(col: str) -> np.ndarray:
        if col not in tel.columns:
            return np.zeros(len(grid), dtype=float)
        src = pd.Series(tel[col].values[valid_mask].astype(float)).ffill().bfill().to_numpy()
        return np.interp(grid, dist, src)

    speed    = _interp("Speed")
    throttle = _interp("Throttle")

    if "Brake" in tel.columns:
        brake_src = pd.Series(tel["Brake"].values[valid_mask].astype(float)).ffill().bfill().to_numpy()
        brake = np.interp(grid, dist, brake_src)
    else:
        brake = np.zeros(len(grid), dtype=float)

    gear = np.round(_interp("nGear")).astype(int)

    return pd.DataFrame({
        "distance": grid,
        "speed":    speed,
        "throttle": throttle,
        "brake":    brake,
        "gear":     gear,
    })


# -- DB helpers ----------------------------------------------------------------

def session_exists(conn, season: int, circuit: str, session_type: str) -> Optional[int]:
    """Return session id if already stored, else None."""
    row = conn.execute(
        "SELECT id FROM sessions WHERE season=? AND circuit=? AND session_type=?",
        (season, circuit, session_type),
    ).fetchone()
    return int(row["id"]) if row else None


def insert_session(conn, season: int, circuit: str,
                   session_type: str, date_str: Optional[str]) -> int:
    cur = conn.execute(
        "INSERT INTO sessions (season, circuit, session_type, date) VALUES (?,?,?,?)",
        (season, circuit, session_type, date_str),
    )
    return cur.lastrowid


def insert_laps_batch(conn, session_id: int, valid_laps) -> dict:
    """Batch-insert valid laps. Returns {(driver, lap_number): db_lap_id}."""
    rows = []
    for i in range(len(valid_laps)):
        lap = valid_laps.iloc[i]
        rows.append((
            session_id,
            str(lap["Driver"]),
            int(lap["LapNumber"]),
            _td_to_seconds(lap.get("LapTime")),
            _td_to_seconds(lap.get("Sector1Time")),
            _td_to_seconds(lap.get("Sector2Time")),
            _td_to_seconds(lap.get("Sector3Time")),
            str(lap.get("Compound") or "") or None,
            int(lap["TyreLife"]) if pd.notna(lap.get("TyreLife")) else None,
            1,
        ))
    conn.executemany(
        """INSERT INTO laps
               (session_id, driver, lap_number, lap_time,
                sector1, sector2, sector3, compound, tyre_age, is_valid)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        rows,
    )
    inserted = conn.execute(
        "SELECT id, driver, lap_number FROM laps WHERE session_id=?", (session_id,)
    ).fetchall()
    return {(r["driver"], int(r["lap_number"])): int(r["id"]) for r in inserted}


def insert_telemetry_batch(conn, rows: list) -> None:
    conn.executemany(
        "INSERT INTO telemetry (lap_id,distance,speed,throttle,brake,gear) VALUES (?,?,?,?,?,?)",
        rows,
    )


# -- clear helper --------------------------------------------------------------

def clear_data() -> None:
    """Delete pitwall.db and all FastF1 cache files."""
    if DB_PATH.exists():
        DB_PATH.unlink()
        log.info("Deleted %s", DB_PATH)
    else:
        log.info("DB not found -- nothing to delete")
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
        log.info("Deleted cache dir %s", CACHE_DIR)
    else:
        log.info("Cache dir not found -- nothing to delete")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Recreated empty cache dir")


# -- core pipeline -------------------------------------------------------------

def process_session(
    season: int,
    circuit: str,
    ff1_event,
    session_type: str,
) -> tuple[int, int]:
    """
    Load, filter, and persist one session.

    Telemetry is extracted only for TELEMETRY_SESSIONS (Race).
    Q is loaded with telemetry=False -- only lap/sector times are saved,
    which keeps Q cache files ~85% smaller.

    Returns (laps_stored, telemetry_rows_stored).
    """
    want_telemetry = session_type in TELEMETRY_SESSIONS

    # skip if already stored
    with managed_connection() as conn:
        existing = session_exists(conn, season, circuit, session_type)
    if existing is not None:
        log.info("      SKIP %s -- already stored (id=%d)", session_type, existing)
        return 0, 0

    log.info("      Loading %-4s (telemetry=%s) ...", session_type, want_telemetry)
    try:
        ff1_sess = ff1_event.get_session(_SESSION_ID_MAP[session_type])
        ff1_sess.load(laps=True, telemetry=want_telemetry, weather=False, messages=False)
    except Exception as exc:
        log.warning("      LOAD FAILED (%s): %s", session_type, exc)
        return 0, 0

    date_str: Optional[str] = None
    try:
        date_str = str(ff1_sess.date.date())
    except Exception:
        pass

    if ff1_sess.laps is None or ff1_sess.laps.empty:
        log.warning("      No laps in %s %s %s", season, circuit, session_type)
        with managed_connection() as conn:
            insert_session(conn, season, circuit, session_type, date_str)
        return 0, 0

    valid_laps = filter_laps(ff1_sess.laps)
    n_valid = len(valid_laps)
    n_total = len(ff1_sess.laps)
    log.info(
        "      %s: %d/%d laps valid%s",
        session_type, n_valid, n_total,
        " -- extracting telemetry ..." if want_telemetry else "",
    )

    if n_valid == 0:
        with managed_connection() as conn:
            insert_session(conn, season, circuit, session_type, date_str)
        return 0, 0

    # collect telemetry before DB write (Race only)
    tel_by_lap: dict[tuple, pd.DataFrame] = {}
    if want_telemetry:
        for i in tqdm(range(n_valid), desc=f"        tel {session_type}",
                      leave=False, unit="lap"):
            lap = valid_laps.iloc[i]
            key = (str(lap["Driver"]), int(lap["LapNumber"]))
            try:
                tel_by_lap[key] = resample_telemetry(lap.get_telemetry())
            except Exception as exc:
                log.debug("        tel skipped %s lap %s: %s", key[0], key[1], exc)
                tel_by_lap[key] = pd.DataFrame(
                    columns=["distance", "speed", "throttle", "brake", "gear"]
                )

    # atomic DB write
    tel_rows: list[tuple] = []
    with managed_connection() as conn:
        session_id = insert_session(conn, season, circuit, session_type, date_str)
        lap_id_map = insert_laps_batch(conn, session_id, valid_laps)

        for (driver, lap_num), tel_df in tel_by_lap.items():
            lap_db_id = lap_id_map.get((driver, lap_num))
            if lap_db_id is None or tel_df.empty:
                continue
            for row in tel_df.itertuples(index=False):
                tel_rows.append((
                    lap_db_id,
                    float(row.distance), float(row.speed),
                    float(row.throttle), float(row.brake), int(row.gear),
                ))

        insert_telemetry_batch(conn, tel_rows)

    n_tel = len(tel_rows)

    # Explicitly release FastF1 session + collected telemetry from RAM.
    # FastF1 keeps the raw high-frequency telemetry stream in memory after load();
    # deleting the reference + forcing GC drops peak usage by 500 MB-1.5 GB per session.
    del ff1_sess, tel_by_lap, tel_rows, valid_laps
    import gc; gc.collect()

    log.info(
        "      %s stored -- session_id=%d, %d laps, %d tel rows",
        session_type, session_id, n_valid, n_tel,
    )
    return n_valid, n_tel


def process_event(season: int, event) -> None:
    """Process all configured session types for one race weekend."""
    circuit = _circuit_name_from_event(str(event["EventName"]))
    log.info("    [%s  %s]", season, circuit)
    for stype in SESSION_TYPES:
        try:
            process_session(season, circuit, event, stype)
        except Exception as exc:
            log.error("      ERROR %s %s %s: %s", season, circuit, stype, exc, exc_info=True)


def ingest_one_session_by_names(
    season: int,
    circuit: str,
    session_type: str,
) -> bool:
    """
    If this session is not already in the DB, load it from FastF1 and insert laps.
    circuit must match stored names, e.g. 'British' (from _circuit_name_from_event).
    session_type: 'Q' or 'Race' (SESSION_TYPES only).

    Returns True if a row exists in sessions after the call, False on failure or
    if the event cannot be found in the schedule.
    """
    with managed_connection() as conn:
        if session_exists(conn, season, circuit, session_type) is not None:
            return True

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    try:
        schedule = fastf1.get_event_schedule(season, include_testing=False)
    except Exception as exc:
        log.warning("ingest: get_event_schedule failed season=%d: %s", season, exc)
        return False

    race_events = schedule[schedule["RoundNumber"] > 0]
    for _, event in race_events.iterrows():
        cname = _circuit_name_from_event(str(event["EventName"]))
        if cname != circuit:
            continue
        try:
            process_session(season, circuit, event, session_type)
        except Exception as exc:
            log.warning(
                "ingest: process_session failed %s %s %s: %s",
                season, circuit, session_type, exc,
            )
            return False
        with managed_connection() as conn:
            return session_exists(conn, season, circuit, session_type) is not None
    log.warning("ingest: no schedule row for circuit=%r season=%d", circuit, season)
    return False


# -- entry point ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PitWall Phase 1 -- extract F1 data into SQLite"
    )
    parser.add_argument(
        "--season", type=int, default=None, metavar="YEAR",
        help="Extract one year only (e.g. --season 2022). "
             "Omit to run all seasons defined in config.py.",
    )
    parser.add_argument(
        "--clear", action="store_true",
        help="Delete pitwall.db and FastF1 cache before running.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("fastf1").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)

    if args.clear:
        log.info("--clear: wiping existing data ...")
        clear_data()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    log.info("FastF1 cache -> %s", CACHE_DIR)
    init_db()

    seasons_to_run = [args.season] if args.season is not None else SEASONS
    if args.season is not None and args.season not in SEASONS:
        log.warning("--season %d not in configured SEASONS %s. Running anyway.",
                    args.season, SEASONS)

    log.info(
        "Seasons: %s | Sessions: %s | Telemetry for: %s | Step: %.0fm",
        seasons_to_run, SESSION_TYPES, TELEMETRY_SESSIONS, TELEMETRY_STEP_M,
    )

    total_laps = total_tel = 0
    for season in seasons_to_run:
        log.info("==================  Season %d  ==================", season)
        try:
            schedule = fastf1.get_event_schedule(season, include_testing=False)
        except Exception as exc:
            log.error("  Could not fetch schedule for %d: %s", season, exc)
            continue

        race_events = schedule[schedule["RoundNumber"] > 0]
        log.info("  %d race events", len(race_events))

        for _, event in race_events.iterrows():
            try:
                process_event(season, event)
            except Exception as exc:
                log.error("  Unhandled error for %s: %s",
                           event.get("EventName", "?"), exc, exc_info=True)

    log.info(
        "==================  Done -- %d laps, %d telemetry rows  ==================",
        total_laps, total_tel,
    )


if __name__ == "__main__":
    main()
