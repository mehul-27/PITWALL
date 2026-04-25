"""
Phase 2 -- stat computation from SQLite raw extraction tables.
Computes ground-truth metrics and writes into *_stats tables.
Safe to re-run: each target row is skipped if already present.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np

# -- project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from pipeline.db import init_db, managed_connection

log = logging.getLogger(__name__)

# Fuel assumptions for rough normalization.
FUEL_BURN_KG_PER_LAP = 1.7
FUEL_SCALE_KG = 10.0

TYRE_DEG_BOUNDS = (0.010, 0.250)
TYRE_CLIFF_BOUNDS = (8, 35)
TYRE_MAX_VIABLE_BOUNDS = (10, 45)
CORNER_MIN_SPEED_BOUNDS = (60.0, 320.0)
TOP_SPEED_BOUNDS = (280.0, 370.0)
SECTOR_DELTA_BOUNDS = (-2.0, 5.0)


# -- generic helpers -----------------------------------------------------------

def _row_exists(conn, table: str, where_clause: str, params: tuple[Any, ...]) -> bool:
    query = f"SELECT 1 FROM {table} WHERE {where_clause} LIMIT 1"
    return conn.execute(query, params).fetchone() is not None


def _linear_slope(x_vals: list[float], y_vals: list[float]) -> float | None:
    if len(x_vals) < 2:
        return None
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)
    if np.allclose(x, x[0]):
        return None
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _in_bounds(value: Any, low: float, high: float) -> bool:
    return _is_finite(value) and low <= float(value) <= high


def _log_rejected_stat(
    table: str,
    circuit: str,
    driver: str | None,
    season: int | None,
    reason: str,
) -> None:
    log.warning(
        "[%s] rejected row circuit=%s driver=%s season=%s reason=%s",
        table,
        circuit,
        driver or "N/A",
        season if season is not None else "N/A",
        reason,
    )


def _valid_tyre_stat_row(
    circuit: str,
    season: int,
    compound: str,
    deg_rate: float,
    cliff_lap: int | None,
    max_viable_laps: int | None,
) -> bool:
    if not _in_bounds(deg_rate, *TYRE_DEG_BOUNDS):
        _log_rejected_stat(
            "tyre_stats", circuit, None, season,
            f"{compound} deg_rate={deg_rate:.4f} outside {TYRE_DEG_BOUNDS}",
        )
        return False
    if cliff_lap is None or not _in_bounds(cliff_lap, *TYRE_CLIFF_BOUNDS):
        _log_rejected_stat(
            "tyre_stats", circuit, None, season,
            f"{compound} cliff_lap={cliff_lap} outside {TYRE_CLIFF_BOUNDS}",
        )
        return False
    if max_viable_laps is None or not _in_bounds(max_viable_laps, *TYRE_MAX_VIABLE_BOUNDS):
        _log_rejected_stat(
            "tyre_stats", circuit, None, season,
            f"{compound} max_viable_laps={max_viable_laps} outside {TYRE_MAX_VIABLE_BOUNDS}",
        )
        return False
    return True


def _valid_sector_stat_row(
    driver: str,
    circuit: str,
    season: int,
    sectors: tuple[float | None, float | None, float | None],
) -> bool:
    for idx, value in enumerate(sectors, start=1):
        if value is not None and not _in_bounds(value, *SECTOR_DELTA_BOUNDS):
            _log_rejected_stat(
                "driver_sector_stats", circuit, driver, season,
                f"sector{idx}_delta={value:.4f} outside {SECTOR_DELTA_BOUNDS}",
            )
            return False
    return True


def _valid_speed_trap_row(driver: str, circuit: str, season: int, top_speed: float) -> bool:
    if not _in_bounds(top_speed, *TOP_SPEED_BOUNDS):
        _log_rejected_stat(
            "speed_trap_stats", circuit, driver, season,
            f"avg_top_speed={top_speed:.2f} outside {TOP_SPEED_BOUNDS}",
        )
        return False
    return True


def _valid_corner_stat_row(driver: str, circuit: str, min_speed: float) -> bool:
    if not _in_bounds(min_speed, *CORNER_MIN_SPEED_BOUNDS):
        _log_rejected_stat(
            "corner_stats", circuit, driver, None,
            f"avg_min_speed={min_speed:.2f} outside {CORNER_MIN_SPEED_BOUNDS}",
        )
        return False
    return True


def _classify_corner(avg_speed: float) -> str:
    if avg_speed < 100.0:
        return "slow"
    if avg_speed <= 180.0:
        return "medium"
    return "high"


def _fetch_weather_join_mode(conn) -> str | None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='weather'"
    ).fetchone()
    if not row:
        return None

    cols = {
        r["name"]
        for r in conn.execute("PRAGMA table_info(weather)").fetchall()
    }
    if {"lap_id", "track_temp"}.issubset(cols):
        return "lap_id"
    if {"session_id", "lap_number", "track_temp"}.issubset(cols):
        return "session_lap"
    return None


# -- tyre stats ----------------------------------------------------------------

def _detect_cliff_lap(age_to_laptime: list[tuple[int, float]]) -> int | None:
    if len(age_to_laptime) < 4:
        return None

    ages = [x[0] for x in age_to_laptime]
    vals = [x[1] for x in age_to_laptime]
    deriv = []
    for i in range(1, len(ages)):
        d_age = ages[i] - ages[i - 1]
        if d_age <= 0:
            continue
        deriv.append((ages[i], (vals[i] - vals[i - 1]) / d_age))
    if len(deriv) < 2:
        return None

    baseline = float(np.median([d[1] for d in deriv]))
    trigger = max(0.15, baseline * 1.75)
    for lap_age, slope in deriv:
        if slope > trigger:
            return int(lap_age)
    return None


def _max_viable_laps(age_to_laptime: list[tuple[int, float]], avg_deg_per_lap: float) -> int:
    ages = [x[0] for x in age_to_laptime]
    if not ages:
        return 0
    if avg_deg_per_lap <= 0:
        return int(max(ages))

    min_age = int(min(ages))
    max_age = int(max(ages))
    for age in range(min_age, max_age + 1):
        loss = (age - min_age) * avg_deg_per_lap
        if loss > 2.0:
            return max(min_age, age - 1)
    return max_age


def compute_tyre_stats(conn) -> None:
    log.info("[tyre_stats] start")
    weather_mode = _fetch_weather_join_mode(conn)
    if weather_mode:
        log.info("[tyre_stats] weather table found (%s join)", weather_mode)
    else:
        log.info("[tyre_stats] no weather table/columns -> temp sensitivity NULL")

    rows = conn.execute(
        """
        SELECT
            s.season,
            s.circuit,
            l.compound,
            l.tyre_age,
            l.lap_time,
            l.id AS lap_id,
            l.session_id,
            l.lap_number
        FROM laps l
        JOIN sessions s ON s.id = l.session_id
        WHERE s.session_type = 'Race'
          AND l.is_valid = 1
          AND l.lap_time IS NOT NULL
          AND l.tyre_age IS NOT NULL
          AND l.compound IS NOT NULL
          AND l.compound != ''
        """
    ).fetchall()
    if not rows:
        log.info("[tyre_stats] no source rows")
        return

    groups: dict[tuple[int, str, str], list[Any]] = defaultdict(list)
    for r in rows:
        groups[(int(r["season"]), str(r["circuit"]), str(r["compound"]))].append(r)
    log.info("[tyre_stats] groups=%d", len(groups))

    # Bounds for sanity-clamping computed values
    _MIN_LAPS       = 15     # minimum valid laps to attempt a regression
    _MAX_CLIFF      = 40     # cliff lap cannot exceed full race distance
    _MIN_VIABLE     = 3      # shortest meaningful stint
    _MAX_VIABLE     = 35     # longest realistic stint
    _MAX_DEG_RATE   = 0.300  # s/lap — above this the data is still noisy
    # (0.0 < slope <= 0.003 is valid flat degradation — store as-is)

    inserted = skipped = rejected = noisy_slope = 0
    for (season, circuit, compound), grp in groups.items():
        if _row_exists(
            conn,
            "tyre_stats",
            "season=? AND circuit=? AND compound=?",
            (season, circuit, compound),
        ):
            skipped += 1
            continue

        # Sort by tyre_age ascending before any processing
        grp_sorted = sorted(grp, key=lambda g: float(g["tyre_age"]))
        raw_x = [float(g["tyre_age"]) for g in grp_sorted]
        raw_y = [float(g["lap_time"]) for g in grp_sorted]

        # Outlier removal: keep laps within 2 std devs of the mean lap time
        if len(raw_y) >= 4:
            mu  = float(np.mean(raw_y))
            sig = float(np.std(raw_y))
            if sig > 0:
                pairs = [
                    (xa, ya) for xa, ya in zip(raw_x, raw_y)
                    if abs(ya - mu) <= 2.0 * sig
                ]
            else:
                pairs = list(zip(raw_x, raw_y))
        else:
            pairs = list(zip(raw_x, raw_y))

        if not pairs:
            rejected += 1
            continue
        x, y = zip(*pairs)
        x, y = list(x), list(y)

        # Enforce minimum sample size
        unique_ages = len(set(x))
        if len(x) < _MIN_LAPS or unique_ages < 3:
            rejected += 1
            continue

        slope = _linear_slope(x, y)
        if slope is None:
            rejected += 1
            continue

        # Negative slope means lap times improve with tyre age — noisy data, skip
        if slope <= 0:
            rejected += 1
            continue

        # Slope above realistic F1 range means outlier removal was insufficient
        if slope > _MAX_DEG_RATE:
            noisy_slope += 1
            continue

        per_age: dict[int, list[float]] = defaultdict(list)
        for xa, ya in zip(x, y):
            per_age[int(xa)].append(ya)
        age_to_laptime = sorted((age, mean(vals)) for age, vals in per_age.items())

        cliff_lap = _detect_cliff_lap(age_to_laptime)
        # Cap cliff_lap at maximum race distance
        if cliff_lap is not None:
            cliff_lap = min(cliff_lap, _MAX_CLIFF)

        max_viable = _max_viable_laps(age_to_laptime, slope)
        if cliff_lap is not None:
            max_viable = min(max_viable, max(1, cliff_lap - 1))

        # Clamp max_viable to [MIN_VIABLE, MAX_VIABLE]; set None if insufficient
        if max_viable < _MIN_VIABLE or max_viable > _MAX_VIABLE:
            max_viable_final: int | None = None
        else:
            max_viable_final = int(max_viable)

        temp_sens = None
        if weather_mode == "lap_id":
            temp_rows = conn.execute(
                """
                SELECT w.track_temp, l.tyre_age, l.lap_time
                FROM weather w
                JOIN laps l ON l.id = w.lap_id
                JOIN sessions s ON s.id = l.session_id
                WHERE s.season = ?
                  AND s.circuit = ?
                  AND s.session_type = 'Race'
                  AND l.compound = ?
                  AND l.is_valid = 1
                  AND l.lap_time IS NOT NULL
                  AND l.tyre_age IS NOT NULL
                  AND w.track_temp IS NOT NULL
                """,
                (season, circuit, compound),
            ).fetchall()
            if len(temp_rows) >= 6:
                age_vals = [float(t["tyre_age"]) for t in temp_rows]
                lap_vals = [float(t["lap_time"]) for t in temp_rows]
                age_slope = _linear_slope(age_vals, lap_vals) or 0.0
                residuals = [
                    lap_vals[i] - (age_slope * age_vals[i]) for i in range(len(lap_vals))
                ]
                temp_slope = _linear_slope(
                    [float(t["track_temp"]) for t in temp_rows], residuals
                )
                temp_sens = temp_slope
        elif weather_mode == "session_lap":
            temp_rows = conn.execute(
                """
                SELECT w.track_temp, l.tyre_age, l.lap_time
                FROM weather w
                JOIN laps l
                  ON l.session_id = w.session_id
                 AND l.lap_number = w.lap_number
                JOIN sessions s ON s.id = l.session_id
                WHERE s.season = ?
                  AND s.circuit = ?
                  AND s.session_type = 'Race'
                  AND l.compound = ?
                  AND l.is_valid = 1
                  AND l.lap_time IS NOT NULL
                  AND l.tyre_age IS NOT NULL
                  AND w.track_temp IS NOT NULL
                """,
                (season, circuit, compound),
            ).fetchall()
            if len(temp_rows) >= 6:
                age_vals = [float(t["tyre_age"]) for t in temp_rows]
                lap_vals = [float(t["lap_time"]) for t in temp_rows]
                age_slope = _linear_slope(age_vals, lap_vals) or 0.0
                residuals = [
                    lap_vals[i] - (age_slope * age_vals[i]) for i in range(len(lap_vals))
                ]
                temp_slope = _linear_slope(
                    [float(t["track_temp"]) for t in temp_rows], residuals
                )
                temp_sens = temp_slope

        if not _valid_tyre_stat_row(
            circuit,
            season,
            compound,
            float(slope),
            cliff_lap,
            max_viable_final,
        ):
            rejected += 1
            continue

        conn.execute(
            """
            INSERT INTO tyre_stats
                (circuit, compound, season, avg_deg_per_lap, cliff_lap, max_viable_laps, track_temp_sensitivity)
            VALUES (?,?,?,?,?,?,?)
            """,
            (
                circuit,
                compound,
                season,
                float(slope),
                cliff_lap,
                max_viable_final,
                temp_sens,
            ),
        )
        inserted += 1

    log.info(
        "[tyre_stats] done inserted=%d skipped=%d rejected=%d noisy_slope=%d",
        inserted, skipped, rejected, noisy_slope,
    )


# -- driver sector stats -------------------------------------------------------

def compute_driver_sector_stats(conn) -> None:
    log.info("[driver_sector_stats] start")
    sessions = conn.execute(
        """
        SELECT id, season, circuit
        FROM sessions
        WHERE session_type='Q'
        """
    ).fetchall()
    if not sessions:
        log.info("[driver_sector_stats] no qualifying sessions")
        return

    # Per-sector deltas keyed by (driver, circuit, season).
    # Each value is a list of (s1_delta|None, s2_delta|None, s3_delta|None)
    # tuples — one entry per session the driver appeared in.
    # Convention: delta = driver_sector_time − session_reference_sector_time (seconds).
    # Reference = the session’s fastest *full* lap (min lap_time across valid laps).
    # Positive = slower than that reference; negative = faster in that sector
    # (possible even if the driver’s overall lap was slower).

    agg: dict[tuple[str, str, int], list[tuple]] = defaultdict(list)
    for sess in sessions:
        sess_id = int(sess["id"])
        sector_rows = conn.execute(
            """
            SELECT driver, lap_time, sector1, sector2, sector3
            FROM laps
            WHERE session_id=?
              AND is_valid=1
              AND lap_time IS NOT NULL
              AND sector1 IS NOT NULL
              AND sector2 IS NOT NULL
              AND sector3 IS NOT NULL
            """,
            (sess_id,),
        ).fetchall()
        if not sector_rows:
            continue

        # Baseline = fastest *lap* in this session (min lap_time; ties → fastest sectors).
        baseline_row = min(sector_rows, key=lambda r: float(r["lap_time"]))
        b1 = float(baseline_row["sector1"])
        b2 = float(baseline_row["sector2"])
        b3 = float(baseline_row["sector3"])

        season  = int(sess["season"])
        circuit = str(sess["circuit"])
        for r in sector_rows:
            d1 = float(r["sector1"]) - b1
            d2 = float(r["sector2"]) - b2
            d3 = float(r["sector3"]) - b3
            lo, hi = SECTOR_DELTA_BOUNDS
            s1 = d1 if lo <= d1 <= hi else None
            s2 = d2 if lo <= d2 <= hi else None
            s3 = d3 if lo <= d3 <= hi else None
            key = (str(r["driver"]), circuit, season)
            agg[key].append((s1, s2, s3))

    inserted = skipped = rejected = 0
    for (driver, circuit, season), vals in agg.items():
        if _row_exists(
            conn,
            "driver_sector_stats",
            "driver=? AND circuit=? AND season=?",
            (driver, circuit, season),
        ):
            skipped += 1
            continue

        # Average only the non-None entries per sector.
        def _avg(idx: int) -> float | None:
            good = [v[idx] for v in vals if v[idx] is not None]
            return mean(good) if good else None

        s1 = _avg(0)
        s2 = _avg(1)
        s3 = _avg(2)

        # Skip if ALL three sectors are suspect.
        if s1 is None and s2 is None and s3 is None:
            _log_rejected_stat(
                "driver_sector_stats", circuit, driver, season,
                "all sector deltas unavailable after bounds filtering",
            )
            rejected += 1
            continue

        if not _valid_sector_stat_row(driver, circuit, season, (s1, s2, s3)):
            rejected += 1
            continue

        conn.execute(
            """
            INSERT INTO driver_sector_stats
                (driver, circuit, season, avg_sector1_delta, avg_sector2_delta, avg_sector3_delta)
            VALUES (?,?,?,?,?,?)
            """,
            (driver, circuit, season, s1, s2, s3),
        )
        inserted += 1

    log.info(
        "[driver_sector_stats] done inserted=%d skipped=%d rejected=%d",
        inserted, skipped, rejected,
    )


# -- speed trap stats ----------------------------------------------------------

def compute_speed_trap_stats(conn) -> None:
    log.info("[speed_trap_stats] start")
    rows = conn.execute(
        """
        WITH lap_top AS (
            SELECT
                l.id AS lap_id,
                l.driver,
                s.circuit,
                s.season,
                s.session_type,
                MAX(t.speed) AS top_speed
            FROM telemetry t
            JOIN laps l ON l.id = t.lap_id
            JOIN sessions s ON s.id = l.session_id
            WHERE l.is_valid = 1
              AND s.session_type = 'Race'
              AND t.speed IS NOT NULL
              AND t.speed BETWEEN 250 AND 380
            GROUP BY l.id, l.driver, s.circuit, s.season, s.session_type
        )
        SELECT
            driver,
            circuit,
            season,
            AVG(top_speed) AS avg_top_speed
        FROM lap_top
        GROUP BY driver, circuit, season
        """
    ).fetchall()
    if not rows:
        log.info("[speed_trap_stats] no telemetry-derived rows")
        return

    # Rank within (circuit, season) only — not across seasons.
    by_field: dict[tuple[str, int], list[Any]] = defaultdict(list)
    for r in rows:
        by_field[(str(r["circuit"]), int(r["season"]))].append(r)

    inserted = skipped = rejected = 0
    for (circuit, season), field_rows in by_field.items():
        ranked = sorted(field_rows, key=lambda x: float(x["avg_top_speed"]), reverse=True)
        for rank, r in enumerate(ranked, start=1):
            driver    = str(r["driver"])
            avg_speed = float(r["avg_top_speed"])
            if _row_exists(
                conn,
                "speed_trap_stats",
                "driver=? AND circuit=? AND season=?",
                (driver, circuit, season),
            ):
                skipped += 1
                continue
            if not _valid_speed_trap_row(driver, circuit, season, avg_speed):
                rejected += 1
                continue
            conn.execute(
                """
                INSERT INTO speed_trap_stats
                    (driver, circuit, season, avg_top_speed, rank_in_field)
                VALUES (?,?,?,?,?)
                """,
                (driver, circuit, season, avg_speed, rank),
            )
            inserted += 1

    log.info(
        "[speed_trap_stats] done inserted=%d skipped=%d rejected=%d",
        inserted, skipped, rejected,
    )


# -- corner stats --------------------------------------------------------------

def _build_corner_map(conn, circuit: str) -> list[tuple[int, float]]:
    rows = conn.execute(
        """
        SELECT corner_number, distance_from_start
        FROM corners
        WHERE circuit=?
          AND distance_from_start IS NOT NULL
        ORDER BY corner_number
        """,
        (circuit,),
    ).fetchall()
    return [(int(r["corner_number"]), float(r["distance_from_start"])) for r in rows]


def _corner_tolerances(corner_map: list[tuple[int, float]]) -> dict[int, float]:
    """Per-corner match radius = half distance to nearest neighbour, clamped [60, 250]m."""
    if not corner_map:
        return {}
    by_distance = sorted(corner_map, key=lambda c: c[1])
    tolerances: dict[int, float] = {}
    for i, (cn, d) in enumerate(by_distance):
        left_gap = d - by_distance[i - 1][1] if i > 0 else float("inf")
        right_gap = by_distance[i + 1][1] - d if i < len(by_distance) - 1 else float("inf")
        half_gap = min(left_gap, right_gap) / 2.0
        tolerances[cn] = max(60.0, min(250.0, half_gap))
    return tolerances


def _extract_local_minima(dist: np.ndarray, speed: np.ndarray) -> list[tuple[float, float]]:
    if len(speed) < 3:
        return []
    p60 = float(np.percentile(speed, 60))
    mins = []
    for i in range(1, len(speed) - 1):
        if speed[i] <= speed[i - 1] and speed[i] < speed[i + 1] and speed[i] <= p60:
            mins.append((float(dist[i]), float(speed[i])))
    return mins


# -- corner map population (FastF1) --------------------------------------------

def _cleanup_empty_corner_rows(conn) -> None:
    """Delete corners rows with NULL/NaN distances (previous failed populate runs)."""
    rows = conn.execute(
        """
        SELECT circuit, COUNT(*) AS total,
               SUM(CASE WHEN distance_from_start IS NOT NULL THEN 1 ELSE 0 END) AS valid
        FROM corners
        GROUP BY circuit
        """
    ).fetchall()
    for r in rows:
        total = int(r["total"])
        valid = int(r["valid"] or 0)
        if total > 0 and valid == 0:
            conn.execute("DELETE FROM corners WHERE circuit=?", (str(r["circuit"]),))
            log.info(
                "[populate_corners] cleaned %s: %d rows with NULL distance",
                r["circuit"], total,
            )


def _load_session_for_circuit_info(event, preferred: str = "Q"):
    """
    Load a session with telemetry enabled so get_circuit_info().corners.Distance
    is populated. Try Q first (much smaller than Race), fall back to Race.
    Returns a loaded FastF1 session or raises last exception.
    """
    order = (preferred, "Race") if preferred == "Q" else ("Race", "Q")
    last_exc: Exception | None = None
    for stype in order:
        try:
            sess = event.get_session(stype)
            sess.load(laps=True, telemetry=True, weather=False, messages=False)
            return sess
        except Exception as exc:
            last_exc = exc
            continue
    raise last_exc if last_exc else RuntimeError("no session could be loaded")


def populate_corners(conn) -> None:
    """
    Populate `corners` table from FastF1 circuit_info for any circuit missing a
    usable map (no rows, or all distance_from_start NULL from a prior bad run).
    Requires telemetry-loaded session because FastF1 derives corner Distance
    from position data; without telemetry the Distance column is all NaN.
    Uses Q session as the source (lighter than Race), falls back to Race.
    """
    log.info("[populate_corners] start")

    _cleanup_empty_corner_rows(conn)

    import fastf1  # lazy: only pay import cost when populating is needed
    from config import CACHE_DIR
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    circuits = [
        str(r["circuit"])
        for r in conn.execute("SELECT DISTINCT circuit FROM sessions").fetchall()
    ]

    inserted = skipped = failed = 0
    for circuit in circuits:
        usable = conn.execute(
            """
            SELECT COUNT(*) AS c
            FROM corners
            WHERE circuit=?
              AND distance_from_start IS NOT NULL
            """,
            (circuit,),
        ).fetchone()["c"]
        if usable > 0:
            skipped += 1
            continue

        sess_row = conn.execute(
            """
            SELECT season, session_type
            FROM sessions
            WHERE circuit=? AND session_type IN ('Q','Race')
            ORDER BY season DESC,
                     CASE session_type WHEN 'Q' THEN 0 ELSE 1 END
            LIMIT 1
            """,
            (circuit,),
        ).fetchone()
        if sess_row is None:
            log.warning("[populate_corners] %s: no Q/Race session recorded -> skip", circuit)
            failed += 1
            continue

        season = int(sess_row["season"])
        preferred = str(sess_row["session_type"])
        try:
            event = fastf1.get_event(season, circuit)
            ff1_sess = _load_session_for_circuit_info(event, preferred=preferred)
            ci = ff1_sess.get_circuit_info()
        except Exception as exc:
            log.warning("[populate_corners] %s %d FAILED: %s", circuit, season, exc)
            failed += 1
            continue

        if ci is None or ci.corners is None or ci.corners.empty:
            log.warning("[populate_corners] %s %d: circuit_info empty", circuit, season)
            failed += 1
            continue

        corners_df = ci.corners
        if "Number" not in corners_df.columns or "Distance" not in corners_df.columns:
            log.warning(
                "[populate_corners] %s %d: unexpected columns %s",
                circuit, season, list(corners_df.columns),
            )
            failed += 1
            continue

        rows: list[tuple[str, int, Any, float]] = []
        for _, row in corners_df.iterrows():
            try:
                number = int(row["Number"])
                distance = float(row["Distance"])
            except (ValueError, TypeError):
                continue
            if not math.isfinite(distance):
                continue
            rows.append((circuit, number, None, distance))

        if not rows:
            log.warning(
                "[populate_corners] %s %d: no usable corner rows (Distance all NaN?)",
                circuit, season,
            )
            failed += 1
            continue

        # Wipe any prior partial rows (incl. NaN-distance rows if cleanup missed them).
        conn.execute("DELETE FROM corners WHERE circuit=?", (circuit,))
        conn.executemany(
            """
            INSERT INTO corners (circuit, corner_number, corner_type, distance_from_start)
            VALUES (?,?,?,?)
            """,
            rows,
        )
        inserted += 1
        log.info(
            "[populate_corners] %s: inserted %d corners (src %s %d)",
            circuit, len(rows), preferred, season,
        )

    log.info(
        "[populate_corners] done circuits_populated=%d already_had=%d failed=%d",
        inserted, skipped, failed,
    )


def _purge_stale_corner_stats(conn) -> None:
    """
    Delete corner_stats rows whose corner_number exceeds the max official
    corner number now present in the `corners` table for that circuit.
    Fixes old rows inserted via the (removed) distance-bin fallback.
    """
    log.info("[corner_stats] checking for stale rows from fallback numbering")
    rows = conn.execute(
        """
        SELECT cs.circuit AS circuit,
               MAX(cs.corner_number) AS stats_max,
               COALESCE((SELECT MAX(c.corner_number)
                         FROM corners c
                         WHERE c.circuit = cs.circuit), 0) AS map_max
        FROM corner_stats cs
        GROUP BY cs.circuit
        """
    ).fetchall()

    purged_circuits = 0
    purged_rows = 0
    for r in rows:
        map_max = int(r["map_max"])
        stats_max = int(r["stats_max"])
        circuit = str(r["circuit"])
        if map_max > 0 and stats_max > map_max:
            cur = conn.execute(
                "DELETE FROM corner_stats WHERE circuit=?", (circuit,)
            )
            purged_circuits += 1
            purged_rows += cur.rowcount
            log.info(
                "[corner_stats] purged %s: stats_max=%d > map_max=%d (rows=%d)",
                circuit, stats_max, map_max, cur.rowcount,
            )
    log.info(
        "[corner_stats] purge done circuits=%d rows=%d",
        purged_circuits, purged_rows,
    )


def compute_corner_stats(conn) -> None:
    log.info("[corner_stats] start")
    circuits = conn.execute("SELECT DISTINCT circuit FROM sessions").fetchall()
    inserted = skipped = rejected = circuits_without_map = 0

    for c_row in circuits:
        circuit = str(c_row["circuit"])

        corner_map = _build_corner_map(conn, circuit)
        if not corner_map:
            log.warning(
                "[corner_stats] %s: no corner map in `corners` table -> SKIP "
                "(run populate_corners to fix)", circuit,
            )
            circuits_without_map += 1
            continue

        max_corner_number = max(cn for cn, _ in corner_map)
        tolerances = _corner_tolerances(corner_map)
        log.info(
            "[corner_stats] circuit=%s corners=%d max_turn=%d",
            circuit, len(corner_map), max_corner_number,
        )

        lap_rows = conn.execute(
            """
            SELECT
                l.id AS lap_id,
                l.driver
            FROM laps l
            JOIN sessions s ON s.id = l.session_id
            WHERE s.circuit=?
              AND l.is_valid=1
            """,
            (circuit,),
        ).fetchall()
        if not lap_rows:
            continue

        driver_corner_values: dict[tuple[str, int], list[float]] = defaultdict(list)

        for lap in lap_rows:
            tel = conn.execute(
                """
                SELECT distance, speed
                FROM telemetry
                WHERE lap_id=?
                  AND distance IS NOT NULL
                  AND speed IS NOT NULL
                ORDER BY distance
                """,
                (int(lap["lap_id"]),),
            ).fetchall()
            if len(tel) < 3:
                continue

            dist = np.asarray([float(t["distance"]) for t in tel], dtype=float)
            speed = np.asarray([float(t["speed"]) for t in tel], dtype=float)
            minima = _extract_local_minima(dist, speed)
            if not minima:
                continue

            lap_corner_min: dict[int, float] = {}
            for d, s in minima:
                nearest_cn, nearest_d = min(corner_map, key=lambda c: abs(c[1] - d))
                if abs(nearest_d - d) > tolerances[nearest_cn]:
                    continue
                prev = lap_corner_min.get(int(nearest_cn))
                lap_corner_min[int(nearest_cn)] = s if prev is None else min(prev, s)

            for corner_number, min_speed in lap_corner_min.items():
                if corner_number < 1 or corner_number > max_corner_number:
                    continue
                driver_corner_values[(str(lap["driver"]), int(corner_number))].append(min_speed)

        field_by_corner: dict[int, list[float]] = defaultdict(list)
        for (_drv, corner_number), vals in driver_corner_values.items():
            field_by_corner[corner_number].append(mean(vals))

        for (driver, corner_number), vals in driver_corner_values.items():
            if _row_exists(
                conn,
                "corner_stats",
                "driver=? AND circuit=? AND corner_number=?",
                (driver, circuit, corner_number),
            ):
                skipped += 1
                continue

            avg_min_speed = mean(vals)
            if not _valid_corner_stat_row(driver, circuit, avg_min_speed):
                rejected += 1
                continue

            field_avg = mean(field_by_corner[corner_number])
            # Speed margin (km/h), not lap-time delta: + = higher min speed vs field mean.
            delta_vs_field = avg_min_speed - field_avg
            conn.execute(
                """
                INSERT INTO corner_stats
                    (driver, circuit, corner_number, avg_min_speed, delta_vs_field)
                VALUES (?,?,?,?,?)
                """,
                (driver, circuit, corner_number, avg_min_speed, delta_vs_field),
            )
            inserted += 1

            ctype = _classify_corner(avg_min_speed)
            conn.execute(
                """
                UPDATE corners
                SET corner_type=?
                WHERE circuit=?
                  AND corner_number=?
                """,
                (ctype, circuit, corner_number),
            )

    log.info(
        "[corner_stats] done inserted=%d skipped=%d rejected=%d circuits_without_map=%d",
        inserted, skipped, rejected, circuits_without_map,
    )


# -- fuel stats ----------------------------------------------------------------

def compute_fuel_stats(conn) -> None:
    """
    Estimate fuel effect (s per 10kg) per driver / circuit / season.

    Method: compare early-stint laps (2-6) against late-race laps on the
    SAME compound and similar tyre age, so lap-time delta is dominated by
    fuel mass. Track evolution is an unavoidable confound and is partly
    mitigated by clipping + median aggregation.

    Filters (avoid noise and DNF contamination):
      - Driver must complete at least MIN_DRIVER_LAPS race laps
      - Driver must finish within FINISH_GAP laps of session winner
      - Exclude driver's final lap (flag/slow-down effects)
      - Same compound required
      - Tyre age must match within TYRE_AGE_TOL
      - Pair lap gap must exceed MIN_FUEL_GAP_LAPS (meaningful fuel delta)
      - Clip absolute per-pair effect to EFFECT_CLIP s/10kg (sanity bound)
      - Require at least MIN_PAIRS valid pairs; aggregate via median
    """
    log.info("[fuel_stats] start")

    MIN_DRIVER_LAPS    = 30
    FINISH_GAP         = 3
    TYRE_AGE_TOL       = 2
    MIN_FUEL_GAP_LAPS  = 20
    EFFECT_CLIP        = 3.0
    MIN_PAIRS          = 3

    race_sessions = conn.execute(
        """
        SELECT id, season, circuit
        FROM sessions
        WHERE session_type='Race'
        """
    ).fetchall()
    if not race_sessions:
        log.info("[fuel_stats] no race sessions")
        return

    agg: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    dnf_skipped = short_skipped = thin_pairs = 0

    for sess in race_sessions:
        sess_id = int(sess["id"])
        season = int(sess["season"])
        circuit = str(sess["circuit"])

        winner_max_row = conn.execute(
            "SELECT MAX(lap_number) AS m FROM laps WHERE session_id=? AND is_valid=1",
            (sess_id,),
        ).fetchone()
        session_max_lap = int(winner_max_row["m"]) if winner_max_row and winner_max_row["m"] is not None else 0
        if session_max_lap < MIN_DRIVER_LAPS:
            continue

        drivers = conn.execute(
            """
            SELECT DISTINCT driver
            FROM laps
            WHERE session_id=?
              AND is_valid=1
            """,
            (sess_id,),
        ).fetchall()

        for drow in drivers:
            driver = str(drow["driver"])
            laps = conn.execute(
                """
                SELECT lap_number, lap_time, compound, tyre_age
                FROM laps
                WHERE session_id=?
                  AND driver=?
                  AND is_valid=1
                  AND lap_time IS NOT NULL
                  AND compound IS NOT NULL
                  AND compound != ''
                  AND tyre_age IS NOT NULL
                ORDER BY lap_number
                """,
                (sess_id, driver),
            ).fetchall()
            if len(laps) < MIN_DRIVER_LAPS:
                short_skipped += 1
                continue

            driver_max_lap = max(int(l["lap_number"]) for l in laps)
            if driver_max_lap < session_max_lap - FINISH_GAP:
                dnf_skipped += 1
                continue

            usable = [l for l in laps if int(l["lap_number"]) < driver_max_lap]
            if not usable:
                continue

            early = [l for l in usable if 2 <= int(l["lap_number"]) <= 6]
            late_threshold = driver_max_lap - 5
            late = [l for l in usable if int(l["lap_number"]) >= late_threshold]
            if not early or not late:
                continue

            pair_effects: list[float] = []
            for e in early:
                e_lap = int(e["lap_number"])
                e_time = float(e["lap_time"])
                e_comp = str(e["compound"])
                e_age = int(e["tyre_age"])
                for l in late:
                    if str(l["compound"]) != e_comp:
                        continue
                    if abs(int(l["tyre_age"]) - e_age) > TYRE_AGE_TOL:
                        continue
                    gap_laps = int(l["lap_number"]) - e_lap
                    if gap_laps < MIN_FUEL_GAP_LAPS:
                        continue
                    fuel_delta_kg = gap_laps * FUEL_BURN_KG_PER_LAP
                    if fuel_delta_kg <= 0:
                        continue
                    raw_delta = e_time - float(l["lap_time"])
                    effect_per_10kg = raw_delta * FUEL_SCALE_KG / fuel_delta_kg
                    if not math.isfinite(effect_per_10kg):
                        continue
                    if abs(effect_per_10kg) > EFFECT_CLIP:
                        continue
                    pair_effects.append(effect_per_10kg)

            if len(pair_effects) >= MIN_PAIRS:
                agg[(driver, circuit, season)].extend(pair_effects)
            elif pair_effects:
                thin_pairs += 1

    inserted = skipped = 0
    for (driver, circuit, season), vals in agg.items():
        if len(vals) < MIN_PAIRS:
            thin_pairs += 1
            continue
        if _row_exists(
            conn,
            "fuel_stats",
            "driver=? AND circuit=? AND season=?",
            (driver, circuit, season),
        ):
            skipped += 1
            continue
        median_effect = float(np.median(vals))
        conn.execute(
            """
            INSERT INTO fuel_stats
                (driver, circuit, season, fuel_effect_per_10kg)
            VALUES (?,?,?,?)
            """,
            (driver, circuit, season, median_effect),
        )
        inserted += 1

    log.info(
        "[fuel_stats] done inserted=%d skipped=%d dnf=%d short=%d thin_pairs=%d",
        inserted, skipped, dnf_skipped, short_skipped, thin_pairs,
    )


# -- quali vs race delta -------------------------------------------------------

def compute_quali_race_delta(conn) -> None:
    log.info("[quali_race_delta] start")

    # Best qualifying lap per driver/circuit/season.
    # MIN(lap_time) across all Q laps reflects the fastest time the driver
    # set regardless of which Q segment knocked them out — functionally
    # equivalent to their best Q3/Q2/Q1 time.
    q_rows = conn.execute(
        """
        SELECT
            s.season,
            s.circuit,
            l.driver,
            MIN(l.lap_time) AS best_quali
        FROM laps l
        JOIN sessions s ON s.id = l.session_id
        WHERE s.session_type='Q'
          AND l.is_valid=1
          AND l.lap_time IS NOT NULL
        GROUP BY s.season, s.circuit, l.driver
        """
    ).fetchall()
    if not q_rows:
        log.info("[quali_race_delta] no qualifying data")
        return

    # Fetch race laps 5 to (total_laps - 3) per driver/session.
    # Compute MEDIAN in Python (SQLite has no native median).
    # Exclude is_valid=0 to skip SC/VSC affected laps already flagged
    # during extraction.
    race_lap_rows = conn.execute(
        """
        WITH race_bounds AS (
            SELECT
                s.id  AS session_id,
                s.season,
                s.circuit,
                l.driver,
                l.lap_number,
                l.lap_time,
                MAX(l.lap_number) OVER (PARTITION BY s.id, l.driver) AS total_laps
            FROM laps l
            JOIN sessions s ON s.id = l.session_id
            WHERE s.session_type = 'Race'
              AND l.is_valid = 1
              AND l.lap_time IS NOT NULL
        )
        SELECT season, circuit, driver, lap_time
        FROM race_bounds
        WHERE lap_number >= 5
          AND lap_number <= (total_laps - 3)
        """
    ).fetchall()

    # Build map: (season, circuit, driver) -> list of valid lap times
    race_laps_map: dict[tuple[int, str, str], list[float]] = defaultdict(list)
    for r in race_lap_rows:
        key = (int(r["season"]), str(r["circuit"]), str(r["driver"]))
        race_laps_map[key].append(float(r["lap_time"]))

    _MIN_DELTA = 2.0    # below this suggests sprint race or data mismatch
    _MAX_DELTA = 15.0   # above this suggests DNF laps or corrupt data

    inserted = skipped = missing = suspect = 0
    for q in q_rows:
        key        = (int(q["season"]), str(q["circuit"]), str(q["driver"]))
        best_quali = float(q["best_quali"])
        season, circuit, driver = key

        lap_times = race_laps_map.get(key)
        if not lap_times:
            missing += 1
            continue

        # Median race pace across laps 5..(total_laps - 3)
        avg_race = float(np.median(lap_times))
        # Reference: best one-lap quali. Positive = race median slower than that quali time.
        delta    = avg_race - best_quali

        # Store None for delta if outside realistic range — data is present
        # but the computed value is unreliable.
        if not (_MIN_DELTA <= delta <= _MAX_DELTA):
            delta_stored: float | None = None
            suspect += 1
        else:
            delta_stored = delta

        if _row_exists(
            conn,
            "quali_race_delta",
            "driver=? AND circuit=? AND season=?",
            (driver, circuit, season),
        ):
            skipped += 1
            continue

        conn.execute(
            """
            INSERT INTO quali_race_delta
                (driver, circuit, season, quali_lap, avg_race_pace, delta)
            VALUES (?,?,?,?,?,?)
            """,
            (driver, circuit, season, best_quali, avg_race, delta_stored),
        )
        inserted += 1

    log.info(
        "[quali_race_delta] done inserted=%d skipped=%d "
        "missing_race=%d suspect_delta=%d",
        inserted, skipped, missing, suspect,
    )


# -- entry point ---------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="PitWall Phase 2 -- compute stats from SQLite"
    )
    args = parser.parse_args()
    _ = args

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    log.info("Initialising DB schema (safe no-op if already ready)")
    init_db()

    with managed_connection() as conn:
        compute_tyre_stats(conn)
        compute_driver_sector_stats(conn)
        compute_speed_trap_stats(conn)
        populate_corners(conn)
        _purge_stale_corner_stats(conn)
        compute_corner_stats(conn)
        compute_fuel_stats(conn)
        compute_quali_race_delta(conn)

    log.info("Stat computation complete")


if __name__ == "__main__":
    main()
