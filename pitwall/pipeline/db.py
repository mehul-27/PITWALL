"""
Database bootstrap — single source of truth for SQLite setup.
Creates all tables and indexes on first run; safe to call repeatedly.
All other modules must obtain connections via get_connection().
"""

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DB_PATH

log = logging.getLogger(__name__)

# ── DDL ────────────────────────────────────────────────────────────────────────

_CREATE_TABLES = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Raw extraction tables --

CREATE TABLE IF NOT EXISTS sessions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    season       INTEGER NOT NULL,
    circuit      TEXT    NOT NULL,
    session_type TEXT    NOT NULL,   -- 'Q', 'FP2', 'Race'
    date         TEXT
);

CREATE TABLE IF NOT EXISTS laps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  INTEGER NOT NULL,
    driver      TEXT    NOT NULL,
    lap_number  INTEGER NOT NULL,
    lap_time    REAL,
    sector1     REAL,
    sector2     REAL,
    sector3     REAL,
    compound    TEXT,
    tyre_age    INTEGER,
    is_valid    INTEGER NOT NULL DEFAULT 1,
    FOREIGN KEY (session_id) REFERENCES sessions (id)
);

CREATE TABLE IF NOT EXISTS telemetry (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    lap_id   INTEGER NOT NULL,
    distance REAL,
    speed    REAL,
    throttle REAL,
    brake    REAL,
    gear     INTEGER,
    FOREIGN KEY (lap_id) REFERENCES laps (id)
);

CREATE TABLE IF NOT EXISTS corners (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit             TEXT NOT NULL,
    corner_number       INTEGER NOT NULL,
    corner_type         TEXT,   -- 'slow', 'medium', 'high'
    distance_from_start REAL
);

-- Computed stat tables (ground truth for fine-tuning) --

CREATE TABLE IF NOT EXISTS tyre_stats (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    circuit               TEXT    NOT NULL,
    compound              TEXT    NOT NULL,
    season                INTEGER NOT NULL,
    avg_deg_per_lap       REAL,
    cliff_lap             INTEGER,
    max_viable_laps       INTEGER,
    track_temp_sensitivity REAL
);

CREATE TABLE IF NOT EXISTS driver_sector_stats (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    driver           TEXT    NOT NULL,
    circuit          TEXT    NOT NULL,
    season           INTEGER NOT NULL,
    avg_sector1_delta REAL,
    avg_sector2_delta REAL,
    avg_sector3_delta REAL
);

CREATE TABLE IF NOT EXISTS speed_trap_stats (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    driver        TEXT    NOT NULL,
    circuit       TEXT    NOT NULL,
    season        INTEGER NOT NULL,
    avg_top_speed REAL,
    rank_in_field INTEGER
);

CREATE TABLE IF NOT EXISTS corner_stats (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    driver        TEXT    NOT NULL,
    circuit       TEXT    NOT NULL,
    corner_number INTEGER NOT NULL,
    avg_min_speed REAL,
    delta_vs_field REAL
);

CREATE TABLE IF NOT EXISTS fuel_stats (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    driver                TEXT    NOT NULL,
    circuit               TEXT    NOT NULL,
    season                INTEGER NOT NULL,
    fuel_effect_per_10kg  REAL
);

CREATE TABLE IF NOT EXISTS quali_race_delta (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    driver         TEXT    NOT NULL,
    circuit        TEXT    NOT NULL,
    season         INTEGER NOT NULL,
    quali_lap      REAL,
    avg_race_pace  REAL,
    delta          REAL
);
"""

# Indexes on every column that appears in WHERE clauses.
# Created separately so each IF NOT EXISTS is independent.
_CREATE_INDEXES = [
    # sessions
    "CREATE INDEX IF NOT EXISTS idx_sessions_circuit ON sessions (circuit)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_season  ON sessions (season)",
    # laps
    "CREATE INDEX IF NOT EXISTS idx_laps_session  ON laps (session_id)",
    "CREATE INDEX IF NOT EXISTS idx_laps_driver   ON laps (driver)",
    "CREATE INDEX IF NOT EXISTS idx_laps_compound ON laps (compound)",
    "CREATE INDEX IF NOT EXISTS idx_laps_valid    ON laps (is_valid)",
    # telemetry
    "CREATE INDEX IF NOT EXISTS idx_telemetry_lap ON telemetry (lap_id)",
    # corners
    "CREATE INDEX IF NOT EXISTS idx_corners_circuit ON corners (circuit)",
    # tyre_stats
    "CREATE INDEX IF NOT EXISTS idx_tyre_circuit  ON tyre_stats (circuit)",
    "CREATE INDEX IF NOT EXISTS idx_tyre_compound ON tyre_stats (compound)",
    "CREATE INDEX IF NOT EXISTS idx_tyre_season   ON tyre_stats (season)",
    # driver_sector_stats
    "CREATE INDEX IF NOT EXISTS idx_dss_driver  ON driver_sector_stats (driver)",
    "CREATE INDEX IF NOT EXISTS idx_dss_circuit ON driver_sector_stats (circuit)",
    "CREATE INDEX IF NOT EXISTS idx_dss_season  ON driver_sector_stats (season)",
    # speed_trap_stats
    "CREATE INDEX IF NOT EXISTS idx_sts_driver  ON speed_trap_stats (driver)",
    "CREATE INDEX IF NOT EXISTS idx_sts_circuit ON speed_trap_stats (circuit)",
    "CREATE INDEX IF NOT EXISTS idx_sts_season  ON speed_trap_stats (season)",
    # corner_stats
    "CREATE INDEX IF NOT EXISTS idx_cs_driver  ON corner_stats (driver)",
    "CREATE INDEX IF NOT EXISTS idx_cs_circuit ON corner_stats (circuit)",
    # fuel_stats
    "CREATE INDEX IF NOT EXISTS idx_fs_driver  ON fuel_stats (driver)",
    "CREATE INDEX IF NOT EXISTS idx_fs_circuit ON fuel_stats (circuit)",
    "CREATE INDEX IF NOT EXISTS idx_fs_season  ON fuel_stats (season)",
    # quali_race_delta
    "CREATE INDEX IF NOT EXISTS idx_qrd_driver  ON quali_race_delta (driver)",
    "CREATE INDEX IF NOT EXISTS idx_qrd_circuit ON quali_race_delta (circuit)",
    "CREATE INDEX IF NOT EXISTS idx_qrd_season  ON quali_race_delta (season)",
]

# ── Connection ─────────────────────────────────────────────────────────────────

def get_connection() -> sqlite3.Connection:
    """
    Return a new SQLite connection to pitwall.db.
    Caller is responsible for closing it (or use as a context manager).
    Foreign keys and WAL mode are enabled on every connection.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row          # column-name access on rows
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def managed_connection():
    """Context manager that commits on success, rolls back on error, and closes."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Initialisation ─────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create all tables and indexes if they don't already exist.
    Safe to call on every startup — uses IF NOT EXISTS throughout.
    """
    log.info("Initialising database at %s", DB_PATH)
    with managed_connection() as conn:
        conn.executescript(_CREATE_TABLES)
        for stmt in _CREATE_INDEXES:
            conn.execute(stmt)
    log.info("Database ready — all tables and indexes exist")


# ── Smoke test ─────────────────────────────────────────────────────────────────

def _smoke_test() -> None:
    """Quick sanity check: init DB, verify all expected tables exist."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    init_db()

    expected = {
        "sessions", "laps", "telemetry", "corners",
        "tyre_stats", "driver_sector_stats", "speed_trap_stats",
        "corner_stats", "fuel_stats", "quali_race_delta",
    }

    with managed_connection() as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        found = {r["name"] for r in rows}

    missing = expected - found
    if missing:
        raise RuntimeError(f"Missing tables after init: {missing}")

    log.info("Smoke test passed — tables: %s", sorted(found))


if __name__ == "__main__":
    _smoke_test()
