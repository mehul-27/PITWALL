"""
Phase 3 -- Template-based dataset generation.
Fills 30 question/answer skeleton templates with computed stats
from SQLite. No Gemini required. Target: exactly 800 JSONL examples.

Each template function:
  - Queries one or more computed-stats tables
  - Returns a list of (question, answer) string pairs
  - Skips rows where data is missing or sparse

Output: data/dataset/templates.jsonl  (one JSON object per line)

Sampling strategy:
  Each template is capped at MAX_PER_TEMPLATE examples (default 40).
  After all templates run, the final list is shuffled and truncated to
  TARGET_EXAMPLES (default 800) so the file never exceeds the target.
  Use --limit N and --per-template N to override from the command line.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable

# -- project imports -----------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_DIR, SYSTEM_PROMPT
from pipeline.db import managed_connection

log = logging.getLogger(__name__)

# Full names for driver codes used in natural-language answers.
_DRIVER_NAMES: dict[str, str] = {
    "ALB": "Alexander Albon",    "ALO": "Fernando Alonso",
    "ANT": "Kimi Antonelli",     "BEA": "Oliver Bearman",
    "BOT": "Valtteri Bottas",    "COL": "Franco Colapinto",
    "DEV": "Nyck de Vries",      "DOO": "Jack Doohan",
    "GAS": "Pierre Gasly",       "GIO": "Antonio Giovinazzi",
    "GRO": "Romain Grosjean",    "HAD": "Isack Hadjar",
    "HAM": "Lewis Hamilton",     "HUL": "Nico Hulkenberg",
    "KVY": "Daniil Kvyat",       "LAT": "Nicholas Latifi",
    "LAW": "Liam Lawson",        "LEC": "Charles Leclerc",
    "MAG": "Kevin Magnussen",    "MAZ": "Nikita Mazepin",
    "MSC": "Mick Schumacher",    "NOR": "Lando Norris",
    "OCO": "Esteban Ocon",       "PER": "Sergio Perez",
    "PIA": "Oscar Piastri",      "RAI": "Kimi Raikkonen",
    "RIC": "Daniel Ricciardo",   "RUS": "George Russell",
    "SAI": "Carlos Sainz",       "SAR": "Logan Sargeant",
    "STR": "Lance Stroll",       "TSU": "Yuki Tsunoda",
    "VER": "Max Verstappen",     "VET": "Sebastian Vettel",
    "ZHO": "Guanyu Zhou",        "AIT": "Jack Aitken",
    "FIT": "Pietro Fittipaldi",  "KUB": "Robert Kubica",
}

def _dn(code: str) -> str:
    return _DRIVER_NAMES.get(code, code)

def _fmt_t(seconds: float) -> str:
    """Format seconds as lap-time string e.g. 1:23.456."""
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:06.3f}"

def _fmt_s(seconds: float, precision: int = 3) -> str:
    """Format a seconds delta with sign e.g. +0.234."""
    return f"{seconds:+.{precision}f}s"

def _sign(val: float) -> str:
    return "faster" if val < 0 else "slower"


# ── Template functions ─────────────────────────────────────────────────────────
# Each returns list[tuple[str, str]] = [(question, answer), ...]

# T01 ── tyre deg rate ─────────────────────────────────────────────────────────
def t01_tyre_deg_rate(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, compound, season, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE avg_deg_per_lap IS NOT NULL
          AND max_viable_laps IS NOT NULL
          AND avg_deg_per_lap > 0
        """
    ).fetchall()
    out = []
    for r in rows:
        compound = r["compound"].capitalize()
        cliff = f" The pace cliff typically arrives around lap {r['cliff_lap']} on the tyre." if r["cliff_lap"] else ""
        q = (f"How do {compound} tyres degrade at {r['circuit']} in {r['season']}?")
        a = (
            f"At {r['circuit']} in {r['season']}, {compound} tyres showed an average "
            f"degradation rate of {r['avg_deg_per_lap']:.3f} seconds per lap "
            f"based on race data. Maximum viable stint length was around "
            f"{r['max_viable_laps']} laps before lap times became uncompetitive.{cliff}"
        )
        out.append((q, a))
    return out


# T02 ── tyre cliff lap ────────────────────────────────────────────────────────
def t02_tyre_cliff(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, compound, season, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE cliff_lap IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        compound = r["compound"].capitalize()
        q = f"When do {compound} tyres hit the cliff at {r['circuit']} in {r['season']}?"
        a = (
            f"In the {r['season']} {r['circuit']} race, {compound} tyres typically hit "
            f"their degradation cliff around lap {r['cliff_lap']} on the tyre. "
            f"Stints beyond {r['max_viable_laps']} laps on this compound were rarely "
            f"competitive, making it important to pit before significant drop-off."
        )
        out.append((q, a))
    return out


# T03 ── compound comparison ───────────────────────────────────────────────────
def t03_compound_compare(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT a.circuit, a.season,
               a.compound AS c1, a.avg_deg_per_lap AS d1, a.max_viable_laps AS m1,
               b.compound AS c2, b.avg_deg_per_lap AS d2, b.max_viable_laps AS m2
        FROM tyre_stats a
        JOIN tyre_stats b
          ON a.circuit = b.circuit
         AND a.season  = b.season
         AND a.compound < b.compound
        WHERE a.avg_deg_per_lap IS NOT NULL
          AND b.avg_deg_per_lap IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        c1, c2 = r["c1"].capitalize(), r["c2"].capitalize()
        faster = c1 if r["d1"] < r["d2"] else c2
        slower = c2 if faster == c1 else c1
        q = f"How do {c1} and {c2} tyres compare at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, {c1} tyres degraded at "
            f"{r['d1']:.3f}s/lap (max stint ~{r['m1']} laps) while {c2} tyres degraded "
            f"at {r['d2']:.3f}s/lap (max stint ~{r['m2']} laps). "
            f"{faster} tyres were more durable, making them the preferred choice "
            f"for longer stints. {slower} tyres offered more initial performance "
            f"but required earlier pit stops."
        )
        out.append((q, a))
    return out


# T04 ── best long-stint compound ─────────────────────────────────────────────
def t04_best_long_stint(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season, compound, avg_deg_per_lap, max_viable_laps
        FROM tyre_stats
        WHERE avg_deg_per_lap IS NOT NULL AND max_viable_laps IS NOT NULL
        ORDER BY circuit, season, avg_deg_per_lap ASC
        """
    ).fetchall()
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["circuit"], r["season"])
        if key in seen:
            continue
        seen.add(key)
        compound = r["compound"].capitalize()
        q = f"What compound works best for a long stint at {r['circuit']} in {r['season']}?"
        a = (
            f"For long stints at {r['circuit']} in {r['season']}, {compound} tyres "
            f"were the most durable option with a degradation rate of "
            f"{r['avg_deg_per_lap']:.3f}s/lap and a viable stint length of up to "
            f"{r['max_viable_laps']} laps. This made them the go-to choice for "
            f"teams aiming to minimise pit stops."
        )
        out.append((q, a))
    return out


# T05 ── track temp sensitivity ────────────────────────────────────────────────
def t05_temp_sensitivity(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, compound, season, track_temp_sensitivity, avg_deg_per_lap
        FROM tyre_stats
        WHERE track_temp_sensitivity IS NOT NULL
          AND ABS(track_temp_sensitivity) > 0.005
        """
    ).fetchall()
    out = []
    for r in rows:
        compound = r["compound"].capitalize()
        direction = "worse" if r["track_temp_sensitivity"] > 0 else "better"
        q = f"How does track temperature affect {compound} tyre performance at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, {compound} tyres showed a track "
            f"temperature sensitivity of {r['track_temp_sensitivity']:.4f}s per degree C "
            f"(after controlling for tyre age). Higher track temperatures generally made "
            f"the compound perform {direction}, compounding the base degradation rate "
            f"of {r['avg_deg_per_lap']:.3f}s/lap."
        )
        out.append((q, a))
    return out


# T06 ── driver sector strengths ───────────────────────────────────────────────
def t06_driver_sector_strengths(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit, season,
               avg_sector1_delta, avg_sector2_delta, avg_sector3_delta
        FROM driver_sector_stats
        WHERE avg_sector1_delta IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        deltas = {
            "Sector 1": r["avg_sector1_delta"],
            "Sector 2": r["avg_sector2_delta"],
            "Sector 3": r["avg_sector3_delta"],
        }
        best = min(deltas, key=deltas.get)
        worst = max(deltas, key=deltas.get)
        name = _dn(r["driver"])
        q = f"Where does {name} gain or lose time at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, {name} averaged "
            f"{_fmt_s(r['avg_sector1_delta'])} in S1, "
            f"{_fmt_s(r['avg_sector2_delta'])} in S2, and "
            f"{_fmt_s(r['avg_sector3_delta'])} in S3 "
            f"relative to the session fastest qualifying lap (reference: overall min lap time; "
            f"+seconds = slower in that sector, − = faster). "
            f"{name}'s strongest sector was {best} "
            f"({_fmt_s(deltas[best])}) and weakest was {worst} "
            f"({_fmt_s(deltas[worst])})."
        )
        out.append((q, a))
    return out


# T07 ── driver sector vs session reference (Q fastest lap) ─────────────────────
def t07_driver_sector_vs_field(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT d.driver, d.circuit, d.season,
               d.avg_sector1_delta, d.avg_sector2_delta, d.avg_sector3_delta
        FROM driver_sector_stats d
        WHERE ABS(d.avg_sector1_delta) > 0.05
           OR ABS(d.avg_sector2_delta) > 0.05
           OR ABS(d.avg_sector3_delta) > 0.05
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        q = f"What are {name}'s sector time weaknesses at {r['circuit']} in {r['season']}?"
        sectors = [
            ("Sector 1", r["avg_sector1_delta"]),
            ("Sector 2", r["avg_sector2_delta"]),
            ("Sector 3", r["avg_sector3_delta"]),
        ]
        weak = [(s, d) for s, d in sectors if d > 0.05]
        strong = [(s, d) for s, d in sectors if d < -0.05]
        parts = []
        if weak:
            parts.append(
                "lost time in " + " and ".join(
                    f"{s} ({_fmt_s(d)})" for s, d in weak
                )
            )
        if strong:
            parts.append(
                "gained time in " + " and ".join(
                    f"{s} ({_fmt_s(d)})" for s, d in strong
                )
            )
        if not parts:
            continue
        a = (
            f"In {r['season']} qualifying at {r['circuit']}, {name} "
            + " and ".join(parts)
            + " versus the session fastest driver."
        )
        out.append((q, a))
    return out


# T08 ── speed trap leader ─────────────────────────────────────────────────────
def t08_speed_trap_leader(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season, driver, avg_top_speed
        FROM speed_trap_stats
        WHERE rank_in_field = 1
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        q = f"Who had the best straight-line speed at {r['circuit']} in {r['season']}?"
        a = (
            f"{name} recorded the highest average top speed at {r['circuit']} in "
            f"{r['season']}, reaching {r['avg_top_speed']:.1f} km/h on average across "
            f"race laps. This straight-line advantage typically reflects a lower "
            f"downforce setup and/or stronger power unit performance on the straights."
        )
        out.append((q, a))
    return out


# T09 ── driver speed rank ─────────────────────────────────────────────────────
def t09_driver_speed_rank(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit, season, avg_top_speed, rank_in_field
        FROM speed_trap_stats
        WHERE rank_in_field <= 3
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        rank_word = {1: "fastest", 2: "second fastest", 3: "third fastest"}.get(r["rank_in_field"], f"rank {r['rank_in_field']}")
        q = f"How fast is {name} through the speed trap at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, {name} was the {rank_word} driver "
            f"through the speed trap with an average top speed of "
            f"{r['avg_top_speed']:.1f} km/h across race laps."
        )
        out.append((q, a))
    return out


# T10 ── corner min speed ──────────────────────────────────────────────────────
def t10_corner_min_speed(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT cs.driver, cs.circuit, cs.corner_number,
               cs.avg_min_speed, cs.delta_vs_field,
               c.corner_type
        FROM corner_stats cs
        LEFT JOIN corners c
          ON c.circuit = cs.circuit
         AND c.corner_number = cs.corner_number
        WHERE cs.avg_min_speed IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        ctype = r["corner_type"] or "unknown"
        direction = "above" if r["delta_vs_field"] >= 0 else "below"
        q = (
            f"How does {name} handle Turn {r['corner_number']} at {r['circuit']}?"
        )
        a = (
            f"At {r['circuit']}, {name} carries an average minimum speed of "
            f"{r['avg_min_speed']:.1f} km/h through Turn {r['corner_number']} "
            f"(classified as a {ctype}-speed corner). "
            f"Field reference: mean min speed across all drivers; "
            f"delta_vs_field is in km/h (not lap time): {abs(r['delta_vs_field']):.1f} km/h "
            f"{direction} that mean (positive = higher min speed vs field, not 'slower lap time')."
        )
        out.append((q, a))
    return out


# T11 ── high-speed corners driver ─────────────────────────────────────────────
def t11_high_speed_corners(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT cs.driver, cs.circuit,
               AVG(cs.avg_min_speed) AS avg_hs_speed,
               AVG(cs.delta_vs_field) AS avg_delta,
               COUNT(*) AS n_corners
        FROM corner_stats cs
        JOIN corners c
          ON c.circuit = cs.circuit
         AND c.corner_number = cs.corner_number
        WHERE c.corner_type = 'high'
        GROUP BY cs.driver, cs.circuit
        HAVING n_corners >= 2
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        direction = "above" if r["avg_delta"] >= 0 else "below"
        q = f"How does {name} handle high-speed corners at {r['circuit']}?"
        a = (
            f"Across {r['n_corners']} high-speed corners at {r['circuit']}, "
            f"{name} averaged {r['avg_hs_speed']:.1f} km/h minimum speed, "
            f"which is {abs(r['avg_delta']):.1f} km/h {direction} the field average "
            f"(field reference: mean min speed; delta in km/h, not seconds—positive = higher min speed). "
            f"{'A positive delta here indicates strong high-speed commitment and confidence through fast sweepers.' if r['avg_delta'] >= 0 else 'A negative delta here suggests a cautious approach or setup compromise through fast sweepers.'}"
        )
        out.append((q, a))
    return out


# T12 ── slow-speed corners driver ─────────────────────────────────────────────
def t12_slow_speed_corners(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT cs.driver, cs.circuit,
               AVG(cs.avg_min_speed) AS avg_ss_speed,
               AVG(cs.delta_vs_field) AS avg_delta,
               COUNT(*) AS n_corners
        FROM corner_stats cs
        JOIN corners c
          ON c.circuit = cs.circuit
         AND c.corner_number = cs.corner_number
        WHERE c.corner_type = 'slow'
        GROUP BY cs.driver, cs.circuit
        HAVING n_corners >= 2
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        direction = "above" if r["avg_delta"] >= 0 else "below"
        q = f"How does {name} perform in slow-speed corners at {r['circuit']}?"
        a = (
            f"Through {r['n_corners']} slow-speed corners at {r['circuit']}, "
            f"{name} averaged {r['avg_ss_speed']:.1f} km/h minimum speed "
            f"({abs(r['avg_delta']):.1f} km/h {direction} field average; "
            f"field reference = mean min speed; delta in km/h, not lap-time seconds). "
            f"Slow corners below 100 km/h are driven mostly on traction and "
            f"mechanical grip, so this reflects {name}'s car setup balance and "
            f"throttle application on exit."
        )
        out.append((q, a))
    return out


# T13 ── circuit corner classification ─────────────────────────────────────────
def t13_circuit_corner_types(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit,
               SUM(CASE WHEN corner_type='slow'   THEN 1 ELSE 0 END) AS n_slow,
               SUM(CASE WHEN corner_type='medium' THEN 1 ELSE 0 END) AS n_med,
               SUM(CASE WHEN corner_type='high'   THEN 1 ELSE 0 END) AS n_high,
               COUNT(*) AS total
        FROM corners
        WHERE corner_type IS NOT NULL
        GROUP BY circuit
        HAVING total >= 5
        """
    ).fetchall()
    out = []
    for r in rows:
        dominant = max(
            [("slow", r["n_slow"]), ("medium", r["n_med"]), ("high", r["n_high"])],
            key=lambda x: x[1],
        )[0]
        q = f"What kind of corners make {r['circuit']} challenging?"
        a = (
            f"{r['circuit']} has {r['total']} mapped corners: "
            f"{r['n_slow']} slow (<100 km/h), {r['n_med']} medium (100–180 km/h), "
            f"and {r['n_high']} high-speed (>180 km/h). "
            f"The circuit is predominantly {dominant}-speed in character, which "
            f"drives the setup philosophy — "
            + {
                "slow": "prioritising mechanical grip and traction out of tight hairpins.",
                "medium": "demanding a balanced setup across a wide range of corner speeds.",
                "high": "rewarding aerodynamic downforce and driver commitment through fast sweepers.",
            }[dominant]
        )
        out.append((q, a))
    return out


# T14 ── fuel effect at circuit ────────────────────────────────────────────────
def t14_fuel_effect_circuit(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season,
               AVG(fuel_effect_per_10kg) AS avg_effect,
               COUNT(*) AS n_drivers
        FROM fuel_stats
        WHERE fuel_effect_per_10kg IS NOT NULL
        GROUP BY circuit, season
        HAVING n_drivers >= 3
        """
    ).fetchall()
    out = []
    for r in rows:
        q = f"How much does fuel load affect lap time at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, carrying an extra 10 kg of fuel "
            f"cost approximately {r['avg_effect']:.3f} seconds per lap on average "
            f"across {r['n_drivers']} drivers. "
            f"This figure is estimated by comparing early-stint lap times "
            f"(laps 2–6) against late-stint laps on the same compound and "
            f"similar tyre age, normalised per 10 kg of fuel burned."
        )
        out.append((q, a))
    return out


# T15 ── driver fuel sensitivity ───────────────────────────────────────────────
def t15_driver_fuel_sensitivity(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit, season, fuel_effect_per_10kg
        FROM fuel_stats
        WHERE fuel_effect_per_10kg IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        q = f"How fuel-sensitive is {name}'s car at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, {name} lost approximately "
            f"{r['fuel_effect_per_10kg']:.3f} seconds per lap for every 10 kg of "
            f"additional fuel carried. "
            f"{'This is a relatively significant fuel penalty, suggesting a setup sensitive to weight.' if r['fuel_effect_per_10kg'] > 0.25 else 'This is a modest fuel penalty for this circuit.'}"
        )
        out.append((q, a))
    return out


# T16 ── quali vs race delta: converts well ────────────────────────────────────
def t16_quali_race_converts(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit, season, quali_lap, avg_race_pace, delta
        FROM quali_race_delta
        WHERE delta IS NOT NULL AND quali_lap IS NOT NULL AND avg_race_pace IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        q = f"Does {name} convert qualifying pace to race pace at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, {name}'s best qualifying lap was "
            f"{_fmt_t(r['quali_lap'])}, with an average race pace of "
            f"{_fmt_t(r['avg_race_pace'])} (excluding lap 1 and final lap). "
            f"The qualifying-to-race delta was {_fmt_s(r['delta'])}, meaning race pace "
            f"was {abs(r['delta']):.3f}s {'slower' if r['delta'] > 0 else 'faster'} "
            f"than qualifying on average."
        )
        out.append((q, a))
    return out


# T17 ── best qualifying performers at circuit ─────────────────────────────────
def t17_best_quali_at_circuit(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season, driver, quali_lap
        FROM quali_race_delta
        WHERE quali_lap IS NOT NULL
        ORDER BY circuit, season, quali_lap ASC
        """
    ).fetchall()
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["circuit"], r["season"])
        if key in seen:
            continue
        seen.add(key)
        name = _dn(r["driver"])
        q = f"Who set the best qualifying lap at {r['circuit']} in {r['season']}?"
        a = (
            f"The best qualifying lap at {r['circuit']} in {r['season']} was set by "
            f"{name} with a time of {_fmt_t(r['quali_lap'])}. "
            f"This represents the benchmark single-lap performance at this circuit "
            f"in this season."
        )
        out.append((q, a))
    return out


# T18 ── best race pace at circuit ─────────────────────────────────────────────
def t18_best_race_pace(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season, driver, avg_race_pace
        FROM quali_race_delta
        WHERE avg_race_pace IS NOT NULL
        ORDER BY circuit, season, avg_race_pace ASC
        """
    ).fetchall()
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["circuit"], r["season"])
        if key in seen:
            continue
        seen.add(key)
        name = _dn(r["driver"])
        q = f"Who had the best race pace at {r['circuit']} in {r['season']}?"
        a = (
            f"{name} showed the strongest average race pace at {r['circuit']} in "
            f"{r['season']}, averaging {_fmt_t(r['avg_race_pace'])} per lap "
            f"(excluding lap 1 and final lap, with safety car laps already filtered)."
        )
        out.append((q, a))
    return out


# T19 ── large quali-race gap ──────────────────────────────────────────────────
def t19_large_quali_race_gap(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit, season, quali_lap, avg_race_pace, delta
        FROM quali_race_delta
        WHERE delta > 5.0
          AND quali_lap IS NOT NULL AND avg_race_pace IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        q = f"Why is {name}'s race pace so different from qualifying at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, {name} had a qualifying lap of "
            f"{_fmt_t(r['quali_lap'])} but an average race pace of "
            f"{_fmt_t(r['avg_race_pace'])}, a delta of {r['delta']:.3f}s. "
            f"A large qualifying-to-race gap like this typically indicates the "
            f"driver or team optimised heavily for a single lap in qualifying, "
            f"using a setup or strategy that is not sustainable over race distance."
        )
        out.append((q, a))
    return out


# T20 ── sector 1 specialist ───────────────────────────────────────────────────
def t20_sector1_specialist(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT d.circuit, d.season, d.driver, d.avg_sector1_delta
        FROM driver_sector_stats d
        WHERE d.avg_sector1_delta IS NOT NULL
        ORDER BY d.circuit, d.season, d.avg_sector1_delta ASC
        """
    ).fetchall()
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["circuit"], r["season"])
        if key in seen:
            continue
        seen.add(key)
        name = _dn(r["driver"])
        q = f"Who is the strongest driver in Sector 1 at {r['circuit']} in {r['season']}?"
        a = (
            f"In {r['season']} qualifying at {r['circuit']}, {name} had the lowest "
            f"average Sector 1 delta at {_fmt_s(r['avg_sector1_delta'])} versus the "
            f"session fastest. Sector 1 performance reflects braking stability and "
            f"traction through the opening sequence of corners."
        )
        out.append((q, a))
    return out


# T21 ── sector 3 specialist ───────────────────────────────────────────────────
def t21_sector3_specialist(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT d.circuit, d.season, d.driver, d.avg_sector3_delta
        FROM driver_sector_stats d
        WHERE d.avg_sector3_delta IS NOT NULL
        ORDER BY d.circuit, d.season, d.avg_sector3_delta ASC
        """
    ).fetchall()
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["circuit"], r["season"])
        if key in seen:
            continue
        seen.add(key)
        name = _dn(r["driver"])
        q = f"Who dominates Sector 3 at {r['circuit']} in {r['season']}?"
        a = (
            f"In {r['season']} qualifying at {r['circuit']}, {name} was strongest "
            f"in Sector 3 with a delta of {_fmt_s(r['avg_sector3_delta'])} versus "
            f"the session fastest. Final-sector pace often reflects setup balance "
            f"and aerodynamic efficiency through the closing sequence."
        )
        out.append((q, a))
    return out


# T22 ── speed trap field range ────────────────────────────────────────────────
def t22_speed_trap_range(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season,
               MAX(avg_top_speed) AS fastest,
               MIN(avg_top_speed) AS slowest,
               COUNT(DISTINCT driver) AS n_drivers
        FROM speed_trap_stats
        GROUP BY circuit, season
        HAVING n_drivers >= 5
        """
    ).fetchall()
    out = []
    for r in rows:
        spread = r["fastest"] - r["slowest"]
        q = f"What is the speed trap range at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, the speed trap spread across "
            f"{r['n_drivers']} drivers ranged from {r['slowest']:.1f} km/h to "
            f"{r['fastest']:.1f} km/h — a field spread of {spread:.1f} km/h. "
            f"A wide spread typically indicates teams running very different "
            f"downforce levels for this circuit."
        )
        out.append((q, a))
    return out


# T23 ── tyre deg season-over-season ───────────────────────────────────────────
def t23_tyre_deg_season_delta(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT a.circuit, a.compound,
               a.season AS s1, a.avg_deg_per_lap AS d1,
               b.season AS s2, b.avg_deg_per_lap AS d2
        FROM tyre_stats a
        JOIN tyre_stats b
          ON a.circuit  = b.circuit
         AND a.compound = b.compound
         AND a.season   < b.season
        WHERE a.avg_deg_per_lap IS NOT NULL
          AND b.avg_deg_per_lap IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        compound = r["compound"].capitalize()
        better_yr = r["s2"] if r["d2"] < r["d1"] else r["s1"]
        worse_yr  = r["s1"] if better_yr == r["s2"] else r["s2"]
        delta = abs(r["d2"] - r["d1"])
        q = f"How did {compound} tyre degradation at {r['circuit']} change from {r['s1']} to {r['s2']}?"
        a = (
            f"{compound} tyre degradation at {r['circuit']} shifted from "
            f"{r['d1']:.3f}s/lap in {r['s1']} to {r['d2']:.3f}s/lap in {r['s2']} "
            f"— a change of {delta:.3f}s/lap. "
            f"{better_yr} was the more tyre-friendly season; {worse_yr} saw higher "
            f"degradation, potentially due to track surface changes, temperature "
            f"differences, or tyre specification updates."
        )
        out.append((q, a))
    return out


# T24 ── driver corner best vs worst ──────────────────────────────────────────
def t24_driver_corner_profile(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT cs.driver, cs.circuit,
               MIN(cs.delta_vs_field) AS best_delta,
               MAX(cs.delta_vs_field) AS worst_delta,
               COUNT(DISTINCT cs.corner_number) AS n_corners,
               (SELECT cs2.corner_number
                FROM corner_stats cs2
                WHERE cs2.driver=cs.driver AND cs2.circuit=cs.circuit
                ORDER BY cs2.delta_vs_field ASC LIMIT 1) AS best_corner,
               (SELECT cs2.corner_number
                FROM corner_stats cs2
                WHERE cs2.driver=cs.driver AND cs2.circuit=cs.circuit
                ORDER BY cs2.delta_vs_field DESC LIMIT 1) AS worst_corner
        FROM corner_stats cs
        GROUP BY cs.driver, cs.circuit
        HAVING n_corners >= 4
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        q = f"Which corners are {name}'s strongest and weakest at {r['circuit']}?"
        a = (
            f"Across {r['n_corners']} corners at {r['circuit']}, {name}'s best corner "
            f"is Turn {r['best_corner']} where they are "
            f"{abs(r['best_delta']):.1f} km/h above the field average. "
            f"Their weakest corner is Turn {r['worst_corner']} where they are "
            f"{abs(r['worst_delta']):.1f} km/h below the field average. "
            f"This reflects driving style, setup preferences, and confidence levels "
            f"through different corner types."
        )
        out.append((q, a))
    return out


# T25 ── quali vs race: best converters ───────────────────────────────────────
def t25_best_race_converters(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season,
               driver,
               delta,
               quali_lap,
               avg_race_pace
        FROM quali_race_delta
        WHERE delta IS NOT NULL AND delta > 0
        ORDER BY circuit, season, delta ASC
        """
    ).fetchall()
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["circuit"], r["season"])
        if key in seen:
            continue
        seen.add(key)
        name = _dn(r["driver"])
        q = f"Which driver best converts qualifying pace to race pace at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, {name} had the smallest "
            f"qualifying-to-race pace gap with a delta of just {r['delta']:.3f}s "
            f"(qualifying {_fmt_t(r['quali_lap'])}, average race pace "
            f"{_fmt_t(r['avg_race_pace'])}). A small delta means the driver "
            f"runs a well-balanced setup that works across both qualifying and race trim."
        )
        out.append((q, a))
    return out


# T26 ── compound viability comparison ────────────────────────────────────────
def t26_compound_viability(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season, compound, max_viable_laps, avg_deg_per_lap
        FROM tyre_stats
        WHERE max_viable_laps IS NOT NULL AND avg_deg_per_lap IS NOT NULL
        ORDER BY circuit, season, max_viable_laps DESC
        """
    ).fetchall()
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["circuit"], r["season"])
        if key in seen:
            continue
        seen.add(key)
        compound = r["compound"].capitalize()
        q = f"What is the maximum viable stint length at {r['circuit']} in {r['season']}?"
        a = (
            f"At {r['circuit']} in {r['season']}, the {compound} compound offered "
            f"the longest viable stint length at {r['max_viable_laps']} laps "
            f"before degradation became a significant strategic liability "
            f"(deg rate {r['avg_deg_per_lap']:.3f}s/lap). "
            f"This sets the baseline for strategy planning and pit-stop window calculations."
        )
        out.append((q, a))
    return out


# T27 ── field average corner speed by type ───────────────────────────────────
def t27_field_corner_speed_by_type(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT cs.circuit, c.corner_type,
               AVG(cs.avg_min_speed) AS field_avg,
               COUNT(DISTINCT cs.corner_number) AS n_corners
        FROM corner_stats cs
        JOIN corners c
          ON c.circuit = cs.circuit
         AND c.corner_number = cs.corner_number
        WHERE c.corner_type IS NOT NULL
        GROUP BY cs.circuit, c.corner_type
        HAVING n_corners >= 2
        """
    ).fetchall()
    out = []
    for r in rows:
        ctype = r["corner_type"]
        q = f"What is the typical minimum speed through {ctype}-speed corners at {r['circuit']}?"
        a = (
            f"At {r['circuit']}, drivers average a minimum speed of "
            f"{r['field_avg']:.1f} km/h through the {r['n_corners']} {ctype}-speed "
            f"corners (classified as "
            + {"slow": "<100 km/h", "medium": "100–180 km/h", "high": ">180 km/h"}[ctype]
            + f"). This field average is derived from telemetry speed traces "
            f"at each corner's apex across race laps."
        )
        out.append((q, a))
    return out


# T28 ── driver speed trap vs teammate ────────────────────────────────────────
def t28_speed_trap_top5(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season, driver, avg_top_speed, rank_in_field
        FROM speed_trap_stats
        WHERE rank_in_field <= 5
        ORDER BY circuit, season, rank_in_field
        """
    ).fetchall()
    groups: dict[tuple, list] = {}
    for r in rows:
        key = (r["circuit"], r["season"])
        groups.setdefault(key, []).append(r)
    out = []
    for (circuit, season), group in groups.items():
        if len(group) < 3:
            continue
        lines = [
            f"{i+1}. {_dn(g['driver'])} — {g['avg_top_speed']:.1f} km/h"
            for i, g in enumerate(group)
        ]
        q = f"Who are the top straight-line speed performers at {circuit} in {season}?"
        a = (
            f"The top average speed trap performers at {circuit} in {season} were:\n"
            + "\n".join(lines)
            + "\nThese rankings reflect average top speed across race laps on the "
            f"main straight, influenced by downforce setup and power unit output."
        )
        out.append((q, a))
    return out


# T29 ── qualifying performance summary ───────────────────────────────────────
def t29_quali_summary(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT d.circuit, d.season,
               d.driver,
               d.avg_sector1_delta + d.avg_sector2_delta + d.avg_sector3_delta AS total_delta,
               d.avg_sector1_delta, d.avg_sector2_delta, d.avg_sector3_delta
        FROM driver_sector_stats d
        WHERE d.avg_sector1_delta IS NOT NULL
        ORDER BY d.circuit, d.season, total_delta ASC
        """
    ).fetchall()
    seen: set[tuple] = set()
    out = []
    for r in rows:
        key = (r["circuit"], r["season"])
        if key in seen:
            continue
        seen.add(key)
        name = _dn(r["driver"])
        total = r["avg_sector1_delta"] + r["avg_sector2_delta"] + r["avg_sector3_delta"]
        q = f"Who had the most complete qualifying performance at {r['circuit']} in {r['season']}?"
        a = (
            f"In {r['season']} qualifying at {r['circuit']}, {name} delivered the "
            f"most balanced performance across all three sectors: "
            f"S1 {_fmt_s(r['avg_sector1_delta'])}, "
            f"S2 {_fmt_s(r['avg_sector2_delta'])}, "
            f"S3 {_fmt_s(r['avg_sector3_delta'])} "
            f"(combined delta {_fmt_s(total)} vs session fastest). "
            f"A consistent performance across all sectors indicates a well-rounded "
            f"car and driver combination for this particular circuit."
        )
        out.append((q, a))
    return out


# T30 ── tyre deg vs speed trade-off ──────────────────────────────────────────
def t30_tyre_deg_vs_speed(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT ts.circuit, ts.season, ts.compound,
               ts.avg_deg_per_lap, ts.max_viable_laps,
               sts.avg_top_speed
        FROM tyre_stats ts
        JOIN speed_trap_stats sts
          ON sts.circuit = ts.circuit
         AND sts.season  = ts.season
         AND sts.rank_in_field = 1
        WHERE ts.avg_deg_per_lap IS NOT NULL
          AND sts.avg_top_speed IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        compound = r["compound"].capitalize()
        q = (
            f"What is the strategic picture for {compound} tyres at "
            f"{r['circuit']} in {r['season']}?"
        )
        a = (
            f"At {r['circuit']} in {r['season']}, {compound} tyres degraded at "
            f"{r['avg_deg_per_lap']:.3f}s/lap with a max viable stint of "
            f"{r['max_viable_laps']} laps. The fastest speed trap reading that "
            f"season was {r['avg_top_speed']:.1f} km/h, giving context to the "
            f"circuit's straight-line emphasis. "
            f"Teams had to balance tyre life against the pace benefit of running "
            f"lower downforce setups optimised for the high-speed sections."
        )
        out.append((q, a))
    return out


# ── Registry ───────────────────────────────────────────────────────────────────

TEMPLATES: list[tuple[str, Callable]] = [
    ("tyre_deg_rate",           t01_tyre_deg_rate),
    ("tyre_cliff",              t02_tyre_cliff),
    ("compound_compare",        t03_compound_compare),
    ("best_long_stint",         t04_best_long_stint),
    ("temp_sensitivity",        t05_temp_sensitivity),
    ("driver_sector_strengths", t06_driver_sector_strengths),
    ("driver_sector_vs_field",  t07_driver_sector_vs_field),
    ("speed_trap_leader",       t08_speed_trap_leader),
    ("driver_speed_rank",       t09_driver_speed_rank),
    ("corner_min_speed",        t10_corner_min_speed),
    ("high_speed_corners",      t11_high_speed_corners),
    ("slow_speed_corners",      t12_slow_speed_corners),
    ("circuit_corner_types",    t13_circuit_corner_types),
    ("fuel_effect_circuit",     t14_fuel_effect_circuit),
    ("driver_fuel_sensitivity", t15_driver_fuel_sensitivity),
    ("quali_race_converts",     t16_quali_race_converts),
    ("best_quali_at_circuit",   t17_best_quali_at_circuit),
    ("best_race_pace",          t18_best_race_pace),
    ("large_quali_race_gap",    t19_large_quali_race_gap),
    ("sector1_specialist",      t20_sector1_specialist),
    ("sector3_specialist",      t21_sector3_specialist),
    ("speed_trap_range",        t22_speed_trap_range),
    ("tyre_deg_season_delta",   t23_tyre_deg_season_delta),
    ("driver_corner_profile",   t24_driver_corner_profile),
    ("best_race_converters",    t25_best_race_converters),
    ("compound_viability",      t26_compound_viability),
    ("field_corner_speed_type", t27_field_corner_speed_by_type),
    ("speed_trap_top5",         t28_speed_trap_top5),
    ("quali_summary",           t29_quali_summary),
    ("tyre_deg_vs_speed",       t30_tyre_deg_vs_speed),
]


# ── JSONL builder ──────────────────────────────────────────────────────────────

def _make_example(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


# ── Constants ──────────────────────────────────────────────────────────────────

TARGET_EXAMPLES   = 800   # hard ceiling on total output
MAX_PER_TEMPLATE  = 40    # max examples taken from any single template
RANDOM_SEED       = 42


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate template-based training examples")
    parser.add_argument("--limit",        type=int, default=TARGET_EXAMPLES,
                        help="Maximum total examples to write (default 800)")
    parser.add_argument("--per-template", type=int, default=MAX_PER_TEMPLATE,
                        help="Maximum examples per template before sampling (default 40)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATASET_DIR / "templates.jsonl"

    log.info("Generating templates -> %s  (limit=%d, per_template=%d)",
             out_path, args.limit, args.per_template)

    rng = random.Random(RANDOM_SEED)
    totals: dict[str, int] = {}
    examples: list[dict] = []

    with managed_connection() as conn:
        for name, fn in TEMPLATES:
            try:
                pairs = fn(conn)
            except Exception as exc:
                log.error("Template %s FAILED: %s", name, exc, exc_info=True)
                totals[name] = 0
                continue

            # Filter empty pairs
            valid = [(q, a) for q, a in pairs if q.strip() and a.strip()]

            # Random sample if over per-template cap
            if len(valid) > args.per_template:
                valid = rng.sample(valid, args.per_template)

            for q, a in valid:
                examples.append(_make_example(q, a))
            totals[name] = len(valid)
            log.info("  %-30s %d examples  (pool %d)", name, len(valid), len(pairs))

    # Shuffle then truncate to global limit
    rng.shuffle(examples)
    examples = examples[: args.limit]

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    total = len(examples)
    log.info("──────────────────────────────────────────────────────")
    log.info("Template counts (after per-template cap):")
    for name, cnt in totals.items():
        log.info("  %-30s %d", name, cnt)
    log.info("──────────────────────────────────────────────────────")
    log.info("Total examples written: %d -> %s", total, out_path)
    if total < 500:
        log.warning(
            "Only %d examples generated — corner_stats may be empty "
            "(run compute_stats.py with populate_corners first).", total
        )


if __name__ == "__main__":
    main()
