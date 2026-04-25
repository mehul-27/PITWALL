"""
Expanded template-based dataset generation for PitWall.
Creates additional structured single-turn Q/A examples from SQLite stats.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_DIR, SYSTEM_PROMPT
from pipeline.db import managed_connection
from pipeline.generate_templates import _dn, _fmt_s, _fmt_t

log = logging.getLogger(__name__)

TARGET_EXAMPLES = 1200
MAX_PER_TEMPLATE = 75
RANDOM_SEED = 84

TYRE_FILTER = """
avg_deg_per_lap BETWEEN 0.010 AND 0.250
AND cliff_lap BETWEEN 8 AND 35
AND max_viable_laps BETWEEN 10 AND 45
"""


def _make_example(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def _compound(value: str) -> str:
    return str(value).capitalize()


def e01_pit_window_open(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE {TYRE_FILTER}
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _compound(r["compound"])
        open_lap = max(1, int(r["cliff_lap"]) - 3)
        q = f"When should we start watching the pit window for {comp} tyres at {r['circuit']} in {r['season']}?"
        a = (
            f"Start watching the pit window around tyre age lap {open_lap}. "
            f"At {r['circuit']} in {r['season']}, the {comp} has a degradation rate of "
            f"{r['avg_deg_per_lap']:.3f}s/lap, a cliff at lap {r['cliff_lap']}, and a "
            f"maximum viable stint of {r['max_viable_laps']} laps. That gives a short "
            f"buffer before the cliff without waiting until the tyre is already falling away."
        )
        out.append((q, a))
    return out


def e02_extend_or_box(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE {TYRE_FILTER}
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _compound(r["compound"])
        age = max(8, int(r["cliff_lap"]) - 2)
        decision = "box soon" if age >= int(r["max_viable_laps"]) - 2 else "extend"
        q = f"{comp} tyres are {age} laps old at {r['circuit']} in {r['season']}. Do we extend or box?"
        a = (
            f"I would {decision}. The {comp} is at tyre age {age}, with the cliff at "
            f"{r['cliff_lap']} and max viable life at {r['max_viable_laps']} laps. "
            f"Deg is {r['avg_deg_per_lap']:.3f}s/lap, so the next laps are still manageable "
            f"only if pace remains stable and traffic after stopping is poor."
        )
        out.append((q, a))
    return out


def e03_low_deg_stretch(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE {TYRE_FILTER}
          AND avg_deg_per_lap <= 0.040
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _compound(r["compound"])
        q = f"Can we stretch the {comp} stint at {r['circuit']} in {r['season']}?"
        a = (
            f"Yes, this is a reasonable compound to stretch. The {comp} degradation rate is "
            f"only {r['avg_deg_per_lap']:.3f}s/lap, with the cliff at lap {r['cliff_lap']} "
            f"and max viable life at {r['max_viable_laps']} laps. The key is to avoid pushing "
            f"too hard before the final third of the stint, because the cliff still defines the limit."
        )
        out.append((q, a))
    return out


def e04_high_deg_protect(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE {TYRE_FILTER}
          AND avg_deg_per_lap >= 0.080
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _compound(r["compound"])
        q = f"How should the driver manage {comp} tyres at {r['circuit']} in {r['season']}?"
        a = (
            f"Protect the tyre early. The {comp} is degrading at {r['avg_deg_per_lap']:.3f}s/lap, "
            f"so sliding or overheating will cost lap time quickly. With a cliff at lap "
            f"{r['cliff_lap']} and max viable life of {r['max_viable_laps']} laps, the driver "
            f"should prioritise clean exits, avoid wheelspin, and keep enough life for the pit window."
        )
        out.append((q, a))
    return out


def e05_best_compound_for_offset(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE {TYRE_FILTER}
        ORDER BY circuit, season, max_viable_laps DESC, avg_deg_per_lap ASC
        """
    ).fetchall()
    seen = set()
    out = []
    for r in rows:
        key = (r["circuit"], r["season"])
        if key in seen:
            continue
        seen.add(key)
        comp = _compound(r["compound"])
        q = f"Which tyre is best for an offset strategy at {r['circuit']} in {r['season']}?"
        a = (
            f"The {comp} is the best offset tyre in this dataset. It can run to about "
            f"{r['max_viable_laps']} laps, with the cliff at lap {r['cliff_lap']} and "
            f"degradation at {r['avg_deg_per_lap']:.3f}s/lap. That gives enough stint length "
            f"to delay the stop and attack later on fresher tyres."
        )
        out.append((q, a))
    return out


def e06_compound_risk_rank(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season,
               GROUP_CONCAT(compound || ':' || ROUND(avg_deg_per_lap, 3) || ':' || max_viable_laps, ' | ') AS summary,
               COUNT(*) AS n
        FROM tyre_stats
        WHERE {TYRE_FILTER}
        GROUP BY circuit, season
        HAVING n >= 2
        """
    ).fetchall()
    out = []
    for r in rows:
        q = f"How do the tyre options rank for stint risk at {r['circuit']} in {r['season']}?"
        a = (
            f"For {r['circuit']} in {r['season']}, the tyre data is compound:deg:max_laps = "
            f"{r['summary']}. Lower degradation and longer max life reduce stint risk. "
            f"The safest call is the compound with the longest viable life; the aggressive "
            f"call is the compound with shorter life but potentially better initial pace."
        )
        out.append((q, a))
    return out


def e07_sector_attack_plan(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit, season, avg_sector1_delta, avg_sector2_delta, avg_sector3_delta
        FROM driver_sector_stats
        WHERE avg_sector1_delta IS NOT NULL
          AND avg_sector2_delta IS NOT NULL
          AND avg_sector3_delta IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        sectors = [(1, r["avg_sector1_delta"]), (2, r["avg_sector2_delta"]), (3, r["avg_sector3_delta"])]
        worst = max(sectors, key=lambda x: x[1])
        name = _dn(r["driver"])
        q = f"What sector should {name} focus on at {r['circuit']} in {r['season']}?"
        a = (
            f"Focus on Sector {worst[0]}. {name}'s deltas are S1 {_fmt_s(r['avg_sector1_delta'])}, "
            f"S2 {_fmt_s(r['avg_sector2_delta'])}, and S3 {_fmt_s(r['avg_sector3_delta'])} "
            f"(vs session fastest Q lap: +s = slower, −s = faster); "
            f"Sector {worst[0]} is the biggest loss at {_fmt_s(worst[1])}. The setup and driving work "
            f"should target the corner sequence in that sector first."
        )
        out.append((q, a))
    return out


def e08_sector_strength_defend(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit, season, avg_sector1_delta, avg_sector2_delta, avg_sector3_delta
        FROM driver_sector_stats
        WHERE avg_sector1_delta IS NOT NULL
          AND avg_sector2_delta IS NOT NULL
          AND avg_sector3_delta IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        sectors = [(1, r["avg_sector1_delta"]), (2, r["avg_sector2_delta"]), (3, r["avg_sector3_delta"])]
        best = min(sectors, key=lambda x: x[1])
        name = _dn(r["driver"])
        q = f"Where can {name} defend lap time at {r['circuit']} in {r['season']}?"
        a = (
            f"{name}'s best area is Sector {best[0]}, with a delta of {_fmt_s(best[1])}. "
            f"The full sector profile is S1 {_fmt_s(r['avg_sector1_delta'])}, "
            f"S2 {_fmt_s(r['avg_sector2_delta'])}, S3 {_fmt_s(r['avg_sector3_delta'])}. "
            f"That sector is where the driver can defend lap time even if tyres or fuel load make "
            f"the rest of the lap more difficult."
        )
        out.append((q, a))
    return out


def e09_head_to_head_sector_edge(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT a.driver AS d1, b.driver AS d2, a.circuit, a.season,
               a.avg_sector1_delta AS a1, a.avg_sector2_delta AS a2, a.avg_sector3_delta AS a3,
               b.avg_sector1_delta AS b1, b.avg_sector2_delta AS b2, b.avg_sector3_delta AS b3
        FROM driver_sector_stats a
        JOIN driver_sector_stats b
          ON b.circuit = a.circuit AND b.season = a.season AND b.driver > a.driver
        WHERE a.avg_sector1_delta IS NOT NULL
          AND a.avg_sector2_delta IS NOT NULL
          AND a.avg_sector3_delta IS NOT NULL
          AND b.avg_sector1_delta IS NOT NULL
          AND b.avg_sector2_delta IS NOT NULL
          AND b.avg_sector3_delta IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        n1, n2 = _dn(r["d1"]), _dn(r["d2"])
        wins = []
        for sector, a_key, b_key in [(1, "a1", "b1"), (2, "a2", "b2"), (3, "a3", "b3")]:
            if r[a_key] <= r[b_key]:
                wins.append(f"Sector {sector}: {n1} by {abs(r[b_key] - r[a_key]):.3f}s")
            else:
                wins.append(f"Sector {sector}: {n2} by {abs(r[a_key] - r[b_key]):.3f}s")
        q = f"Compare {n1} and {n2} by sector at {r['circuit']} in {r['season']}."
        a = (
            f"Sector comparison at {r['circuit']} in {r['season']}: "
            + "; ".join(wins)
            + ". Lower delta is better because these figures are measured relative to the session fastest."
        )
        out.append((q, a))
    return out


def e10_top_speed_vs_sector_loss(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT d.driver, d.circuit, d.season,
               d.avg_sector1_delta, d.avg_sector2_delta, d.avg_sector3_delta,
               s.avg_top_speed, s.rank_in_field
        FROM driver_sector_stats d
        JOIN speed_trap_stats s
          ON s.driver = d.driver AND s.circuit = d.circuit AND s.season = d.season
        WHERE s.avg_top_speed BETWEEN 280 AND 370
          AND s.rank_in_field <= 5
          AND d.avg_sector1_delta IS NOT NULL
          AND d.avg_sector2_delta IS NOT NULL
          AND d.avg_sector3_delta IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        total = r["avg_sector1_delta"] + r["avg_sector2_delta"] + r["avg_sector3_delta"]
        q = f"Is {name}'s top speed translating into lap time at {r['circuit']} in {r['season']}?"
        a = (
            f"{name} ranked P{r['rank_in_field']} in the speed trap at {r['avg_top_speed']:.1f} km/h. "
            f"The sector deltas were S1 {_fmt_s(r['avg_sector1_delta'])}, S2 {_fmt_s(r['avg_sector2_delta'])}, "
            f"S3 {_fmt_s(r['avg_sector3_delta'])}, for a combined delta of {_fmt_s(total)}. "
            f"Good top speed helps, but the sector deltas show whether that straight-line pace is actually "
            f"being converted across the full lap."
        )
        out.append((q, a))
    return out


def e11_speed_setup_signal(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season,
               MAX(avg_top_speed) AS fastest,
               MIN(avg_top_speed) AS slowest,
               AVG(avg_top_speed) AS avg_speed,
               COUNT(*) AS n
        FROM speed_trap_stats
        WHERE avg_top_speed BETWEEN 280 AND 370
        GROUP BY circuit, season
        HAVING n >= 8
        """
    ).fetchall()
    out = []
    for r in rows:
        spread = r["fastest"] - r["slowest"]
        q = f"What does the speed trap spread say about setup at {r['circuit']} in {r['season']}?"
        a = (
            f"The field averaged {r['avg_speed']:.1f} km/h, with a range from "
            f"{r['slowest']:.1f} to {r['fastest']:.1f} km/h across {r['n']} drivers. "
            f"That {spread:.1f} km/h spread suggests how differently teams traded drag against "
            f"corner load. A bigger spread usually means setup choice mattered strongly."
        )
        out.append((q, a))
    return out


def e12_corner_attack(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit, corner_number, avg_min_speed, delta_vs_field
        FROM corner_stats
        WHERE avg_min_speed BETWEEN 60 AND 320
          AND ABS(delta_vs_field) >= 3.0
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        direction = "above" if r["delta_vs_field"] > 0 else "below"
        q = f"What should {name} change through Turn {r['corner_number']} at {r['circuit']}?"
        a = (
            f"{name}'s average minimum speed through Turn {r['corner_number']} is "
            f"{r['avg_min_speed']:.1f} km/h, which is {abs(r['delta_vs_field']):.1f} km/h "
            f"{direction} the field average. If below the field, the priority is entry stability "
            f"and earlier throttle; if above the field, preserve that strength without overheating the tyre."
        )
        out.append((q, a))
    return out


def e13_corner_type_setup(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT cs.driver, cs.circuit, c.corner_type,
               AVG(cs.avg_min_speed) AS avg_speed,
               AVG(cs.delta_vs_field) AS avg_delta,
               COUNT(*) AS n
        FROM corner_stats cs
        JOIN corners c ON c.circuit = cs.circuit AND c.corner_number = cs.corner_number
        WHERE c.corner_type IS NOT NULL
          AND cs.avg_min_speed BETWEEN 60 AND 320
        GROUP BY cs.driver, cs.circuit, c.corner_type
        HAVING n >= 2
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        direction = "above" if r["avg_delta"] >= 0 else "below"
        q = f"What does {name}'s {r['corner_type']}-corner profile look like at {r['circuit']}?"
        a = (
            f"Across {r['n']} {r['corner_type']}-speed corners at {r['circuit']}, {name} averages "
            f"{r['avg_speed']:.1f} km/h minimum speed, {abs(r['avg_delta']):.1f} km/h "
            f"{direction} the field. This points to the car balance and driver confidence in that "
            f"corner-speed range."
        )
        out.append((q, a))
    return out


def e14_corner_consistency(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit,
               MIN(delta_vs_field) AS min_delta,
               MAX(delta_vs_field) AS max_delta,
               COUNT(*) AS n
        FROM corner_stats
        WHERE avg_min_speed BETWEEN 60 AND 320
        GROUP BY driver, circuit
        HAVING n >= 6
        """
    ).fetchall()
    out = []
    for r in rows:
        spread = r["max_delta"] - r["min_delta"]
        name = _dn(r["driver"])
        q = f"How consistent is {name} across corners at {r['circuit']}?"
        a = (
            f"{name}'s corner-speed delta range at {r['circuit']} spans from "
            f"{r['min_delta']:+.1f} to {r['max_delta']:+.1f} km/h versus the field across "
            f"{r['n']} corners, a spread of {spread:.1f} km/h. A smaller spread indicates "
            f"a more consistent corner profile; a bigger spread shows clear strengths and weaknesses."
        )
        out.append((q, a))
    return out


def e15_fuel_phase_strategy(conn) -> list[tuple[str, str]]:
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
        q = f"How should {name} adjust driving as fuel burns off at {r['circuit']} in {r['season']}?"
        a = (
            f"{name}'s estimated fuel effect is {r['fuel_effect_per_10kg']:.3f}s per lap per 10 kg. "
            f"As fuel burns off, lap time should improve by roughly that amount per 10 kg removed, "
            f"assuming tyre age is controlled. The race engineer should protect tyres early, then "
            f"let the driver attack more once the car is lighter."
        )
        out.append((q, a))
    return out


def e16_quali_to_race_setup(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT driver, circuit, season, quali_lap, avg_race_pace, delta
        FROM quali_race_delta
        WHERE delta BETWEEN 2.0 AND 15.0
          AND quali_lap IS NOT NULL
          AND avg_race_pace IS NOT NULL
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        q = f"What does {name}'s quali-to-race delta say at {r['circuit']} in {r['season']}?"
        a = (
            f"{name} qualified at {_fmt_t(r['quali_lap'])} and averaged {_fmt_t(r['avg_race_pace'])} "
            f"in race pace, a delta of {_fmt_s(r['delta'])}. A smaller delta points to a setup that "
            f"carries well into race trim; a larger delta suggests the car was more optimised for one-lap pace."
        )
        out.append((q, a))
    return out


def e17_race_pace_order(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season, driver, avg_race_pace
        FROM quali_race_delta
        WHERE avg_race_pace IS NOT NULL
        ORDER BY circuit, season, avg_race_pace ASC
        """
    ).fetchall()
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["circuit"], r["season"]), []).append(r)
    out = []
    for (circuit, season), group in groups.items():
        if len(group) < 3:
            continue
        top = group[:3]
        q = f"Who were the top three race-pace drivers at {circuit} in {season}?"
        a = (
            f"The top three average race-pace drivers at {circuit} in {season} were "
            f"{_dn(top[0]['driver'])} at {_fmt_t(top[0]['avg_race_pace'])}, "
            f"{_dn(top[1]['driver'])} at {_fmt_t(top[1]['avg_race_pace'])}, and "
            f"{_dn(top[2]['driver'])} at {_fmt_t(top[2]['avg_race_pace'])}. "
            f"These averages exclude lap 1 and final-lap effects."
        )
        out.append((q, a))
    return out


def e18_quali_pace_order(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        """
        SELECT circuit, season, driver, quali_lap
        FROM quali_race_delta
        WHERE quali_lap IS NOT NULL
        ORDER BY circuit, season, quali_lap ASC
        """
    ).fetchall()
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r["circuit"], r["season"]), []).append(r)
    out = []
    for (circuit, season), group in groups.items():
        if len(group) < 3:
            continue
        top = group[:3]
        q = f"Who were the top three qualifying benchmarks at {circuit} in {season}?"
        a = (
            f"The qualifying benchmarks at {circuit} in {season} were "
            f"{_dn(top[0]['driver'])} at {_fmt_t(top[0]['quali_lap'])}, "
            f"{_dn(top[1]['driver'])} at {_fmt_t(top[1]['quali_lap'])}, and "
            f"{_dn(top[2]['driver'])} at {_fmt_t(top[2]['quali_lap'])}. "
            f"That gives the one-lap reference before race-fuel and tyre degradation are considered."
        )
        out.append((q, a))
    return out


def e19_strategy_snapshot(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT ts.circuit, ts.season, ts.compound, ts.avg_deg_per_lap,
               ts.cliff_lap, ts.max_viable_laps,
               sp.fastest_speed
        FROM tyre_stats ts
        LEFT JOIN (
            SELECT circuit, season, MAX(avg_top_speed) AS fastest_speed
            FROM speed_trap_stats
            WHERE avg_top_speed BETWEEN 280 AND 370
            GROUP BY circuit, season
        ) sp ON sp.circuit = ts.circuit AND sp.season = ts.season
        WHERE {TYRE_FILTER}
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _compound(r["compound"])
        q = f"Give me a race-engineer strategy snapshot for {r['circuit']} {r['season']} on {comp}."
        speed = f" Top speed benchmark is {r['fastest_speed']:.1f} km/h." if r["fastest_speed"] is not None else ""
        a = (
            f"{comp} at {r['circuit']} in {r['season']}: deg {r['avg_deg_per_lap']:.3f}s/lap, "
            f"cliff lap {r['cliff_lap']}, max viable stint {r['max_viable_laps']} laps."
            f"{speed} Strategy should avoid running past the cliff unless track position is worth more "
            f"than the tyre-life loss."
        )
        out.append((q, a))
    return out


def e20_lap_query_from_real_laps(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT l.driver, l.lap_number, l.tyre_age, l.compound, s.circuit, s.season,
               ts.avg_deg_per_lap, ts.cliff_lap, ts.max_viable_laps
        FROM laps l
        JOIN sessions s ON s.id = l.session_id
        JOIN tyre_stats ts
          ON ts.circuit = s.circuit AND ts.season = s.season AND ts.compound = l.compound
        WHERE s.session_type = 'Race'
          AND l.is_valid = 1
          AND l.tyre_age IS NOT NULL
          AND l.tyre_age BETWEEN 6 AND 30
          AND {TYRE_FILTER.replace('avg_deg_per_lap', 'ts.avg_deg_per_lap').replace('cliff_lap', 'ts.cliff_lap').replace('max_viable_laps', 'ts.max_viable_laps')}
        ORDER BY RANDOM()
        LIMIT 500
        """
    ).fetchall()
    out = []
    for r in rows:
        name = _dn(r["driver"])
        comp = _compound(r["compound"])
        margin = int(r["cliff_lap"]) - int(r["tyre_age"])
        q = (
            f"{name} is on lap {r['lap_number']} at {r['circuit']} in {r['season']}, "
            f"{comp} tyres age {r['tyre_age']}. What is the tyre risk?"
        )
        a = (
            f"The tyre risk is driven by margin to the cliff. The {comp} is age {r['tyre_age']}, "
            f"with cliff lap {r['cliff_lap']} and max viable life {r['max_viable_laps']} laps, "
            f"so margin to cliff is {margin} laps. Deg is {r['avg_deg_per_lap']:.3f}s/lap; "
            f"if the margin is small, prepare the box call, otherwise keep managing."
        )
        out.append((q, a))
    return out


def e21_clean_air_priority(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE {TYRE_FILTER}
          AND avg_deg_per_lap >= 0.050
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _compound(r["compound"])
        q = f"How important is clean air for {comp} tyre management at {r['circuit']} in {r['season']}?"
        a = (
            f"Clean air matters a lot here. The {comp} is degrading at {r['avg_deg_per_lap']:.3f}s/lap, "
            f"with the cliff at lap {r['cliff_lap']} and viable life around {r['max_viable_laps']} laps. "
            f"Running in dirty air can raise tyre temperature and make that degradation worse, so avoid "
            f"traffic if the pit window gives a clean-air option."
        )
        out.append((q, a))
    return out


def e22_push_lap_timing(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE {TYRE_FILTER}
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _compound(r["compound"])
        push_age = max(1, int(r["cliff_lap"]) - 1)
        q = f"When is the latest safe push lap on {comp} tyres at {r['circuit']} in {r['season']}?"
        a = (
            f"The latest safe push point is around tyre age lap {push_age}, one lap before the "
            f"modelled cliff at {r['cliff_lap']}. The max viable life is {r['max_viable_laps']} laps "
            f"and deg is {r['avg_deg_per_lap']:.3f}s/lap, so using the tyre hard after that risks "
            f"crossing into non-linear drop-off."
        )
        out.append((q, a))
    return out


def e23_undercut_pressure(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE {TYRE_FILTER}
          AND avg_deg_per_lap >= 0.035
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _compound(r["compound"])
        q = f"Is the undercut threat strong on {comp} tyres at {r['circuit']} in {r['season']}?"
        a = (
            f"Yes, the undercut threat is meaningful because degradation is {r['avg_deg_per_lap']:.3f}s/lap. "
            f"If a rival pits before the cliff at lap {r['cliff_lap']}, fresh tyres can quickly offset "
            f"the pit timing disadvantage. With max viable life at {r['max_viable_laps']} laps, do not "
            f"leave the response too late if track position is close."
        )
        out.append((q, a))
    return out


def e24_overcut_case(conn) -> list[tuple[str, str]]:
    rows = conn.execute(
        f"""
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE {TYRE_FILTER}
          AND avg_deg_per_lap <= 0.035
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = _compound(r["compound"])
        q = f"Can the overcut work on {comp} tyres at {r['circuit']} in {r['season']}?"
        a = (
            f"The overcut can work if the car has clean air. Deg is only {r['avg_deg_per_lap']:.3f}s/lap, "
            f"so staying out can preserve track position until closer to the cliff at lap {r['cliff_lap']}. "
            f"The hard limit is the max viable stint of {r['max_viable_laps']} laps; beyond that, the tyre "
            f"life risk outweighs the overcut benefit."
        )
        out.append((q, a))
    return out


EXPANDED_TEMPLATES: list[tuple[str, Callable]] = [
    ("pit_window_open", e01_pit_window_open),
    ("extend_or_box", e02_extend_or_box),
    ("low_deg_stretch", e03_low_deg_stretch),
    ("high_deg_protect", e04_high_deg_protect),
    ("best_compound_for_offset", e05_best_compound_for_offset),
    ("compound_risk_rank", e06_compound_risk_rank),
    ("sector_attack_plan", e07_sector_attack_plan),
    ("sector_strength_defend", e08_sector_strength_defend),
    ("head_to_head_sector_edge", e09_head_to_head_sector_edge),
    ("top_speed_vs_sector_loss", e10_top_speed_vs_sector_loss),
    ("speed_setup_signal", e11_speed_setup_signal),
    ("corner_attack", e12_corner_attack),
    ("corner_type_setup", e13_corner_type_setup),
    ("corner_consistency", e14_corner_consistency),
    ("fuel_phase_strategy", e15_fuel_phase_strategy),
    ("quali_to_race_setup", e16_quali_to_race_setup),
    ("race_pace_order", e17_race_pace_order),
    ("quali_pace_order", e18_quali_pace_order),
    ("strategy_snapshot", e19_strategy_snapshot),
    ("lap_query_from_real_laps", e20_lap_query_from_real_laps),
    ("clean_air_priority", e21_clean_air_priority),
    ("push_lap_timing", e22_push_lap_timing),
    ("undercut_pressure", e23_undercut_pressure),
    ("overcut_case", e24_overcut_case),
]


def _dedupe_examples(examples: list[dict]) -> list[dict]:
    seen: set[str] = set()
    unique: list[dict] = []
    for ex in examples:
        user = ex["messages"][1]["content"]
        if user in seen:
            continue
        seen.add(user)
        unique.append(ex)
    return unique


def generate_expanded_templates(limit: int, per_template: int, output_path: Path) -> int:
    rng = random.Random(RANDOM_SEED)
    examples: list[dict] = []
    totals: dict[str, int] = {}

    with managed_connection() as conn:
        for name, fn in EXPANDED_TEMPLATES:
            try:
                pairs = fn(conn)
            except Exception as exc:
                log.error("Expanded template %s FAILED: %s", name, exc, exc_info=True)
                totals[name] = 0
                continue

            valid = [(q, a) for q, a in pairs if q.strip() and a.strip()]
            if len(valid) > per_template:
                valid = rng.sample(valid, per_template)
            for question, answer in valid:
                examples.append(_make_example(question, answer))
            totals[name] = len(valid)
            log.info("  %-30s %d examples  (pool %d)", name, len(valid), len(pairs))

    examples = _dedupe_examples(examples)
    rng.shuffle(examples)
    examples = examples[:limit]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    log.info("Expanded template counts:")
    for name, count in totals.items():
        log.info("  %-30s %d", name, count)
    log.info("Total expanded examples written: %d -> %s", len(examples), output_path)
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate expanded template-based PitWall examples")
    parser.add_argument("--limit", type=int, default=TARGET_EXAMPLES)
    parser.add_argument("--per-template", type=int, default=MAX_PER_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DATASET_DIR / "expanded_templates.jsonl")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    generate_expanded_templates(args.limit, args.per_template, args.output)


if __name__ == "__main__":
    main()
