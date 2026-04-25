"""
Generate extra examples for weak categories: strategy, corner types, sparse comparisons.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import DATASET_DIR, SYSTEM_PROMPT
from pipeline.db import managed_connection
from pipeline.generate_templates import _dn, _fmt_s, _fmt_t

log = logging.getLogger(__name__)

OUTPUT_PATH = DATASET_DIR / "failure_boost.jsonl"
RANDOM_SEED = 321


def _make_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _strategy_examples(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT l.driver, l.lap_number, l.tyre_age, s.circuit, s.season, l.compound,
               ts.avg_deg_per_lap, ts.cliff_lap, ts.max_viable_laps
        FROM laps l
        JOIN sessions s ON s.id = l.session_id
        JOIN tyre_stats ts
          ON ts.circuit = s.circuit AND ts.season = s.season AND ts.compound = l.compound
        WHERE s.session_type = 'Race'
          AND l.is_valid = 1
          AND l.tyre_age IS NOT NULL
          AND l.tyre_age BETWEEN 8 AND 28
          AND ts.avg_deg_per_lap BETWEEN 0.010 AND 0.250
          AND ts.cliff_lap BETWEEN 8 AND 35
          AND ts.max_viable_laps BETWEEN 10 AND 45
        ORDER BY RANDOM()
        LIMIT 600
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = str(r["compound"]).capitalize()
        name = _dn(r["driver"])
        margin = int(r["cliff_lap"]) - int(r["tyre_age"])
        q = (
            f"{name} is on lap {r['lap_number']} at {r['circuit']} in {r['season']} "
            f"on {comp} tyres aged {r['tyre_age']} laps. What matters most for the strategy call?"
        )
        a = (
            f"Key facts: deg {r['avg_deg_per_lap']:.3f}s/lap, cliff lap {r['cliff_lap']}, max viable {r['max_viable_laps']} laps. "
            f"The tyre age is {r['tyre_age']} laps, so margin to the cliff is {margin} laps. "
            f"The engineer should judge whether that remaining tyre life is enough to protect track position, or whether the stop needs to become defensive before the tyre falls off."
        )
        out.append(_make_example(q, a))
    return out


def _corner_type_examples(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT circuit,
               SUM(CASE WHEN corner_type='slow' THEN 1 ELSE 0 END) AS slow_n,
               SUM(CASE WHEN corner_type='medium' THEN 1 ELSE 0 END) AS medium_n,
               SUM(CASE WHEN corner_type='high' THEN 1 ELSE 0 END) AS high_n,
               COUNT(*) AS total
        FROM corners
        WHERE corner_type IS NOT NULL
        GROUP BY circuit
        HAVING total >= 5
        """
    ).fetchall()
    out = []
    for r in rows:
        q = f"What setup story does the corner mix tell us for {r['circuit']}?"
        a = (
            f"Key facts: {r['slow_n']} slow-speed corners, {r['medium_n']} medium-speed corners, {r['high_n']} high-speed corners. "
            f"That mix tells you where the circuit demands grip and where it demands aero efficiency. "
            f"The setup should lean toward the dominant corner-speed class without making the weakest part of the lap unmanageable."
        )
        out.append(_make_example(q, a))
    return out


def _sparse_comparison_examples(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT a.driver AS d1, b.driver AS d2, a.circuit, a.season,
               a.avg_sector1_delta AS a1, a.avg_sector2_delta AS a2, a.avg_sector3_delta AS a3,
               b.avg_sector1_delta AS b1, b.avg_sector2_delta AS b2, b.avg_sector3_delta AS b3,
               qa.quali_lap AS q1, qb.quali_lap AS q2,
               ra.avg_race_pace AS r1, rb.avg_race_pace AS r2
        FROM driver_sector_stats a
        JOIN driver_sector_stats b
          ON b.circuit = a.circuit AND b.season = a.season AND b.driver > a.driver
        LEFT JOIN quali_race_delta qa
          ON qa.driver = a.driver AND qa.circuit = a.circuit AND qa.season = a.season
        LEFT JOIN quali_race_delta qb
          ON qb.driver = b.driver AND qb.circuit = b.circuit AND qb.season = b.season
        LEFT JOIN quali_race_delta ra
          ON ra.driver = a.driver AND ra.circuit = a.circuit AND ra.season = a.season
        LEFT JOIN quali_race_delta rb
          ON rb.driver = b.driver AND rb.circuit = b.circuit AND rb.season = b.season
        WHERE a.avg_sector1_delta IS NOT NULL
          AND a.avg_sector2_delta IS NOT NULL
          AND a.avg_sector3_delta IS NOT NULL
          AND b.avg_sector1_delta IS NOT NULL
          AND b.avg_sector2_delta IS NOT NULL
          AND b.avg_sector3_delta IS NOT NULL
          AND qa.quali_lap IS NOT NULL
          AND qb.quali_lap IS NOT NULL
          AND ra.avg_race_pace IS NOT NULL
          AND rb.avg_race_pace IS NOT NULL
        LIMIT 400
        """
    ).fetchall()
    out = []
    for r in rows:
        n1, n2 = _dn(r["d1"]), _dn(r["d2"])
        q = f"Give me a compact comparison of {n1} vs {n2} at {r['circuit']} in {r['season']}."
        a = (
            f"Key facts: {n1} sectors {_fmt_s(r['a1'])}, {_fmt_s(r['a2'])}, {_fmt_s(r['a3'])}; "
            f"{n2} sectors {_fmt_s(r['b1'])}, {_fmt_s(r['b2'])}, {_fmt_s(r['b3'])}. "
            f"Qualifying laps were {_fmt_t(r['q1'])} for {n1} and {_fmt_t(r['q2'])} for {n2}; "
            f"race pace averaged {_fmt_t(r['r1'])} vs {_fmt_t(r['r2'])}. "
            f"That gives a direct read on where each driver gains time and whether the one-lap edge also carries into race trim."
        )
        out.append(_make_example(q, a))
    return out


def generate_failure_boost(output_path: Path, strategy_n: int, corner_n: int, compare_n: int) -> int:
    rng = random.Random(RANDOM_SEED)
    with managed_connection() as conn:
        strategy = _strategy_examples(conn)
        corner = _corner_type_examples(conn)
        compare = _sparse_comparison_examples(conn)
    rng.shuffle(strategy)
    rng.shuffle(corner)
    rng.shuffle(compare)
    examples = strategy[:strategy_n] + corner[:corner_n] + compare[:compare_n]
    rng.shuffle(examples)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info(
        "Wrote %d failure-boost examples -> %s (strategy=%d corner=%d compare=%d)",
        len(examples), output_path, min(len(strategy), strategy_n), min(len(corner), corner_n), min(len(compare), compare_n),
    )
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate extra examples for weak categories")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--strategy", type=int, default=40)
    parser.add_argument("--corner", type=int, default=30)
    parser.add_argument("--compare", type=int, default=40)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    generate_failure_boost(args.output, args.strategy, args.corner, args.compare)


if __name__ == "__main__":
    main()
