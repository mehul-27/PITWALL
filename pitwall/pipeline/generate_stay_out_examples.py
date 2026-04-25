"""
Generate stay-out strategy training examples from real SQLite tyre and lap data.
Appends single-turn JSONL examples to the training split.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYSTEM_PROMPT, TRAIN_PATH
from pipeline.db import managed_connection

log = logging.getLogger(__name__)

HELD_OUT_CIRCUIT_ALIASES = {
    "Bahrain", "Bahraini",
    "Monaco", "Monegasque",
    "Belgium", "Belgian",
    "Hungary", "Hungarian",
    "Italy", "Italian",
    "Singapore",
}


def _driver_name(code: str) -> str:
    names = {
        "ALB": "Albon", "ALO": "Alonso", "BOT": "Bottas", "GAS": "Gasly",
        "HAM": "Hamilton", "HUL": "Hulkenberg", "LAW": "Lawson", "LEC": "Leclerc",
        "MAG": "Magnussen", "NOR": "Norris", "OCO": "Ocon", "PER": "Perez",
        "PIA": "Piastri", "RIC": "Ricciardo", "RUS": "Russell", "SAI": "Sainz",
        "STR": "Stroll", "TSU": "Tsunoda", "VER": "Verstappen", "VET": "Vettel",
        "ZHO": "Zhou", "MSC": "Schumacher", "LAT": "Latifi",
    }
    return names.get(code, code)


def _fetch_candidates(conn) -> list[dict[str, Any]]:
    placeholders = ",".join("?" for _ in HELD_OUT_CIRCUIT_ALIASES)
    rows = conn.execute(
        f"""
        SELECT
            l.session_id,
            l.driver,
            l.lap_number,
            l.tyre_age,
            l.compound,
            l.lap_time,
            s.circuit,
            s.season,
            ts.avg_deg_per_lap,
            ts.cliff_lap,
            ts.max_viable_laps
        FROM laps l
        JOIN sessions s ON s.id = l.session_id
        JOIN tyre_stats ts
          ON ts.circuit = s.circuit
         AND ts.season = s.season
         AND ts.compound = l.compound
        WHERE s.session_type = 'Race'
          AND s.circuit NOT IN ({placeholders})
          AND l.is_valid = 1
          AND l.tyre_age IS NOT NULL
          AND l.lap_time IS NOT NULL
          AND ts.avg_deg_per_lap BETWEEN 0.010 AND 0.120
          AND ts.cliff_lap BETWEEN 12 AND 35
          AND ts.max_viable_laps BETWEEN 12 AND 45
          AND l.tyre_age <= ts.cliff_lap - 3
          AND l.tyre_age <= ts.max_viable_laps - 3
        ORDER BY RANDOM()
        LIMIT 1000
        """,
        tuple(sorted(HELD_OUT_CIRCUIT_ALIASES)),
    ).fetchall()

    candidates: list[dict[str, Any]] = []
    for row in rows:
        gap_row = conn.execute(
            """
            SELECT MIN(other.lap_time - ?) AS gap_behind
            FROM laps other
            WHERE other.session_id = ?
              AND other.lap_number = ?
              AND other.driver != ?
              AND other.is_valid = 1
              AND other.lap_time IS NOT NULL
              AND other.lap_time > ?
            """,
            (
                float(row["lap_time"]),
                int(row["session_id"]),
                int(row["lap_number"]),
                str(row["driver"]),
                float(row["lap_time"]),
            ),
        ).fetchone()
        if gap_row is None or gap_row["gap_behind"] is None:
            continue
        gap = float(gap_row["gap_behind"])
        if 0.1 <= gap <= 5.0:
            candidates.append({**dict(row), "gap_behind": gap})
    return candidates


def _make_example(row: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    driver = _driver_name(str(row["driver"]))
    compound = str(row["compound"]).capitalize()
    tyre_age = int(row["tyre_age"])
    deg_rate = float(row["avg_deg_per_lap"])
    cliff = int(row["cliff_lap"])
    gap = float(row["gap_behind"])
    margin = cliff - tyre_age

    user = (
        f"{driver} is on lap {int(row['lap_number'])} at {row['circuit']} in {int(row['season'])}. "
        f"Tyres are {tyre_age} laps old on the {compound} compound, deg rate is "
        f"{deg_rate:.3f} s/lap, cliff is at lap {cliff}. Gap behind is {gap:.1f} seconds. "
        "Recommend stay out with reasoning based on margin to cliff and traffic cost of pitting."
    )
    assistant = (
        f"Stay out. The {compound} tyres are {tyre_age} laps old with {margin} laps of margin "
        f"to the cliff, and the degradation rate is controlled at {deg_rate:.3f} s/lap. "
        f"With only {gap:.1f}s back to the car behind, pitting now risks dropping into traffic "
        "and giving away track position, so extend the stint and reassess next lap."
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def append_stay_out_examples(output_path: Path, count: int) -> int:
    with managed_connection() as conn:
        candidates = _fetch_candidates(conn)
    if len(candidates) < count:
        raise RuntimeError(f"Only {len(candidates)} stay-out candidates available; need {count}")

    random.shuffle(candidates)
    selected = candidates[:count]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as f:
        for row in selected:
            f.write(json.dumps(_make_example(row), ensure_ascii=False) + "\n")
    log.info("Appended %d stay-out examples -> %s", count, output_path)
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="Append stay-out strategy examples")
    parser.add_argument("--output", type=Path, default=TRAIN_PATH)
    parser.add_argument("--count", type=int, default=100)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    append_stay_out_examples(args.output, args.count)


if __name__ == "__main__":
    main()
