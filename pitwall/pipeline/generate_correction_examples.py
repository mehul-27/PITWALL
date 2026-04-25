"""
Generate explicit correction examples so the model learns to push back on bad premises.
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
from pipeline.generate_templates import _dn

log = logging.getLogger(__name__)

OUTPUT_PATH = DATASET_DIR / "correction_examples.jsonl"
RANDOM_SEED = 123


def _make_example(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _tyre_corrections(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT circuit, season, compound, avg_deg_per_lap, cliff_lap, max_viable_laps
        FROM tyre_stats
        WHERE avg_deg_per_lap BETWEEN 0.010 AND 0.250
          AND cliff_lap BETWEEN 8 AND 35
          AND max_viable_laps BETWEEN 10 AND 45
        """
    ).fetchall()
    out = []
    for r in rows:
        comp = str(r["compound"]).capitalize()
        wrong_max = max(3, int(r["max_viable_laps"]) - 6)
        user = (
            f"The {comp} at {r['circuit']} in {r['season']} is basically unusable after "
            f"{wrong_max} laps, right?"
        )
        assistant = (
            f"No, that premise is too pessimistic. Key facts: deg {r['avg_deg_per_lap']:.3f}s/lap, "
            f"cliff lap {r['cliff_lap']}, max viable {r['max_viable_laps']} laps. "
            f"The tyre does not become unusable after {wrong_max} laps; the workable window runs "
            f"much deeper into the stint before the cliff and viability limits arrive."
        )
        out.append(_make_example(user, assistant))
    return out


def _corner_type_corrections(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT circuit,
               SUM(CASE WHEN corner_type='slow' THEN 1 ELSE 0 END) AS slow_n,
               SUM(CASE WHEN corner_type='medium' THEN 1 ELSE 0 END) AS medium_n,
               SUM(CASE WHEN corner_type='high' THEN 1 ELSE 0 END) AS high_n
        FROM corners
        WHERE corner_type IS NOT NULL
        GROUP BY circuit
        HAVING COUNT(*) >= 5
        """
    ).fetchall()
    out = []
    for r in rows:
        dominant = max(
            [("slow", r["slow_n"]), ("medium", r["medium_n"]), ("high", r["high_n"])],
            key=lambda item: item[1],
        )[0]
        wrong = {"slow": "high-speed", "medium": "slow-speed", "high": "slow-speed"}[dominant]
        user = f"{r['circuit']} is mainly a {wrong} circuit, isn't it?"
        assistant = (
            f"Not from the corner map. Key facts: {r['slow_n']} slow-speed corners, "
            f"{r['medium_n']} medium-speed corners, {r['high_n']} high-speed corners. "
            f"That makes {r['circuit']} predominantly {dominant}-speed in character, so the setup story "
            f"should follow that balance rather than the {wrong} assumption."
        )
        out.append(_make_example(user, assistant))
    return out


def _driver_comparison_corrections(conn) -> list[dict]:
    rows = conn.execute(
        """
        SELECT a.driver AS d1, b.driver AS d2, a.circuit, a.season,
               a.avg_sector1_delta + a.avg_sector2_delta + a.avg_sector3_delta AS total_a,
               b.avg_sector1_delta + b.avg_sector2_delta + b.avg_sector3_delta AS total_b
        FROM driver_sector_stats a
        JOIN driver_sector_stats b
          ON b.circuit = a.circuit AND b.season = a.season AND b.driver > a.driver
        WHERE a.avg_sector1_delta IS NOT NULL
          AND a.avg_sector2_delta IS NOT NULL
          AND a.avg_sector3_delta IS NOT NULL
          AND b.avg_sector1_delta IS NOT NULL
          AND b.avg_sector2_delta IS NOT NULL
          AND b.avg_sector3_delta IS NOT NULL
        LIMIT 200
        """
    ).fetchall()
    out = []
    for r in rows:
        stronger = r["d1"] if r["total_a"] <= r["total_b"] else r["d2"]
        weaker = r["d2"] if stronger == r["d1"] else r["d1"]
        user = f"{_dn(weaker)} clearly had the edge over {_dn(stronger)} at {r['circuit']} in {r['season']}, right?"
        assistant = (
            f"No, the data points the other way. Key facts: {_dn(r['d1'])} total sector delta {r['total_a']:+.3f}s, "
            f"{_dn(r['d2'])} total sector delta {r['total_b']:+.3f}s. Lower combined delta is stronger, "
            f"so {_dn(stronger)} had the better qualifying profile at {r['circuit']} in {r['season']}."
        )
        out.append(_make_example(user, assistant))
    return out


def generate_corrections(output_path: Path, limit: int) -> int:
    rng = random.Random(RANDOM_SEED)
    with managed_connection() as conn:
        examples = _tyre_corrections(conn) + _corner_type_corrections(conn) + _driver_comparison_corrections(conn)
    rng.shuffle(examples)
    examples = examples[:limit]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    log.info("Wrote %d correction examples -> %s", len(examples), output_path)
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate correction examples for bad premises")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    generate_corrections(args.output, args.limit)


if __name__ == "__main__":
    main()
