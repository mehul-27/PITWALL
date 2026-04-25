"""
Regenerate only multi-turn JSONL examples through Groq.
Each multi-turn dialogue is rewritten as a single user situation plus one
assistant analysis/recommendation, with all facts upfront.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import SYSTEM_PROMPT, TRAIN_PATH
from pipeline.db import managed_connection
from pipeline.generate_groq import DailyQuotaExceeded, _call_groq, _init_groq

log = logging.getLogger(__name__)

DRIVER_TO_CODE = {
    "albon": "ALB", "alonso": "ALO", "bottas": "BOT", "gasly": "GAS",
    "hamilton": "HAM", "hulk": "HUL", "hulkenberg": "HUL", "lawson": "LAW",
    "leclerc": "LEC", "magnussen": "MAG", "norris": "NOR", "ocon": "OCO",
    "perez": "PER", "piastri": "PIA", "ricciardo": "RIC", "russell": "RUS",
    "sainz": "SAI", "stroll": "STR", "tsunoda": "TSU", "verstappen": "VER",
    "vettel": "VET", "zhou": "ZHO", "schumacher": "MSC", "latifi": "LAT",
}

PROMPT = """\
Rewrite this flawed multi-turn PitWall training example into exactly one
single-turn chat example.

FACTS:
{facts}

SOURCE DIALOGUE:
{source}

STRICT RULES:
- Output valid JSON only, with keys "user" and "assistant".
- The user message must contain the full situation upfront: lap number, tyre age, gap behind, and track temperature.
- The assistant reply must contain analysis and a clear recommendation.
- Do not drip-feed facts across turns.
- Use only FACTS. If gap behind or track temperature is unavailable, say it is unavailable; do not invent a number.
- Do not add any numeric value not present in FACTS.
- Keep the assistant reply to 3-5 sentences.
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def _source_text(example: dict[str, Any]) -> str:
    return "\n".join(
        f"{m.get('role', '').upper()}: {m.get('content', '')}"
        for m in example.get("messages", [])
        if m.get("role") != "system"
    )


def _extract_basic_facts(text: str) -> dict[str, Any]:
    lower = text.lower()
    driver = None
    for name, code in DRIVER_TO_CODE.items():
        if re.search(rf"\b{re.escape(name)}\b", lower):
            driver = code
            break
    if driver is None:
        m_code = re.search(r"\b(ALB|ALO|BOT|GAS|HAM|HUL|LAW|LEC|MAG|NOR|OCO|PER|PIA|RIC|RUS|SAI|STR|TSU|VER|VET|ZHO|MSC|LAT)\b", text)
        driver = m_code.group(1) if m_code else None

    def num(pattern: str, cast=float):
        m = re.search(pattern, text, re.IGNORECASE)
        return cast(m.group(1)) if m else None

    return {
        "driver": driver,
        "lap_number": num(r"\blap\s+(\d+)\b", int),
        "tyre_age": num(r"(\d+)\s+laps?(?:\s+old|\s+on the tyres|\s+into the stint)", int),
        "compound": (re.search(r"\b(soft|medium|hard|intermediate|wet)\b", lower) or [None])[0],
        "deg_rate": num(r"(?:degradation(?: rate)?|degradation's|deg rate(?:'s)?)\D+(\d+(?:\.\d+)?)"),
        "cliff_lap": num(r"(?:cliff(?: lap)?(?: is| at|, number)?|tyre cliff(?: lap)?(?: is| at)?)\D+(\d+)", int),
        "max_viable_laps": num(r"max viable(?: tyre)? stint(?: is| of)?\D+(\d+)", int),
    }


def _weather_join_mode(conn) -> str | None:
    row = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='weather'").fetchone()
    if not row:
        return None
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(weather)").fetchall()}
    if {"lap_id", "track_temp"}.issubset(cols):
        return "lap_id"
    if {"session_id", "lap_number", "track_temp"}.issubset(cols):
        return "session_lap"
    return None


def _augment_from_db(facts: dict[str, Any]) -> dict[str, Any]:
    if not facts.get("driver") or not facts.get("lap_number"):
        return facts
    with managed_connection() as conn:
        row = conn.execute(
            """
            SELECT l.id AS lap_id, l.session_id, l.driver, l.lap_number, l.tyre_age,
                   l.compound, l.lap_time, s.circuit, s.season,
                   ts.avg_deg_per_lap, ts.cliff_lap, ts.max_viable_laps
            FROM laps l
            JOIN sessions s ON s.id = l.session_id
            LEFT JOIN tyre_stats ts
              ON ts.circuit = s.circuit AND ts.season = s.season AND ts.compound = l.compound
            WHERE l.driver = ?
              AND l.lap_number = ?
              AND (? IS NULL OR l.tyre_age = ?)
              AND l.lap_time IS NOT NULL
            ORDER BY s.season DESC
            LIMIT 1
            """,
            (facts["driver"], facts["lap_number"], facts.get("tyre_age"), facts.get("tyre_age")),
        ).fetchone()
        if row is None:
            return facts
        facts.update({k: row[k] for k in row.keys() if row[k] is not None})

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
            (float(row["lap_time"]), int(row["session_id"]), int(row["lap_number"]), str(row["driver"]), float(row["lap_time"])),
        ).fetchone()
        if gap_row and gap_row["gap_behind"] is not None:
            facts["gap_behind"] = round(float(gap_row["gap_behind"]), 1)

        mode = _weather_join_mode(conn)
        if mode == "lap_id":
            temp = conn.execute("SELECT track_temp FROM weather WHERE lap_id=? LIMIT 1", (int(row["lap_id"]),)).fetchone()
        elif mode == "session_lap":
            temp = conn.execute(
                "SELECT track_temp FROM weather WHERE session_id=? AND lap_number=? LIMIT 1",
                (int(row["session_id"]), int(row["lap_number"])),
            ).fetchone()
        else:
            temp = None
        if temp and temp["track_temp"] is not None:
            facts["track_temp"] = round(float(temp["track_temp"]), 1)
    return facts


def _facts_block(facts: dict[str, Any]) -> str:
    labels = {
        "driver": "Driver", "circuit": "Circuit", "season": "Season",
        "lap_number": "Lap number", "tyre_age": "Tyre age",
        "compound": "Tyre compound", "avg_deg_per_lap": "Deg rate",
        "deg_rate": "Deg rate", "cliff_lap": "Cliff lap",
        "max_viable_laps": "Max viable laps", "gap_behind": "Gap behind",
        "track_temp": "Track temp",
    }
    lines = []
    for key, label in labels.items():
        if key in facts and facts[key] is not None:
            lines.append(f"{label}: {facts[key]}")
    if "gap_behind" not in facts:
        lines.append("Gap behind: unavailable")
    if "track_temp" not in facts:
        lines.append("Track temp: unavailable")
    return "\n".join(lines)


def _parse_response(response: str) -> tuple[str, str] | None:
    response = response.strip()
    if response.startswith("```"):
        response = re.sub(r"^```(?:json)?\s*", "", response, flags=re.IGNORECASE)
        response = re.sub(r"\s*```$", "", response)
    try:
        payload = json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            payload = json.loads(match.group(0))
        else:
            user_match = re.search(r"(?:user|driver|situation)\s*:\s*(.+?)(?:\n\s*(?:assistant|engineer|recommendation)\s*:)", response, re.IGNORECASE | re.DOTALL)
            assistant_match = re.search(r"(?:assistant|engineer|recommendation)\s*:\s*(.+)$", response, re.IGNORECASE | re.DOTALL)
            if not user_match or not assistant_match:
                return None
            payload = {
                "user": user_match.group(1).strip().strip('"'),
                "assistant": assistant_match.group(1).strip().strip('"'),
            }
    user = payload.get("user")
    assistant = payload.get("assistant")
    if not isinstance(user, str) or not isinstance(assistant, str):
        return None
    user_lower = user.lower()
    has_required = (
        "lap" in user_lower
        and ("tyre" in user_lower or "tire" in user_lower)
        and "gap" in user_lower
        and ("track temp" in user_lower or "track temperature" in user_lower)
    )
    if not has_required:
        return None
    return user.strip(), assistant.strip()


def regenerate_file(input_path: Path, output_path: Path) -> int:
    examples = _read_jsonl(input_path)
    client = _init_groq()
    rewritten = 0
    for idx, ex in enumerate(examples):
        if len(ex.get("messages", [])) <= 3:
            continue
        source = _source_text(ex)
        facts = _augment_from_db(_extract_basic_facts(source))
        prompt = PROMPT.format(facts=_facts_block(facts), source=source)
        try:
            response = _call_groq(client, prompt)
        except DailyQuotaExceeded:
            log.warning("Daily quota hit after rewriting %d examples", rewritten)
            break
        if response is None:
            log.warning("Example %d skipped: Groq returned no response", idx + 1)
            continue
        parsed = _parse_response(response)
        if parsed is None:
            log.warning("Example %d skipped: invalid rewrite format", idx + 1)
            continue
        user, assistant = parsed
        examples[idx] = {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
        rewritten += 1
        log.info("Rewritten multi-turn example %d (%d total)", idx + 1, rewritten)

    _write_jsonl(output_path, examples)
    log.info("Wrote %s with %d rewrites", output_path, rewritten)
    return rewritten


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate multi-turn JSONL examples via Groq")
    parser.add_argument("--input", type=Path, default=TRAIN_PATH)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    output = args.output or args.input.with_name(args.input.stem + "_singleturn.jsonl")
    regenerate_file(args.input, output)


if __name__ == "__main__":
    main()
