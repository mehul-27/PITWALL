"""
Phase 4 -- Groq-based dataset generation.

Method 2 — Groq formats facts  (~800 examples)
  Pass computed stats explicitly; Groq writes engineer-style responses.
  Question types:
    1. Tyre behaviour deep dives
    2. Undercut/overcut strategy per circuit
    3. Driver characteristic analysis
    4. Circuit-specific strategy overview
    5. Driver head-to-head at a circuit

Method 3 — Multi-turn race scenarios  (~400 examples)
  Query real race moments (lap, tyre age, compound, gap, pace) from SQLite.
  Groq generates a 4-6 turn pit-wall dialogue grounded only in those facts.

Total target: ~1200 examples -> data/dataset/groq.jsonl

Checkpoint: data/raw/groq_checkpoint.json  (resumes after crash)
Rate limit: time.sleep(2) between every API call  (<= 30 RPM)
Hallucination guard: discard any response that introduces numbers not
  present in the FACTS block passed to the model.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any


def _load_env() -> None:
    """
    Read key=value pairs from pitwall/.env without requiring python-dotenv.
    Only sets variables that are not already in the environment.
    """
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        return
    with env_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key   = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_env()

# -- project imports (after env is loaded) -------------------------------------
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import (
    DATASET_DIR, DB_PATH,
    GROQ_MODEL, GROQ_RATE_LIMIT_SLEEP,
    SYSTEM_PROMPT,
)
from pipeline.db import managed_connection

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

M2_TARGET   = 800   # Method 2 examples
M3_TARGET   = 400   # Method 3 examples

# Cap per question-type bucket so one type doesn't dominate
M2_PER_TYPE = M2_TARGET // 5   # 160 per type × 5 types = 800

# Checkpoint and output paths
RAW_DIR         = DB_PATH.parent.parent / "data" / "raw"
CHECKPOINT_PATH = RAW_DIR / "groq_checkpoint.json"
OUTPUT_PATH     = DATASET_DIR / "groq.jsonl"

# ── Groq client setup ──────────────────────────────────────────────────────────

def _init_groq():
    """Initialise the Groq client. Raises if key missing or SDK not installed."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY not found in environment. "
            "Add GROQ_API_KEY=<your_key> to your .env file."
        )
    try:
        import groq as groq_lib
        client = groq_lib.Groq(api_key=api_key)
        log.info("Groq client ready — model: %s", GROQ_MODEL)
        return client
    except ImportError as exc:
        raise ImportError(
            "groq package not installed. Run: pip install groq"
        ) from exc


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _load_checkpoint() -> dict[str, Any]:
    if CHECKPOINT_PATH.exists():
        with CHECKPOINT_PATH.open(encoding="utf-8") as f:
            return json.load(f)
    return {"m2_done": {}, "m3_done": 0, "total_discarded": 0}


def _save_checkpoint(cp: dict[str, Any]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    with CHECKPOINT_PATH.open("w", encoding="utf-8") as f:
        json.dump(cp, f, indent=2)


# ── Hallucination guard (consistent with training/evaluate.py) ─────────────────

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
_LAPTIME_RE = re.compile(r"\b(\d{1,2}):(\d{2}\.\d+)\b")
_YEAR_RANGE = set(range(2018, 2028))
_ORDINAL_RE = re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.IGNORECASE)
_HALLUC_TOL = 0.15   # ±15% relative tolerance
_HALLUC_ABS = 1.0    # ±1.0 absolute tolerance


def _ns(val, fmt: str) -> str:
    """None-safe formatter: returns the formatted value or 'data unavailable'."""
    return f"{val:{fmt}}" if val is not None else "data unavailable"


def _ns_lap(seconds) -> str:
    """Format a lap-time float (seconds) as M:SS.mmm, or 'data unavailable'."""
    if seconds is None:
        return "data unavailable"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m}:{s:06.3f}"


def _clean_response(text: str) -> str:
    """
    Strip outer quote characters and escaped quotes that some LLMs wrap
    their responses in, so the saved assistant content is always plain text.
    """
    text = text.strip()
    # Remove surrounding double-quote wrapper (e.g. "\"response here\"")
    if text.startswith('"') and text.endswith('"') and len(text) > 1:
        text = text[1:-1]
    # Remove escaped quotes left after the above strip
    text = text.replace('\\"', '"').replace("\\'", "'")
    return text.strip()


def _extract_numbers(text: str) -> set[float]:
    """Extract meaningful numeric values from text (consistent with evaluate.py).

    - Converts lap times (M:SS.sss) to total seconds.
    - Filters out season years (2018-2027) and ordinal rankings.
    """
    def _laptime_to_secs(m: re.Match) -> str:
        mins = int(m.group(1))
        secs = float(m.group(2))
        return f"{mins * 60 + secs:.3f}"

    normalised = _LAPTIME_RE.sub(_laptime_to_secs, text)

    ordinals: set[str] = set()
    for m in _ORDINAL_RE.finditer(text):
        ordinals.add(m.group(1))

    result: set[float] = set()
    for tok in _NUM_RE.findall(normalised):
        val = float(tok)
        if val == int(val) and int(val) in _YEAR_RANGE:
            continue
        if tok in ordinals:
            continue
        result.add(val)
    return result


def _numbers_close(pred: float, ref: float) -> bool:
    """True if pred is acceptably close to ref (±15% rel OR ±1.0 abs)."""
    if ref == 0:
        return abs(pred) < max(0.001, _HALLUC_ABS)
    if abs(pred - ref) / abs(ref) <= _HALLUC_TOL:
        return True
    if abs(pred - ref) <= _HALLUC_ABS:
        return True
    return False


def _is_hallucination(facts: str, response: str) -> bool:
    """
    Return True if the response contains any numeric value that is
    not within tolerance of any number in the facts string.
    Uses ±15% relative OR ±1.0 absolute tolerance (whichever is more
    generous), consistent with the evaluation metric in training/evaluate.py.
    """
    fact_nums = _extract_numbers(facts)
    resp_nums = _extract_numbers(response)
    for p in resp_nums:
        grounded = any(_numbers_close(p, r) for r in fact_nums)
        if not grounded:
            log.debug("Hallucinated number: %s (not near any fact number)", p)
            return True
    return False


# ── Daily quota sentinel ───────────────────────────────────────────────────────

class DailyQuotaExceeded(Exception):
    """Raised when the daily free-tier request or token limit is hit."""


# ── Groq call wrapper ──────────────────────────────────────────────────────────

def _call_groq(client, prompt: str, _rpm_retries: int = 3) -> str | None:
    """
    Call Groq with rate-limiting and smart 429 handling.

    - Sleeps GROQ_RATE_LIMIT_SLEEP seconds before every call (RPM guard).
    - On 429 RPM (per-minute) quota: sleeps 60 seconds, retries up to
      _rpm_retries times.
    - On 429 daily quota (tokens_per_day): raises DailyQuotaExceeded so the
      caller exits cleanly — no point retrying until the quota resets.
    - All other errors: logs warning and returns None.
    """
    import groq as groq_lib

    time.sleep(GROQ_RATE_LIMIT_SLEEP)
    for attempt in range(1, _rpm_retries + 2):
        try:
            response = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                temperature=0.3,
            )
            return _clean_response(response.choices[0].message.content)
        except groq_lib.RateLimitError as exc:
            msg = str(exc)
            # Daily token quota — no point retrying until tomorrow
            if "tokens_per_day" in msg or "per_day" in msg or "daily" in msg.lower():
                raise DailyQuotaExceeded(
                    "Daily free-tier token quota exhausted. "
                    "Run again tomorrow."
                ) from exc
            # Per-minute rate limit — sleep 60s and retry
            if attempt <= _rpm_retries:
                log.warning(
                    "RPM quota hit — sleeping 60s then retry %d/%d",
                    attempt, _rpm_retries,
                )
                time.sleep(60)
                continue
            log.warning("Groq RPM quota exceeded after %d retries", _rpm_retries)
            return None
        except Exception as exc:
            log.warning("Groq API error: %s", exc)
            return None
    return None


# ── Prompt builders ────────────────────────────────────────────────────────────

# Four opening styles; one is chosen at random per call to prevent
# all responses having the same structure.
_M2_OPENING_STYLES = [
    "Write a direct race engineer response to this question:",
    "Write a conversational race engineer response to this question:",
    "Write a brief, technical race engineer response to this question:",
    "Write a detailed race engineer debrief response to this question:",
]

_M2_PROMPT_TEMPLATE = """\
You are formatting F1 engineering knowledge into natural conversation.
Use ONLY these exact statistics — do not add any numbers or facts not in this list:

FACTS:
{facts}

{opening}
{question}

RULES:
- Use only the facts provided above
- Sound like a real race engineer (direct, technical, concise)
- Be specific and reference the numbers provided
- 3-5 sentences maximum
- Do not invent any numbers
- Do not add caveats like "based on the data provided"
"""

_M3_PROMPT_TEMPLATE = """\
You are generating a realistic F1 pit-wall radio conversation.
Use ONLY these exact statistics — do not add any numbers or facts not listed:

RACE FACTS:
{facts}

Generate a 4-6 turn conversation between a race engineer (ENG) and a driver (DRV).
The conversation should be about the strategic situation described in the facts:
current lap, tyre age, compound, pace, and whether to pit.

RULES:
- Use only the numbers provided in RACE FACTS
- Each turn is 1-2 sentences maximum
- Sound like a real F1 pit-wall radio exchange
- ENG starts, DRV replies, alternate turns
- End with a clear decision (pit / stay out)
- Do not invent lap counts, gaps, or temperatures not in the facts

Format exactly as:
ENG: <message>
DRV: <message>
ENG: <message>
...
"""


# ── Method 2 — fact queries ────────────────────────────────────────────────────

def _m2_tyre_behaviour(conn) -> list[dict]:
    """Q-type 1: Tyre behaviour deep dive (compound × circuit × season)."""
    rows = conn.execute(
        """
        SELECT circuit, compound, season,
               avg_deg_per_lap, cliff_lap, max_viable_laps, track_temp_sensitivity
        FROM tyre_stats
        WHERE avg_deg_per_lap IS NOT NULL AND max_viable_laps IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 200
        """
    ).fetchall()
    out = []
    for r in rows:
        compound = r["compound"].capitalize()
        facts_parts = [
            f"Circuit: {r['circuit']}",
            f"Season: {r['season']}",
            f"Compound: {compound}",
            f"Average degradation rate: {r['avg_deg_per_lap']:.4f} seconds per lap",
            f"Maximum viable stint length: {r['max_viable_laps']} laps",
        ]
        if r["cliff_lap"]:
            facts_parts.append(f"Pace cliff lap: {r['cliff_lap']}")
        if r["track_temp_sensitivity"] is not None:
            facts_parts.append(
                f"Track temperature sensitivity: {r['track_temp_sensitivity']:.4f} s/°C"
            )
        facts = "\n".join(facts_parts)
        question = (
            f"How do {compound} tyres behave at {r['circuit']} in {r['season']}? "
            f"Include degradation profile, stint strategy, and any temperature sensitivity."
        )
        out.append({"question": question, "facts": facts, "qtype": "tyre_behaviour"})
    return out


def _m2_undercut_strategy(conn) -> list[dict]:
    """Q-type 2: Undercut/overcut strategy per circuit."""
    rows = conn.execute(
        """
        SELECT ts.circuit, ts.season, ts.compound,
               ts.avg_deg_per_lap, ts.max_viable_laps, ts.cliff_lap,
               fs.fuel_effect_per_10kg,
               qr.avg_race_pace
        FROM tyre_stats ts
        LEFT JOIN fuel_stats fs
          ON fs.circuit = ts.circuit AND fs.season = ts.season
        LEFT JOIN quali_race_delta qr
          ON qr.circuit = ts.circuit AND qr.season = ts.season
        WHERE ts.avg_deg_per_lap IS NOT NULL
          AND ts.max_viable_laps IS NOT NULL
        GROUP BY ts.circuit, ts.season, ts.compound
        ORDER BY RANDOM()
        LIMIT 200
        """
    ).fetchall()
    out = []
    for r in rows:
        compound = r["compound"].capitalize()
        facts_parts = [
            f"Circuit: {r['circuit']}",
            f"Season: {r['season']}",
            f"Compound: {compound}",
            f"Tyre degradation rate: {r['avg_deg_per_lap']:.4f} s/lap",
            f"Max viable stint: {r['max_viable_laps']} laps",
        ]
        if r["cliff_lap"]:
            facts_parts.append(f"Cliff lap: {r['cliff_lap']}")
        if r["fuel_effect_per_10kg"] is not None:
            facts_parts.append(
                f"Fuel effect per 10 kg: {r['fuel_effect_per_10kg']:.3f} s/lap"
            )
        if r["avg_race_pace"] is not None:
            facts_parts.append(f"Average race pace: {_ns_lap(r['avg_race_pace'])}")
        facts = "\n".join(facts_parts)
        question = (
            f"What is the undercut and overcut opportunity at {r['circuit']} in "
            f"{r['season']} on {compound} tyres? When should a team trigger the undercut?"
        )
        out.append({"question": question, "facts": facts, "qtype": "undercut_strategy"})
    return out


def _m2_driver_characteristics(conn) -> list[dict]:
    """Q-type 3: Driver characteristic analysis at a circuit."""
    rows = conn.execute(
        """
        SELECT d.driver, d.circuit, d.season,
               d.avg_sector1_delta, d.avg_sector2_delta, d.avg_sector3_delta,
               s.avg_top_speed, s.rank_in_field,
               qr.quali_lap, qr.avg_race_pace, qr.delta
        FROM driver_sector_stats d
        LEFT JOIN speed_trap_stats s
          ON s.driver = d.driver AND s.circuit = d.circuit AND s.season = d.season
        LEFT JOIN quali_race_delta qr
          ON qr.driver = d.driver AND qr.circuit = d.circuit AND qr.season = d.season
        WHERE d.avg_sector1_delta IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 200
        """
    ).fetchall()
    _DN = {
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
        "ZHO": "Guanyu Zhou",
    }
    out = []
    for r in rows:
        name = _DN.get(r["driver"], r["driver"])
        facts_parts = [
            f"Driver: {name}",
            f"Circuit: {r['circuit']}",
            f"Season: {r['season']}",
            f"Sector 1 delta vs fastest: {_ns(r['avg_sector1_delta'], '+.3f')}s",
            f"Sector 2 delta vs fastest: {_ns(r['avg_sector2_delta'], '+.3f')}s",
            f"Sector 3 delta vs fastest: {_ns(r['avg_sector3_delta'], '+.3f')}s",
        ]
        if r["avg_top_speed"] is not None:
            facts_parts.append(
                f"Average top speed (speed trap): {r['avg_top_speed']:.1f} km/h "
                f"(rank {r['rank_in_field']} in field)"
            )
        if r["quali_lap"] is not None:
            facts_parts.append(f"Best qualifying lap: {_ns_lap(r['quali_lap'])}")
        if r["avg_race_pace"] is not None:
            facts_parts.append(f"Average race pace: {_ns_lap(r['avg_race_pace'])}")
        if r["delta"] is not None:
            facts_parts.append(f"Qualifying-to-race delta: {r['delta']:+.3f}s")
        facts = "\n".join(facts_parts)
        question = (
            f"Describe {name}'s driving characteristics and strengths at "
            f"{r['circuit']} in {r['season']}. Where does he gain and lose time?"
        )
        out.append({"question": question, "facts": facts, "qtype": "driver_characteristics"})
    return out


def _m2_circuit_strategy(conn) -> list[dict]:
    """Q-type 4: Circuit-specific race strategy overview (all compounds)."""
    rows = conn.execute(
        """
        SELECT ts.circuit, ts.season,
               GROUP_CONCAT(ts.compound || ':' ||
                   ROUND(ts.avg_deg_per_lap, 4) || ':' ||
                   COALESCE(ts.max_viable_laps, 'N/A'),
                   ' | ')  AS compounds_summary,
               MIN(ts.avg_deg_per_lap) AS min_deg,
               AVG(fs.fuel_effect_per_10kg) AS avg_fuel_effect,
               MIN(qr.avg_race_pace) AS fastest_race_pace
        FROM tyre_stats ts
        LEFT JOIN fuel_stats fs
          ON fs.circuit = ts.circuit AND fs.season = ts.season
        LEFT JOIN quali_race_delta qr
          ON qr.circuit = ts.circuit AND qr.season = ts.season
        WHERE ts.avg_deg_per_lap IS NOT NULL
        GROUP BY ts.circuit, ts.season
        ORDER BY RANDOM()
        LIMIT 200
        """
    ).fetchall()
    out = []
    for r in rows:
        facts_parts = [
            f"Circuit: {r['circuit']}",
            f"Season: {r['season']}",
            f"Tyre compounds (compound:deg_s_per_lap:max_viable_laps): {r['compounds_summary']}",
        ]
        if r["avg_fuel_effect"] is not None:
            facts_parts.append(
                f"Average fuel effect per 10 kg: {r['avg_fuel_effect']:.3f} s/lap"
            )
        if r["fastest_race_pace"] is not None:
            facts_parts.append(f"Fastest average race pace in field: {_ns_lap(r['fastest_race_pace'])}")
        facts = "\n".join(facts_parts)
        question = (
            f"What does an optimal race strategy look like at {r['circuit']} in "
            f"{r['season']}? Consider compound choices, stint lengths, and fuel effect."
        )
        out.append({"question": question, "facts": facts, "qtype": "circuit_strategy"})
    return out


def _m2_driver_comparison(conn) -> list[dict]:
    """Q-type 5: Head-to-head driver comparison at same circuit/season."""
    rows = conn.execute(
        """
        SELECT a.driver AS d1, b.driver AS d2,
               a.circuit, a.season,
               a.avg_sector1_delta AS a_s1, a.avg_sector2_delta AS a_s2, a.avg_sector3_delta AS a_s3,
               b.avg_sector1_delta AS b_s1, b.avg_sector2_delta AS b_s2, b.avg_sector3_delta AS b_s3,
               sa.avg_top_speed AS a_speed, sa.rank_in_field AS a_rank,
               sb.avg_top_speed AS b_speed, sb.rank_in_field AS b_rank
        FROM driver_sector_stats a
        JOIN driver_sector_stats b
          ON b.circuit = a.circuit AND b.season = a.season AND b.driver > a.driver
        LEFT JOIN speed_trap_stats sa
          ON sa.driver = a.driver AND sa.circuit = a.circuit AND sa.season = a.season
        LEFT JOIN speed_trap_stats sb
          ON sb.driver = b.driver AND sb.circuit = b.circuit AND sb.season = b.season
        WHERE a.avg_sector1_delta IS NOT NULL
          AND b.avg_sector1_delta IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 200
        """
    ).fetchall()
    _DN = {
        "ALB": "Alexander Albon",    "ALO": "Fernando Alonso",
        "ANT": "Kimi Antonelli",     "BOT": "Valtteri Bottas",
        "GAS": "Pierre Gasly",       "HAM": "Lewis Hamilton",
        "HUL": "Nico Hulkenberg",    "LAW": "Liam Lawson",
        "LEC": "Charles Leclerc",    "MAG": "Kevin Magnussen",
        "NOR": "Lando Norris",       "OCO": "Esteban Ocon",
        "PER": "Sergio Perez",       "PIA": "Oscar Piastri",
        "RIC": "Daniel Ricciardo",   "RUS": "George Russell",
        "SAI": "Carlos Sainz",       "STR": "Lance Stroll",
        "TSU": "Yuki Tsunoda",       "VER": "Max Verstappen",
        "VET": "Sebastian Vettel",   "ZHO": "Guanyu Zhou",
        "MSC": "Mick Schumacher",
    }
    out = []
    for r in rows:
        n1 = _DN.get(r["d1"], r["d1"])
        n2 = _DN.get(r["d2"], r["d2"])
        facts_parts = [
            f"Circuit: {r['circuit']}",
            f"Season: {r['season']}",
            (f"{n1} — S1: {_ns(r['a_s1'], '+.3f')}s"
             f"  S2: {_ns(r['a_s2'], '+.3f')}s"
             f"  S3: {_ns(r['a_s3'], '+.3f')}s"),
            (f"{n2} — S1: {_ns(r['b_s1'], '+.3f')}s"
             f"  S2: {_ns(r['b_s2'], '+.3f')}s"
             f"  S3: {_ns(r['b_s3'], '+.3f')}s"),
        ]
        if r["a_speed"] is not None:
            facts_parts.append(
                f"{n1} speed trap: {r['a_speed']:.1f} km/h (rank {r['a_rank']})"
            )
        if r["b_speed"] is not None:
            facts_parts.append(
                f"{n2} speed trap: {r['b_speed']:.1f} km/h (rank {r['b_rank']})"
            )
        facts = "\n".join(facts_parts)
        question = (
            f"How do {n1} and {n2} compare at {r['circuit']} in {r['season']}? "
            f"Who has the edge in each sector and who is stronger overall?"
        )
        out.append({"question": question, "facts": facts, "qtype": "driver_comparison"})
    return out


# ── Method 3 — multi-turn race scenario queries ────────────────────────────────

def _m3_race_moments(conn) -> list[dict]:
    """
    Query real race moments suitable for multi-turn strategy conversations.
    Each row becomes one scenario (one 4-6 turn conversation).
    """
    rows = conn.execute(
        """
        SELECT
            l.driver,
            s.circuit,
            s.season,
            l.lap_number,
            l.tyre_age,
            l.compound,
            l.lap_time,
            l.sector1,
            l.sector2,
            l.sector3,
            ts.avg_deg_per_lap,
            ts.cliff_lap,
            ts.max_viable_laps,
            fs.fuel_effect_per_10kg,
            qr.avg_race_pace
        FROM laps l
        JOIN sessions s ON s.id = l.session_id
        LEFT JOIN tyre_stats ts
          ON ts.circuit = s.circuit AND ts.season = s.season AND ts.compound = l.compound
        LEFT JOIN fuel_stats fs
          ON fs.circuit = s.circuit AND fs.season = s.season AND fs.driver = l.driver
        LEFT JOIN quali_race_delta qr
          ON qr.circuit = s.circuit AND qr.season = s.season AND qr.driver = l.driver
        WHERE s.session_type = 'Race'
          AND l.is_valid = 1
          AND l.tyre_age BETWEEN 8 AND 25
          AND l.lap_number BETWEEN 20 AND 50
          AND l.lap_time IS NOT NULL
          AND l.compound IS NOT NULL
          AND ts.avg_deg_per_lap IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 600
        """
    ).fetchall()

    _DN = {
        "ALB": "Albon",   "ALO": "Alonso",  "BOT": "Bottas",  "GAS": "Gasly",
        "HAM": "Hamilton","HUL": "Hulk",    "LAW": "Lawson",  "LEC": "Leclerc",
        "MAG": "Magnussen","NOR": "Norris",  "OCO": "Ocon",    "PER": "Perez",
        "PIA": "Piastri", "RIC": "Ricciardo","RUS": "Russell", "SAI": "Sainz",
        "STR": "Stroll",  "TSU": "Tsunoda", "VER": "Verstappen","VET": "Vettel",
        "ZHO": "Zhou",    "MSC": "Schumacher",
    }

    out = []
    for r in rows:
        lap_str = _ns_lap(r["lap_time"])

        facts_parts = [
            f"Driver: {_DN.get(r['driver'], r['driver'])}",
            f"Circuit: {r['circuit']}",
            f"Season: {r['season']}",
            f"Current lap: {r['lap_number']}",
            f"Current lap time: {lap_str}",
            f"Tyre compound: {r['compound'].capitalize()}",
            f"Tyre age: {r['tyre_age']} laps",
            f"Tyre degradation rate: {r['avg_deg_per_lap']:.4f} s/lap",
            f"Max viable tyre stint: {r['max_viable_laps']} laps",
        ]
        if r["cliff_lap"]:
            facts_parts.append(f"Tyre cliff lap: {r['cliff_lap']}")
        if r["fuel_effect_per_10kg"] is not None:
            facts_parts.append(
                f"Fuel effect: {r['fuel_effect_per_10kg']:.3f} s per 10 kg"
            )
        if r["avg_race_pace"] is not None:
            facts_parts.append(f"Driver's average race pace: {_ns_lap(r['avg_race_pace'])}")

        facts = "\n".join(facts_parts)
        out.append({
            "driver":  r["driver"],
            "circuit": r["circuit"],
            "season":  r["season"],
            "lap":     r["lap_number"],
            "facts":   facts,
        })
    return out


# ── Multi-turn response parser ─────────────────────────────────────────────────

def _parse_multi_turn(response: str) -> list[dict] | None:
    """
    Parse ENG:/DRV: formatted response into messages list.
    Returns None if format is invalid or fewer than 4 turns found.
    """
    messages = []
    for line in response.splitlines():
        line = line.strip()
        if line.startswith("ENG:"):
            messages.append({"role": "user", "content": line[4:].strip()})
        elif line.startswith("DRV:"):
            messages.append({"role": "assistant", "content": line[4:].strip()})
    if len(messages) < 4:
        return None
    return messages


# ── JSONL writer ───────────────────────────────────────────────────────────────

def _make_single_turn(question: str, answer: str) -> dict:
    return {
        "messages": [
            {"role": "system",    "content": SYSTEM_PROMPT},
            {"role": "user",      "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def _make_multi_turn(turns: list[dict]) -> dict:
    return {
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + turns
    }


def _append_example(path: Path, example: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(example, ensure_ascii=False) + "\n")


# ── Method 2 runner ────────────────────────────────────────────────────────────

_M2_QUERIES = [
    ("tyre_behaviour",         _m2_tyre_behaviour),
    ("undercut_strategy",      _m2_undercut_strategy),
    ("driver_characteristics", _m2_driver_characteristics),
    ("circuit_strategy",       _m2_circuit_strategy),
    ("driver_comparison",      _m2_driver_comparison),
]


def run_method2(client, conn, cp: dict[str, Any], stats: dict[str, int]) -> None:
    log.info("[Method 2] Starting — target %d examples (%d per type)", M2_TARGET, M2_PER_TYPE)

    for qtype, query_fn in _M2_QUERIES:
        done   = cp["m2_done"].get(qtype, 0)
        target = M2_PER_TYPE

        if done >= target:
            log.info("  [%s] already at %d/%d — skipping", qtype, done, target)
            continue

        log.info("  [%s] fetching candidates …", qtype)
        candidates = query_fn(conn)
        log.info("  [%s] %d candidates available", qtype, len(candidates))

        for item in candidates:
            if done >= target:
                break

            prompt = _M2_PROMPT_TEMPLATE.format(
                facts=item["facts"],
                question=item["question"],
                opening=random.choice(_M2_OPENING_STYLES),
            )
            try:
                response = _call_groq(client, prompt)
            except DailyQuotaExceeded as e:
                log.warning("Daily quota hit during [%s]: %s", qtype, e)
                log.info("Checkpoint saved — re-run tomorrow to resume.")
                return
            if response is None:
                stats["api_errors"] += 1
                continue

            if _is_hallucination(item["facts"], response):
                stats["discard_reasons"]["hallucination"] += 1
                log.info(
                    "  [%s] DISCARDED hallucination  (hallu=%d  bad_fmt=%d)",
                    qtype,
                    stats["discard_reasons"]["hallucination"],
                    stats["discard_reasons"]["bad_format"],
                )
                stats["discarded"] += 1
                cp["total_discarded"] += 1
                _save_checkpoint(cp)
                continue

            example = _make_single_turn(item["question"], response)
            _append_example(OUTPUT_PATH, example)
            done += 1
            stats["generated"] += 1
            cp["m2_done"][qtype] = done
            _save_checkpoint(cp)
            log.info(
                "  [%s] %d/%d  (total %d  discarded %d)",
                qtype, done, target, stats["generated"], stats["discarded"],
            )

        log.info("  [%s] finished — %d examples", qtype, done)


# ── Method 3 runner ────────────────────────────────────────────────────────────

def run_method3(client, conn, cp: dict[str, Any], stats: dict[str, int]) -> None:
    log.info("[Method 3] Starting — target %d multi-turn scenarios", M3_TARGET)

    done = cp["m3_done"]
    if done >= M3_TARGET:
        log.info("[Method 3] Already complete (%d/%d) — skipping", done, M3_TARGET)
        return

    scenarios = _m3_race_moments(conn)
    log.info("[Method 3] %d race moment candidates available", len(scenarios))

    for sc in scenarios:
        if done >= M3_TARGET:
            break

        prompt = _M3_PROMPT_TEMPLATE.format(facts=sc["facts"])
        try:
            response = _call_groq(client, prompt)
        except DailyQuotaExceeded as e:
            log.warning("Daily quota hit during Method 3: %s", e)
            log.info("Checkpoint saved — re-run tomorrow to resume.")
            return
        if response is None:
            stats["api_errors"] += 1
            continue

        if _is_hallucination(sc["facts"], response):
            stats["discard_reasons"]["hallucination"] += 1
            log.info(
                "[Method 3] DISCARDED hallucination  (hallu=%d  bad_fmt=%d)",
                stats["discard_reasons"]["hallucination"],
                stats["discard_reasons"]["bad_format"],
            )
            stats["discarded"] += 1
            cp["total_discarded"] += 1
            _save_checkpoint(cp)
            continue

        turns = _parse_multi_turn(response)
        if turns is None:
            stats["discard_reasons"]["bad_format"] += 1
            log.info(
                "[Method 3] DISCARDED bad format  (hallu=%d  bad_fmt=%d)",
                stats["discard_reasons"]["hallucination"],
                stats["discard_reasons"]["bad_format"],
            )
            stats["discarded"] += 1
            continue

        example = _make_multi_turn(turns)
        _append_example(OUTPUT_PATH, example)
        done += 1
        stats["generated"] += 1
        cp["m3_done"] = done
        _save_checkpoint(cp)
        log.info(
            "[Method 3] %d/%d  (total %d  discarded %d)",
            done, M3_TARGET, stats["generated"], stats["discarded"],
        )

    log.info("[Method 3] finished — %d examples", done)


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Initialise output file only if starting fresh
    cp = _load_checkpoint()
    fresh_start = cp == {"m2_done": {}, "m3_done": 0, "total_discarded": 0}
    if fresh_start and OUTPUT_PATH.exists():
        OUTPUT_PATH.unlink()
        log.info("Cleared previous output file for fresh run")

    log.info("Output: %s", OUTPUT_PATH)
    log.info("Checkpoint: %s", CHECKPOINT_PATH)

    # Count already-written examples (for resume)
    existing = 0
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(encoding="utf-8") as f:
            existing = sum(1 for _ in f)
    log.info("Resuming from %d existing examples", existing)

    client = _init_groq()

    stats: dict[str, Any] = {
        "generated":       existing,
        "discarded":       cp.get("total_discarded", 0),
        "api_errors":      0,
        "discard_reasons": {"hallucination": 0, "bad_format": 0},
    }

    with managed_connection() as conn:
        run_method2(client, conn, cp, stats)
        run_method3(client, conn, cp, stats)

    total     = stats["generated"]
    discarded = stats["discarded"]
    halluc_rate = discarded / max(1, total + discarded)
    dr = stats["discard_reasons"]

    log.info("══════════════════════════════════════════════════════")
    log.info("Examples generated  : %d", total)
    log.info("Examples discarded  : %d total", discarded)
    log.info("  — hallucination   : %d", dr["hallucination"])
    log.info("  — bad format      : %d", dr["bad_format"])
    log.info("API errors          : %d", stats["api_errors"])
    log.info("Discard rate        : %.1f%%", halluc_rate * 100)
    log.info("Output              : %s", OUTPUT_PATH)
    log.info("══════════════════════════════════════════════════════")


if __name__ == "__main__":
    main()
