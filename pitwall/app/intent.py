"""
Intent detection — classifies user query as either:
  - 'general'      → answered from model weights alone
  - 'telemetry'    → requires SQL retrieval before inference

Also extracts structured intent for telemetry queries (driver codes,
circuit names, year, session, lap number) and topic keywords for UI badges.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import ALL_CIRCUITS, ALL_DRIVERS

# ---------------------------------------------------------------------------
# Mode detection
# ---------------------------------------------------------------------------

TELEMETRY_TRIGGERS = [
    r'\b(?:lap\s*\d+|\d+(?:st|nd|rd|th)?\s*lap)\b',
    r'\b(compare|vs|versus)\b',
    r'\bsector\s*[123]\b',
    r'\b(Q1|Q2|Q3|FP1|FP2|FP3)\b',
    r'\b(fastest lap|telemetry|speed trap|mini.?sector)\b',
    r'\b(20[12][0-9])\b',
]


def detect_mode(message: str) -> str:
    """Return 'telemetry' if message matches any trigger, else 'general'."""
    for pattern in TELEMETRY_TRIGGERS:
        if re.search(pattern, message, re.IGNORECASE):
            return "telemetry"
    return "general"


# ---------------------------------------------------------------------------
# Telemetry intent parsing
# ---------------------------------------------------------------------------

# Build a lookup of lowercase circuit names/aliases → canonical name
_CIRCUIT_ALIASES: dict[str, str] = {}
for _c in ALL_CIRCUITS:
    _CIRCUIT_ALIASES[_c.lower()] = _c
    # Also add common short forms
    for _word in _c.lower().split():
        if len(_word) > 3:
            _CIRCUIT_ALIASES[_word] = _c

# Extra aliases not covered by ALL_CIRCUITS
_EXTRA_ALIASES = {
    "spa": "Belgium", "monza": "Italy", "silverstone": "Great Britain",
    "suzuka": "Japan", "baku": "Azerbaijan", "jeddah": "Saudi Arabia",
    "interlagos": "Sao Paulo", "albert park": "Australia",
    "hungaroring": "Hungary", "zandvoort": "Netherlands",
    "barcelona": "Spain", "montreal": "Canada", "cota": "United States",
    "austin": "United States", "marina bay": "Singapore",
    "yas marina": "Abu Dhabi", "miami": "Miami", "vegas": "Las Vegas",
    "las vegas": "Las Vegas", "imola": "Emilia Romagna",
    "monte carlo": "Monaco", "red bull ring": "Austria",
    "paul ricard": "France", "portimao": "Portugal",
    "mugello": "Tuscany", "nurburgring": "Eifel",
    "losail": "Qatar", "sochi": "Russia",
}
_CIRCUIT_ALIASES.update(_EXTRA_ALIASES)

# Driver code set for fast lookup
_DRIVER_CODES = {d.upper() for d in ALL_DRIVERS}

# Common full name → code mapping
_DRIVER_NAMES: dict[str, str] = {
    "verstappen": "VER", "hamilton": "HAM", "leclerc": "LEC",
    "norris": "NOR", "sainz": "SAI", "russell": "RUS",
    "piastri": "PIA", "alonso": "ALO", "stroll": "STR",
    "gasly": "GAS", "ocon": "OCO", "perez": "PER",
    "tsunoda": "TSU", "ricciardo": "RIC", "albon": "ALB",
    "bottas": "BOT", "zhou": "ZHO", "magnussen": "MAG",
    "hulkenberg": "HUL", "sargeant": "SAR", "lawson": "LAW",
    "bearman": "BEA", "antonelli": "ANT", "schumacher": "MSC",
    "vettel": "VET", "raikkonen": "RAI", "latifi": "LAT",
    "de vries": "DEV", "doohan": "DOO", "hadjar": "HAD",
    "colapinto": "COL",
}


def parse_telemetry_intent(message: str) -> dict:
    """Extract structured entities from a telemetry-mode message.

    Returns dict with keys: drivers, circuit, year, session, lap.
    Any field may be None/empty if not detected.
    """
    msg_lower = message.lower()
    result: dict = {
        "drivers": [],
        "circuit": None,
        "year": None,
        "session": None,
        "lap": None,
    }

    # ── Drivers ──────────────────────────────────────────────────────────
    # Check 3-letter codes in the message (case-insensitive)
    for token in re.findall(r'\b[A-Za-z]{3}\b', message):
        if token.upper() in _DRIVER_CODES and token.upper() not in result["drivers"]:
            result["drivers"].append(token.upper())

    # Check full last names
    for name, code in _DRIVER_NAMES.items():
        if name in msg_lower and code not in result["drivers"]:
            result["drivers"].append(code)

    # ── Circuit ──────────────────────────────────────────────────────────
    # Try multi-word aliases first (longer matches win)
    for alias in sorted(_CIRCUIT_ALIASES.keys(), key=len, reverse=True):
        if alias in msg_lower:
            result["circuit"] = _CIRCUIT_ALIASES[alias]
            break

    # ── Year ─────────────────────────────────────────────────────────────
    year_match = re.search(r'\b(20[12]\d)\b', message)
    if year_match:
        result["year"] = int(year_match.group(1))

    # ── Session ──────────────────────────────────────────────────────────
    session_map = {
        r'\bQ1\b': 'Q', r'\bQ2\b': 'Q', r'\bQ3\b': 'Q',
        r'\bquali': 'Q', r'\bqualifying\b': 'Q',
        r'\bFP1\b': 'FP1', r'\bFP2\b': 'FP2', r'\bFP3\b': 'FP3',
        r'\bpractice\b': 'FP2',
        r'\brace\b': 'Race', r'\bsprint\b': 'Race',
    }
    for pat, sess in session_map.items():
        if re.search(pat, message, re.IGNORECASE):
            result["session"] = sess
            break

    # ── Lap number ───────────────────────────────────────────────────────
    lap_match = re.search(r'\b(?:lap\s*(\d+)|(\d+)(?:st|nd|rd|th)?\s*lap)\b', message, re.IGNORECASE)
    if lap_match:
        result["lap"] = int(lap_match.group(1) or lap_match.group(2))
    elif re.search(r'\bfastest\s*lap\b', message, re.IGNORECASE):
        result["lap"] = "fastest"

    return result


# ---------------------------------------------------------------------------
# Topic extraction (for UI badge chips)
# ---------------------------------------------------------------------------

_TOPIC_KEYWORDS = {
    "strategy": r'\b(strategy|undercut|overcut|pit stop|stint)\b',
    "tyres": r'\b(tyre|tire|compound|soft|medium|hard|intermediate|wet|deg)\b',
    "pace": r'\b(pace|lap time|race pace|quali pace)\b',
    "sectors": r'\b(sector|s1|s2|s3)\b',
    "speed": r'\b(speed|speed trap|top speed|straight.?line)\b',
    "corners": r'\b(corner|turn|apex|braking|throttle)\b',
    "weather": r'\b(rain|wet|dry|weather|conditions)\b',
    "fuel": r'\b(fuel|fuel effect|weight)\b',
    "driver": r'\b(driver|driving|style|characteristics)\b',
}


def extract_topics(message: str) -> list[str]:
    """Extract topic keywords from message for UI badge display."""
    topics = []
    for topic, pattern in _TOPIC_KEYWORDS.items():
        if re.search(pattern, message, re.IGNORECASE):
            topics.append(topic)
    return topics[:4]  # Cap at 4 badges
