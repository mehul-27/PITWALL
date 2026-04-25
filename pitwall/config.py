"""
PitWall project configuration.
All constants, paths, and settings used across the pipeline.
No hardcoded secrets -- all API keys go in .env.
"""

import os
from pathlib import Path

# -- Project root --------------------------------------------------------------
BASE_DIR = Path(__file__).parent


def _load_env_file(path: Path) -> None:
    """Minimal .env reader when python-dotenv is not installed (KEY=VAL lines only)."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].strip()
        key, _, value = line.partition("=")
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] in "\"'" and value[0] == value[-1]:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


try:
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env", override=False)
except ModuleNotFoundError:
    _load_env_file(BASE_DIR / ".env")

# -- Paths ---------------------------------------------------------------------
CACHE_DIR    = BASE_DIR / 'data' / 'cache'
DB_PATH      = BASE_DIR / 'data' / 'pitwall.db'
RAW_DIR      = BASE_DIR / 'data' / 'raw'
DATASET_DIR  = BASE_DIR / 'data' / 'dataset'
TRAIN_PATH   = DATASET_DIR / 'train.jsonl'
VAL_PATH     = DATASET_DIR / 'val.jsonl'
TEST_PATH    = DATASET_DIR / 'test.jsonl'

# -- Seasons -------------------------------------------------------------------
# 2020-2021 dropped: budget-cap era (2022+) has better data consistency and
# avoids downloading extra seasons of cache.
SEASONS = list(range(2022, 2026))  # 2022-2025 inclusive

# -- Time delta sign (seconds) vs reference -----------------------------------
# All PitWall *lap / sector* deltas use the same rule:
#   + = slower than the stated reference;  − = faster than the reference.
# driver_sector_stats: reference = the session’s fastest full lap in qualifying
#   (minimum lap time among all valid Q laps, then that lap’s sector splits).
# Live SQL telemetry: reference is given explicitly in the fetched block
#   (session fastest lap, or field average of per-driver means when “average” mode).
# corner_stats.delta_vs_field is a *speed* offset in km/h, not a lap-time delta—see
# CORNER_SPEED_DELTA_CONVENTION (sign meaning differs from time).
SECTOR_TIME_DELTA_CONVENTION_BANNER = "Lap-time sector deltas (seconds) vs reference:"
TIME_DELTA_SIGN_RULE = (
    "Positive = slower than the reference in that sector; "
    "negative = faster than the reference in that sector."
)
CORNER_SPEED_DELTA_CONVENTION = (
    "Corner delta_vs_field (km/h) = this driver’s mean minimum speed through the corner "
    "minus the field mean. Positive = higher min speed than the field average; "
    "this is a speed margin, not the lap-time sign convention (positive = slower) above."
)
LLM_TIME_DELTA_FOLLOW = (
    "When the prompt includes a 'SECTOR TIME DELTAS' block, the plain-English "
    "'FASTER' / 'SLOWER' lines and the sign on the number are the source of truth—do not reinterpret."
)
LLM_TELEMETRY_ANSWER_BODY = (
    "When a lap table (Lap, LapTime, S1–S3) is present, your answer must *quote those numbers* "
    "— each driver's lap time, the overall gap, and a brief sector readout where relevant. "
    "A single-sentence paraphrase of the user's question with no times is an invalid answer."
)

# -- Session types -------------------------------------------------------------
# FP2 dropped: Q + Race cover all stat categories needed for fine-tuning.
#   Q    -> sector times, qualifying pace, quali/race delta
#   Race -> tyre deg, fuel effect, corner speeds, race pace
SESSION_TYPES = ['Q', 'Race']

# Sessions for which full car telemetry is downloaded and stored.
# Q is excluded -- we only need Q lap times, not car data.
# This keeps Q cache files ~85% smaller (no car-data download).
TELEMETRY_SESSIONS = ['Q', 'Race']

# -- All circuits on the F1 calendar 2020-2025 ---------------------------------
# Names match FastF1 event location strings accepted by fastf1.get_event().
ALL_CIRCUITS = [
    '70th Anniversary',   # 2020 only - Silverstone (second race)
    'Abu Dhabi',          # Yas Marina
    'Australia',          # Albert Park
    'Austria',            # Red Bull Ring
    'Azerbaijan',         # Baku City Circuit
    'Bahrain',            # Bahrain International Circuit (mixed)
    'Belgium',            # Spa-Francorchamps
    'Canada',             # Circuit Gilles Villeneuve
    'China',              # Shanghai - 2024+
    'Eifel',              # 2020 only - Nurburgring
    'Emilia Romagna',     # Imola
    'France',             # 2021-2022 - Paul Ricard
    'Great Britain',      # Silverstone
    'Hungary',            # Hungaroring
    'Italy',              # Monza (high speed)
    'Japan',              # Suzuka
    'Las Vegas',          # Las Vegas Strip - 2023+
    'Mexico City',        # Autodromo Hermanos Rodriguez
    'Miami',              # Miami International Autodrome - 2022+
    'Monaco',             # Circuit de Monaco (street)
    'Netherlands',        # Zandvoort - 2021+
    'Portugal',           # Algarve/Portimao - 2020-2021
    'Qatar',              # Losail - 2021 & 2023+
    'Russia',             # Sochi Autodrom - 2020-2021
    'Sakhir',             # 2020 only - Bahrain outer circuit
    'Saudi Arabia',       # Jeddah Corniche Circuit - 2021+
    'Singapore',          # Marina Bay Street Circuit
    'Spain',              # Circuit de Barcelona-Catalunya
    'Sao Paulo',          # Interlagos (also known as Brazil)
    'Styria',             # 2020-2021 - Red Bull Ring (second race)
    'Turkey',             # Istanbul Park - 2020-2021
    'Tuscany',            # 2020 only - Mugello
    'United States',      # Circuit of the Americas
]

# -- All driver codes who raced 2020-2025 --------------------------------------
# Three-letter codes as used by FastF1. Includes mid-season replacements.
ALL_DRIVERS = [
    # Multi-season regulars
    'ALB',  # Alexander Albon
    'ALO',  # Fernando Alonso
    'ANT',  # Andrea Kimi Antonelli (2025+)
    'BEA',  # Oliver Bearman (2024 replacement, 2025+)
    'BOT',  # Valtteri Bottas
    'COL',  # Franco Colapinto (2024)
    'DEV',  # Nyck de Vries (2023)
    'DOO',  # Jack Doohan (2025)
    'GAS',  # Pierre Gasly
    'GIO',  # Antonio Giovinazzi (2020-2021)
    'GRO',  # Romain Grosjean (2020)
    'HAD',  # Isack Hadjar (2025+)
    'HAM',  # Lewis Hamilton
    'HUL',  # Nico Hulkenberg
    'KVY',  # Daniil Kvyat (2020)
    'LAT',  # Nicholas Latifi (2020-2022)
    'LAW',  # Liam Lawson (2023 replacement, 2025+)
    'LEC',  # Charles Leclerc
    'MAG',  # Kevin Magnussen
    'MAZ',  # Nikita Mazepin (2021-2022)
    'MSC',  # Mick Schumacher (2021-2022)
    'NOR',  # Lando Norris
    'OCO',  # Esteban Ocon
    'PER',  # Sergio Perez
    'PIA',  # Oscar Piastri (2023+)
    'RAI',  # Kimi Raikkonen (2020-2021)
    'RIC',  # Daniel Ricciardo
    'RUS',  # George Russell
    'SAI',  # Carlos Sainz
    'SAR',  # Logan Sargeant (2023-2024)
    'STR',  # Lance Stroll
    'TSU',  # Yuki Tsunoda (2021+)
    'VER',  # Max Verstappen
    'VET',  # Sebastian Vettel (2020-2022)
    'ZHO',  # Guanyu Zhou (2022-2024)
    # One-race / short-stint replacements
    'AIT',  # Jack Aitken (2020 Williams replacement)
    'FIT',  # Pietro Fittipaldi (2020 Haas replacement)
    'KUB',  # Robert Kubica (2021 Alfa Romeo replacement)
]

# -- Dataset split -------------------------------------------------------------
# Split strictly by circuit -- never by driver, season, or random shuffle.
# Test circuits decided upfront and never changed.
TEST_CIRCUITS = [
    'Italy',    # Monza  - high speed archetype
    'Monaco',   # Monaco - street circuit archetype
    'Bahrain',  # Bahrain - mixed circuit archetype
]

VAL_CIRCUITS: list[str] = [
    'Belgium',     # Spa - high speed archetype
    'Singapore',   # Marina Bay - street circuit archetype
    'Hungary',     # Hungaroring - high downforce / mixed archetype
]

TRAIN_CIRCUITS = [
    c for c in ALL_CIRCUITS
    if c not in TEST_CIRCUITS and c not in VAL_CIRCUITS
]

# -- Groq API settings ---------------------------------------------------------
GROQ_MODEL              = 'llama-3.1-8b-instant'
GROQ_RPM                = 30
GROQ_TPM                = 6000
GROQ_RATE_LIMIT_SLEEP   = 2   # seconds between calls; stays comfortably under 30 RPM

# -- Model & inference settings ------------------------------------------------
BASE_MODEL_ID  = 'meta-llama/Llama-3.2-3B-Instruct'
# Slightly lower than default 350 = shorter, faster tail latency; raise if answers truncate.
MAX_NEW_TOKENS = 256
# All local inference (chat, evaluate, baseline) must use 0.1 — set explicitly, do not rely on model defaults.
INFERENCE_TEMPERATURE = 0.1
TEMPERATURE = INFERENCE_TEMPERATURE

# Inference backend:
#   ollama     -> current local REST-based flow
#   local_qwen -> Transformers + PEFT loading from local disk, fully offline
INFERENCE_BACKEND = os.getenv('PITWALL_INFERENCE_BACKEND', 'ollama').strip().lower()

# -- Local Qwen inference ------------------------------------------------------
# These paths should point to local, already-downloaded files.
# Example:
#   PITWALL_QWEN_BASE_MODEL_PATH=C:\models\Qwen2.5-3B-Instruct
#   PITWALL_QWEN_ADAPTER_PATH=C:\models\pitwall-qwen25-3b-adapter
QWEN_BASE_MODEL_PATH = os.getenv('PITWALL_QWEN_BASE_MODEL_PATH', '').strip()
QWEN_ADAPTER_PATH    = os.getenv('PITWALL_QWEN_ADAPTER_PATH', '').strip()
QWEN_LOCAL_DTYPE     = os.getenv('PITWALL_QWEN_DTYPE', 'auto').strip().lower()
QWEN_LOAD_IN_4BIT    = os.getenv('PITWALL_QWEN_LOAD_IN_4BIT', '1').strip() not in {'0', 'false', 'False'}

# -- Ollama inference (used when running locally on Windows) --------------------
# Change OLLAMA_MODEL to 'pitwall' once the fine-tuned GGUF is loaded into Ollama
OLLAMA_MODEL   = 'llama3.2:3b'
OLLAMA_URL     = 'http://localhost:11434'

# -- System prompt -------------------------------------------------------------
# Short form: use in pipeline, training JSONL generation, evaluate / baseline so
# metrics and dataset system lines stay aligned.
SYSTEM_PROMPT = (
    "You are an F1 race engineer with deep knowledge of tyre behaviour, race strategy, "
    "driving mechanics, and circuit characteristics across the 2022-2025 seasons. "
    "Give specific, data-grounded answers. Never guess -- if you don't know, say so. "
    "For any lap/sector *time* delta in the provided data, use the sign rule given there "
    "(positive = slower than the stated reference unless explicitly described otherwise; "
    "corner speed margins in km/h are labelled separately and are not lap-time signs)."
)

# -- PitWall chat: split so general (strategy / compounds) is not given telemetry-only rules.
CHAT_FORMAT_AND_TONE = (
    "Format every reply for readability. Start with one short summary line, then a blank line, then details. "
    "Use short paragraphs and put a blank line between distinct ideas. "
    "For several facts or numbers, use markdown bullet lists (each line starting with '- '). "
    "You may use **bold** for key terms sparingly. "
    "For longer or multi-part answers, use this structure (markdown headings on their own line): "
    "**Summary** then your overview; then **Facts** with bullets or tight paragraphs; "
    "then **Notes** only for caveats, uncertainty, or what is not in the data. "
    "Omit empty sections. Keep short answers to one or two short paragraphs when the question is simple. "
)

CHAT_BASE_CONTEXT_RULES = (
    "Context: The application may attach a '--- TELEMETRY DATA ---' block in this same prompt, or it may not. "
    "If you do *not* see that block, you are in general race-engineer mode: answer strategy, pit calls, and "
    "starting-grip / compound trade-offs from normal F1 knowledge. Do not say you are 'limited to the numbers "
    "in telemetry' or that you must 'use only telemetry'—that applies only when a telemetry block is actually "
    "present. Do not paraphrase the user’s question as your entire answer. "
    "Do not invent specific session lap times, sector splits, or speed-trap km/h as if from our database. "
    "If the user is wrong, correct them. If you are uncertain, say so. Never self-label a data source. "
    "Sources are set by the application, not the model."
)

# App always starts from this; TELEMETRY_RULES is appended only when a block is joined.
CHAT_SYSTEM_PROMPT_BASE = (
    SYSTEM_PROMPT + " " + CHAT_FORMAT_AND_TONE + " " + CHAT_BASE_CONTEXT_RULES
)

# Injected in app.py only with live SQL context.
CHAT_SYSTEM_TELEMETRY_RULES = (
    "A telemetry data block is attached below. For those numbers only: use the exact values—no substitutes "
    "from memory. If asked to verify, re-check the block. "
    "For lap/sector *time* deltas, follow the sign rule stated in the block (positive = slower vs the reference). "
    "In head-to-head lines, the block lists the only driver codes in scope; do not add a third driver. "
    "For corner speed 'delta' in km/h, that is a speed margin, not a lap-time sign—see the block wording."
)

# Backward compatibility for imports expecting one symbol.
CHAT_SYSTEM_PROMPT = CHAT_SYSTEM_PROMPT_BASE
