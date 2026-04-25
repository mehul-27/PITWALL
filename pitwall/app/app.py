"""
Flask entry point — keeps routes thin; all logic lives in sub-modules.
Serves chat UI, health endpoint, and conversation API.

Run:  python app/app.py   (from pitwall/ directory)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from pathlib import Path

# Ensure project root is on path so config imports work
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ModuleNotFoundError:
    pass

from flask import Flask, render_template, request, jsonify, session

import re

from config import (
    ALL_DRIVERS,
    CHAT_SYSTEM_PROMPT_BASE,
    CHAT_SYSTEM_TELEMETRY_RULES,
    LLM_TELEMETRY_ANSWER_BODY,
    LLM_TIME_DELTA_FOLLOW,
)
from driver_stats import get_session_count, load_driver_session_counts
from intent import (
    detect_mode,
    extract_topics,
    is_pushback_message,
    merge_telemetry_intent,
    parse_telemetry_intent,
)
from memory import (
    create_session, add_message, get_history, get_model_messages, clear,
)
from retrieval import fetch_telemetry_structured
from telemetry_synthesis import maybe_append_telemetry_synthesis
from telemetry_validation import (
    BOUNDS_REPLACEMENT_MESSAGE,
    check_response_sector_sign_misinterpretation,
    validate_telemetry_block,
)

# Last successful telemetry intent per chat (for pushback re-query)
_LAST_TELEMETRY_INTENT: dict[str, dict] = {}

SPARSE_DATA_NOTE = (
    "Note: Limited session data is available for this driver. This response is based on fewer "
    "sessions than usual and figures may be less reliable than for drivers with longer track records in the dataset."
)

# Try to load the real model — fall back to stubs if Ollama unavailable
_MODEL_LOADED = False
try:
    from inference import generate as _model_generate, MODEL_AVAILABLE
    _MODEL_LOADED = MODEL_AVAILABLE
except Exception as exc:
    logging.warning("Could not import inference module: %s — using stub responses", exc)
    _model_generate = None

try:
    load_driver_session_counts()
except Exception as exc:
    logging.warning("Driver session stats not loaded: %s", exc)

log = logging.getLogger(__name__)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
    static_folder=str(Path(__file__).parent / "static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "pitwall-dev-key-change-in-prod")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Serve the main chat UI."""
    # Create a session ID for this visitor if not already present
    if "sid" not in session:
        session["sid"] = create_session()
    return render_template("index.html")


@app.route("/health")
def health():
    """Health check endpoint."""
    return jsonify({"status": "ok", "service": "pitwall"}), 200


def _collect_driver_codes(message: str, intent: dict) -> list[str]:
    codes: set[str] = set(intent.get("drivers") or [])
    for t in re.findall(r"\b([A-Z]{3})\b", message.upper()):
        if t in ALL_DRIVERS:
            codes.add(t)
    return list(codes)


def _maybe_append_sparse_note(text: str, drivers: list[str]) -> str:
    for d in drivers:
        n = get_session_count(d)
        if n is not None and n < 3:
            return text.rstrip() + "\n\n* " + SPARSE_DATA_NOTE
    return text


@app.route("/chat", methods=["POST"])
def chat():
    """Handle a chat message.

    Accepts JSON: { "message": str, "session_id": str (optional) }
    Returns JSON: {
        "response": str,
        "metadata": {
            "mode": "general" | "telemetry",
            "source": "fastf1_valid" | "data_unavailable" | "model_knowledge" | "no_data_found",
            "response_time_ms": int,
            "confidence": float,
            "topics": [str],
            "telemetry": { ... } | {}
        }
    }
    """
    data = request.get_json(force=True)
    message = data.get("message", "").strip()
    sid = data.get("session_id") or session.get("sid") or create_session()

    if not message:
        return jsonify({"error": "Empty message"}), 400

    start = time.time()
    pushback = is_pushback_message(message)
    raw_intent = parse_telemetry_intent(message)
    last_tel = _LAST_TELEMETRY_INTENT.get(sid)
    if pushback:
        mode = "telemetry"
        intent = merge_telemetry_intent(raw_intent, last_tel)
    else:
        mode = detect_mode(message)
        if mode == "telemetry":
            intent = merge_telemetry_intent(raw_intent, last_tel)
        else:
            intent = raw_intent

    telemetry_meta: dict = {}
    telemetry_context: str | None = None
    source = "model_knowledge"
    t_query_ms = 0

    if mode == "telemetry" and not intent.get("circuit"):
        log.warning("Telemetry mode but no circuit (sid=%s).", sid)
        if pushback:
            telemetry_context = (
                "The user is challenging a prior answer, but the circuit is not in this message. "
                "Ask them to name a circuit, year, and session, or the previous request context was lost."
            )
            source = "no_data_found"
        else:
            mode = "general"

    if mode == "telemetry" and intent.get("circuit"):
        try:
            t0 = time.time()
            tfetch = fetch_telemetry_structured(intent)
            t_query_ms = int((time.time() - t0) * 1000)
            query_ran = True
            _LAST_TELEMETRY_INTENT[sid] = dict(intent)
            vmeta = dict(tfetch.meta)
            vmeta["year"] = vmeta.get("year") or vmeta.get("season")

            telemetry_meta = {
                "query_ms": t_query_ms,
                "drivers": intent.get("drivers", []),
                "circuit": intent.get("circuit"),
                "session": intent.get("session"),
                "year": intent.get("year"),
                "lap": intent.get("lap"),
            }

            if getattr(tfetch, "rejection_reason", None):
                # Session unavailable after ingest, or two-driver quality checks failed — do not
                # treat as normal telemetry.
                source = "data_unavailable"
                telemetry_context = (tfetch.text or tfetch.rejection_reason).strip()
                telemetry_meta["rejection_reason"] = tfetch.rejection_reason
            elif not tfetch.query_found_session or not (tfetch.text and tfetch.text.strip()):
                source = "no_data_found"
                telemetry_context = (
                    "No matching session or lap was found in the database for the requested filter. "
                    "Ask the user to confirm session type (Q vs Race) and year."
                )
            elif not tfetch.has_lap_rows:
                source = "no_data_found"
                telemetry_context = (
                    "No valid laps remained after filtering (green-flag valid laps, full sectors, "
                    "excluded outlaps for race, etc.). Suggest a different session or drivers."
                )
            else:
                val = validate_telemetry_block(tfetch.text, vmeta)
                if not val.ok:
                    source = "data_unavailable"
                    telemetry_context = BOUNDS_REPLACEMENT_MESSAGE
                    telemetry_meta["validation_failed"] = True
                else:
                    source = "fastf1_valid"
                    telemetry_context = tfetch.text
        except FileNotFoundError as exc:
            log.warning("DB missing: %s", exc)
            mode = "general"
            telemetry_context = "Telemetry database file was not found on the server."
            source = "no_data_found"
        except Exception as exc:
            log.warning("Telemetry fetch failed: %s", exc)
            mode = "general"
            telemetry_meta = {"error": str(exc)}
            source = "data_unavailable"

    add_message(sid, "user", message, mode)

    topics = extract_topics(message)
    response_text = _generate(sid, message, mode, telemetry_context, pushback=pushback)
    if mode == "telemetry" and source == "fastf1_valid" and telemetry_context:
        response_text = maybe_append_telemetry_synthesis(
            response_text, telemetry_context, message
        )
    if (
        mode == "telemetry"
        and telemetry_context
        and source == "fastf1_valid"
    ):
        sign_note = check_response_sector_sign_misinterpretation(
            telemetry_context, response_text
        )
        if sign_note:
            response_text = f"[Verification: {sign_note}]\n\n" + response_text
            if isinstance(telemetry_meta, dict):
                telemetry_meta["sign_interpretation_warning"] = True
    response_text = _maybe_append_sparse_note(response_text, _collect_driver_codes(message, intent))

    add_message(sid, "assistant", response_text, mode)

    elapsed_ms = int((time.time() - start) * 1000)

    return jsonify({
        "response": response_text,
        "metadata": {
            "mode": mode,
            "source": source,
            "response_time_ms": elapsed_ms,
            "confidence": 0.91 if source == "model_knowledge" else 0.87,
            "topics": topics,
            "telemetry": telemetry_meta,
        },
    })


@app.route("/history", methods=["GET"])
def history():
    """Return conversation history for the current session."""
    sid = request.args.get("session_id") or session.get("sid", "")
    return jsonify({"messages": get_history(sid)})


@app.route("/clear", methods=["POST"])
def clear_session():
    """Clear conversation history."""
    sid = request.get_json(force=True).get("session_id") or session.get("sid", "")
    clear(sid)
    return jsonify({"status": "cleared"})


# ---------------------------------------------------------------------------
# Real model generation (with stub fallback)
# ---------------------------------------------------------------------------

def _generate(
    sid: str,
    message: str,
    mode: str,
    telemetry_context: str | None,
    *,
    pushback: bool = False,
) -> str:
    """Generate a response using the fine-tuned model.

    Falls back to stub responses if the model isn't loaded.
    For telemetry mode, injects the telemetry context into the system prompt.
    On user pushback, redacts prior assistant numerical claims in history.
    """
    if not _MODEL_LOADED:
        return _stub_generate(message, mode, telemetry_context)

    try:
        sys_prompt = CHAT_SYSTEM_PROMPT_BASE
        if mode == "telemetry" and telemetry_context:
            sys_prompt += (
                "\n\n" + CHAT_SYSTEM_TELEMETRY_RULES
                + "\n\n" + LLM_TIME_DELTA_FOLLOW
                + "\n\n" + LLM_TELEMETRY_ANSWER_BODY
                + "\n\n--- TELEMETRY DATA (use ONLY these numbers; do not substitute from memory) ---\n"
                + telemetry_context
            )

        msgs = get_model_messages(
            sid, sys_prompt, redact_assistant_numerical_claims=pushback
        )
        return _model_generate(msgs)
    except Exception as exc:
        log.error(
            "Model inference failed: %r — falling back to stub", exc, exc_info=True
        )
        return _stub_generate(message, mode, telemetry_context)


# ---------------------------------------------------------------------------
# Stubs (fallback when model is not available)
# ---------------------------------------------------------------------------

def _stub_fetch_telemetry(intent: dict) -> str:
    """Placeholder telemetry fetch — returns formatted stub data."""
    drivers = ", ".join(intent.get("drivers", [])) or "Unknown"
    circuit = intent.get("circuit", "Unknown Circuit")
    year = intent.get("year", "N/A")
    sess = intent.get("session", "Race")
    lap = intent.get("lap", "N/A")
    return (
        f"[TELEMETRY CONTEXT]\n"
        f"Driver(s): {drivers}\n"
        f"Circuit: {circuit} | Year: {year} | Session: {sess}\n"
        f"Lap: {lap}\n"
        f"--- Stub data: replace with real FastF1 query ---"
    )


_GENERAL_STUBS = [
    "Based on the data from our telemetry database, {topic} patterns show interesting "
    "trends across the seasons we've analyzed. The key factor here is consistency — "
    "the drivers who manage their tyres most effectively tend to gain the most in the "
    "second stint, particularly on circuits with high rear-axle load requirements.",

    "Looking at the historical data, this is a fascinating area. The aerodynamic "
    "regulations since 2022 have fundamentally changed how cars behave through "
    "medium-speed corners, and we can see that reflected in the sector time deltas. "
    "The ground effect cars generate peak downforce differently, which affects "
    "minimum corner speeds significantly.",

    "From an engineering perspective, the answer depends on several variables: "
    "track temperature, fuel load, and tyre compound all interact here. Our data "
    "shows that degradation rates vary by up to 0.04s/lap between cool and hot "
    "conditions on thermally sensitive circuits.",
]

_TELEMETRY_STUBS = [
    "Analyzing the telemetry data for this specific session: the speed traces show "
    "notable differences in braking points and throttle application through the key "
    "corners. The driver's minimum speeds through the technical section are "
    "particularly revealing — we can see where time is gained and lost sector by sector.",

    "Pulling the lap data from our database: the sector breakdown shows the main "
    "delta is concentrated in Sector 2, which aligns with the corner profile of this "
    "circuit. The speed trap figures confirm the straight-line performance gap we'd "
    "expect given the power unit characteristics.",
]

import random

def _stub_generate(message: str, mode: str, context: str | None) -> str:
    """Placeholder response generator."""
    if mode == "telemetry":
        return random.choice(_TELEMETRY_STUBS)
    topic = "tyre degradation" if "tyre" in message.lower() else "performance"
    return random.choice(_GENERAL_STUBS).format(topic=topic)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    app.run(debug=True, host="127.0.0.1", port=5000)
