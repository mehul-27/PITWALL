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

from config import CHAT_SYSTEM_PROMPT
from intent import detect_mode, parse_telemetry_intent, extract_topics
from memory import (
    create_session, add_message, get_history, get_model_messages, clear,
)
from retrieval import fetch_telemetry

# Try to load the real model — fall back to stubs if Ollama unavailable
_MODEL_LOADED = False
try:
    from inference import generate as _model_generate, MODEL_AVAILABLE
    _MODEL_LOADED = MODEL_AVAILABLE
except Exception as exc:
    logging.warning("Could not import inference module: %s — using stub responses", exc)
    _model_generate = None

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


@app.route("/chat", methods=["POST"])
def chat():
    """Handle a chat message.

    Accepts JSON: { "message": str, "session_id": str (optional) }
    Returns JSON: {
        "response": str,
        "metadata": {
            "mode": "general" | "telemetry",
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

    # ── Detect mode ──────────────────────────────────────────────────────
    mode = detect_mode(message)
    telemetry_meta: dict = {}
    telemetry_context: str | None = None

    if mode == "telemetry":
        intent = parse_telemetry_intent(message)
        try:
            t0 = time.time()
            telemetry_context = fetch_telemetry(intent)
            telemetry_meta = {
                "query_ms": int((time.time() - t0) * 1000),
                "drivers": intent.get("drivers", []),
                "circuit": intent.get("circuit"),
                "session": intent.get("session"),
                "year": intent.get("year"),
                "lap": intent.get("lap"),
            }
        except Exception as exc:
            log.warning("Telemetry fetch failed: %s — falling back to general", exc)
            mode = "general"
            telemetry_meta = {"error": str(exc)}

    # ── Record user message ──────────────────────────────────────────────
    add_message(sid, "user", message, mode)

    # ── Generate response ────────────────────────────────────────────────
    topics = extract_topics(message)
    response_text = _generate(sid, message, mode, telemetry_context)

    # ── Record assistant message ─────────────────────────────────────────
    add_message(sid, "assistant", response_text, mode)

    elapsed_ms = int((time.time() - start) * 1000)

    return jsonify({
        "response": response_text,
        "metadata": {
            "mode": mode,
            "response_time_ms": elapsed_ms,
            "confidence": 0.91 if mode == "general" else 0.87,
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
    sid: str, message: str, mode: str, telemetry_context: str | None
) -> str:
    """Generate a response using the fine-tuned model.

    Falls back to stub responses if the model isn't loaded.
    For telemetry mode, injects the telemetry context into the system prompt.
    """
    if not _MODEL_LOADED:
        return _stub_generate(message, mode, telemetry_context)

    try:
        # Build the system prompt — inject telemetry context if available
        sys_prompt = CHAT_SYSTEM_PROMPT
        if mode == "telemetry" and telemetry_context:
            sys_prompt += (
                "\n\n--- TELEMETRY DATA (use this to ground your answer) ---\n"
                + telemetry_context
            )

        # Build message list: system + conversation history
        # (history already includes the current user message from add_message)
        msgs = get_model_messages(sid, sys_prompt)

        response = _model_generate(msgs)
        return response

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
