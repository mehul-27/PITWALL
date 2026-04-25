"""
Conversation memory — stores multi-turn history server-side,
keyed by session_id. Provides context window management so the
full conversation history is included in each inference call.
"""

from __future__ import annotations

import re
import time
import uuid
from collections import defaultdict

# In-memory store: session_id -> list of {role, content, timestamp, mode}
_store: dict[str, list[dict]] = defaultdict(list)

# Max messages kept per session (prevents unbounded memory growth)
MAX_HISTORY = 50


def create_session() -> str:
    """Create a new conversation session, return its ID."""
    sid = uuid.uuid4().hex[:12]
    _store[sid] = []
    return sid


def add_message(
    session_id: str,
    role: str,
    content: str,
    mode: str = "general",
) -> None:
    """Append a message to the session history."""
    _store[session_id].append({
        "role": role,
        "content": content,
        "mode": mode,
        "timestamp": time.time(),
    })
    # Trim oldest messages if over limit
    if len(_store[session_id]) > MAX_HISTORY:
        _store[session_id] = _store[session_id][-MAX_HISTORY:]


def get_history(session_id: str) -> list[dict]:
    """Return full message history for a session."""
    return list(_store.get(session_id, []))


# When user challenges prior answer, strip numbers the model may have hallucinated
_NUM_CLAIM = re.compile(
    r"(\d+:\d{2}\.\d+)|(\d+\.?\d*)\s*(s|sec|/lap|s/lap|s\s*per|km/h|kph|kmh|%)|(\b\d{1,2}\.?\d*)\s*laps?\b",
    re.IGNORECASE,
)


def _redact_numerical_assistant_bodies(text: str) -> str:
    return _NUM_CLAIM.sub("[prior numeric claim redacted: verify with fresh data]", text)


def get_model_messages(
    session_id: str,
    system_prompt: str,
    *,
    redact_assistant_numerical_claims: bool = False,
) -> list[dict]:
    """Build the messages list for model inference (system + history)."""
    msgs: list[dict] = [{"role": "system", "content": system_prompt}]
    for m in _store.get(session_id, []):
        content = m["content"]
        if m["role"] == "assistant" and redact_assistant_numerical_claims:
            content = _redact_numerical_assistant_bodies(content)
        msgs.append({"role": m["role"], "content": content})
    return msgs


def clear(session_id: str) -> None:
    """Clear all history for a session."""
    _store.pop(session_id, None)
