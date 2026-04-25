"""
Conversation memory — stores multi-turn history server-side,
keyed by session_id. Provides context window management so the
full conversation history is included in each inference call.
"""

from __future__ import annotations

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


def get_model_messages(session_id: str, system_prompt: str) -> list[dict]:
    """Build the messages list for model inference (system + history)."""
    msgs = [{"role": "system", "content": system_prompt}]
    for m in _store.get(session_id, []):
        msgs.append({"role": m["role"], "content": m["content"]})
    return msgs


def clear(session_id: str) -> None:
    """Clear all history for a session."""
    _store.pop(session_id, None)
