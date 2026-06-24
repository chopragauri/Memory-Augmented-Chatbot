"""
Tracks per-session conversation history (in-memory, can swap to Redis/Mongo).
"""

from collections import defaultdict

_sessions: dict[str, list[dict]] = defaultdict(list)
MAX_HISTORY = 10  # keep last N turns


def add_turn(session_id: str, role: str, content: str):
    _sessions[session_id].append({"role": role, "content": content})
    if len(_sessions[session_id]) > MAX_HISTORY * 2:
        _sessions[session_id] = _sessions[session_id][-MAX_HISTORY * 2 :]


def get_history(session_id: str) -> list[dict]:
    return _sessions[session_id]


def format_for_prompt(session_id: str) -> str:
    history = get_history(session_id)
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:  # last 3 turns
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)
