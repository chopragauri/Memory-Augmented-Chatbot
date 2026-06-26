"""
Per-session chat history backed by Supabase (PostgreSQL).
"""

import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

_client = None
MAX_HISTORY = 10


def _sb():
    global _client
    if _client is None:
        _client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _client


def add_turn(session_id: str, role: str, content: str, user_id: str = "default"):
    _sb().table("chat_history").insert({
        "session_id": session_id,
        "user_id": user_id,
        "role": role,
        "content": content,
    }).execute()


def get_history(session_id: str) -> list[dict]:
    result = (
        _sb().table("chat_history")
        .select("role, content, created_at")
        .eq("session_id", session_id)
        .order("created_at", desc=False)
        .limit(MAX_HISTORY * 2)
        .execute()
    )
    return result.data or []


def format_for_prompt(session_id: str) -> str:
    history = get_history(session_id)
    if not history:
        return ""
    lines = []
    for msg in history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    return "\n".join(lines)
