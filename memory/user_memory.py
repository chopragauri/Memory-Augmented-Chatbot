"""
Long-term user memory backed by Supabase (PostgreSQL).
"""

import os
from datetime import datetime, timezone
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

_client = None


def _sb():
    global _client
    if _client is None:
        _client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    return _client


class UserMemory:
    def get(self, user_id: str) -> dict:
        result = _sb().table("user_memory").select("*").eq("user_id", user_id).execute()
        if result.data:
            return result.data[0]
        return {"user_id": user_id, "facts": [], "preferences": [], "last_seen": None}

    def update_facts(self, user_id: str, new_facts: list[str]):
        existing = self.get(user_id)
        merged = list(set(existing.get("facts") or []) | set(new_facts))
        _sb().table("user_memory").upsert({
            "user_id": user_id,
            "facts": merged,
            "preferences": existing.get("preferences") or [],
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="user_id").execute()

    def update_preferences(self, user_id: str, preferences: list[str]):
        existing = self.get(user_id)
        _sb().table("user_memory").upsert({
            "user_id": user_id,
            "facts": existing.get("facts") or [],
            "preferences": preferences,
            "last_seen": datetime.now(timezone.utc).isoformat(),
        }, on_conflict="user_id").execute()

    def format_for_prompt(self, user_id: str) -> str:
        mem = self.get(user_id)
        parts = []
        if mem.get("preferences"):
            parts.append("User preferences: " + ", ".join(mem["preferences"]))
        if mem.get("facts"):
            parts.append("Known facts about user: " + "; ".join(mem["facts"][-5:]))
        return "\n".join(parts) if parts else ""
