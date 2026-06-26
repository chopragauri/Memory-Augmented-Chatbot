"""
Long-term user memory backed by a local JSON file — no MongoDB needed.
Drop-in replaceable with MongoDB later by swapping this class.
"""

import os
import json
from datetime import datetime

MEMORY_PATH = os.path.join(os.path.dirname(__file__), "../data/user_memory.json")


def _load() -> dict:
    if os.path.exists(MEMORY_PATH):
        with open(MEMORY_PATH) as f:
            return json.load(f)
    return {}


def _save(data: dict):
    os.makedirs(os.path.dirname(MEMORY_PATH), exist_ok=True)
    with open(MEMORY_PATH, "w") as f:
        json.dump(data, f, indent=2)


class UserMemory:
    def get(self, user_id: str) -> dict:
        store = _load()
        return store.get(user_id, {"user_id": user_id, "preferences": [], "facts": [], "last_seen": None})

    def update_facts(self, user_id: str, new_facts: list[str]):
        store = _load()
        user = store.get(user_id, {"preferences": [], "facts": []})
        existing = set(user["facts"])
        user["facts"] = list(existing | set(new_facts))
        user["last_seen"] = datetime.utcnow().isoformat()
        store[user_id] = user
        _save(store)

    def update_preferences(self, user_id: str, preferences: list[str]):
        store = _load()
        user = store.get(user_id, {"preferences": [], "facts": []})
        user["preferences"] = preferences
        user["last_seen"] = datetime.utcnow().isoformat()
        store[user_id] = user
        _save(store)

    def format_for_prompt(self, user_id: str) -> str:
        mem = self.get(user_id)
        parts = []
        if mem.get("preferences"):
            parts.append("User preferences: " + ", ".join(mem["preferences"]))
        if mem.get("facts"):
            parts.append("Known facts about user: " + "; ".join(mem["facts"][-5:]))
        return "\n".join(parts) if parts else ""
