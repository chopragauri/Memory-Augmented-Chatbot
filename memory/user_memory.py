"""
Manages long-term user memory in MongoDB.
"""

import os
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()


class UserMemory:
    def __init__(self):
        client = MongoClient(os.getenv("MONGO_URI", "mongodb://localhost:27017"))
        db = client["memory_chatbot"]
        self.collection = db["users"]

    def get(self, user_id: str) -> dict:
        doc = self.collection.find_one({"user_id": user_id})
        if doc:
            doc.pop("_id", None)
        return doc or {"user_id": user_id, "preferences": [], "facts": [], "last_seen": None}

    def update_facts(self, user_id: str, new_facts: list[str]):
        self.collection.update_one(
            {"user_id": user_id},
            {
                "$addToSet": {"facts": {"$each": new_facts}},
                "$set": {"last_seen": datetime.utcnow().isoformat()},
            },
            upsert=True,
        )

    def update_preferences(self, user_id: str, preferences: list[str]):
        self.collection.update_one(
            {"user_id": user_id},
            {"$set": {"preferences": preferences, "last_seen": datetime.utcnow().isoformat()}},
            upsert=True,
        )

    def format_for_prompt(self, user_id: str) -> str:
        mem = self.get(user_id)
        parts = []
        if mem.get("preferences"):
            parts.append("User preferences: " + ", ".join(mem["preferences"]))
        if mem.get("facts"):
            parts.append("Known facts about user: " + "; ".join(mem["facts"][-5:]))
        return "\n".join(parts) if parts else ""
