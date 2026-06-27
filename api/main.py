"""
FastAPI backend with structured response — sources, KG triples, retrieval metrics.
Run: python3 -m uvicorn api.main:app --reload
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime, timezone
from supabase import create_client

from agent.graph import chat_full
from memory.user_memory import UserMemory
from memory.chat_history import get_history

app = FastAPI(title="Memory-Augmented Chatbot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

user_mem = UserMemory()
_sb = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))


def _ensure_session(session_id: str, user_id: str, title: str = "New Chat"):
    existing = _sb.table("sessions").select("id").eq("id", session_id).execute()
    if not existing.data:
        _sb.table("sessions").insert({
            "id": session_id, "user_id": user_id, "title": title,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
    else:
        _sb.table("sessions").update({
            "updated_at": datetime.now(timezone.utc).isoformat()
        }).eq("id", session_id).execute()


def _set_session_title(session_id: str, title: str):
    _sb.table("sessions").update({"title": title[:60]}).eq("id", session_id).execute()


class ChatRequest(BaseModel):
    user_id: str = "default"
    session_id: str = "default"
    message: str


class MemoryUpdateRequest(BaseModel):
    facts: list[str] = []
    preferences: list[str] = []


@app.get("/", response_class=HTMLResponse)
def index():
    return open(os.path.join(os.path.dirname(__file__), "index.html")).read()


@app.post("/chat")
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    # Auto-create session, set title from first message
    _ensure_session(req.session_id, req.user_id, req.message[:60])
    result = chat_full(req.message, user_id=req.user_id, session_id=req.session_id)
    return result


@app.get("/sessions/{user_id}")
def list_sessions(user_id: str):
    result = (
        _sb.table("sessions")
        .select("id, title, created_at, updated_at")
        .eq("user_id", user_id)
        .order("updated_at", desc=True)
        .limit(50)
        .execute()
    )
    return result.data or []


@app.get("/sessions/{user_id}/{session_id}/messages")
def get_session_messages(user_id: str, session_id: str):
    return get_history(session_id)


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    _sb.table("chat_history").delete().eq("session_id", session_id).execute()
    _sb.table("sessions").delete().eq("id", session_id).execute()
    return {"status": "deleted"}


@app.post("/sessions/new")
def new_session(user_id: str = "default"):
    session_id = str(uuid.uuid4())[:8]
    _ensure_session(session_id, user_id, "New Chat")
    return {"session_id": session_id}


@app.get("/memory/{user_id}")
def get_memory(user_id: str):
    return user_mem.get(user_id)


@app.post("/memory/{user_id}")
def update_memory(user_id: str, req: MemoryUpdateRequest):
    if req.facts:
        user_mem.update_facts(user_id, req.facts)
    if req.preferences:
        user_mem.update_preferences(user_id, req.preferences)
    return {"status": "updated", "user_id": user_id}


@app.get("/health")
def health():
    return {"status": "ok"}
