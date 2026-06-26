"""
FastAPI backend with structured response — sources, KG triples, retrieval metrics.
Run: python3 -m uvicorn api.main:app --reload
"""

import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from agent.graph import chat_full
from memory.user_memory import UserMemory

app = FastAPI(title="Memory-Augmented Chatbot API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

user_mem = UserMemory()


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
    return chat_full(req.message, user_id=req.user_id, session_id=req.session_id)


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
