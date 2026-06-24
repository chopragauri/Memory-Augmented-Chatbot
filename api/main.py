"""
FastAPI backend.
Run: uvicorn api.main:app --reload
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent.graph import chat
from memory.user_memory import UserMemory

app = FastAPI(title="Memory-Augmented Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

user_mem = UserMemory()


class ChatRequest(BaseModel):
    user_id: str = "default"
    session_id: str = "default"
    message: str


class ChatResponse(BaseModel):
    answer: str
    user_id: str
    session_id: str


class MemoryUpdateRequest(BaseModel):
    facts: list[str] = []
    preferences: list[str] = []


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    answer = chat(req.message, user_id=req.user_id, session_id=req.session_id)
    return ChatResponse(answer=answer, user_id=req.user_id, session_id=req.session_id)


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
