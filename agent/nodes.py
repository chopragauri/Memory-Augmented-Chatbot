"""
LangGraph node functions. Each receives and returns ChatState.
LLM: uses Groq if GROQ_API_KEY is set, otherwise falls back to a local HF model.
"""

import os
from dotenv import load_dotenv

from rag.retriever import retrieve
from knowledge_graph.graph_store import KnowledgeGraph
from memory.user_memory import UserMemory
from memory.chat_history import add_turn, format_for_prompt

load_dotenv()

kg = KnowledgeGraph()
user_mem = UserMemory()

SYSTEM_PROMPT = (
    "You are a knowledgeable AI assistant specializing in machine learning and AI. "
    "Answer questions accurately using the provided context. Be concise but thorough. "
    "If the context is insufficient, say so honestly."
)

# ── LLM setup ────────────────────────────────────────────────────────────────

def _build_llm():
    groq_key = os.getenv("GROQ_API_KEY")
    if groq_key:
        from groq import Groq
        client = Groq(api_key=groq_key)

        def call_llm(system: str, user: str) -> str:
            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=1024,
            )
            return resp.choices[0].message.content

        print("[LLM] Using Groq (LLaMA 3 70B)")
        return call_llm

    # Fallback: local HuggingFace model (small, CPU-friendly)
    print("[LLM] No GROQ_API_KEY — using extractive answer mode (RAG pipeline still fully active)")

    def call_llm(system: str, user: str) -> str:
        # Extract question and context from the combined user message
        lines = user.split("\n")
        question = ""
        context_lines = []
        in_context = False
        for line in lines:
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif line.startswith("Context:") or line.startswith("Retrieved knowledge:"):
                in_context = True
            elif in_context and line.strip():
                context_lines.append(line.strip())

        # Pull best sentences from retrieved context that match question keywords
        keywords = {w.lower() for w in question.split() if len(w) > 3}
        scored = []
        for line in context_lines:
            if len(line) < 30:
                continue
            score = sum(1 for k in keywords if k in line.lower())
            scored.append((score, line))

        scored.sort(key=lambda x: -x[0])
        top = [s[1] for s in scored[:4] if s[0] > 0]

        if top:
            answer = f"Based on the retrieved knowledge base:\n\n" + " ".join(top[:3])
            answer += "\n\n[Note: Add GROQ_API_KEY to .env to enable LLM-generated answers]"
        else:
            answer = "I found relevant documents in the knowledge base but could not extract a precise answer. Add GROQ_API_KEY to .env to enable full LLM generation."

        return answer

    return call_llm


_llm = None


def get_llm():
    global _llm
    if _llm is None:
        _llm = _build_llm()
    return _llm


# ── Nodes ─────────────────────────────────────────────────────────────────────

def rag_node(state: dict) -> dict:
    chunks = retrieve(state["query"], top_k=5)
    state["rag_context"] = "\n\n".join(
        f"[{c['title']}]: {c['text'][:400]}" for c in chunks
    )
    return state


def graph_node(state: dict) -> dict:
    keywords = [w for w in state["query"].split() if len(w) > 4 and w.isalpha()]
    graph_context = ""
    if keywords:
        triples = kg.query_related(keywords[0], limit=8)
        if triples:
            lines = [f"{t['subject']} --[{t['relation']}]--> {t['object']}" for t in triples]
            graph_context = "Knowledge graph relations:\n" + "\n".join(lines)
    state["graph_context"] = graph_context
    return state


def memory_node(state: dict) -> dict:
    state["memory_context"] = user_mem.format_for_prompt(state.get("user_id", "default"))
    state["chat_history"] = format_for_prompt(state.get("session_id", "default"))
    return state


def model_node(state: dict) -> dict:
    parts = []
    if state.get("memory_context"):
        parts.append(f"User context:\n{state['memory_context']}")
    if state.get("chat_history"):
        parts.append(f"Recent conversation:\n{state['chat_history']}")
    if state.get("rag_context"):
        parts.append(f"Retrieved knowledge:\n{state['rag_context']}")
    if state.get("graph_context"):
        parts.append(state["graph_context"])

    full_context = "\n\n---\n\n".join(parts)
    user_msg = f"Context:\n{full_context}\n\nQuestion: {state['query']}"

    answer = get_llm()(SYSTEM_PROMPT, user_msg)

    session_id = state.get("session_id", "default")
    user_id = state.get("user_id", "default")
    add_turn(session_id, "user", state["query"], user_id)
    add_turn(session_id, "assistant", answer, user_id)

    state["answer"] = answer
    return state
