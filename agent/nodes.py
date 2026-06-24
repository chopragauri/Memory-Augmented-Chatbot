"""
LangGraph node functions. Each receives and returns ChatState.
"""

import os
from groq import Groq
from dotenv import load_dotenv

from rag.retriever import retrieve
from knowledge_graph.graph_store import KnowledgeGraph
from memory.user_memory import UserMemory
from memory.chat_history import add_turn, format_for_prompt

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
kg = KnowledgeGraph()
user_mem = UserMemory()

SYSTEM_PROMPT = """You are a knowledgeable AI assistant with access to a rich knowledge base about ML/AI.
Answer questions accurately using the provided context. Be concise but thorough.
If context is insufficient, say so honestly."""


def rag_node(state: dict) -> dict:
    query = state["query"]
    chunks = retrieve(query, top_k=5)
    rag_context = "\n\n".join(
        f"[{c['title']}]: {c['text'][:400]}" for c in chunks
    )
    state["rag_context"] = rag_context
    return state


def graph_node(state: dict) -> dict:
    # Extract key noun from query for graph lookup
    query = state["query"]
    # Simple keyword extraction — take first multi-word noun phrase
    keywords = [w for w in query.split() if len(w) > 4 and w.isalpha()]
    graph_context = ""
    if keywords:
        keyword = keywords[0]
        triples = kg.query_related(keyword, limit=8)
        if triples:
            lines = [f"{t['subject']} --[{t['relation']}]--> {t['object']}" for t in triples]
            graph_context = "Knowledge graph relations:\n" + "\n".join(lines)
    state["graph_context"] = graph_context
    return state


def memory_node(state: dict) -> dict:
    user_id = state.get("user_id", "default")
    session_id = state.get("session_id", "default")
    state["memory_context"] = user_mem.format_for_prompt(user_id)
    state["chat_history"] = format_for_prompt(session_id)
    return state


def model_node(state: dict) -> dict:
    query = state["query"]
    rag_context = state.get("rag_context", "")
    graph_context = state.get("graph_context", "")
    memory_context = state.get("memory_context", "")
    chat_history = state.get("chat_history", "")

    context_parts = []
    if memory_context:
        context_parts.append(f"User context:\n{memory_context}")
    if chat_history:
        context_parts.append(f"Recent conversation:\n{chat_history}")
    if rag_context:
        context_parts.append(f"Retrieved knowledge:\n{rag_context}")
    if graph_context:
        context_parts.append(graph_context)

    full_context = "\n\n---\n\n".join(context_parts)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{full_context}\n\nQuestion: {query}"},
    ]

    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    answer = response.choices[0].message.content

    # Save to chat history
    session_id = state.get("session_id", "default")
    add_turn(session_id, "user", query)
    add_turn(session_id, "assistant", answer)

    state["answer"] = answer
    return state
