"""
LangGraph StateGraph definition — the core orchestrator.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict

from agent.nodes import rag_node, graph_node, memory_node, model_node
from rag.retriever import retrieve
from knowledge_graph.graph_store import KnowledgeGraph
from sentence_transformers import SentenceTransformer
import numpy as np


class ChatState(TypedDict):
    query: str
    user_id: str
    session_id: str
    rag_context: str
    graph_context: str
    memory_context: str
    chat_history: str
    answer: str
    raw_chunks: list
    raw_triples: list


def build_graph():
    g = StateGraph(ChatState)
    g.add_node("memory", memory_node)
    g.add_node("rag", rag_node)
    g.add_node("graph", graph_node)
    g.add_node("model", model_node)
    g.set_entry_point("memory")
    g.add_edge("memory", "rag")
    g.add_edge("rag", "graph")
    g.add_edge("graph", "model")
    g.add_edge("model", END)
    return g.compile()


chat_graph = build_graph()
_embed_model = None
_kg = KnowledgeGraph()


def _get_embed():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _embed_model


def _cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def chat(query: str, user_id: str = "default", session_id: str = "default") -> str:
    return chat_full(query, user_id, session_id)["answer"]


def chat_full(query: str, user_id: str = "default", session_id: str = "default") -> dict:
    state = {
        "query": query, "user_id": user_id, "session_id": session_id,
        "rag_context": "", "graph_context": "", "memory_context": "",
        "chat_history": "", "answer": "", "raw_chunks": [], "raw_triples": [],
    }
    result = chat_graph.invoke(state)
    answer = result["answer"]

    # Retrieve sources + metrics for structured response
    chunks = retrieve(query, top_k=5)
    sources = [{"title": c["title"], "text": c["text"][:200], "score": round(c["score"], 3)} for c in chunks]

    # Compute retrieval metrics
    model = _get_embed()
    q_emb = model.encode([query])[0]
    ctx_embs = model.encode([c["text"] for c in chunks])
    ctx_sims = [_cosine(q_emb, e) for e in ctx_embs]
    context_relevance = round(float(np.mean(ctx_sims)), 3)
    context_recall = round(float(max(ctx_sims)), 3)

    # KG triples
    keywords = [w for w in query.split() if len(w) > 4 and w.isalpha()]
    kg_triples = []
    if keywords:
        kg_triples = _kg.query_related(keywords[0], limit=8)

    return {
        "answer": answer,
        "user_id": user_id,
        "session_id": session_id,
        "sources": sources,
        "kg_triples": kg_triples,
        "context_relevance": context_relevance,
        "context_recall": context_recall,
    }


if __name__ == "__main__":
    print("Chatbot ready. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if q:
            r = chat_full(q, user_id="test_user", session_id="demo")
            print(f"\nBot: {r['answer']}\n")
            print(f"Sources: {[s['title'] for s in r['sources']]}")
            print(f"Relevance: {r['context_relevance']} | Recall: {r['context_recall']}\n")
