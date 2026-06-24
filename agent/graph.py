"""
LangGraph StateGraph definition — the core orchestrator.
"""

from langgraph.graph import StateGraph, END
from typing import TypedDict

from agent.nodes import rag_node, graph_node, memory_node, model_node


class ChatState(TypedDict):
    query: str
    user_id: str
    session_id: str
    rag_context: str
    graph_context: str
    memory_context: str
    chat_history: str
    answer: str


def build_graph():
    g = StateGraph(ChatState)

    g.add_node("memory", memory_node)
    g.add_node("rag", rag_node)
    g.add_node("graph", graph_node)
    g.add_node("model", model_node)

    # All retrieval nodes run in sequence, then model generates
    g.set_entry_point("memory")
    g.add_edge("memory", "rag")
    g.add_edge("rag", "graph")
    g.add_edge("graph", "model")
    g.add_edge("model", END)

    return g.compile()


chat_graph = build_graph()


def chat(query: str, user_id: str = "default", session_id: str = "default") -> str:
    state = {
        "query": query,
        "user_id": user_id,
        "session_id": session_id,
        "rag_context": "",
        "graph_context": "",
        "memory_context": "",
        "chat_history": "",
        "answer": "",
    }
    result = chat_graph.invoke(state)
    return result["answer"]


if __name__ == "__main__":
    print("Chatbot ready. Type 'quit' to exit.\n")
    while True:
        q = input("You: ").strip()
        if q.lower() in ("quit", "exit"):
            break
        if q:
            answer = chat(q, user_id="test_user", session_id="demo")
            print(f"\nBot: {answer}\n")
