"""
In-memory knowledge graph using networkx. Saves/loads from JSON — no Neo4j needed.
Run: python knowledge_graph/graph_store.py  (to ingest)
"""

import os
import json
import networkx as nx

TRIPLES_PATH = os.path.join(os.path.dirname(__file__), "../data/chunks/triples.json")
GRAPH_PATH = os.path.join(os.path.dirname(__file__), "../data/chunks/graph.json")

_graph: nx.DiGraph = None


def _load_graph() -> nx.DiGraph:
    global _graph
    if _graph is not None:
        return _graph
    if os.path.exists(GRAPH_PATH):
        with open(GRAPH_PATH) as f:
            data = json.load(f)
        _graph = nx.node_link_graph(data, edges="links")
    else:
        _graph = nx.DiGraph()
    return _graph


class KnowledgeGraph:
    def __init__(self):
        self.g = _load_graph()

    def ingest_triples(self, triples: list[dict]):
        for t in triples:
            s, r, o = t["subject"], t["relation"], t["object"]
            if s and o:
                self.g.add_node(s)
                self.g.add_node(o)
                self.g.add_edge(s, o, relation=r, source=t.get("source", ""))
        # Persist
        data = nx.node_link_data(self.g, edges="links")
        with open(GRAPH_PATH, "w") as f:
            json.dump(data, f)

    def query_related(self, entity: str, limit: int = 10) -> list[dict]:
        g = self.g
        entity_lower = entity.lower()
        results = []
        for s, o, attrs in g.edges(data=True):
            if entity_lower in s.lower() or entity_lower in o.lower():
                results.append({"subject": s, "relation": attrs.get("relation", ""), "object": o})
            if len(results) >= limit:
                break
        return results

    def query_path(self, from_entity: str, to_entity: str) -> list[str]:
        try:
            path = nx.shortest_path(self.g, from_entity, to_entity)
            return path
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def stats(self) -> dict:
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges()}


def run():
    with open(TRIPLES_PATH) as f:
        triples = json.load(f)

    kg = KnowledgeGraph()
    print(f"Ingesting {len(triples)} triples into local graph...")
    kg.ingest_triples(triples)
    print(f"Graph stats: {kg.stats()}")

    results = kg.query_related("neural network", limit=5)
    print("\nSample — 'neural network' relations:")
    for r in results:
        print(f"  {r['subject']} --[{r['relation']}]--> {r['object']}")


if __name__ == "__main__":
    run()
