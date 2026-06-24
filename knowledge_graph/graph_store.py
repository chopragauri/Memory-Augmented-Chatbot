"""
Loads triples into Neo4j and provides query interface.
Run: python knowledge_graph/graph_store.py  (to ingest)
"""

import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv

load_dotenv()

TRIPLES_PATH = os.path.join(os.path.dirname(__file__), "../data/chunks/triples.json")


class KnowledgeGraph:
    def __init__(self):
        self.driver = GraphDatabase.driver(
            os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
        )

    def close(self):
        self.driver.close()

    def ingest_triples(self, triples: list[dict]):
        with self.driver.session() as session:
            for t in triples:
                session.run(
                    """
                    MERGE (s:Entity {name: $subject})
                    MERGE (o:Entity {name: $object})
                    MERGE (s)-[r:RELATES {type: $relation, source: $source}]->(o)
                    """,
                    subject=t["subject"],
                    object=t["object"],
                    relation=t["relation"],
                    source=t.get("source", ""),
                )

    def query_related(self, entity: str, limit: int = 10) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH (s:Entity)-[r:RELATES]->(o:Entity)
                WHERE toLower(s.name) CONTAINS toLower($entity)
                   OR toLower(o.name) CONTAINS toLower($entity)
                RETURN s.name AS subject, r.type AS relation, o.name AS object
                LIMIT $limit
                """,
                entity=entity,
                limit=limit,
            )
            return [dict(row) for row in result]

    def query_path(self, from_entity: str, to_entity: str) -> list[dict]:
        with self.driver.session() as session:
            result = session.run(
                """
                MATCH path = shortestPath(
                  (a:Entity {name: $from_e})-[*..5]-(b:Entity {name: $to_e})
                )
                RETURN [n IN nodes(path) | n.name] AS nodes,
                       [r IN relationships(path) | r.type] AS relations
                """,
                from_e=from_entity,
                to_e=to_entity,
            )
            return [dict(row) for row in result]


def run():
    with open(TRIPLES_PATH) as f:
        triples = json.load(f)

    kg = KnowledgeGraph()
    print(f"Ingesting {len(triples)} triples into Neo4j...")
    kg.ingest_triples(triples)
    print("Done.")

    # Quick test
    results = kg.query_related("transformer")
    print(f"\nSample query - 'transformer' relations:")
    for r in results[:5]:
        print(f"  {r['subject']} --[{r['relation']}]--> {r['object']}")

    kg.close()


if __name__ == "__main__":
    run()
