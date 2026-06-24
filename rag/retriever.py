"""
Retrieves top-k relevant chunks from FAISS given a query.
"""

import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_DIR = os.path.join(os.path.dirname(__file__), "../data/embeddings")
MODEL_NAME = "all-MiniLM-L6-v2"

_model = None
_index = None
_chunks = None


def _load():
    global _model, _index, _chunks
    if _index is None:
        _model = SentenceTransformer(MODEL_NAME)
        _index = faiss.read_index(os.path.join(EMBED_DIR, "faiss.index"))
        with open(os.path.join(EMBED_DIR, "chunks_meta.pkl"), "rb") as f:
            _chunks = pickle.load(f)


def retrieve(query: str, top_k: int = 5) -> list[dict]:
    _load()
    query_vec = _model.encode([query], convert_to_numpy=True).astype("float32")
    distances, indices = _index.search(query_vec, top_k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < len(_chunks):
            chunk = _chunks[idx].copy()
            chunk["score"] = float(dist)
            results.append(chunk)
    return results


if __name__ == "__main__":
    results = retrieve("What is retrieval augmented generation?")
    for r in results:
        print(f"[{r['title']}] score={r['score']:.2f}\n{r['text'][:200]}\n")
