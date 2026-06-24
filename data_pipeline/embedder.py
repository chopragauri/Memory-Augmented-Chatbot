"""
Generates embeddings for all chunks and stores them in a FAISS index.
Run: python data_pipeline/embedder.py
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "../data/chunks/all_chunks.json")
EMBED_DIR = os.path.join(os.path.dirname(__file__), "../data/embeddings")

MODEL_NAME = "all-MiniLM-L6-v2"  # free, fast, 384-dim


def run():
    os.makedirs(EMBED_DIR, exist_ok=True)

    print(f"Loading chunks from {CHUNKS_PATH}...")
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks with {MODEL_NAME}...")

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)
    embeddings = np.array(embeddings, dtype="float32")

    # Build FAISS index (flat L2 — good for <100k chunks)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, os.path.join(EMBED_DIR, "faiss.index"))

    # Save chunk metadata for lookup by index position
    with open(os.path.join(EMBED_DIR, "chunks_meta.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    print(f"Saved FAISS index ({index.ntotal} vectors) and metadata to {EMBED_DIR}")


if __name__ == "__main__":
    run()
