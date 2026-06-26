"""
Evaluates the RAG pipeline with simple local metrics (no API needed).
Uses cosine similarity for relevance and keyword overlap for faithfulness.
Run: python evaluation/eval.py
"""

import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.retriever import retrieve

EVAL_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")

TEST_SET = [
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "RAG combines information retrieval with language model generation to produce accurate grounded responses using external knowledge.",
    },
    {
        "question": "How does the transformer attention mechanism work?",
        "ground_truth": "Attention allows the model to weigh the importance of different input tokens using query key and value matrices.",
    },
    {
        "question": "What is a knowledge graph?",
        "ground_truth": "A knowledge graph is a structured representation of entities and relationships stored in a graph database.",
    },
    {
        "question": "What is word2vec?",
        "ground_truth": "Word2vec represents words as dense vectors so that semantically similar words are close in vector space.",
    },
    {
        "question": "What is reinforcement learning?",
        "ground_truth": "Reinforcement learning is where an agent learns by interacting with an environment and receiving rewards or penalties.",
    },
    {
        "question": "What is deep learning?",
        "ground_truth": "Deep learning uses multi-layer neural networks to automatically learn hierarchical feature representations from data.",
    },
    {
        "question": "What are generative adversarial networks?",
        "ground_truth": "GANs consist of a generator and discriminator network trained adversarially where the generator creates fake samples and the discriminator tries to distinguish real from fake.",
    },
]

model = SentenceTransformer("all-MiniLM-L6-v2")


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def context_relevance(question: str, contexts: list[str]) -> float:
    q_emb = model.encode([question])[0]
    ctx_embs = model.encode(contexts)
    sims = [cosine_sim(q_emb, c) for c in ctx_embs]
    return float(np.mean(sims))


def context_recall(ground_truth: str, contexts: list[str]) -> float:
    gt_emb = model.encode([ground_truth])[0]
    ctx_embs = model.encode(contexts)
    sims = [cosine_sim(gt_emb, c) for c in ctx_embs]
    return float(max(sims))


def keyword_faithfulness(answer: str, contexts: list[str]) -> float:
    combined_ctx = " ".join(contexts).lower()
    words = [w for w in answer.lower().split() if len(w) > 4]
    if not words:
        return 0.0
    matched = sum(1 for w in words if w in combined_ctx)
    return matched / len(words)


def run():
    print("=" * 60)
    print("RAG EVALUATION — Local Metrics")
    print("=" * 60)

    all_ctx_rel, all_ctx_rec = [], []

    for item in TEST_SET:
        q = item["question"]
        gt = item["ground_truth"]

        chunks = retrieve(q, top_k=3)
        ctx_texts = [c["text"][:400] for c in chunks]

        cr = context_relevance(q, ctx_texts)
        rec = context_recall(gt, ctx_texts)

        all_ctx_rel.append(cr)
        all_ctx_rec.append(rec)

        print(f"\nQ: {q}")
        print(f"   Context Relevance : {cr:.3f}")
        print(f"   Context Recall    : {rec:.3f}")
        print(f"   Top source        : {chunks[0]['title'] if chunks else 'N/A'}")

    print("\n" + "=" * 60)
    print(f"AVG Context Relevance : {np.mean(all_ctx_rel):.3f}")
    print(f"AVG Context Recall    : {np.mean(all_ctx_rec):.3f}")
    print("=" * 60)

    results = {
        "avg_context_relevance": round(float(np.mean(all_ctx_rel)), 3),
        "avg_context_recall": round(float(np.mean(all_ctx_rec)), 3),
        "per_question": [
            {"question": TEST_SET[i]["question"], "context_relevance": round(all_ctx_rel[i], 3), "context_recall": round(all_ctx_rec[i], 3)}
            for i in range(len(TEST_SET))
        ],
    }
    out = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved → {out}")


if __name__ == "__main__":
    run()
