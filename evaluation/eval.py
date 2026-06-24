"""
Evaluates the RAG pipeline using RAGAS metrics.
Run: python evaluation/eval.py
"""

import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from groq import Groq
from dotenv import load_dotenv

from rag.retriever import retrieve

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Small gold-standard test set
TEST_SET = [
    {
        "question": "What is retrieval-augmented generation?",
        "ground_truth": "RAG is a technique that combines information retrieval with language model generation to produce more accurate, grounded responses.",
    },
    {
        "question": "How does the transformer attention mechanism work?",
        "ground_truth": "Attention allows the model to weigh the importance of different input tokens when generating each output token using query, key, and value matrices.",
    },
    {
        "question": "What is a knowledge graph?",
        "ground_truth": "A knowledge graph is a structured representation of entities and the relationships between them, stored in a graph database.",
    },
    {
        "question": "What is word2vec?",
        "ground_truth": "Word2vec is a technique to represent words as dense vectors such that semantically similar words are close in vector space.",
    },
    {
        "question": "What is reinforcement learning?",
        "ground_truth": "Reinforcement learning is a type of machine learning where an agent learns by interacting with an environment and receiving rewards or penalties.",
    },
]


def generate_answer(question: str, context: str) -> str:
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {"role": "system", "content": "Answer the question based on the provided context only."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
        max_tokens=512,
    )
    return response.choices[0].message.content


def run():
    questions, answers, contexts, ground_truths = [], [], [], []

    for item in TEST_SET:
        q = item["question"]
        chunks = retrieve(q, top_k=3)
        ctx_texts = [c["text"] for c in chunks]
        answer = generate_answer(q, "\n\n".join(ctx_texts))

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx_texts)
        ground_truths.append(item["ground_truth"])
        print(f"[eval] Q: {q[:60]}...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS evaluation...")
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])
    print("\n=== Evaluation Results ===")
    print(result)
    return result


if __name__ == "__main__":
    run()
