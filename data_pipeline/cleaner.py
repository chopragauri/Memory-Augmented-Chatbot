"""
Cleans raw scraped text and splits into overlapping chunks.
Run: python data_pipeline/cleaner.py
"""

import os
import re
import json

RAW_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
CHUNKS_DIR = os.path.join(os.path.dirname(__file__), "../data/chunks")

CHUNK_SIZE = 500      # tokens (approximate via word count)
CHUNK_OVERLAP = 100


def clean_text(text: str) -> str:
    text = re.sub(r"\[\d+\]", "", text)          # remove citation markers [1]
    text = re.sub(r"\s+", " ", text)             # collapse whitespace
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
    return text.strip()


def split_into_chunks(text: str, source: str, title: str) -> list[dict]:
    words = text.split()
    chunks = []
    i = 0
    chunk_id = 0
    while i < len(words):
        chunk_words = words[i : i + CHUNK_SIZE]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "chunk_id": f"{source}_{chunk_id}",
            "source": source,
            "title": title,
            "text": chunk_text,
            "word_count": len(chunk_words),
        })
        chunk_id += 1
        i += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def run():
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    all_chunks = []

    for fname in os.listdir(RAW_DIR):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(RAW_DIR, fname)) as f:
            doc = json.load(f)

        cleaned = clean_text(doc["text"])
        chunks = split_into_chunks(cleaned, doc["topic"], doc["title"])
        all_chunks.extend(chunks)
        print(f"[chunk] {doc['title']}: {len(chunks)} chunks")

    out_path = os.path.join(CHUNKS_DIR, "all_chunks.json")
    with open(out_path, "w") as f:
        json.dump(all_chunks, f, indent=2)

    print(f"\nTotal chunks: {len(all_chunks)} → saved to {out_path}")


if __name__ == "__main__":
    run()
