"""
Extracts (subject, relation, object) triples from chunks using Groq LLM.
Run: python knowledge_graph/extractor.py
"""

import os
import json
import re
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "../data/chunks/all_chunks.json")
TRIPLES_PATH = os.path.join(os.path.dirname(__file__), "../data/chunks/triples.json")

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a knowledge graph builder. Given a text passage, extract factual relationships.
Return ONLY a JSON array of triples, no explanation. Format:
[{"subject": "...", "relation": "...", "object": "..."}]
Extract at most 5 triples per passage. Focus on factual, general knowledge."""


def extract_triples(text: str) -> list[dict]:
    try:
        response = client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Extract triples from:\n\n{text[:1000]}"},
            ],
            temperature=0,
            max_tokens=512,
        )
        raw = response.choices[0].message.content.strip()
        # Extract JSON array from response
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[error] {e}")
    return []


def run():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    # Process every 5th chunk to stay within rate limits
    sampled = chunks[::5]
    all_triples = []

    for i, chunk in enumerate(sampled):
        print(f"[{i+1}/{len(sampled)}] {chunk['title']}")
        triples = extract_triples(chunk["text"])
        for t in triples:
            t["source"] = chunk["source"]
            t["title"] = chunk["title"]
        all_triples.extend(triples)

    with open(TRIPLES_PATH, "w") as f:
        json.dump(all_triples, f, indent=2)

    print(f"\nExtracted {len(all_triples)} triples → {TRIPLES_PATH}")


if __name__ == "__main__":
    run()
