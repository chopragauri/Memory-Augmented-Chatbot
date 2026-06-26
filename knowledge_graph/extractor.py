"""
Extracts (subject, relation, object) triples from chunks using spaCy (fully offline).
Uses dependency parsing: nsubj → ROOT verb → dobj/attr pattern.
Run: python knowledge_graph/extractor.py
"""

import os
import json
import spacy

CHUNKS_PATH = os.path.join(os.path.dirname(__file__), "../data/chunks/all_chunks.json")
TRIPLES_PATH = os.path.join(os.path.dirname(__file__), "../data/chunks/triples.json")

nlp = spacy.load("en_core_web_sm")


def extract_triples(text: str, source: str, title: str) -> list[dict]:
    doc = nlp(text[:2000])
    triples = []

    for sent in doc.sents:
        for token in sent:
            if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                subjects = [c for c in token.children if c.dep_ in ("nsubj", "nsubjpass")]
                objects = [c for c in token.children if c.dep_ in ("dobj", "attr", "pobj", "acomp")]

                for subj in subjects:
                    for obj in objects:
                        subj_text = " ".join(t.text for t in subj.subtree if not t.is_stop or t == subj)
                        obj_text = " ".join(t.text for t in obj.subtree if not t.is_stop or t == obj)
                        relation = token.lemma_

                        if len(subj_text) < 60 and len(obj_text) < 60:
                            triples.append({
                                "subject": subj_text.strip(),
                                "relation": relation,
                                "object": obj_text.strip(),
                                "source": source,
                                "title": title,
                            })

    return triples[:10]


def run():
    with open(CHUNKS_PATH) as f:
        chunks = json.load(f)

    all_triples = []
    for i, chunk in enumerate(chunks):
        triples = extract_triples(chunk["text"], chunk["source"], chunk["title"])
        all_triples.extend(triples)
        if (i + 1) % 20 == 0:
            print(f"  processed {i+1}/{len(chunks)} chunks, {len(all_triples)} triples so far")

    seen = set()
    unique = []
    for t in all_triples:
        key = (t["subject"].lower(), t["relation"], t["object"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(t)

    with open(TRIPLES_PATH, "w") as f:
        json.dump(unique, f, indent=2)

    print(f"\nExtracted {len(unique)} unique triples → {TRIPLES_PATH}")


if __name__ == "__main__":
    run()
