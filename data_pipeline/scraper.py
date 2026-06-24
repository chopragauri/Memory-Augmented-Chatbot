"""
Scrapes ML/AI Wikipedia articles and saves raw text to data/raw/.
Run: python data_pipeline/scraper.py
"""

import os
import json
import time
import requests
from bs4 import BeautifulSoup

TOPICS = [
    "Machine_learning",
    "Deep_learning",
    "Natural_language_processing",
    "Transformer_(machine_learning_model)",
    "Retrieval-augmented_generation",
    "Large_language_model",
    "Knowledge_graph",
    "Recurrent_neural_network",
    "Convolutional_neural_network",
    "Reinforcement_learning",
    "Generative_adversarial_network",
    "BERT_(language_model)",
    "GPT-3",
    "Attention_(machine_learning)",
    "Word2vec",
]

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")


def scrape_wikipedia(topic: str) -> dict:
    url = f"https://en.wikipedia.org/wiki/{topic}"
    response = requests.get(url, headers={"User-Agent": "MemoryChatbot/1.0"})
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove unwanted sections
    for tag in soup.find_all(["table", "sup", "span.mw-editsection"]):
        tag.decompose()

    content_div = soup.find("div", {"id": "mw-content-text"})
    paragraphs = content_div.find_all("p") if content_div else []

    text = "\n\n".join(
        p.get_text(strip=True) for p in paragraphs if len(p.get_text(strip=True)) > 50
    )

    title = soup.find("h1").get_text(strip=True) if soup.find("h1") else topic

    return {"title": title, "url": url, "topic": topic, "text": text}


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for topic in TOPICS:
        out_path = os.path.join(OUTPUT_DIR, f"{topic}.json")
        if os.path.exists(out_path):
            print(f"[skip] {topic} already scraped")
            continue
        try:
            print(f"[scrape] {topic}")
            doc = scrape_wikipedia(topic)
            with open(out_path, "w") as f:
                json.dump(doc, f, indent=2)
            time.sleep(1)  # polite delay
        except Exception as e:
            print(f"[error] {topic}: {e}")
    print(f"\nDone. Files saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run()
