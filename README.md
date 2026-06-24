# Memory-Augmented Chatbot with Knowledge Graph and Hybrid RAG

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cp .env.example .env  # fill in your keys
```

## Run Order (Day by Day)

### Day 1 — Scrape & Clean
```bash
python data_pipeline/scraper.py
python data_pipeline/cleaner.py
```

### Day 2 — Embed & Index
```bash
python data_pipeline/embedder.py
# Test retrieval:
python rag/retriever.py
```

### Day 3 — Knowledge Graph
```bash
# Start Neo4j (Docker):
docker run -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/password neo4j:latest

python knowledge_graph/extractor.py
python knowledge_graph/graph_store.py
```

### Day 4 — Memory + Agent
```bash
# Start MongoDB:
docker run -p 27017:27017 mongo:latest

python agent/graph.py  # interactive CLI test
```

### Day 5-6 — Evaluation + API
```bash
python evaluation/eval.py
uvicorn api.main:app --reload
# API docs at http://localhost:8000/docs
```

## Architecture

```
User Query
    ↓
LangGraph Orchestrator
    ├── memory_node  → MongoDB (user prefs + facts)
    ├── rag_node     → FAISS vector search
    ├── graph_node   → Neo4j knowledge graph
    └── model_node   → Groq LLaMA 3 (answer generation)
         ↓
    FastAPI /chat endpoint
```

## Tech Stack
- **LLM**: Groq (LLaMA 3 70B) — free tier
- **Orchestration**: LangGraph
- **Vector DB**: FAISS
- **Graph DB**: Neo4j
- **Memory**: MongoDB
- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Evaluation**: RAGAS
- **API**: FastAPI
