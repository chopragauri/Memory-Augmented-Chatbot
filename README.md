# Memory-Augmented Chatbot with Knowledge Graph and Hybrid RAG

A domain-specific AI assistant for **Machine Learning & AI** topics, built with a full hybrid RAG pipeline, LangGraph orchestration, Supabase-backed memory, and Google OAuth — developed as a Celebal Technologies internship project.

---

## Domain

**Machine Learning & Artificial Intelligence** — the knowledge base was built by scraping 15 Wikipedia articles:
Transformers, BERT, GPT-3, RAG, GANs, CNNs, RNNs, Word2Vec, Reinforcement Learning, Knowledge Graphs, Attention Mechanisms, Large Language Models, NLP, Deep Learning, Machine Learning.

---

## Architecture

```
User Query
    ↓
LangGraph StateGraph Orchestrator
    ├── memory_node  → Supabase (long-term user facts + preferences)
    ├── rag_node     → FAISS vector search (sentence-transformers, 163 chunks)
    ├── graph_node   → networkx Knowledge Graph (762 triples, offline)
    └── model_node   → Groq API — LLaMA 3.3 70B (cloud inference)
                    → Auto memory extraction (facts saved back to Supabase)
         ↓
    FastAPI /chat → index.html (3-panel UI)
```

---

## Tech Stack (Actual)

| Component | Tool | Notes |
|---|---|---|
| LLM | LLaMA 3.3 70B via **Groq API** | Cloud inference, free tier |
| Orchestration | **LangGraph** StateGraph | 4-node pipeline |
| Vector Search | **FAISS** + sentence-transformers | all-MiniLM-L6-v2, 384-dim, local |
| Knowledge Graph | **networkx** DiGraph | 762 triples, offline (no Neo4j) |
| Memory / History | **Supabase** PostgreSQL | user_memory + chat_history + sessions |
| Auth | **Supabase Auth** + Google OAuth | Guest mode also supported |
| Embeddings | sentence-transformers | all-MiniLM-L6-v2 |
| Data Pipeline | Wikipedia API scraper | 15 articles → 163 chunks |
| NLP / KG extraction | **spaCy** en_core_web_sm | dependency parsing, offline |
| Evaluation | Cosine similarity | context_relevance + context_recall |
| API | **FastAPI** | /chat, /sessions, /memory endpoints |
| UI | Vanilla HTML/CSS/JS | ChatGPT-style 3-panel layout |

---

## Data Pipeline

```
Wikipedia API scrape (15 articles)
    → clean & chunk (≤500 tokens)           163 chunks
    → embed (sentence-transformers)          384-dim vectors → FAISS index
    → NLP triple extraction (spaCy)          762 (subject, relation, object) triples
    → networkx KG                            1200 nodes, 760 edges
    → Supabase                               sessions, chat_history, user_memory tables
```

---

## Features

- **Hybrid RAG**: FAISS semantic search + Knowledge Graph triples combined in every prompt
- **Long-term memory**: User facts auto-extracted from conversation and persisted in Supabase
- **Session history**: ChatGPT-style sidebar — New Chat, session list, click to reload, delete
- **Google OAuth + Guest mode**: Sign in with Google for personal memory; guest mode for anonymous use
- **Pipeline visualization**: Real-time 4-step animation (Memory → RAG → KG → LLM)
- **Retrieval metrics**: Cosine-similarity-based Relevance and Recall shown per response
- **Domain-specific**: Trained knowledge base on ML/AI Wikipedia corpus

---

## Setup

```bash
pip install -r requirements.txt
python3 -m spacy download en_core_web_sm
cp .env.example .env   # add GROQ_API_KEY, SUPABASE_URL, SUPABASE_KEY
```

### Run (from project root)
```bash
python3 -m uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000
```

### Re-build data pipeline (optional — pre-built indexes included)
```bash
python3 -m data_pipeline.scraper
python3 -m data_pipeline.cleaner
python3 -m data_pipeline.embedder
python3 -m knowledge_graph.extractor
python3 -m knowledge_graph.graph_store
```

---

## Supabase Tables Required

```sql
-- Sessions
create table sessions (
  id text primary key,
  user_id text not null,
  title text default 'New Chat',
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Chat history
create table chat_history (
  id bigserial primary key,
  session_id text not null,
  user_id text,
  role text not null,
  content text not null,
  created_at timestamptz default now()
);

-- User memory
create table user_memory (
  user_id text primary key,
  facts jsonb default '[]',
  preferences jsonb default '[]',
  last_seen timestamptz
);
```

---

## Evaluation

Evaluation uses cosine similarity (no external API needed):
- **Context Relevance** — similarity between retrieved chunks and the query
- **Context Recall** — similarity between the answer and retrieved chunks

Run: `python3 -m evaluation.eval`

### Results (7 domain queries)

| Query | Relevance | Recall | Top Source |
|---|---|---|---|
| What is RAG? | 0.430 | 0.685 | Retrieval-augmented generation |
| How does transformer attention work? | 0.503 | 0.491 | Transformer (deep learning) |
| What is a knowledge graph? | 0.649 | 0.715 | Knowledge graph |
| What is Word2Vec? | 0.639 | 0.737 | Word2vec |
| What is reinforcement learning? | 0.581 | 0.696 | Reinforcement learning |
| What is deep learning? | 0.583 | 0.603 | Deep learning |
| What are GANs? | 0.486 | 0.544 | Generative adversarial network |
| **Average** | **0.553** | **0.639** | |

> Metrics computed using cosine similarity between sentence-transformer embeddings of query, retrieved chunks, and generated answer — no external evaluation API required.

---

## Problem Statement Compliance

| Requirement | Status |
|---|---|
| Memory-augmented chatbot | ✅ Supabase long-term memory + auto-extraction |
| Knowledge Graph | ✅ spaCy + networkx, 762 triples |
| Hybrid RAG | ✅ FAISS vector search + KG context combined |
| LangGraph agent | ✅ 4-node StateGraph |
| FastAPI backend | ✅ /chat, /sessions, /memory |
| Evaluation metrics | ✅ Cosine similarity relevance + recall |
| UI | ✅ 3-panel ChatGPT-style with pipeline visualization |
| Domain-specific KB | ✅ 15 ML/AI Wikipedia articles, web-scraped |

---

## Screenshots

<!-- Add screenshots here after recording -->

### Login Screen (Google OAuth + Guest Mode)
<!-- ![Login](assets/login.png) -->

### Chat with Pipeline Visualization
<!-- ![Chat](assets/chat.png) -->

### Session History Sidebar
<!-- ![Sessions](assets/sessions.png) -->

### User Memory (different accounts)
<!-- ![Memory](assets/memory.png) -->

### Evaluation Results
<!-- ![Eval](assets/eval.png) -->
