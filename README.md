# RAG Evaluation

Compares three RAG approaches on a Saudi Aramco knowledge base and evaluates them with [RAGAS](https://docs.ragas.io) metrics.

**Research question:** Can better RAG techniques improve answer quality even with suboptimal data quality?

## Approaches

| Approach | Description |
|---|---|
| Simple RAG | Direct pgvector cosine similarity search + OpenAI |
| LangChain RAG | LangChain LCEL chain with custom pgvector retriever |
| Haystack Semantic | Haystack pipeline with Elasticsearch embedding retrieval |
| Haystack BM25 | Haystack pipeline with Elasticsearch keyword (BM25) retrieval |

## Results (RAGAS metrics, 0–1)

| Approach | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Average |
|---|---|---|---|---|---|
| LangChain RAG | 1.000 | 0.921 | 0.823 | 0.850 | **0.899** |
| Simple RAG | 0.971 | 0.920 | 0.824 | 0.850 | 0.891 |
| Haystack Semantic | 0.879 | 0.888 | 0.825 | 0.850 | 0.861 |
| Haystack BM25 | 0.872 | 0.857 | 0.704 | **0.950** | 0.846 |

BM25 leads on context recall (finds more relevant chunks) but lags on precision. Semantic and pgvector approaches score more consistently across all metrics.

## Stack

- **Vector DB:** PostgreSQL + pgvector
- **Search:** Elasticsearch 8.x (BM25 + embedding)
- **Embeddings / LLM:** OpenAI (`text-embedding-3-small`, `gpt-4o-mini`)
- **Frameworks:** Haystack 2.x, LangChain
- **Evaluation:** RAGAS

## Setup

**Prerequisites:** Python 3.13+, Docker, PostgreSQL with pgvector, OpenAI API key.

```bash
# Install dependencies
pip install uv
uv sync

# Start Elasticsearch
docker compose up -d

# Configure environment
cp .env.example .env  # fill in DB_URL and OPENAI_API_KEY
```

Then open and run `src/notebooks/rag_evaluation.ipynb` top to bottom.

## Project structure

```
src/
  data/                    # Source knowledge base (Saudi Aramco)
  notebooks/
    rag_evaluation.ipynb   # Main experiment notebook
    ragas_summary.csv      # Aggregated metric scores
    ragas_detail_*.csv     # Per-question scores per approach
```
