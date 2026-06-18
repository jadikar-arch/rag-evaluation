# RAG Evaluation: Comparing Retrieval Approaches on a Real Knowledge Base

A hands-on research project comparing how different retrieval strategies affect answer quality in a Retrieval-Augmented Generation (RAG) system. We're testing whether better retrieval techniques can improve results even when working with imperfect data.

**The main question:** Can smarter retrieval compensate for data quality issues?

---

## What This Project Does

We take a Saudi Aramco knowledge base, split it into chunks, and run it through four different retrieval setups:

1. **Simple RAG** — Just embed the query and search pgvector for the most similar chunks
2. **Haystack Semantic** — Use Elasticsearch with embeddings (dense retrieval)
3. **Haystack BM25** — Use Elasticsearch keyword matching (sparse retrieval)
4. **Haystack Hybrid** — Combine both dense and BM25 results using rank fusion (RRF)

Then we ask the same 10 questions to all four setups (using the same LLM for generation to keep things fair) and score the answers using RAGAS metrics to see which retrieval strategy works best.

---

## Quick Overview

| Component | Tech | Purpose |
|-----------|------|---------|
| **Vector Database** | PostgreSQL + pgvector | Store and search embeddings |
| **Search Engine** | Elasticsearch 8.x | BM25 keyword search + dense retrieval |
| **Embeddings & LLM** | OpenAI (text-embedding-3-small, gpt-4o-mini) | Create vectors and generate answers |
| **Evaluation** | RAGAS + gpt-5.5 | Score answer quality |
| **Framework** | Haystack 2.x | Orchestrate the retrieval pipelines |

---

## Setup

### Prerequisites

- Python 3.13+
- Docker (for Elasticsearch & PostgreSQL)
- OpenAI API key
- ~10GB disk space

### Installation Steps

1. **Clone and navigate:**
   ```bash
   cd /Users/munirdin/Documents/projects/rag-evaluation
   ```

2. **Install dependencies:**
   ```bash
   pip install uv
   uv sync
   ```

3. **Start services:**
   ```bash
   docker compose up -d
   ```

4. **Configure environment:**
   ```bash
   cp .env.example .env
   # Edit .env and add:
   # - DB_URL (PostgreSQL connection)
   # - OPENAI_API_KEY
   ```

5. **Initialize pgvector extension:**
   ```bash
   docker compose exec postgres psql -U postgres -d rag_evaluation \
     -c "CREATE EXTENSION IF NOT EXISTS vector;"
   ```

### Troubleshooting

- **Can't connect to Elasticsearch?** — Wait 30 seconds, it takes time to start. Check `docker compose ps`.
- **pgvector extension not found?** — Make sure you ran the extension creation command above.
- **Ports already in use?** — Edit `docker-compose.yml` and change the port mappings.

---

## Running the Evaluation

Open `src/notebooks/rag_evaluation_clean_text.ipynb` and run the cells top to bottom:

1. **Setup & Configuration** — Load env vars, connect to databases
2. **Data Exploration** — Look at the knowledge base chunks
3. **Golden Test Set** — Generate 10 Q&A pairs using gpt-5.5
4. **Simple RAG** — Run pgvector retrieval + OpenAI generation
5. **Haystack Semantic** — Run Elasticsearch embedding search
6. **Haystack BM25** — Run Elasticsearch keyword search
7. **Haystack Hybrid** — Run both retrievers + rank fusion
8. **RAGAS Evaluation** — Score all answers on 4 metrics
9. **Visualization** — Charts comparing the approaches

The notebook generates CSV files with results:
- `ragas_summary.csv` — Average scores for each approach
- `ragas_detail_*.csv` — Per-question scores for debugging

---

## What the Code Actually Does

### 1. Loading & Chunking Knowledge Base
Reads the Saudi Aramco knowledge base from `src/data/aramco_knowledge_base.txt`, stores it in PostgreSQL with embeddings, and indexes it in Elasticsearch. The chunks are already in the database table `document_chunks_v2`.

### 2. Generating a Golden Test Set
Uses gpt-5.5 to create 10 diverse Q&A pairs from the knowledge base. The prompts are carefully worded to ask questions in a different way than they appear in the docs (to avoid accidentally making keyword search look better than it is). This golden set is cached to disk so you don't regenerate it on every run.

### 3. Running Four Retrieval Approaches
For each of the 10 questions:

**Simple RAG:**
- Embed query → Search pgvector with cosine similarity → Get top 10 chunks
- Send chunks + question to gpt-4o-mini → Get answer

**Haystack Semantic:**
- Embed query → Search Elasticsearch for most similar embeddings → Get top 10
- Send to gpt-4o-mini with template → Get answer

**Haystack BM25:**
- Run query through Elasticsearch BM25 (keyword matching) → Get top 10
- Send to gpt-4o-mini with template → Get answer

**Haystack Hybrid:**
- Run both semantic AND BM25 in parallel
- Fuse results using Reciprocal Rank Fusion (RRF) → Keep top 10
- Send to gpt-4o-mini with template → Get answer

**Note:** Generation is held constant (same model, same prompt, temperature=0) so differences in scores come purely from which chunks were retrieved.

### 4. Evaluating with RAGAS
Uses gpt-5.5 as the judge (separate from the gpt-4o-mini that generated answers) to score:

- **Faithfulness** — Is the answer based on the retrieved context, or is it making stuff up?
- **Answer Relevancy** — Does it actually answer the question?
- **Context Precision** — Are the retrieved chunks actually relevant?
- **Context Recall** — Did it retrieve chunks that cover the ground truth answer?

---

## Understanding the Metrics

After running the notebook, here are the results from the latest run:

| Approach | Faithfulness | Answer Relevancy | Context Precision | Context Recall | Average |
|---|---:|---:|---:|---:|---:|
| Haystack Semantic | 0.866884 | 0.815824 | 0.614963 | 0.7 | 0.749418 |
| Simple RAG | 0.862591 | 0.813795 | 0.608901 | 0.7 | 0.746322 |
| Haystack Hybrid | 0.759834 | 0.795360 | 0.534923 | 0.7 | 0.697529 |
| Haystack BM25 | 0.771184 | 0.733620 | 0.460476 | 0.5 | 0.616320 |

(Numbers come from the notebook `src/notebooks/rag_evaluation_clean_text.ipynb`.)

### What Each Score Means

**Faithfulness (0-1):** Does the LLM actually use the retrieved context, or is it hallucinating? A score of 0.95 means 95% of the answer is grounded in the documents.

**Answer Relevancy (0-1):** Does the answer actually address the question asked? A score of 0.90 means the answer is mostly on-topic.

**Context Precision (0-1):** What percentage of the retrieved chunks are actually useful? If you retrieve 10 chunks and 7 are relevant, that's ~0.70.

**Context Recall (0-1):** Did you retrieve the chunks needed to answer? Hard to measure perfectly, but the evaluator tries to estimate how much of the ground truth is covered by what was retrieved.

---

## Important Limitations & Caveats

1. **Only 10 test questions** — This is a tiny evaluation set. You should probably have 30-50+ for solid conclusions. Results can be noisy with such a small sample.

2. **Single run** — We run each question once. LLM judges aren't perfectly deterministic, so you might get slightly different scores if you re-run. For real research, run it multiple times and report mean ± standard deviation.

3. **Can't directly answer "does better RAG fix bad data?"** — To really answer that question, you'd need to:
   - Run the evaluation on the original messy data
   - Clean/re-chunk it
   - Run again
   - Compare the deltas
   
   This notebook only tests different retrieval methods on one fixed corpus.

4. **Evaluation judge might be biased** — We use gpt-5.5 to score, which is what all the RAG approaches also reference. A truly independent evaluator would be more trustworthy, but cost-prohibitive for this project.

5. **Generation is held constant** — We use the same LLM for all approaches so retrieval is the only variable. But in practice, you might tune prompts differently per approach, which could change results.

---

## Project Layout

```
rag-evaluation/
├── README.md                              ← You are here
├── docker-compose.yml                     ← Elasticsearch + PostgreSQL config
├── pyproject.toml                         ← Python dependencies
├── .env.example                           ← Copy to .env and fill in secrets
│
├── src/
│   ├── data/
│   │   └── aramco_knowledge_base.txt      ← The knowledge base (raw text)
│   │
│   └── notebooks/
│       ├── rag_evaluation_clean_text.ipynb  ← ⭐ Main notebook — run this
│       ├── embed_to_openwebui_v2.ipynb      ← (Not part of main flow)
│       ├── golden_set.json                  ← Test questions (auto-generated)
│       │
│       └── Results (generated after running):
│           ├── ragas_summary.csv
│           ├── ragas_detail_simple_rag.csv
│           ├── ragas_detail_haystack_semantic.csv
│           ├── ragas_detail_haystack_bm25.csv
│           └── ragas_detail_haystack_hybrid.csv
```

---

## What's in the Knowledge Base?

The file `src/data/aramco_knowledge_base.txt` contains real Saudi Aramco documentation — things like:
- Technical specifications for equipment
- Safety procedures
- Operational guidelines
- Business information

It's moderately messy (inconsistent formatting, some OCR artifacts) which is intentional — we wanted to test whether better retrieval can work even with imperfect source data.

---

## Interpreting Results

After the notebook runs, look for patterns:

- **Is one approach consistently better?** → That retrieval method probably handles your specific knowledge base better.
- **Does hybrid outperform both semantic and BM25?** → Rank fusion is working; the approaches are capturing different relevant chunks.
- **Does BM25 score low on context precision but high on recall?** → It's retrieving more chunks, but some are off-topic. That could be good (find everything) or bad (noisy context) depending on your use case.
- **Does semantic do well on answer relevancy?** → Embeddings are good at understanding paraphrased questions.

---

## Tips for Extending This

1. **Larger test set** — Add more Q&A pairs to `golden_set.json` manually or by adjusting the golden set generation prompt.

2. **Test on degraded data** — Copy the knowledge base, intentionally make it messier, re-chunk it, and run the same evaluation to see if retrieval improvements can overcome data quality drops.

3. **Try different Top-K** — Change `TOP_K = 10` in the notebook to 5 or 20 and see how context recall/precision trade off.

4. **Add re-ranking** — After hybrid retrieval, feed the 10 chunks through a cross-encoder to re-rank them. Could improve precision.

5. **Query expansion** — Try generating multiple variants of the question and retrieving for each, then deduplicating. (Technique called HyDE or multi-query.)

6. **Custom prompts** — Tweak the `SYSTEM_PROMPT` to see if better grounding instructions improve faithfulness.

---

## Troubleshooting

**Notebook hangs on "Evaluating: ..."**
- RAGAS evaluation is slow because it calls the LLM many times. First run on 10 questions takes ~5-10 minutes. Grab coffee. ☕

**All RAGAS scores are NaN**
- Check that `EVAL_MODEL` (gpt-5.5) is available and your API key is valid.
- Check that retrieved contexts are actually being passed correctly.

**Database connection refused**
- Make sure PostgreSQL is running: `docker compose ps`
- Check `DB_URL` in `.env` is correct.

**Elasticsearch not reachable**
- Make sure Elasticsearch is running: `docker compose ps`
- Wait 30 seconds after starting; it takes time to initialize.

**Out of memory during evaluation**
- The notebook loads a lot of data. Close other apps or increase Docker's memory limit.

**Golden set.json not regenerating**
- The notebook caches it to avoid re-running expensive generation. Delete `golden_set.json` to regenerate.

---

## References & Further Reading

- [RAGAS: Automated Evaluation Framework](https://arxiv.org/abs/2309.15217) — The metrics we use
- [Haystack Documentation](https://docs.haystack.deepset.ai/) — Framework and pipeline docs
- [Elasticsearch BM25](https://www.elastic.co/guide/en/elasticsearch/reference/current/index.html) — Keyword search
- [PostgreSQL pgvector](https://github.com/pgvector/pgvector) — Vector database
- [OpenAI Embeddings API](https://platform.openai.com/docs/models/embeddings) — Embedding model docs

---

## Who Built This

Munirdin Jadikar — June 2026


