# 🧠 RAG-Based Financial News Question Answering System

This project implements a lightweight Retrieval-Augmented Generation (RAG) pipeline for answering finance-related questions based on recent Hong Kong news articles.

Built with:
- `FastAPI` for API serving
- `FAISS` for dense vector retrieval
- `BAAI/bge-m3` as a bi-encoder for document embeddings
- `cross-encoder/ms-marco-MiniLM-L-6-v2` for re-ranking
- `Qwen/Qwen1.5-0.5B-Chat` as the LLM for answer generation

---

## 🔧 Features

- ✅ Semantic search using FAISS and bi-encoder embeddings
- ✅ Cross-encoder re-ranking for better relevance
- ✅ LLM response generation with prompt control
- ✅ FastAPI backend for deployment and testing
- ✅ Chinese-language question handling supported

---

## 🚀 How It Works

1. News articles are pre-scraped and embedded using `bge-m3`, then stored in a FAISS index.
2. When a user asks a question, the query is embedded and used to search the top-k relevant news via FAISS.
3. The top-k results are re-ranked using a cross-encoder for better semantic match.
4. The top result(s) are inserted into a prompt template and sent to Qwen for final answer generation.
5. A response is returned via `/ask` endpoint.

---

## 📦 API Usage

### POST `/ask`

**Body:**

```json
{
  "query": "五一黃金週有多少人來港？"
}
