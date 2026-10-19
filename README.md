# RAG Assignment: Baseline vs. Re-Ranking

Two retrieval pipelines over the same chunked **Magic: The Gathering Comprehensive Rules** (19 September 2025):

- **Pipeline 1:** Bi-Encoder (`all-MiniLM-L6-v2`) → Top 10 cosine hits → Top 3 → LLM
- **Pipeline 2:** same Top 10 → Cross-Encoder (`ms-marco-MiniLM-L-6-v2`) → new Top 3 → same LLM prompt

Submit `rag_baseline_vs_rerank.ipynb`. Tables for the report are written to `outputs/` (`/kaggle/working/outputs` on Kaggle).

## Run on Kaggle (GPU)

1. New notebook → **Accelerator: GPU T4**, **Internet: On**.
2. File → Import notebook → `rag_baseline_vs_rerank.ipynb`.
3. Run All.

The notebook installs `pypdf`, `sentence-transformers`, and `faiss-cpu`, then downloads the official rules PDF. Embeddings, the Cross-Encoder, and `Qwen/Qwen2.5-1.5B-Instruct` all use the GPU. No API key is required.

After the run, download `outputs/rank_shifts.csv`, `outputs/latency.csv`, and `outputs/run_trace.json` for the written report.

## Report

Write the report yourself from those outputs. The brief asks for a non-AI report covering architecture, rank shifts, latency, and why a Cross-Encoder is not run over the full collection.


GITHUB: https://github.com/nradhakrishna/rag-assignment