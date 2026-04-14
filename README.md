# Semantic File Search Engine

An AI-powered file search engine that understands what your files are *about*, not just their names. Search for "budget report" and find `Q3_final_v2.xlsx` — even though the filename never mentions "budget."

Runs entirely offline on your machine. No cloud, no API calls, no data leaves your system.

## How It Works

**Indexing (background)**
```
File System → Crawler → Text Extractor → Chunker → Embedding Model → FAISS + SQLite
```

**Searching (on query)**
```
User Query → Query Expansion → Dense Search (FAISS) + Sparse Search (BM25) → RRF Fusion → Cross-Encoder Reranking → Results
```

## Features

- Hybrid retrieval: combines semantic understanding (FAISS) with keyword matching (BM25)
- Cross-encoder reranking for precision
- Supports PDF, DOCX, PPTX, XLSX, IPYNB, TXT, MD, PY, JS
- Real-time file monitoring with watchdog
- Incremental indexing — only re-processes changed files
- Query expansion via WordNet synonyms
- Fully offline and privacy-first

## Setup and Usage

```bash
git clone https://github.com/yourusername/semantic-search.git
cd semantic-search
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac
pip install -r requirements.txt
```

Edit `config.yaml` to set your directories then run:

```bash
python main.py
```

## Tech Stack

- **Embeddings**: sentence-transformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (IndexIDMap2 + IndexFlatL2)
- **Metadata Store**: SQLite
- **Lexical Search**: BM25 (rank-bm25)
- **Reranking**: cross-encoder (ms-marco-MiniLM-L-6-v2)
- **Query Expansion**: NLTK WordNet
- **File Monitoring**: watchdog
- **Text Extraction**: PyMuPDF, python-docx, python-pptx, openpyxl
- **Evaluation**: pytrec-eval-terrier

## References

- [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663) (NeurIPS 2021)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [MS MARCO: A Human Generated MAchine Reading COmprehension Dataset](https://arxiv.org/abs/1611.09268)

## License

MIT
