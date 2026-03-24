# evaluate.py
#
# Evaluates the search pipeline against any BEIR dataset.
# Computes nDCG@10, MAP@10, MRR, Recall@10, and Precision@10
# using pytrec_eval (same library used by the BEIR research paper).
#
# Configure the dataset in config_eval.yaml:
#   dataset_name: "scifact"
#   dataset_url: "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
#
# Usage:
#   python evaluate.py                  (downloads dataset, indexes, evaluates)
#   python evaluate.py --skip-index     (skip indexing if already done)
#
# Install: pip install pytrec-eval-terrier

import os
import sys
import json
import zipfile
import urllib.request

# Suppress HF warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_VERBOSITY"] = "error"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import yaml
import pytrec_eval
from indexer.embedder import Embedder
from indexer.chunker import Chunker
from indexer.store import Store
from search.expander import QueryExpander
from search.dense import DenseSearch
from search.sparse import SparseSearch
from search.fusion import RRFFusion
from search.reranker import Reranker


# ──────────────────────────────────────────────
#  0. Load eval config
# ──────────────────────────────────────────────

EVAL_CONFIG = "config_eval.yaml"

def load_eval_config():
    """Load evaluation config and derive paths."""
    with open(EVAL_CONFIG, "r") as f:
        config = yaml.safe_load(f)

    dataset_name = config["dataset_name"]
    dataset_url = config["dataset_url"]
    dataset_dir = os.path.join("datasets", dataset_name)
    data_dir = config["data_dir"]

    return config, dataset_name, dataset_url, dataset_dir, data_dir


# ──────────────────────────────────────────────
#  1. Download & Load any BEIR dataset
# ──────────────────────────────────────────────

def download_dataset(dataset_name, dataset_url, dataset_dir):
    """Download and extract a BEIR dataset if not already present."""
    if os.path.exists(dataset_dir):
        print(f"{dataset_name} already downloaded at {dataset_dir}")
        return

    os.makedirs("datasets", exist_ok=True)
    zip_path = os.path.join("datasets", f"{dataset_name}.zip")

    print(f"Downloading {dataset_name} dataset...")
    urllib.request.urlretrieve(dataset_url, zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall("datasets")

    os.remove(zip_path)
    print(f"{dataset_name} ready at {dataset_dir}")


def load_corpus(dataset_dir):
    """Load corpus.jsonl → dict of {doc_id: {"title": ..., "text": ...}}"""
    corpus = {}
    with open(os.path.join(dataset_dir, "corpus.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line.strip())
            corpus[doc["_id"]] = {
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
            }
    print(f"Loaded {len(corpus)} documents from corpus")
    return corpus


def load_queries(dataset_dir):
    """Load queries.jsonl → dict of {query_id: query_text}"""
    queries = {}
    with open(os.path.join(dataset_dir, "queries.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line.strip())
            queries[q["_id"]] = q["text"]
    print(f"Loaded {len(queries)} queries")
    return queries


def load_qrels(dataset_dir):
    """
    Load qrels/test.tsv → dict of {query_id: {doc_id: relevance_score}}
    """
    qrels = {}
    qrels_path = os.path.join(dataset_dir, "qrels", "test.tsv")
    with open(qrels_path, "r", encoding="utf-8") as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split("\t")
            query_id, doc_id, score = parts[0], parts[1], int(parts[2])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = score
    print(f"Loaded qrels for {len(qrels)} queries")
    return qrels


# ──────────────────────────────────────────────
#  2. Index dataset into our pipeline
# ──────────────────────────────────────────────

def index_dataset(corpus, embedder, store, chunker, dataset_name):
    """
    Index a BEIR corpus directly into FAISS + SQLite.
    Each document is chunked, embedded, and stored with doc_id as the filepath
    to map search results back to BEIR document IDs.
    """
    print(f"\nIndexing {len(corpus)} documents from {dataset_name}...")

    doc_ids = list(corpus.keys())
    batch_size = 100
    total_chunks = 0

    for batch_start in range(0, len(doc_ids), batch_size):
        batch_ids = doc_ids[batch_start : batch_start + batch_size]
        all_chunks = []

        for doc_id in batch_ids:
            doc = corpus[doc_id]
            full_text = doc["title"] + " " + doc["text"] if doc["title"] else doc["text"]
            chunks = chunker.chunk_file(full_text, doc_id)
            all_chunks.extend(chunks)

        if not all_chunks:
            continue

        chunk_texts = [c["text"] for c in all_chunks]
        embeddings = embedder.embed_chunks(chunk_texts)
        store.add_chunks(all_chunks, embeddings)

        for doc_id in batch_ids:
            doc = corpus[doc_id]
            full_text = doc["title"] + " " + doc["text"]
            store.save_file_info(doc_id, "eval", len(chunker.chunk_text(full_text)))

        total_chunks += len(all_chunks)

        if (batch_start // batch_size + 1) % 10 == 0:
            print(f"  Indexed {batch_start + len(batch_ids)}/{len(doc_ids)} documents...")

    print(f"Indexing complete: {len(doc_ids)} documents → {total_chunks} chunks → {store.get_total_vectors()} vectors")


# ──────────────────────────────────────────────
#  3. Run Search
# ──────────────────────────────────────────────

def search_query(query_text, expander, dense_search, sparse_search, fusion, reranker, top_k=10):
    """Run a single query through the full search pipeline."""
    expanded = expander.expand_query(query_text)
    dense_results = dense_search.search(expanded, top_k=20)
    sparse_results = sparse_search.search(query_text, top_k=20)
    fused = fusion.fuse(dense_results, sparse_results)
    reranked = reranker.rerank(query_text, fused, top_k=top_k)
    return reranked


# ──────────────────────────────────────────────
#  4. Evaluate 
# ──────────────────────────────────────────────

def evaluate_pytrec(queries, qrels, expander, dense_search, sparse_search, fusion, reranker):
    """
    Run all queries and compute metrics using pytrec_eval.

    pytrec_eval expects:
      qrel = { "query_id": { "doc_id": int_relevance, ... }, ... }
      run  = { "query_id": { "doc_id": float_score, ... }, ... }

    It returns per-query metrics which we average.
    """
    eval_queries = {qid: text for qid, text in queries.items() if qid in qrels}
    total = len(eval_queries)

    print(f"\nEvaluating {total} queries...")

    # Build the run dict for pytrec_eval
    run = {}
    for i, (query_id, query_text) in enumerate(eval_queries.items()):
        results = search_query(
            query_text, expander, dense_search, sparse_search, fusion, reranker, top_k=10
        )

        # Convert results to {doc_id: score} format
        # Multiple chunks from the same doc → take max score
        doc_scores = {}
        for r in results:
            doc_id = str(r["filepath"])
            score = float(r["rerank_score"])
            if doc_id not in doc_scores or score > doc_scores[doc_id]:
                doc_scores[doc_id] = score

        run[query_id] = doc_scores

        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{total} queries...")

    # Define metrics to compute
    metrics_to_compute = {
        "ndcg_cut.10",       # nDCG@10
        "map_cut.10",        # MAP@10
        "recall.10",         # Recall@10
        "P.10",              # Precision@10
        "recip_rank",        # MRR
    }

    # Create evaluator and run evaluation
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics_to_compute)
    per_query_results = evaluator.evaluate(run)

    # Aggregate: compute mean across all queries
    metric_sums = {}
    for query_id, query_metrics in per_query_results.items():
        for metric_name, value in query_metrics.items():
            if metric_name not in metric_sums:
                metric_sums[metric_name] = 0.0
            metric_sums[metric_name] += value

    num_queries = len(per_query_results)
    metric_averages = {k: v / num_queries for k, v in metric_sums.items()}

    return metric_averages, per_query_results, run


# ──────────────────────────────────────────────
#  5. Manual metric implementations 
# ──────────────────────────────────────────────

# def compute_ndcg(ranked_doc_ids, qrel, k=10):
#     """
#     Compute nDCG@k (normalized Discounted Cumulative Gain).
#
#     This is the standard IR metric:
#     - DCG rewards relevant documents appearing higher in the ranking
#     - nDCG normalizes by the ideal DCG (perfect ranking)
#     - Score of 1.0 = perfect ranking, 0.0 = no relevant docs found
#     """
#     dcg = 0.0
#     for i, doc_id in enumerate(ranked_doc_ids[:k]):
#         rel = qrel.get(str(doc_id), 0)
#         dcg += rel / math.log2(i + 2)
#
#     ideal_rels = sorted(qrel.values(), reverse=True)[:k]
#     idcg = 0.0
#     for i, rel in enumerate(ideal_rels):
#         idcg += rel / math.log2(i + 2)
#
#     if idcg == 0:
#         return 0.0
#     return dcg / idcg


# def compute_recall(ranked_doc_ids, qrel, k=10):
#     """Recall@k: fraction of relevant docs that appear in top-k results."""
#     relevant = set(qrel.keys())
#     retrieved = set(str(d) for d in ranked_doc_ids[:k])
#     if len(relevant) == 0:
#         return 0.0
#     return len(relevant & retrieved) / len(relevant)


# def compute_precision(ranked_doc_ids, qrel, k=10):
#     """Precision@k: fraction of top-k results that are relevant."""
#     relevant = set(qrel.keys())
#     retrieved = [str(d) for d in ranked_doc_ids[:k]]
#     if len(retrieved) == 0:
#         return 0.0
#     hits = sum(1 for d in retrieved if d in relevant)
#     return hits / len(retrieved)


# def compute_mrr(ranked_doc_ids, qrel, k=10):
#     """
#     MRR (Mean Reciprocal Rank):
#     The reciprocal of the rank of the first relevant document.
#     """
#     relevant = set(qrel.keys())
#     for i, doc_id in enumerate(ranked_doc_ids[:k]):
#         if str(doc_id) in relevant:
#             return 1.0 / (i + 1)
#     return 0.0


# def compute_map(ranked_doc_ids, qrel, k=10):
#     """
#     MAP@k (Mean Average Precision):
#     Average of precision values computed at each rank where a relevant doc appears.
#     """
#     relevant = set(qrel.keys())
#     hits = 0
#     sum_precisions = 0.0
#     for i, doc_id in enumerate(ranked_doc_ids[:k]):
#         if str(doc_id) in relevant:
#             hits += 1
#             sum_precisions += hits / (i + 1)
#     if len(relevant) == 0:
#         return 0.0
#     return sum_precisions / len(relevant)


# ──────────────────────────────────────────────
#  6. Main
# ──────────────────────────────────────────────

def main():
    skip_index = "--skip-index" in sys.argv

    # Step 1: Load eval config
    config, dataset_name, dataset_url, dataset_dir, data_dir = load_eval_config()

    print(f"Dataset: {dataset_name}")
    print(f"URL: {dataset_url}")

    # Step 2: Download dataset
    download_dataset(dataset_name, dataset_url, dataset_dir)

    # Step 3: Load dataset
    corpus = load_corpus(dataset_dir)
    queries = load_queries(dataset_dir)
    qrels = load_qrels(dataset_dir)

    # Step 4: Initialize components
    print("\nInitializing models...")
    embedder = Embedder(EVAL_CONFIG)
    store = Store(EVAL_CONFIG)
    chunker = Chunker(
        chunk_size=config.get("chunk_size", 200),
        overlap=config.get("overlap", 30),
    )

    # Step 5: Index (unless skipped)
    if skip_index:
        store = Store(EVAL_CONFIG)
        if store.get_total_vectors() > 0:
            print(f"Skipping indexing — {store.get_total_vectors()} vectors already in index")
        else:
            skip_index = False  # no vectors found, need to index

    if not skip_index:
        import shutil
        if os.path.exists(data_dir):
            shutil.rmtree(data_dir)
        store = Store(EVAL_CONFIG)
        index_dataset(corpus, embedder, store, chunker, dataset_name)

    # Step 6: Initialize search components
    print("\nInitializing search pipeline...")
    expander = QueryExpander(max_synonyms_per_word=3)
    dense_search = DenseSearch.__new__(DenseSearch)
    dense_search.embedder = embedder
    dense_search.store = store

    sparse_search = SparseSearch(EVAL_CONFIG)
    fusion = RRFFusion(k=60)
    reranker = Reranker(EVAL_CONFIG)

    # Step 7: Evaluate using pytrec_eval
    metrics, per_query_results, run = evaluate_pytrec(
        queries, qrels, expander, dense_search, sparse_search, fusion, reranker
    )

    # Step 8: Print results
    print("\n" + "=" * 50)
    print(f"  EVALUATION RESULTS — {dataset_name} (pytrec_eval)")
    print("=" * 50)
    print(f"  Queries evaluated:  {len(per_query_results)}")
    print(f"  nDCG@10:            {metrics.get('ndcg_cut_10', 0):.4f}")
    print(f"  MAP@10:             {metrics.get('map_cut_10', 0):.4f}")
    print(f"  MRR:                {metrics.get('recip_rank', 0):.4f}")
    print(f"  Recall@10:          {metrics.get('recall_10', 0):.4f}")
    print(f"  Precision@10:       {metrics.get('P_10', 0):.4f}")
    print("=" * 50)

    # Step 9: Show some example results
    print("\n--- Example Queries ---\n")
    eval_queries = {qid: text for qid, text in queries.items() if qid in qrels}
    sample_queries = list(eval_queries.items())[:3]

    for query_id, query_text in sample_queries:
        results = search_query(
            query_text, expander, dense_search, sparse_search, fusion, reranker, top_k=5
        )

        qrel = qrels[query_id]
        relevant_docs = set(qrel.keys())

        q_metrics = per_query_results.get(query_id, {})

        print(f"Query [{query_id}]: {query_text[:80]}...")
        print(f"  Relevant docs: {relevant_docs}")
        print(f"  nDCG@10: {q_metrics.get('ndcg_cut_10', 0):.4f}  MAP@10: {q_metrics.get('map_cut_10', 0):.4f}  MRR: {q_metrics.get('recip_rank', 0):.4f}")
        for rank, r in enumerate(results, 1):
            is_relevant = "✅" if str(r["filepath"]) in relevant_docs else "❌"
            print(f"  {rank}. {is_relevant} doc:{r['filepath']} [{r['rerank_score']:.2f}] {r['chunk_text'][:60]}...")
        print()


if __name__ == "__main__":
    main()